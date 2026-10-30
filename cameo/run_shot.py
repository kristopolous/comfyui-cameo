"""Put the cameo into one window of one shot, end to end, at the LOWEST resolution.

    python -m cameo.run_shot work/nd 10 --person work/gpu/standin_00001_.png --box 380,640,900,770 \
        --where "dancing on the stage to the right of the dancer" --action "dances energetically next to him"

1. Klein inserts the person into the window's first frame (box record, 1-2 s).
2. LTX-2.5 regenerates only the box across the window, anchored on that frame (CameoBoxNoiseMask).
3. The box region is pasted back over the ORIGINAL frames with a feathered mask; nothing else comes from LTX.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter

from cameo import comfy, graphs

WINDOW = 25


def frames(clip, start, n, dest):
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vf", f"select=gte(n\\,{start})",
                    "-fps_mode", "passthrough", "-frames:v", str(n), str(dest / "%03d.png")], check=True)
    return sorted(dest.glob("*.png"))


def encode(pattern, out, fps, audio=None, start_s=0.0, dur_s=None):
    cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(pattern)]
    if audio:
        cmd += ["-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", str(audio), "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]
    subprocess.run(cmd + ["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out)], check=True)


def box_mask(size, box, feather=4, grow=0.04):
    """Feathered rectangle for a [y0, x0, y1, x1] 0-1000 box, grown a little to cover limbs."""
    w, h = size
    y0, x0, y1, x1 = [v / 1000 for v in box]
    gy, gx = grow, grow
    rect = (max(0, (x0 - gx) * w), max(0, (y0 - gy) * h), min(w, (x1 + gx) * w), min(h, (y1 + gy) * h))
    m = Image.new("L", size, 0)
    m.paste(255, [round(v) for v in rect])
    return m.filter(ImageFilter.GaussianBlur(feather))


def masks(paths):
    return [Image.open(p).convert("L") for p in sorted(paths) if str(p).endswith(".png")]


def fill_gaps(ms, ratio=0.5):
    """Detection flickers: a frame whose mask area drops below ratio x median borrows its nearest good neighbors."""
    area = [sum(m.point(lambda v: 255 if v > 127 else 0).histogram()[255:]) for m in ms]
    med = sorted(area)[len(area) // 2] or 1
    good = [a >= ratio * med for a in area]
    out = []
    for i, m in enumerate(ms):
        if good[i]:
            out.append(m)
            continue
        prev = next((ms[j] for j in range(i - 1, -1, -1) if good[j]), None)
        nxt = next((ms[j] for j in range(i + 1, len(ms)) if good[j]), None)
        nb = [x for x in (prev, nxt) if x is not None]
        out.append(ImageChops.lighter(*nb) if len(nb) == 2 else (nb[0] if nb else m))
    return out


def composite(orig, gen, newp, people, region_mask, size, dest):
    W, H = size
    dest.mkdir(exist_ok=True)
    cut = []
    for i in range(len(orig)):
        m = ImageChops.multiply(newp[min(i, len(newp) - 1)].resize((W, H)), region_mask)
        cut.append(ImageChops.subtract(m, people[min(i, len(people) - 1)].resize((W, H))))
    for i, (f, m) in enumerate(zip(orig, fill_gaps(cut))):
        base = Image.open(f).convert("RGB")
        src = Image.open(gen[min(i, len(gen) - 1)]).convert("RGB").resize((W, H))
        m = m.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1))
        Image.composite(src, base, m).save(dest / f"{i + 1:03d}.png")


def run(work, shot_id, person, box, where, action, desc, scene_prompt, seed=1, region=None, seeds=(1,)):
    region = region or box
    work = Path(work)
    meta = json.load(open(work / "shots.json"))
    shot = meta["shots"][shot_id]
    fps, W, H = meta["fps"], meta["width"], meta["height"]
    n = min(WINDOW, shot["frames"] - (shot["frames"] - 1) % 8)  # 8n+1 frames for LTX
    out = work / f"shot{shot_id:04d}"
    shutil.rmtree(out, ignore_errors=True)
    orig = frames(meta["clip"], shot["start"], n, out / "orig")

    # 1. Klein: the person into frame 0
    scene = comfy.upload(orig[0], f"s{shot_id}_f0.png")
    me = comfy.upload(person, "person.png")
    record = graphs.insert_record(box, desc, where)
    (p,), t1 = comfy.run(graphs.klein_edit(record, [scene, me], W, H, seed=seed, prefix=f"cameo/s{shot_id}_klein"), out)
    first = Image.open(p).convert("RGB").resize((W, H))

    # 2. LTX: carry the box through the window
    win = out / "win"
    win.mkdir()
    for i, f in enumerate(orig):
        (first if i == 0 else Image.open(f).convert("RGB")).save(win / f"{i + 1:03d}.png")
    encode(win / "%03d.png", out / "window.mp4", fps)
    vid = comfy.upload(out / "window.mp4", f"s{shot_id}_window.mp4")
    prompt = f"{scene_prompt} {desc} He {action}."
    raw = work / "raw.mp4" if (work / "raw.mp4").exists() else None
    start_s, dur_s = shot["start"] / fps, n / fps
    encode(orig[0].parent / "%03d.png", out / "orig.mp4", fps, raw, start_s, dur_s)
    ov = comfy.upload(out / "orig.mp4", f"s{shot_id}_orig.mp4")
    people, _ = comfy.run(graphs.sam3_mask(ov, prefix=f"cameo/s{shot_id}_people"), out / "people")
    people = masks(people)
    region_mask = box_mask((W, H), region, feather=0, grow=0.0)
    t1s = []
    for sd in seeds:
        gen, t2 = comfy.run(graphs.ltx_box_inpaint(vid, [region], prompt, frames=n, fps=fps, seed=sd,
                                                   prefix=f"cameo/s{shot_id}_ltx{sd}"), out / f"ltx{sd}")
        gen = sorted(g for g in gen if str(g).endswith(".png"))
        encode(out / f"ltx{sd}" / f"{Path(gen[0]).name.rsplit('_', 2)[0]}_%05d_.png", out / f"ltx{sd}.mp4", fps)
        gv = comfy.upload(out / f"ltx{sd}.mp4", f"s{shot_id}_ltx{sd}.mp4")
        newp, _ = comfy.run(graphs.sam3_mask(gv, threshold=0.25, prefix=f"cameo/s{shot_id}_new{sd}"), out / f"new{sd}")
        newp = masks(newp)
        comp = out / f"comp{sd}"
        composite(orig, gen, newp, people, region_mask, (W, H), comp)
        encode(comp / "%03d.png", out / f"cameo{sd}.mp4", fps, raw, start_s, dur_s)
        t1s.append(t2)
    t2 = t1s
    print(f"klein {t1}s, ltx {t2}s -> {out / 'cameo.mp4'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("shot", type=int)
    ap.add_argument("--person", required=True)
    ap.add_argument("--box", required=True, help="y0,x0,y1,x1 on 0-1000")
    ap.add_argument("--where", required=True)
    ap.add_argument("--action", default="moves naturally")
    ap.add_argument("--desc", default="A man in his thirties with light skin, short dark hair and a short dark beard, "
                                       "wearing a navy hoodie, blue jeans and white sneakers.")
    ap.add_argument("--scene", default="A static wide shot.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--region", help="y0,x0,y1,x1 the video model may change (default: --box)")
    ap.add_argument("--seeds", default="1", help="comma-separated LTX seeds")
    a = ap.parse_args()
    run(a.work, a.shot, a.person, [int(v) for v in a.box.split(",")], a.where, a.action, a.desc, a.scene, a.seed,
        [int(v) for v in a.region.split(",")] if a.region else None, [int(v) for v in a.seeds.split(",")])
