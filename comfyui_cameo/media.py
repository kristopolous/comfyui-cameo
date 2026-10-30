"""Fetching footage, splitting shots, and putting the generated cameo back into the original video."""
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction

import torch
import torch.nn.functional as F

import folder_paths
from comfy_api.latest import InputImpl, Types, io

from .cast import rate_take


def _run(cmd):
    env = dict(os.environ, PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-2000:]}")


def _shots(path, threshold, min_len):
    from scenedetect import ContentDetector, detect
    scenes = detect(path, ContentDetector(threshold=threshold, min_scene_len=min_len))
    return [[a.frame_num, b.frame_num] for a, b in scenes]


class CameoFetch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CameoFetch",
            display_name="Cameo Fetch Video",
            category="cameo",
            description="Download a video section from a URL (YouTube or anything yt-dlp reads) at the LOWEST "
                        "resolution, normalize it, and split it into shots.",
            inputs=[
                io.String.Input("url", default="https://www.youtube.com/watch?v=WGdc92EOdm0"),
                io.String.Input("start", default="", tooltip="Optional start time, e.g. 0:30."),
                io.String.Input("end", default="", tooltip="Optional end time, e.g. 1:10."),
                io.Int.Input("height", default=192, min=96, max=1080, step=32, tooltip="Keep it tiny while iterating."),
                io.Int.Input("fps", default=12, min=6, max=30),
                io.Float.Input("cut_threshold", default=15.0, min=1.0, max=100.0, tooltip="Shot-cut sensitivity."),
            ],
            outputs=[io.Video.Output(display_name="video"), io.String.Output(display_name="shots")],
        )

    @classmethod
    def execute(cls, url, start, end, height, fps, cut_threshold) -> io.NodeOutput:
        work = tempfile.mkdtemp(prefix="cameo_", dir=folder_paths.get_temp_directory())
        raw = os.path.join(work, "raw.mp4")
        cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", "worstvideo[height>=144]+worstaudio/worst[height>=144]/worst",
               "--merge-output-format", "mp4", "-o", raw, "--force-overwrites"]
        if start or end:
            cmd += ["--download-sections", f"*{start or '0'}-{end or 'inf'}"]
        _run(cmd + [url])
        clip = os.path.join(work, "clip.mp4")
        _run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-vf", f"fps={fps},scale=-2:{height}:flags=area,"
              f"crop=trunc(iw/32)*32:{height}", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
              "-c:a", "aac", clip])
        shots = _shots(clip, cut_threshold, 6)
        return io.NodeOutput(InputImpl.VideoFromFile(clip), json.dumps(shots))


class CameoReplaceFirstFrame(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CameoReplaceFirstFrame",
            display_name="Cameo Replace First Frame",
            category="cameo",
            inputs=[io.Image.Input("frames"), io.Image.Input("first")],
            outputs=[io.Image.Output(display_name="frames")],
        )

    @classmethod
    def execute(cls, frames, first) -> io.NodeOutput:
        first = first[:1].movedim(-1, 1)
        first = F.interpolate(first, size=frames.shape[1:3], mode="bilinear", align_corners=False).movedim(1, -1)
        out = frames.clone()
        out[0] = first[0].to(out.dtype)
        return io.NodeOutput(out)


def _box_mask(box, h, w, device):
    y0, x0, y1, x1 = box
    m = torch.zeros((h, w), device=device)
    m[int(y0 * h / 1000):max(int(y0 * h / 1000) + 1, int(y1 * h / 1000)),
      int(x0 * w / 1000):max(int(x0 * w / 1000) + 1, int(x1 * w / 1000))] = 1.0
    return m


def _fill_gaps(masks, ratio=0.5):
    """Detection flickers: a frame whose mask area drops below ratio x median borrows its nearest good neighbors."""
    area = masks.flatten(1).gt(0.5).sum(1).float()
    med = area.median().clamp(min=1)
    good = area >= ratio * med
    out = masks.clone()
    for i in range(len(masks)):
        if good[i]:
            continue
        prev = next((j for j in range(i - 1, -1, -1) if good[j]), None)
        nxt = next((j for j in range(i + 1, len(masks)) if good[j]), None)
        nb = [masks[j] for j in (prev, nxt) if j is not None]
        if nb:
            out[i] = torch.stack(nb).amax(0)
    return out


def _soften(m):
    m = F.max_pool2d(m[:, None], 3, 1, 1)
    k = torch.tensor([1., 2., 1.], device=m.device)
    k = (k[:, None] * k[None]) / 16
    return F.conv2d(m, k[None, None], padding=1)[:, 0].clamp(0, 1)


class CameoAssemble(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CameoAssemble",
            display_name="Cameo Assemble",
            category="cameo",
            description="Paste each shot's cameo back over the ORIGINAL frames using only the new person's matte "
                        "(minus the original people, who stay in front). When a shot has several takes, the decision "
                        "model rates each one (is the person there, is anyone distorted) and keeps the best. "
                        "Then rebuild the full video with its audio.",
            is_input_list=True,
            inputs=[
                io.Video.Input("video"),
                io.String.Input("plan", force_input=True),
                io.Image.Input("generated", tooltip="Per cast shot: the video model's frames."),
                io.Mask.Input("cameo_masks", tooltip="Per cast shot: SAM3 'person' masks on the generated frames."),
                io.Mask.Input("people_masks", tooltip="Per cast shot: SAM3 'person' masks on the original frames."),
                io.Float.Input("min_score", default=0.3, min=0.0, max=1.0, step=0.05,
                               tooltip="Shots whose best take the decision model rates below this stay untouched."),
                io.Boolean.Input("only_cast_shots", default=False,
                                 tooltip="Output just the cast shots back to back instead of the whole video."),
            ],
            outputs=[io.Video.Output(display_name="video"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, video, plan, generated, cameo_masks, people_masks, min_score, only_cast_shots) -> io.NodeOutput:
        comp = video[0].get_components()
        frames = comp.images.clone()
        plan = json.loads(plan[0])
        H, W = frames.shape[1:3]
        best = {}
        for item, gen, new, ppl in zip(plan["shots"], generated, cameo_masks, people_masks):
            s, n = item["start"], item["frames"]
            gen = F.interpolate(gen[:n].movedim(-1, 1), size=(H, W), mode="bilinear").movedim(1, -1)
            region = _box_mask(item["region"], H, W, new.device)
            new = F.interpolate(new[:n, None], size=(H, W), mode="bilinear")[:, 0]
            ppl = F.interpolate(ppl[:n, None], size=(H, W), mode="bilinear")[:, 0]
            m = _soften(_fill_gaps((new * region - ppl).clamp(0, 1)))[..., None]
            n = min(n, len(gen), len(m), len(frames) - s)
            take = gen[:n] * m[:n] + frames[s:s + n] * (1 - m[:n])
            score, why = rate_take(take, plan["desc"])
            item.update(score=round(score, 3), **why)
            k = item.get("shot", s)
            if k not in best or score > best[k][0]:
                best[k] = (score, s, n, take)
        spans = []
        for score, s, n, take in sorted(best.values(), key=lambda b: b[1]):
            if score < min_score[0]:
                continue
            frames[s:s + n] = take
            spans.append((s, s + n))
        audio = comp.audio
        if only_cast_shots[0]:
            frames = torch.cat([frames[a:b] for a, b in spans])
            audio = None
        return io.NodeOutput(InputImpl.VideoFromComponents(
            Types.VideoComponents(images=frames, audio=audio, frame_rate=Fraction(comp.frame_rate))),
            json.dumps(plan, indent=1))
