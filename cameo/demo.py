"""Cut a demo reel from finished runs: title cards, App Mode screenshot, before/after clips.

    python -m cameo.demo work/demo
"""
import json
import subprocess
import sys
from pathlib import Path

W, H, FPS = 1280, 720, 24
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def esc(t):
    return t.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def card(out, title, lines=(), seconds=4.0):
    """Dark title card: a bold title and a few lines under it."""
    draws = [f"drawtext=fontfile={BOLD}:text='{esc(title)}':fontcolor=white:fontsize=56:x=(w-tw)/2:y=h*0.30"]
    for i, line in enumerate(lines):
        draws.append(f"drawtext=fontfile={FONT}:text='{esc(line)}':fontcolor=0xd0d0d0:fontsize=30:"
                     f"x=(w-tw)/2:y=h*0.46+{i * 46}")
    ff("-f", "lavfi", "-i", f"color=c=0x15161a:s={W}x{H}:d={seconds}:r={FPS}",
       "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo", "-t", str(seconds),
       "-vf", ",".join(draws), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out))


def still(out, image, caption, seconds=5.0):
    ff("-loop", "1", "-i", str(image), "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", str(seconds),
       "-vf", f"scale={W}:{H - 80}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:0:color=0x15161a,"
              f"drawtext=fontfile={FONT}:text='{esc(caption)}':fontcolor=white:fontsize=24:x=(w-tw)/2:y=h-54",
       "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out))


def before_after(out, orig, cameo, spans, fps, caption):
    """Original on the left, cameo on the right, for each [start, end) frame span; audio from the original."""
    parts = []
    for i, (a, b) in enumerate(spans):
        parts.append(f"[0:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[o{i}];"
                     f"[1:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[c{i}];"
                     f"[0:a]atrim=start={a / fps}:end={b / fps},asetpts=PTS-STARTPTS[a{i}];"
                     f"[o{i}][c{i}]hstack[v{i}]")
    cat = "".join(f"[v{i}][a{i}]" for i in range(len(spans)))
    label = (f"scale={W}:-2:flags=lanczos,pad={W}:{H}:0:(oh-ih)/2:color=0x15161a,"
             f"drawtext=fontfile={BOLD}:text='original':fontcolor=0xb0b0b0:fontsize=28:x=w*0.25-tw/2:y=40,"
             f"drawtext=fontfile={BOLD}:text='with Cameo':fontcolor=white:fontsize=28:x=w*0.75-tw/2:y=40,"
             f"drawtext=fontfile={FONT}:text='{esc(caption)}':fontcolor=white:fontsize=28:x=(w-tw)/2:y=h-60,"
             f"fps={FPS}")
    ff("-i", str(orig), "-i", str(cameo), "-filter_complex",
       ";".join(parts) + f";{cat}concat=n={len(spans)}:v=1:a=1[v][a];[v]{label}[vo]",
       "-map", "[vo]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100",
       "-ac", "2", str(out))


def reel(out_dir, segments):
    out_dir = Path(out_dir)
    lst = out_dir / "list.txt"
    lst.write_text("".join(f"file '{Path(s).resolve()}'\n" for s in segments))
    ff("-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
       str(out_dir / "cameo_demo.mp4"))
    return out_dir / "cameo_demo.mp4"


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "work/demo")
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"out": str(out)}))
