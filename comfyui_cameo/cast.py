"""Casting with an open decision model (LiquidAI d1-2B): which shots get the cameo, where they stand,
and a description of the person built from typed answers about their photo."""
import json

import numpy as np
import torch
from PIL import Image

import comfy.model_management
from comfy_api.latest import io

D1 = "LiquidAI/d1-2B"
SHOTS = io.Custom("CAMEO_SHOTS")
PLAN = io.Custom("CAMEO_PLAN")
_model = None


def _d1():
    global _model
    if _model is None:
        from transformers import AutoModel
        device = comfy.model_management.get_torch_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        _model = AutoModel.from_pretrained(D1, trust_remote_code=True, dtype=dtype).to(device)
    return _model


def _pil(t):
    return Image.fromarray((t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))


def _ask(image, questions):
    return _d1().system_one(None, questions, media=[_pil(image)])["answers"]


SHOT = {
    "framing": {"type": "choice", "instructions": "How is this shot framed?",
                "criteria": {"wide": "Full bodies and plenty of the setting are visible",
                             "medium": "People from about the waist up",
                             "closeup": "A face or a single object fills the frame"}},
    "room": {"type": "noul", "instructions": "Is there open space where one more standing person could fit naturally?"},
    "people": {"type": "number", "instructions": "How many people are visible?", "range": [0, 20], "step": 1},
    "everyone": {"type": "boxes", "instructions": "Every person.", "max": 6},
    "spot": {"type": "box", "instructions": "Where is the best open space for one more person to stand?"},
}

PERSON = {
    "gender": {"type": "choice", "instructions": "Who is in the photo?",
               "criteria": {"man": "A man or boy", "woman": "A woman or girl"}},
    "age": {"type": "choice", "instructions": "About how old is the person?",
            "criteria": {"teenage": "A teenager", "young": "In their twenties", "adult": "In their thirties or forties",
                         "older": "Fifty or older"}},
    "hair": {"type": "choice", "instructions": "What is the person's hair like?",
             "criteria": {"short dark": "Short dark hair", "long dark": "Long dark hair", "short blond": "Short blond hair",
                          "long blond": "Long blond hair", "red": "Red hair", "gray": "Gray or white hair",
                          "no": "Bald or shaved head"}},
    "top": {"type": "choice", "instructions": "What is the person wearing on top?",
            "criteria": {"t-shirt": "A t-shirt", "hoodie": "A hoodie or sweatshirt", "shirt": "A button shirt or blouse",
                         "jacket": "A jacket or coat", "sweater": "A sweater", "dress": "A dress"}},
    "color": {"type": "choice", "instructions": "What color is the top?",
              "criteria": {c: f"Mostly {c}" for c in ("black", "white", "gray", "navy", "blue", "red", "green",
                                                       "yellow", "brown", "pink", "purple")}},
    "bottom": {"type": "choice", "instructions": "What is the person wearing below the waist?",
               "criteria": {"jeans": "Jeans", "trousers": "Trousers or slacks", "shorts": "Shorts", "a skirt": "A skirt",
                            "": "Not visible or a dress"}},
}


def describe(photo):
    a = {k: v["choice"] for k, v in _ask(photo, PERSON).items()}
    who = {"teenage": "teenage", "young": "young", "adult": "", "older": "older"}[a["age"]]
    desc = f"A {who} {a['gender']} with {a['hair']} hair, wearing a {a['color']} {a['top']}".replace("  ", " ")
    desc += f" and {a['bottom']}." if a["bottom"] and a["top"] != "dress" else "."
    return desc, ("She" if a["gender"] == "woman" else "He")


def _grid(b):
    return [round(b["top"] * 1000), round(b["left"] * 1000), round(b["bottom"] * 1000), round(b["right"] * 1000)]


def _boxes(answer):
    """d1 returns {"box": {...}} for one box and a list for "boxes"; items may wrap the box. Normalize to dicts."""
    items = answer.get("boxes") or answer.get("items") or ([answer["box"]] if answer.get("box") else [])
    out = []
    for b in items:
        b = b.get("box", b) if isinstance(b, dict) else b
        if isinstance(b, dict) and all(k in b for k in ("top", "left", "bottom", "right")):
            out.append(b)
        elif isinstance(b, (list, tuple)) and len(b) == 4:
            out.append(dict(zip(("top", "left", "bottom", "right"), b)))
    return out


def place(answers):
    """Stand next to the biggest person, on the side with more room, at their scale. Returns (box, region, where)."""
    people = _boxes(answers.get("everyone", {}))
    if people:
        p = max(people, key=lambda b: (b["bottom"] - b["top"]) * (b["right"] - b["left"]))
        w = max(p["right"] - p["left"], 0.08)
        right = (1 - p["right"]) >= p["left"]
        x0 = min(p["right"] + 0.03, 1 - w) if right else max(p["left"] - 0.03 - w, 0)
        box = {"top": p["top"], "bottom": p["bottom"], "left": x0, "right": x0 + w}
        side = "right" if right else "left"
        region = {"top": max(0, p["top"] - 0.15), "bottom": 1.0,
                  "left": p["right"] - 0.02 if right else 0.0, "right": 1.0 if right else p["left"] + 0.02}
        return _grid(box), _grid(region), f"standing to the {side} of the person <ref_image_0>, at the same distance from the camera and the same size"
    s = (_boxes(answers.get("spot", {})) or [{"top": 0.3, "left": 0.6, "bottom": 0.95, "right": 0.8}])[0]
    h = max(s["bottom"] - s["top"], 0.4)
    box = {"top": max(0, s["bottom"] - h), "bottom": s["bottom"], "left": s["left"], "right": s["right"]}
    region = {"top": max(0, box["top"] - 0.15), "bottom": 1.0, "left": max(0, s["left"] - 0.15), "right": min(1, s["right"] + 0.15)}
    return _grid(box), _grid(region), "standing in the open space"


def cast_ok(a, min_room=0.35, max_people=4):
    return (a["framing"]["choice"] in ("wide", "medium") and a["room"]["noul"] >= min_room
            and a["people"]["number"] <= max_people)


def rate_take(frames, desc):
    """Decision-model QC of one composited take: is the cameo there in every sampled frame, and is nobody distorted?"""
    qs = {"present": {"type": "noul", "instructions": f"Is this person clearly visible in the frame: {desc}"},
          "natural": {"type": "noul", "instructions": "Do all the people in the frame look like normal, undistorted, "
                                                      "complete human beings?"}}
    picks = [frames[int(i)] for i in torch.linspace(0, len(frames) - 1, 4)]
    answers = [_ask(f, qs) for f in picks]
    present = min(a["present"]["noul"] for a in answers)
    natural = sum(a["natural"]["noul"] for a in answers) / len(answers)
    return present * natural, {"present": round(present, 3), "natural": round(natural, 3)}


def insert_record(tgt_bbox, desc, where):
    para = (f"Add the person <person_1> from <ref_image_1> to the scene, {where}, matching the scene's lighting, "
            f"color grade, film grain and perspective. Everything else stays identical to the source.")
    rows = [{"id": "person_1", "from": "ref_image_1", "src_bbox": [0, 0, 1000, 1000], "tgt_bbox": tgt_bbox, "desc": desc}]
    return f"{para} {json.dumps(rows)}"


class CameoCast(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        L = dict(is_output_list=True)
        return io.Schema(
            node_id="CameoCast",
            display_name="Cameo Cast (decision model)",
            category="cameo",
            description="A decision model (LiquidAI d1-2B, one forward pass per question set, no generated tokens) "
                        "judges every shot: framing, room for one more person, everyone's boxes. It casts the best "
                        "shots, places you beside the people already there, and describes you from your photo. "
                        "Outputs are lists, so everything downstream runs once per cast shot.",
            inputs=[
                io.Video.Input("video"),
                SHOTS.Input("shots"),
                io.Image.Input("person", tooltip="A photo of you."),
                io.String.Input("action", default="dances along, goofy and over the top", multiline=True,
                                tooltip="What you do in every shot."),
                io.Int.Input("max_shots", default=4, min=1, max=500),
                io.Int.Input("max_frames", default=49, min=9, max=257, step=8, tooltip="Frames per shot (8n+1)."),
                io.Int.Input("min_frames", default=17, min=9, max=257, step=8),
                io.Int.Input("takes", default=2, min=1, max=8,
                             tooltip="Video-model takes per shot; Cameo Assemble keeps the one the decision model rates best."),
                io.Int.Input("seed", default=1, min=0, max=2**31),
            ],
            outputs=[
                io.Image.Output(display_name="windows", **L),
                io.Image.Output(display_name="first_frames", **L),
                io.String.Output(display_name="insert_prompts", **L),
                io.String.Output(display_name="video_prompts", **L),
                io.String.Output(display_name="regions", **L),
                io.Int.Output(display_name="frame_counts", **L),
                io.Int.Output(display_name="seeds", **L),
                PLAN.Output(display_name="plan"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Float.Output(display_name="fps"),
            ],
        )

    @classmethod
    def execute(cls, video, shots, person, action, max_shots, max_frames, min_frames, takes, seed) -> io.NodeOutput:
        comp = video.get_components()
        frames, fps = comp.images, float(comp.frame_rate)
        H, W = frames.shape[1:3]
        desc, pronoun = describe(person[0])
        judged = []
        for s, e in json.loads(shots):
            n = min(max_frames, e - s)
            n = n - (n - 1) % 8
            if n < min_frames:
                continue
            a = _ask(frames[s], SHOT)
            if cast_ok(a):
                judged.append((a["room"]["noul"] * n, s, n, a))
        judged.sort(reverse=True)
        picked = sorted(judged[:max_shots], key=lambda j: j[1])
        if not picked:
            raise ValueError("The decision model found no shot with room for one more person.")
        out = {k: [] for k in ("windows", "first", "insert", "video", "regions", "counts", "seeds")}
        plan = {"desc": desc, "shots": []}
        for i, (_, s, n, a) in enumerate(picked):
            box, region, where = place(a)
            for take in range(takes):
                out["windows"].append(frames[s:s + n])
                out["first"].append(frames[s:s + 1])
                out["insert"].append(insert_record(box, desc, where))
                out["video"].append(f"A static shot. {desc} {pronoun} {action}.")
                out["regions"].append(json.dumps([region]))
                out["counts"].append(n)
                out["seeds"].append(seed + 1000 * i + take)
                plan["shots"].append({"shot": i, "take": take, "start": s, "frames": n, "box": box, "region": region,
                                      "framing": a["framing"]["choice"], "room": round(a["room"]["noul"], 3)})
        return io.NodeOutput(out["windows"], out["first"], out["insert"], out["video"], out["regions"],
                             out["counts"], out["seeds"], json.dumps(plan), W, H, fps)
