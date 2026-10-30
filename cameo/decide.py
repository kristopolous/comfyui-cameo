"""Per-shot casting decisions with an open decision model (LiquidAI d1-2B).

    python -m cameo.decide work/test/shots.json

One forward pass per shot answers every question about its keyframe; no tokens are generated.
Adds a "decision" block to each shot in shots.json.
"""
import argparse
import json
import time

import torch
from PIL import Image
from transformers import AutoModel

MODEL = "LiquidAI/d1-2B"

QUESTIONS = {
    "framing": {"type": "choice", "instructions": "How is this shot framed?",
                "criteria": {"wide": "Full bodies and plenty of the setting are visible",
                             "medium": "People from about the waist up",
                             "closeup": "A face or a single object fills the frame"}},
    "room": {"type": "noul", "instructions": "Is there open space where one more standing person could fit naturally?"},
    "people": {"type": "number", "instructions": "How many people are visible?", "range": [0, 20], "step": 1},
    "spot": {"type": "box", "instructions": "Where is the best open space for one more person to stand?"},
    "lighting": {"type": "choice", "instructions": "What is the lighting?",
                 "criteria": {"day": "Daylight or bright light", "night": "Night or dark",
                              "interior": "Indoor artificial light", "stage": "Stage or concert lights"}},
}


def load(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    return AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=dtype).to(device)


def decide(model, image_path):
    image = Image.open(image_path).convert("RGB")
    out = model.system_one(None, QUESTIONS, media=[image])
    return out["answers"]


def cast(answers, min_room=0.35, max_people=4):
    """Should this shot get the cameo? Wide or medium, some room, not a crowd, and a spot found.
    d1's "room" is conservative (empty stages score about 0.4-0.5), hence 0.35."""
    return (answers["framing"]["choice"] in ("wide", "medium")
            and answers["room"]["noul"] >= min_room
            and answers["people"]["number"] <= max_people
            and answers.get("spot", {}).get("box") is not None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("shots")
    a = ap.parse_args()
    data = json.load(open(a.shots))
    t = time.time()
    model = load()
    print(f"loaded in {time.time() - t:.1f}s")
    for shot in data["shots"]:
        t = time.time()
        ans = decide(model, shot["keyframe"])
        shot["decision"] = {"answers": ans, "cast": cast(ans), "seconds": round(time.time() - t, 2)}
        print(shot["id"], json.dumps(shot["decision"], default=str)[:400])
    json.dump(data, open(a.shots, "w"), indent=2, default=str)
