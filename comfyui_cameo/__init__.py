"""ComfyUI-Cameo: put yourself in any video.

Graph:  Cameo Fetch -> Cameo Cast (decision model picks shots and spots) -> [per cast shot, via list execution:
        FLUX 3 Klein inserts you into the first frame -> Cameo Replace First Frame -> LTX-2.5 regenerates only
        Cameo Box Noise Mask's region -> SAM3 mattes] -> Cameo Assemble (paste mattes back, stitch, keep audio).
Everything defaults to the LOWEST resolution.
"""
import json

import torch
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .cast import CameoCast
from .media import CameoAssemble, CameoFetch, CameoPhoto, CameoReplaceFirstFrame


WEB_DIRECTORY = "./web"


class CameoBoxNoiseMask(io.ComfyNode):
    """Regenerate only a box of a video latent; everything else stays the original footage."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CameoBoxNoiseMask",
            display_name="Cameo Box Noise Mask",
            category="cameo",
            description="Set a video latent's noise_mask to 1 inside a box (FLUX 3 grid: [y_min, x_min, y_max, x_max] "
                        "on 0-1000) and 0 elsewhere, so the sampler only regenerates that region. "
                        "The first keep_frames latent frames stay fully frozen (e.g. a pre-edited first frame).",
            inputs=[
                io.Latent.Input("latent", tooltip="5D video latent [B, C, F, H, W]."),
                io.String.Input("boxes", default="[[0, 0, 1000, 1000]]",
                                tooltip="JSON: one box for every latent frame, or a single box for all of them."),
                io.Int.Input("dilate", default=1, min=0, max=8, tooltip="Grow the box by this many latent cells."),
                io.Int.Input("keep_frames", default=1, min=0, max=64, tooltip="Leading latent frames to freeze."),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, latent, boxes, dilate, keep_frames) -> io.NodeOutput:
        samples = latent["samples"]
        if samples.ndim != 5:
            raise ValueError(f"Expected a 5D video latent, got {list(samples.shape)}")
        b, _, f, h, w = samples.shape
        boxes = json.loads(boxes)
        if boxes and isinstance(boxes[0], (int, float)):
            boxes = [boxes]
        if len(boxes) == 1:
            boxes = boxes * f
        if len(boxes) != f:
            raise ValueError(f"Got {len(boxes)} boxes for {f} latent frames")
        mask = torch.zeros((b, 1, f, h, w), dtype=torch.float32)
        for t, (y0, x0, y1, x1) in enumerate(boxes):
            if t < keep_frames:
                continue
            r0 = max(0, int(y0 * h / 1000) - dilate)
            r1 = min(h, -(-int(y1 * h) // 1000) + dilate)
            c0 = max(0, int(x0 * w / 1000) - dilate)
            c1 = min(w, -(-int(x1 * w) // 1000) + dilate)
            mask[:, :, t, r0:r1, c0:c1] = 1.0
        out = latent.copy()
        out["noise_mask"] = mask
        return io.NodeOutput(out)


class CameoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CameoFetch, CameoPhoto, CameoCast, CameoReplaceFirstFrame, CameoBoxNoiseMask, CameoAssemble]


async def comfy_entrypoint() -> CameoExtension:
    return CameoExtension()
