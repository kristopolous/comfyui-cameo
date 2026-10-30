# Cameo: put yourself in any video

Paste a video link, upload a photo of yourself, say what you're doing ("dances along, goofy and over the top"), and get the video back **with you in it**. You stand next to the people already on screen, lit like the scene, moving through the shots, while every original pixel outside your silhouette stays exactly as it was filmed.

This doesn't swap a person out. It **adds** a new one. And since the original footage holds the timeline together, it works at any length: a clip, a scene, or a whole film, one shot at a time.

It's a single ComfyUI workflow that opens in **App Mode**.

## How it works

```
Cameo Fetch        URL ─► lowest-res clip + shot cuts (yt-dlp, PySceneDetect)
Cameo Cast         decision model (LiquidAI d1-2B) judges every shot in one forward pass each:
                   framing, room for one more person, everyone's boxes; casts the best shots,
                   places you beside the people already there, and describes you from your photo
   │  (outputs are lists: everything below runs once per cast shot)
   ├─ FLUX 3 Klein        you, inserted into the shot's first frame from a box layout record
   ├─ Cameo Replace First Frame
   ├─ LTX-2.5             regenerates ONLY the region Cast chose (Cameo Box Noise Mask), anchored on that frame
   └─ SAM3 ×2             your matte on the generated frames, and the original people (who stay in front)
Cameo Assemble     for every take: pastes only your matte back over the ORIGINAL frames; the decision model rates
                   each take (are you there in every sampled frame? is anyone distorted?) and keeps the best;
                   shots with no good take stay untouched; stitches everything and keeps the audio
```

Why it's built this way:

- **A decision model is the director.** d1 returns calibrated, typed answers (choices, yes/no, counts, boxes) in a single pass with no generated text. That makes it cheap enough to judge every shot of a feature film. It decides which shots get you, where you stand, and how big you are. It also turns your photo into a description ("a man with short dark hair, wearing a navy hoodie and jeans") without a captioning LLM.
- **Boxes are the shared language.** d1 writes boxes, FLUX 3 Klein's edit records read them (`tgt_bbox` on a 0–1000 grid), `Cameo Box Noise Mask` turns them into an LTX-2.5 latent noise mask, and SAM3 refines them into mattes.
- **Only you come from the generators.** LTX's VAE changes pixels everywhere, so the final frame is the original film plus your SAM3 matte, minus the original people's mattes, so they stay in front of you.
- **The decision model is also quality control.** Each shot is generated in several takes (different seeds). d1 scores each one by asking whether this specific person ("a man with short dark hair, wearing a navy hoodie and jeans") is clearly visible in sampled frames and whether everyone looks undistorted. In testing, it picked the take where you stay on screen (0.77) over the one where you vanish (0.08), and it left alone a shot where both takes failed.
- **List execution keeps it one graph.** `Cameo Cast` emits lists, and ComfyUI runs Comfy's stock Klein, LTX and SAM3 nodes once per cast shot. No external orchestrator is needed.
- **Lowest resolution by default.** 192 px tall at 12 fps, the distilled LTX with one stage and no upscaler, and Klein at 16 steps. A full 3-minute video with 6 cast shots renders in about 5 minutes on one RTX 4090.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kristopolous/comfyui-cameo
pip install -r comfyui-cameo/requirements.txt
```

You need ComfyUI 0.43 or later (for SAM3, FLUX 3 and LTX-2.5 support in core). Models go in `ComfyUI/models/`:

| Folder | File | Source |
|---|---|---|
| `diffusion_models` | `flux3-image-klein-int8-convrot.safetensors` | [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) |
| `text_encoders` | `qwen3vl_4b_int8_convrot.safetensors` | same repo |
| `vae` | `flux2-vae.safetensors` | same repo |
| `diffusion_models` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) |
| `text_encoders` | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | same repo |
| `vae` | `ltx-2.5-video-vae-bf16.safetensors`, `ltx-2.5-audio-vae-bf16.safetensors` | same repo |
| `checkpoints` | `sam3.1_multiplex_fp16.safetensors` | [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) |

d1-2B ([LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B)) downloads itself on first use. Each model has its own license; read them before you use the output commercially.

## Use

Open `workflows/cameo_app.json` in ComfyUI. It opens in App Mode with these inputs: video URL, optional start and end times, your photo, what you do, how many shots, and how many takes per shot. The outputs are the video and a JSON report of every decision: which shots were cast, where you stand, and each take's score.

## Repo layout

- `comfyui_cameo/`: the node pack (`CameoFetch`, `CameoCast`, `CameoReplaceFirstFrame`, `CameoBoxNoiseMask`, `CameoAssemble`).
- `workflows/`: the app (`cameo_app.json`, plus the API version `cameo_app_api.json`), and the two building blocks on their own (`cameo_insert`: Klein; `cameo_ltx_box`: LTX box regeneration).
- `cameo/`: the Python scripts we prototyped with. They include a small ComfyUI API client (`comfy.py`), the graph builders (`graphs.py`), the step-by-step pipeline (`fetch.py`, `shots.py`, `decide.py`, `run_shot.py`), and `ui.py`, which turns API graphs into editable workflows using comfy-cli's edit ops.

## Limits

- Static or slow cameras work best. A fast camera move needs per-frame boxes, which `Cameo Box Noise Mask` already accepts.
- Sometimes the video model drops the person after the first frame. Several takes plus d1 scoring cover most of these failures; shots where every take fails are left as filmed.
- No video files are in this repo. Footage is fetched from a URL at run time.

## License

MIT for the code in this repo. Models and fetched footage carry their own licenses.
