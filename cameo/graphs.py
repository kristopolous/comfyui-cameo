"""API-format ComfyUI graphs, built as plain dicts. Every default is the LOWEST resolution."""
import json
import math

KLEIN = "flux3-image-klein-int8-convrot.safetensors"
KLEIN_TE = "qwen3vl_4b_int8_convrot.safetensors"
FLUX_VAE = "flux2-vae.safetensors"


class Graph(dict):
    """Node ids are assigned in insertion order; add() returns an output reference factory."""

    def add(self, class_type, **inputs):
        nid = str(len(self) + 1)
        self[nid] = {"class_type": class_type, "inputs": inputs}
        return lambda slot=0: [nid, slot]


def _klein_base(g, width, height, prompt):
    unet = g.add("UNETLoader", unet_name=KLEIN, weight_dtype="default")
    clip = g.add("CLIPLoader", clip_name=KLEIN_TE, type="flux3", device="default")
    vae = g.add("VAELoader", vae_name=FLUX_VAE)
    mu = math.log(16.0)  # alpha=16, the Klein reference workflow's timestep shift
    model = g.add("ModelSamplingFlux", model=unet(), max_shift=mu, base_shift=mu, width=width, height=height)
    pos = g.add("CLIPTextEncode", clip=clip(), text=prompt)
    neg = g.add("CLIPTextEncode", clip=clip(), text="")
    latent = g.add("EmptyFlux2LatentImage", width=width, height=height, batch_size=1)
    return model, vae, pos, neg, latent


def _sample(g, model, vae, pos, neg, latent, steps, seed, prefix):
    s = g.add("KSampler", model=model(), seed=seed, steps=steps, cfg=1.0, sampler_name="euler",
              scheduler="simple", positive=pos, negative=neg, latent_image=latent, denoise=1.0)
    img = g.add("VAEDecode", samples=s(), vae=vae())
    g.add("SaveImage", images=img(), filename_prefix=prefix)
    return g


def klein_t2i(prompt, width=256, height=256, steps=16, seed=1, prefix="cameo/t2i"):
    g = Graph()
    model, vae, pos, neg, latent = _klein_base(g, width, height, prompt)
    return _sample(g, model, vae, pos(), neg(), latent(), steps, seed, prefix)


def klein_edit(prompt, images, width, height, steps=16, seed=1, ref_megapixels=0.07, prefix="cameo/edit"):
    """images: uploaded names in order; images[0] is ref_image_0 (the scene), images[1] is ref_image_1, ..."""
    g = Graph()
    model, vae, pos, neg, latent = _klein_base(g, width, height, prompt)
    refs = {f"images.image{i}": g.add("LoadImage", image=name)() for i, name in enumerate(images)}
    ref = g.add("ReferenceMultiLatent", positive=pos(), negative=neg(), vae=vae(),
                resolution="resize", **{"resolution.megapixels": ref_megapixels}, **refs)
    return _sample(g, model, vae, ref(0), ref(1), latent(), steps, seed, prefix)


def to_grid(box):
    """d1 box {top,left,bottom,right} in 0..1 -> FLUX 3 [y_min, x_min, y_max, x_max] on 0..1000."""
    return [round(box["top"] * 1000), round(box["left"] * 1000),
            round(box["bottom"] * 1000), round(box["right"] * 1000)]


def insert_record(tgt_bbox, desc, where, person_src_bbox=(0, 0, 1000, 1000)):
    """The i2i_bbox annotation record Klein was trained on: instruction paragraph + one JSON table."""
    para = (f"Add the person <person_1> from <ref_image_1> to the scene, {where}, matching the scene's "
            f"lighting, color grade, film grain and perspective. Everything else stays identical to the source.")
    rows = [{"id": "person_1", "from": "ref_image_1", "src_bbox": list(person_src_bbox),
             "tgt_bbox": list(tgt_bbox), "desc": desc}]
    return f"{para} {json.dumps(rows)}"


LTX = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
LTX_TE = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
LTX_VAE = "ltx-2.5-video-vae-bf16.safetensors"
LTX_AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
LTX_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"  # distilled, stage 1 only


def ltx_box_inpaint(video, boxes, prompt, frames, fps, seed=1, dilate=1, keep_frames=1,
                    negative="pc game, console game, video game, cartoon, childish, ugly", prefix="cameo/ltx"):
    """Regenerate only `boxes` of an uploaded window video whose frame 0 already contains the cameo.
    No upscaler, no second stage: LOWEST resolution."""
    g = Graph()
    vid = g.add("LoadVideo", file=video)
    comp = g.add("GetVideoComponents", video=vid())
    vae = g.add("VAELoader", vae_name=LTX_VAE)
    avae = g.add("VAELoader", vae_name=LTX_AUDIO_VAE)
    lat = g.add("VAEEncode", pixels=comp(0), vae=vae())
    masked = g.add("CameoBoxNoiseMask", latent=lat(), boxes=json.dumps(boxes), dilate=dilate, keep_frames=keep_frames)
    audio = g.add("LTXVEmptyLatentAudio", frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=avae())
    av = g.add("LTXVConcatAVLatent", video_latent=masked(), audio_latent=audio())
    clip = g.add("CLIPLoader", clip_name=LTX_TE, type="ltxv", device="default")
    pos = g.add("CLIPTextEncode", clip=clip(), text=prompt)
    neg = g.add("CLIPTextEncode", clip=clip(), text=negative)
    cond = g.add("LTXVConditioning", positive=pos(), negative=neg(), frame_rate=float(fps))
    model = g.add("UNETLoader", unet_name=LTX, weight_dtype="default")
    guider = g.add("CFGGuider", model=model(), positive=cond(0), negative=cond(1), cfg=1.0)
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seed)(), guider=guider(),
                sampler=g.add("KSamplerSelect", sampler_name="euler_ancestral")(),
                sigmas=g.add("ManualSigmas", sigmas=LTX_SIGMAS)(), latent_image=av())
    sep = g.add("LTXVSeparateAVLatent", av_latent=out(0))
    img = g.add("VAEDecode", samples=sep(0), vae=vae())
    g.add("SaveImage", images=img(), filename_prefix=prefix)
    return g


SAM3 = "sam3.1_multiplex_fp16.safetensors"


def sam3_mask(video, text="person", threshold=0.4, prefix="cameo/sam3"):
    """Union mask of every `text` instance in every frame of an uploaded video, saved as grayscale frames."""
    g = Graph()
    comp = g.add("GetVideoComponents", video=g.add("LoadVideo", file=video)())
    ckpt = g.add("CheckpointLoaderSimple", ckpt_name=SAM3)
    cond = g.add("CLIPTextEncode", clip=ckpt(1), text=text)
    det = g.add("SAM3_Detect", model=ckpt(0), image=comp(0), conditioning=cond(), threshold=threshold,
                refine_iterations=2, individual_masks=False)
    g.add("SaveImage", images=g.add("MaskToImage", mask=det(0))(), filename_prefix=prefix)
    return g


def cameo_app(url, photo, action="", start="", end="", max_shots=4, make_cameo=True, suggestion="(my own video)",
              webcam=False,
              max_frames=49, height=192, fps=12, seed=1, only_cast_shots=False, prefix="cameo/app", debug=False, takes=2):
    """The whole app as ONE ComfyUI graph: inputs are a video URL, a photo and an action; output is a video.
    CameoCast outputs lists, so the Klein / LTX / SAM3 sections run once per cast shot (list execution)."""
    g = Graph()
    fetch = g.add("CameoFetch", suggestion=suggestion, url=url, start=start, end=end, height=height, fps=fps,
                  cut_threshold=15.0)
    g.add("PreviewAny", source=fetch(2))
    upload = g.add("LoadImage", image=photo)
    # webcam only in the App Mode workflow: API runs have no browser camera
    cam = {"webcam": g.add("WebcamCapture", image=photo, width=640, height=480, capture_on_queue=True)()} if webcam else {}
    person = g.add("CameoPhoto", source="photo", photo=upload(0), **cam)
    cast = g.add("CameoCast", video=fetch(0), shots=fetch(1), hint=fetch(3), make_cameo=make_cameo, person=person(0),
                 action=action,
                 max_shots=max_shots, max_frames=max_frames, min_frames=17, takes=takes, seed=seed)
    windows, firsts, inserts, vprompts, regions, counts, seeds = (cast(i) for i in range(7))
    plan, W, H, FPS = cast(7), cast(8), cast(9), cast(10)

    # FLUX 3 Klein: you, inserted into each cast shot's first frame
    unet = g.add("UNETLoader", unet_name=KLEIN, weight_dtype="default")
    kclip = g.add("CLIPLoader", clip_name=KLEIN_TE, type="flux3", device="default")
    kvae = g.add("VAELoader", vae_name=FLUX_VAE)
    mu = math.log(16.0)
    kmodel = g.add("ModelSamplingFlux", model=unet(), max_shift=mu, base_shift=mu, width=W, height=H)
    kpos = g.add("CLIPTextEncode", clip=kclip(), text=inserts)
    kneg = g.add("CLIPTextEncode", clip=kclip(), text="")
    ref = g.add("ReferenceMultiLatent", positive=kpos(), negative=kneg(), vae=kvae(), resolution="resize",
                **{"resolution.megapixels": 0.07, "images.image0": firsts, "images.image1": person(0)})
    klat = g.add("EmptyFlux2LatentImage", width=W, height=H, batch_size=1)
    ks = g.add("KSampler", model=kmodel(), seed=seed, steps=16, cfg=1.0, sampler_name="euler", scheduler="simple",
               positive=ref(0), negative=ref(1), latent_image=klat(), denoise=1.0)
    inserted = g.add("VAEDecode", samples=ks(), vae=kvae())
    window = g.add("CameoReplaceFirstFrame", frames=windows, first=inserted())

    # LTX-2.5: carry you through each shot, regenerating only the region the decision model chose
    vae = g.add("VAELoader", vae_name=LTX_VAE)
    avae = g.add("VAELoader", vae_name=LTX_AUDIO_VAE)
    lat = g.add("VAEEncode", pixels=window(), vae=vae())
    masked = g.add("CameoBoxNoiseMask", latent=lat(), boxes=regions, dilate=1, keep_frames=1)
    audio = g.add("LTXVEmptyLatentAudio", frames_number=counts, frame_rate=FPS, batch_size=1, audio_vae=avae())
    av = g.add("LTXVConcatAVLatent", video_latent=masked(), audio_latent=audio())
    lclip = g.add("CLIPLoader", clip_name=LTX_TE, type="ltxv", device="default")
    lpos = g.add("CLIPTextEncode", clip=lclip(), text=vprompts)
    lneg = g.add("CLIPTextEncode", clip=lclip(), text="pc game, console game, video game, cartoon, childish, ugly")
    cond = g.add("LTXVConditioning", positive=lpos(), negative=lneg(), frame_rate=FPS)
    lmodel = g.add("UNETLoader", unet_name=LTX, weight_dtype="default")
    guider = g.add("CFGGuider", model=lmodel(), positive=cond(0), negative=cond(1), cfg=1.0)
    out = g.add("SamplerCustomAdvanced", noise=g.add("RandomNoise", noise_seed=seeds)(), guider=guider(),
                sampler=g.add("KSamplerSelect", sampler_name="euler_ancestral")(),
                sigmas=g.add("ManualSigmas", sigmas=LTX_SIGMAS)(), latent_image=av())
    sep = g.add("LTXVSeparateAVLatent", av_latent=out(0))
    gen = g.add("VAEDecode", samples=sep(0), vae=vae())

    # SAM3: the new person's matte, and the original people who stay in front
    sam = g.add("CheckpointLoaderSimple", ckpt_name=SAM3)
    who = g.add("CLIPTextEncode", clip=sam(1), text="person")
    cameo = g.add("SAM3_Detect", model=sam(0), image=gen(), conditioning=who(), threshold=0.25,
                  refine_iterations=2, individual_masks=False)
    people = g.add("SAM3_Detect", model=sam(0), image=windows, conditioning=who(), threshold=0.4,
                   refine_iterations=2, individual_masks=False)

    if debug:
        g.add("SaveImage", images=gen(), filename_prefix=f"{prefix}_dbg_gen")
        g.add("SaveImage", images=g.add("MaskToImage", mask=cameo(0))(), filename_prefix=f"{prefix}_dbg_cameo")
        g.add("SaveImage", images=g.add("MaskToImage", mask=people(0))(), filename_prefix=f"{prefix}_dbg_people")
    final = g.add("CameoAssemble", video=fetch(0), plan=plan, generated=gen(), cameo_masks=cameo(0),
                  people_masks=people(0), min_score=0.3, only_cast_shots=only_cast_shots)
    g.add("SaveVideo", video=final(0), filename_prefix=prefix, format="mp4", **{"format.codec": "h264"})
    g.add("PreviewAny", source=final(1))
    return g
