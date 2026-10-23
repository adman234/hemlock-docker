"""Batch runner: wires real models into the pipeline. Requires the 'models' extra."""
from __future__ import annotations

import os

import numpy as np
import torch

from . import concept as concept_mod
from . import device as device_mod
from . import imageio, models
from .pipeline import Components, shade_image
from .repro import log, seed_everything
from .target import generate_target_image


def _blip_caption(processor, model, img, device):
    inputs = processor(img, concept_mod.BLIP_PREFIX, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


def run_batch(paths, out_dir, params, args):
    device = args.device or device_mod.detect_device()
    half = device_mod.use_half(device)
    sd_id = args.sd_model or models.DEFAULT_SD_ID
    seed_everything(args.seed)
    log.info("device=%s half=%s", device, half)

    encode = models.load_vae_encoder(sd_id, device, half)
    perceptual = models.load_lpips(device, half)
    pipe = models.load_img2img_pipeline(sd_id, device, half)

    def run_pipeline(**kw):
        with torch.no_grad():
            return pipe(**kw).images[0]

    def make_target(small, target_concept):
        return generate_target_image(small, target_concept, run_pipeline, device=device)

    comps = Components(encode, perceptual, make_target, device=device, half=half)

    pool, pool_emb, clip_proc, clip_model = None, None, None, None
    if not args.target:
        if not args.concepts_file:
            raise SystemExit("provide -t/--target or --concepts-file (the original's concept pool is proprietary)")
        with open(args.concepts_file, encoding="utf8") as f:
            pool = [ln.strip() for ln in f if ln.strip()]
        clip_proc, clip_model = models.load_clip(models.DEFAULT_CLIP_ID, device)
        with torch.no_grad():
            tok = clip_proc(text=pool, return_tensors="pt", padding=True).to(device)
            pool_emb = clip_model.get_text_features(**tok).float().cpu().numpy()

    blip = None
    results = []
    for path in paths:
        cur = args.concept
        if cur is None:
            if blip is None:
                import nltk

                blip = models.load_blip(models.DEFAULT_BLIP_ID, device) + (nltk.pos_tag,)
            raw = imageio.load_img(path)
            cap = _blip_caption(blip[0], blip[1], raw, device)
            nouns = concept_mod.extract_nouns(cap, blip[2])
            cur = concept_mod.pick_concept(nouns)  # CLIP re-ranking omitted without models; see known-limitations
        if args.target:
            tgt = args.target
        else:
            with torch.no_grad():
                tok = clip_proc(text=[cur], return_tensors="pt", padding=True).to(device)
                emb = clip_model.get_text_features(**tok).float().cpu().numpy()
            sim = (emb / np.linalg.norm(emb, axis=1, keepdims=True)) @ (
                pool_emb / np.linalg.norm(pool_emb, axis=1, keepdims=True)
            ).T
            tgt = concept_mod.pick_target(cur, pool, sim)
        log.info("concept=%s target=%s", cur, tgt)
        out = shade_image(path, out_dir, params, comps, tgt, detected_concept=cur, seed=args.seed)
        results.append(f"{out}: {cur}")

    header = ("To achieve maximal effect, please try to include the poison tags below as part of the "
              "ALT text field when you post corresponding images online. ")
    with open(os.path.join(out_dir, "nightshade-result.txt"), "w+") as f:
        f.write("\n".join([header] + results))
