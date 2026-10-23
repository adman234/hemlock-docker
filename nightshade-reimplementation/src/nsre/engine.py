"""High-level engine: loads the models once and shades images one at a time.

Used by the CLI (runner.py) and the web app (docker/app.py). BLIP and CLIP stay on the CPU, like the
original's ImageAnalyzer('cpu'); the VAE, LPIPS and the SD img2img pipeline use the selected device.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from . import concept as concept_mod
from . import device as device_mod
from . import imageio, models
from .optimize import core_optimize, get_mask
from .params import Params
from .repro import environment_info, log, sha256_file, write_manifest
from .target import generate_target_image

Stage = Callable[[str], None]
Progress = Callable[[float], None]  # 0..100 within the optimisation loop


@dataclass
class ShadeResult:
    output: str
    concept: str
    target: str
    seconds: float
    params: dict


class Engine:
    def __init__(
        self,
        sd_dir: str,
        device: Optional[str] = None,
        blip_dir: Optional[str] = None,
        clip_dir: Optional[str] = None,
        use_concept_models: bool = True,
        lpips_pickle: Optional[str] = None,
        pool: Optional[Sequence[str]] = None,
        mapping: Optional[dict] = None,
    ):
        self.device = device or device_mod.detect_device()
        self.half = device_mod.use_half(self.device)
        self.sd_dir, self.blip_dir, self.clip_dir = sd_dir, blip_dir, clip_dir
        self.use_concept_models = use_concept_models
        self.lpips_pickle = lpips_pickle
        self.pool = list(pool) if pool else []
        self.mapping = mapping or {}
        self._blip = self._clip = self._pool_emb = None
        self._loaded = False

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        if self._loaded:
            return
        self._encode = models.load_vae_encoder(self.sd_dir, self.device, self.half)
        self._perceptual = models.load_lpips(self.device, self.half, pickle_path=self.lpips_pickle)
        self._pipe = models.load_img2img_pipeline(self.sd_dir, self.device, self.half)
        self._blip = self._clip = self._pool_emb = None
        if self.use_concept_models:
            self._clip = models.load_clip(self.clip_dir or models.DEFAULT_CLIP_ID, "cpu")
            self._blip = models.load_blip(self.blip_dir or models.DEFAULT_BLIP_ID, "cpu")
            if self.pool:
                self._pool_emb = self._text_features(self.pool)
        self._loaded = True

    def _text_features(self, texts: list[str]) -> np.ndarray:
        proc, model = self._clip
        with torch.no_grad():
            tok = proc(text=texts, return_tensors="pt", padding=True)
            return model.get_text_features(**tok).float().numpy()

    # ------------------------------------------------------------------ concept / target
    def detect_concept(self, img: Image.Image) -> str:
        if not self._blip:
            return concept_mod.FALLBACK_CONCEPT
        import nltk

        proc, model = self._blip
        with torch.no_grad():
            inputs = proc(img, concept_mod.BLIP_PREFIX, return_tensors="pt")
            caption = proc.decode(model.generate(**inputs)[0], skip_special_tokens=True)
        nouns = concept_mod.extract_nouns(caption, nltk.pos_tag)
        cproc, cmodel = self._clip

        def scores(prompts):
            with torch.no_grad():
                inp = cproc(text=prompts, images=img, return_tensors="pt", padding=True)
                return cmodel(**inp).logits_per_image.softmax(-1).float().numpy()

        return concept_mod.pick_concept(nouns, scores)

    def pick_target(self, concept: str) -> str:
        if concept in self.mapping:
            return self.mapping[concept]
        if not (self._clip and self._pool_emb is not None):
            if not self.pool:
                raise RuntimeError("no target given and no concept pool available")
            # degraded mode without CLIP: deterministic pick per concept (not similarity based)
            import zlib

            choices = [p for p in self.pool if p != concept]
            return random.Random(zlib.crc32(concept.encode())).choice(choices)
        emb = self._text_features([concept])
        sim = (emb / np.linalg.norm(emb, axis=1, keepdims=True)) @ (
            self._pool_emb / np.linalg.norm(self._pool_emb, axis=1, keepdims=True)
        ).T
        return concept_mod.pick_target(concept, self.pool, sim, mapping=self.mapping)

    # ------------------------------------------------------------------ shading
    def shade(
        self,
        path: str,
        out_dir: str,
        params: Params,
        tag: Optional[str] = None,
        target: Optional[str] = None,
        seed: Optional[int] = None,
        on_stage: Optional[Stage] = None,
        on_progress: Optional[Progress] = None,
        reset_exif_orientation: bool = False,
        write_json: bool = True,
    ) -> ShadeResult:
        self.load()
        t0 = time.time()
        stage = on_stage or (lambda s: None)
        full = imageio.load_img(path)
        if full is None:
            raise ValueError(f"cannot load image: {os.path.basename(path)}")
        exif = full.getexif()
        small = imageio.shrink_shortside(full)

        stage("Reading the image")
        concept = tag or self.detect_concept(full)
        tgt_name = target or self.pick_target(concept)

        def run_pipeline(**kw):
            with torch.no_grad():
                return self._pipe(**kw).images[0]

        stage("Making the target image")
        target_img = generate_target_image(small, tgt_name, run_pipeline, device=self.device)
        src = imageio.img2tensor(small, self.device)
        tgt = imageio.img2tensor(target_img, self.device)
        if self.half:
            src, tgt = src.half(), tgt.half()
        _mask, masked_target = get_mask(src, tgt)

        stage("Crafting the protection")
        res = core_optimize(
            src, masked_target, params, self._encode, self._perceptual, rng=random.Random(seed),
            progress=on_progress, og_size=(full.size[1], full.size[0]),
        )
        final = imageio.add_perturbation_full_size(full, src, res.adv)
        stage("Saving")
        os.makedirs(out_dir, exist_ok=True)
        out = imageio.save_image(final, path, out_dir, params.protection_level, exif=exif,
                                 reset_exif_orientation=reset_exif_orientation)
        secs = time.time() - t0
        if write_json:
            write_manifest(out + ".json", {
                "input": path, "input_sha256": sha256_file(path), "output": out, "output_sha256": sha256_file(out),
                "params": params.as_dict(), "target_concept": tgt_name, "detected_concept": concept, "seed": seed,
                "env": environment_info(), "runtime_s": secs,
            })
        log.info("saved %s in %.1fs", out, secs)
        return ShadeResult(out, concept, tgt_name, secs, params.as_dict())

