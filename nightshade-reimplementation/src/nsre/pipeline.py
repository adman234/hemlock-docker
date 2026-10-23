"""End-to-end single-image pipeline: Optimizer.generate_perturbation equivalent."""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

from PIL import Image

from . import imageio
from .optimize import core_optimize, get_mask
from .params import Params
from .repro import environment_info, log, sha256_file, write_manifest


@dataclass
class Components:
    encode_latent: Callable
    perceptual: Callable
    make_target: Callable[[Image.Image, str], Image.Image]  # (small_image, target_concept) -> target PIL
    device: str = "cpu"
    half: bool = False


def shade_image(
    path: str,
    out_dir: str,
    params: Params,
    comps: Components,
    target_concept: str,
    detected_concept: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    t0 = time.time()
    full = imageio.load_img(path)
    if full is None:
        raise ValueError(f"cannot load image: {path}")
    exif = full.getexif()
    small = imageio.shrink_shortside(full)
    target_img = comps.make_target(small, target_concept)

    src = imageio.img2tensor(small, comps.device)
    tgt = imageio.img2tensor(target_img, comps.device)
    if comps.half:
        src, tgt = src.half(), tgt.half()
    _mask, masked_target = get_mask(src, tgt)

    rng = random.Random(seed)
    res = core_optimize(src, masked_target, params, comps.encode_latent, comps.perceptual, rng=rng)
    final = imageio.add_perturbation_full_size(full, src, res.adv)
    os.makedirs(out_dir, exist_ok=True)
    out = imageio.save_image(final, path, out_dir, params.protection_level, exif=exif)
    log.info("saved %s in %.1fs", out, time.time() - t0)
    write_manifest(
        out + ".json",
        {
            "input": path,
            "input_sha256": sha256_file(path),
            "output": out,
            "output_sha256": sha256_file(out),
            "params": params.as_dict(),
            "target_concept": target_concept,
            "detected_concept": detected_concept,
            "seed": seed,
            "env": environment_info(),
            "runtime_s": time.time() - t0,
            "final_loss": res.losses[-1] if res.losses else None,
            "final_lpips": res.lpips_losses[-1] if res.lpips_losses else None,
        },
    )
    return out
