"""Optimizer-only parity: ORIGINAL opt.Optimizer.core_optimize vs nsre.optimize.core_optimize.

usage: python -I optimizer_parity.py REFDIR NSRE_SRC CORPUS_DIR   (GPU; USERPROFILE = scratch home with assets)
Identical inputs: same source image, same fixed synthetic target (no Stable Diffusion), same VAE and LPIPS
objects/weights, same Python RNG state (random.seed(678121)). Prints one JSON line per case.
"""
import contextlib
import io
import json
import os
import random
import sys
import time

import numpy as np
import torch

refdir, nsre_src, corpus = sys.argv[1:4]
sys.path[:0] = [refdir, nsre_src]
home = os.path.join(os.path.expanduser("~"), ".nightshade")
dev = "cuda"


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


import opt as ref_opt  # noqa: E402
import utils as ref_utils  # noqa: E402
from diffusers import AutoencoderKL  # noqa: E402
from nsre import imageio  # noqa: E402
from nsre.optimize import core_optimize, get_mask  # noqa: E402
from nsre.params import get_params  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import make_corpus  # noqa: E402

ref = ref_opt.Optimizer(None, [dev, dev], None, home)
ref.vae = AutoencoderKL.from_pretrained(os.path.join(home, "base", "base"), subfolder="vae").to(dev).half()
ref.vae.requires_grad_(False)


def encode(x):
    return ref.vae.encode(x.half()).latent_dist.mean


def lp(a, b):
    return ref.loss_fn(a, b)


cases = [("sq512.png", "DEFAULT", "1"), ("wide768.png", "DEFAULT", "1"), ("tall512x768.png", "LOW", "2"),
         ("big2000x1500.png", "HIGH", "1"), ("wide768.png", "DEFAULT", "3")]
for fname, level, optset in cases:
    full = ref_utils.load_img(os.path.join(corpus, fname), home)
    small = imageio.shrink_shortside(full)
    src = ref_utils.img2tensor(small, dev).half()
    tgt_img = make_corpus.scene(small.size[0], small.size[1], 99)  # fixed synthetic target
    tgt = ref_utils.img2tensor(tgt_img, dev).half()
    mask, mtgt = ref_opt.Optimizer.get_mask(None, src, tgt)
    ref.og_size = (full.size[1], full.size[0])
    p = quiet(ref_utils.get_app_parameters, level, optset)
    mine = get_params(level, optset)
    assert mine.tot_steps == p["tot_steps"] and mine.eot == p.get("eot", 1)

    random.seed(678121)
    t0 = time.time()
    a = quiet(ref.core_optimize, src, mtgt, mask, dict(p))
    t_ref = time.time() - t0
    random.seed(678121)
    t0 = time.time()
    b = core_optimize(src, mtgt, mine, encode, lp, rng=random.Random(678121), og_size=ref.og_size).adv
    t_our = time.time() - t0

    da, db = (a - src).float().flatten(), (b - src).float().flatten()
    cos = float(da @ db / (da.norm() * db.norm() + 1e-12))
    # second ORIGINAL run with the same RNG state: GPU nondeterminism floor
    random.seed(678121)
    a2 = quiet(ref.core_optimize, src, mtgt, mask, dict(p))
    dc = (a2 - src).float().flatten()
    cos_floor = float(da @ dc / (da.norm() * dc.norm() + 1e-12))
    print(json.dumps({
        "case": f"{fname}/{level}/{optset}", "steps": p["tot_steps"], "eot": p.get("eot", 1),
        "cos_ref_vs_nsre": round(cos, 5), "max_abs_diff_tensor": float((a - b).abs().max()),
        "cos_ref_vs_ref_rerun": round(cos_floor, 5), "max_abs_diff_ref_rerun": float((a - a2).abs().max()),
        "t_ref_s": round(t_ref, 1), "t_nsre_s": round(t_our, 1),
        "delta_linf_ref": float((a - src).abs().max()), "delta_linf_nsre": float((b - src).abs().max()),
    }), flush=True)
