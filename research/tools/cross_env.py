"""Run fixed inputs through nsre under one library set and save the results, to compare library versions.

usage: python -I cross_env.py NSRE_SRC CORPUS_DIR OUT.npz     (GPU; USERPROFILE = scratch home with assets)
Saves: VAE latent of a fixed tensor, optimizer output for two cases (fixed target and RNG), and the SD img2img
target image generated with a fixed torch seed. Compare two such files with research/tools/cross_env_diff.py.
"""
import os
import random
import sys

import numpy as np
import torch

nsre_src, corpus, out = sys.argv[1:4]
sys.path[:0] = [nsre_src, os.path.dirname(os.path.abspath(__file__))]
home = os.path.join(os.path.expanduser("~"), ".nightshade")
sd = os.path.join(home, "base", "base")

import diffusers  # noqa: E402
import make_corpus  # noqa: E402
import transformers  # noqa: E402
from nsre import imageio, models  # noqa: E402
from nsre.optimize import core_optimize, get_mask  # noqa: E402
from nsre.params import get_params  # noqa: E402
from nsre.target import generate_target_image  # noqa: E402
from PIL import Image  # noqa: E402

res = {"versions": np.array([diffusers.__version__, transformers.__version__, torch.__version__])}
enc = models.load_vae_encoder(sd, "cuda", True)
lp = models.load_lpips("cuda", True, pickle_path=os.path.join(home, "lpips_fn.p"))

g = torch.Generator().manual_seed(0)
x = (torch.rand(1, 3, 512, 512, generator=g) * 2 - 1).cuda()
res["latent"] = enc(x).float().cpu().numpy()

for name, fname in (("opt_sq", "sq512.png"), ("opt_wide", "wide768.png")):
    full = imageio.load_img(os.path.join(corpus, fname))
    small = imageio.shrink_shortside(full)
    src = imageio.img2tensor(small, "cuda").half()
    tgt = imageio.img2tensor(make_corpus.scene(small.size[0], small.size[1], 99), "cuda").half()
    _m, mt = get_mask(src, tgt)
    p = get_params("DEFAULT", "1")
    adv = core_optimize(src, mt, p, enc, lp, rng=random.Random(678121), og_size=(full.size[1], full.size[0])).adv
    res[name] = (adv - src).float().cpu().numpy()

pipe = models.load_img2img_pipeline(sd, "cuda", True)


def run(**kw):
    torch.manual_seed(1)
    with torch.no_grad():
        return pipe(**kw).images[0]


small = imageio.shrink_shortside(imageio.load_img(os.path.join(corpus, "sq512.png")))
res["sd_target"] = np.array(generate_target_image(small, "dog", run, device="cuda"))
np.savez(out, **res)
print("saved", out, res["versions"])
