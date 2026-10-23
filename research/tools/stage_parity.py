"""Exact-equality checks of deterministic stages: ORIGINAL (unpacked .pyc) vs nsre.

usage: python -I stage_parity.py REFDIR NSRE_SRC CORPUS_DIR   (run on CPU, USERPROFILE=scratch home)
Prints one JSON object per check and a final summary line.
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch

refdir, nsre_src, corpus = sys.argv[1:4]
sys.path[:0] = [refdir, nsre_src]
home = os.path.join(os.path.expanduser("~"), ".nightshade")

import opt as ref_opt  # noqa: E402
import utils as ref_utils  # noqa: E402
from nsre import imageio, optimize, params  # noqa: E402

results = []
import contextlib
import io


def quiet(fn, *a, **k):
    """Call a reference function with its stdout chatter suppressed; return (value, exception)."""
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return fn(*a, **k), None
        except Exception as e:  # noqa: BLE001
            return None, repr(e)


DEVIATIONS = {"load_img fmt_still.gif": "original raises AttributeError(_getexif) on GIF; nsre accepts it"}


def check(name, ok, detail=""):
    if name in DEVIATIONS:
        print(json.dumps({"check": name, "ok": True, "deviation": DEVIATIONS[name], "detail": detail}))
        results.append(True)
        return
    results.append(ok)
    print(json.dumps({"check": name, "ok": bool(ok), "detail": detail}))


# parameters: every (level, opt_setting) combination
for lvl, o in itertools.product(["LOW", "DEFAULT", "HIGH"], ["0", "1", "2", "3", "4"]):
    r, _ = quiet(ref_utils.get_app_parameters, lvl, o)
    m = params.get_params(lvl, o)
    same = (r["max_change"], r["lr_initial"], r["tot_steps"], r["penalty_initial"], r["style_transfer_iter"]) == (
        m.max_change, m.lr_initial, m.tot_steps, m.penalty_initial, m.style_transfer_iter)
    same = same and (r.get("eot", 1 if o == "0" else None) == m.eot)
    check(f"params {lvl}/{o}", same, f"ref={ {k: r[k] for k in ('max_change','lr_initial','tot_steps','penalty_initial')} }")

files = sorted(f for f in glob.glob(os.path.join(corpus, "*")) if os.path.isfile(f))
fake_gen = torch.Generator().manual_seed(0)
for f in files:
    base = os.path.basename(f)
    a, ea = quiet(ref_utils.load_img, f, home)
    try:
        b, eb = imageio.load_img(f), None
    except Exception as e:  # noqa: BLE001
        b, eb = None, repr(e)
    if ea or eb:
        check(f"load_img {base}", bool(ea) == bool(eb), f"ref_exc={ea} our_exc={eb}")
        continue
    if a is None or b is None:
        check(f"load_img {base}", (a is None) == (b is None), "both None" if a is None and b is None else f"ref={a} ours={b}")
        continue
    check(f"load_img {base}", a.size == b.size and np.array_equal(np.array(a), np.array(b)), f"size={a.size}")
    t_a = ref_utils.img2tensor(a, "cpu")
    t_b = imageio.img2tensor(b, "cpu")
    check(f"img2tensor {base}", torch.equal(t_a, t_b), str(tuple(t_a.shape)))
    check(f"tensor2img {base}", np.array_equal(np.array(ref_utils.tensor2img(t_a)), np.array(imageio.tensor2img(t_b))))
    check(f"contain_image {base}", np.array_equal(np.array(ref_utils.contain_image(a.copy())), np.array(imageio.contain_image(b.copy()))))
    import torchvision.transforms as T

    sh_ref = T.Resize(512, interpolation=T.InterpolationMode.BILINEAR)(a)
    check(f"shrink_shortside {base}", np.array_equal(np.array(sh_ref), np.array(imageio.shrink_shortside(b))), f"->{sh_ref.size}")
    # lift-to-full-size and mask on perturbed tensors
    small = imageio.shrink_shortside(b)
    src = ref_utils.img2tensor(small, "cpu")
    adv = torch.clamp(src + (torch.rand(src.shape, generator=fake_gen) - 0.5) * 0.2, -1, 1)
    fr = ref_opt.Optimizer.add_perturbation_full_size(None, a, src, adv)
    fo = imageio.add_perturbation_full_size(b, src, adv)
    check(f"add_perturbation_full_size {base}", np.array_equal(np.array(fr), np.array(fo)))
    tgt = torch.clamp(src + (torch.rand(src.shape, generator=fake_gen) - 0.5) * 0.9, -1, 1)
    mr, mtr = ref_opt.Optimizer.get_mask(None, src, tgt)
    mo, mto = optimize.get_mask(src, tgt)
    check(f"get_mask {base}", torch.equal(mr, mo) and torch.equal(mtr, mto))

# output naming through the original save_image
class Dummy:
    output_dir = None


import tempfile
from PIL import Image

def _exif():
    ex = Image.Exif()
    ex[0x0112] = 1  # a non-empty EXIF block, as every real image yields; Pillow 9.3 rejects an empty one for JPEG
    return ex


with tempfile.TemporaryDirectory() as td:
    d = Dummy(); d.output_dir = td
    for name, lvl in [("cat.png", "LOW"), ("my.photo.jpg", "HIGH"), ("noext", "DEFAULT"), ("UP.PNG", "DEFAULT"), ("a.gif", "DEFAULT")]:
        img = Image.new("RGB", (8, 8), (1, 2, 3))
        try:
            p_ref = ref_opt.Optimizer.save_image(d, img, os.path.join("x", name), {"protection_level": lvl}, meta=_exif())
        except Exception as e:  # noqa: BLE001
            p_ref = f"ERR {e!r}"
        p_our = imageio.save_image(img, os.path.join("x", name), td, lvl, exif=_exif())
        fmt_ref = Image.open(p_ref).format if os.path.exists(str(p_ref)) else None
        check(f"save_image {name}", os.path.basename(str(p_ref)) == os.path.basename(p_our) and fmt_ref == Image.open(p_our).format, f"{os.path.basename(str(p_ref))} {fmt_ref}")

print(json.dumps({"summary": True, "passed": int(sum(results)), "total": len(results)}))


