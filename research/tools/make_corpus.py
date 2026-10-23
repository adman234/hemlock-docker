"""Deterministic synthetic test corpus (no third-party artwork).

usage: python make_corpus.py OUTDIR
Scenes are gradients, shapes and mild texture so SD img2img and codecs behave sensibly.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def scene(w, h, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 200 // w + 30, yy * 160 // h + 60, (xx + yy) * 120 // (w + h) + 90], -1).astype(np.uint8)
    img = Image.fromarray(base)
    d = ImageDraw.Draw(img)
    for _ in range(6):
        x0, y0 = int(rng.integers(0, w * 3 // 4)), int(rng.integers(0, h * 3 // 4))
        x1, y1 = x0 + int(rng.integers(w // 8, w // 3)), y0 + int(rng.integers(h // 8, h // 3))
        col = tuple(int(c) for c in rng.integers(20, 235, 3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([x0, y0, x1, y1], fill=col)
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    arr = np.array(img).astype(np.int16) + rng.integers(-6, 7, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


SPECS = {
    "sq512.png": (512, 512, 1), "wide768.png": (768, 512, 2), "tall512x768.png": (512, 768, 3),
    "sq1024.png": (1024, 1024, 4), "big2000x1500.png": (2000, 1500, 5),
}

if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for name, (w, h, s) in SPECS.items():
        scene(w, h, s).save(os.path.join(out, name))
    scene(768, 512, 2).save(os.path.join(out, "wide768.jpg"), quality=92)
    base = scene(768, 512, 2)
    a = np.zeros((512, 768), np.uint8); a[:, :] = np.linspace(40, 255, 768).astype(np.uint8)[None, :]
    base.convert("RGBA").putalpha(Image.fromarray(a))
    rgba = base.convert("RGBA"); rgba.putalpha(Image.fromarray(a)); rgba.save(os.path.join(out, "fmt_rgba.png"))
    base.convert("L").save(os.path.join(out, "fmt_gray.png"))
    base.convert("P", palette=Image.ADAPTIVE).save(os.path.join(out, "fmt_palette.png"))
    ex = Image.Exif(); ex[0x0112] = 6
    scene(512, 768, 7).save(os.path.join(out, "fmt_exif6.jpg"), quality=92, exif=ex)
    base.convert("P", palette=Image.ADAPTIVE).save(os.path.join(out, "fmt_still.gif"))
    base.save(os.path.join(out, "fmt_upper.PNG"), format="PNG")
    open(os.path.join(out, "bad_zero.png"), "wb").close()
    data = open(os.path.join(out, "wide768.png"), "rb").read()
    open(os.path.join(out, "bad_trunc.png"), "wb").write(data[: len(data) // 3])
    open(os.path.join(out, "bad_text.png"), "w").write("this is not an image")
    scene(512, 512, 1).save(os.path.join(out, "a-nightshade-intensity.png"))
    scene(512, 512, 1).save(os.path.join(out, "a-protected-intensity-LOW-V1.png"))
    print(sorted(os.listdir(out)))

