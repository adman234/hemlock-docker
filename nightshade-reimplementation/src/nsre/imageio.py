"""Image I/O and preprocessing.

Mirrors utils.load_img / reduce_quality / contain_image / img2tensor / tensor2img and
opt.Optimizer.save_image from Nightshade 1.1 (evidence E-030..E-037). Independently written.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from PIL import Image, ExifTags, UnidentifiedImageError

from .params import MAX_RESOLUTION, PROCESS_SIZE

SUPPORTED_SAVE_FORMATS = {".png": "PNG"}  # everything else is saved as JPEG (quality 100)


def check_is_img(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def reduce_quality(img: Image.Image, max_res: int = MAX_RESOLUTION) -> Image.Image:
    """Downscale in place (Lanczos thumbnail) when the long side exceeds max_res."""
    if max(img.size) > max_res:
        img.thumbnail((max_res, max_res), Image.LANCZOS)
    return img


def load_img(path: str) -> Optional[Image.Image]:
    """Open, apply EXIF orientation 3/6/8 via rotate(expand=True), convert to RGB, cap size.

    Returns None for missing paths, directories, unidentified images or OS errors,
    like the original.
    """
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path)
    except (UnidentifiedImageError, IsADirectoryError, OSError):
        return None
    try:
        exif = img._getexif()  # noqa: SLF001 - same private API the original uses
    except OSError:
        return None  # original: OSError (e.g. truncated PNG) -> skip the image
    except AttributeError:
        # Formats without _getexif (GIF, ...) crash the original with AttributeError even though
        # its file dialog offers *.gif (E-080). We deliberately accept them instead.
        exif = None
    if exif:
        orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)
        o = exif.get(orientation_key) if orientation_key is not None else None
        if o == 3:
            img = img.rotate(180, expand=True)
        elif o == 6:
            img = img.rotate(270, expand=True)
        elif o == 8:
            img = img.rotate(90, expand=True)
    img = img.convert("RGB")
    return reduce_quality(img)


def contain_image(img: Image.Image, size: int = PROCESS_SIZE) -> Image.Image:
    """Fit inside a size x size box: the LONGER side becomes `size`, aspect preserved."""
    w, h = img.size
    if w < h:
        new_h = size
        new_w = int(w / (h / size))
    elif w > h:
        new_w = size
        new_h = int(h / (w / size))
    else:
        new_w = new_h = size
    return img.resize((new_w, new_h))


def shrink_shortside(img: Image.Image, size: int = PROCESS_SIZE) -> Image.Image:
    """torchvision Resize(512, BILINEAR) on a PIL image: SHORTER side becomes 512."""
    import torchvision.transforms as T

    return T.Resize(size, interpolation=T.InterpolationMode.BILINEAR)(img)


def img2tensor(img: Image.Image, device: str | torch.device = "cpu") -> torch.Tensor:
    """PIL RGB -> [1,3,H,W] float32 in [-1,1] using x/127.5 - 1."""
    # divide in float64, then cast: bit-identical to the original (float32 division differs by 1 ulp)
    arr = (np.array(img) / 127.5 - 1.0).astype(np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr).unsqueeze(0).to(device)


def tensor2img(t: torch.Tensor) -> Image.Image:
    """[-1,1] tensor -> PIL. Truncating (not rounding) uint8 cast, as in the original."""
    if t.dim() == 3:
        t = t.unsqueeze(0)
    t = torch.clamp((t.detach() + 1.0) / 2.0, 0.0, 1.0)
    arr = (t[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def output_name(og_path: str, level: str, tag: str = "V1", is_target: bool = False) -> str:
    """'<stem>-protected-intensity-<LEVEL>-V1.<ext>' (target variant adds 'target-')."""
    base = os.path.basename(og_path)
    if "." in base:
        stem, ext = ".".join(base.split(".")[:-1]), base.split(".")[-1]
    else:
        stem, ext = base, None
    mid = "-protected-intensity-target-{}-{}" if is_target else "-protected-intensity-{}-{}"
    name = stem + mid.format(level, tag)
    return name if ext is None else name + "." + ext


def save_image(img: Image.Image, og_path: str, out_dir: str, level: str, exif=None,
               reset_exif_orientation: bool = False) -> str:
    """PNG when the original path ends in '.png' (case-sensitive, like the original), else JPEG q=100.

    The original keeps the source EXIF verbatim even though load_img already rotated the pixels, so an
    Orientation tag of 3/6/8 makes viewers rotate the result a second time (E-081). With
    reset_exif_orientation=True the tag is set to 1; the default keeps parity with the original.
    """
    out = os.path.join(out_dir, output_name(og_path, level))
    fmt = "PNG" if og_path.endswith(".png") else "JPEG"
    kwargs = {"quality": 100}
    if exif is not None:
        if reset_exif_orientation and 0x0112 in exif:
            exif[0x0112] = 1
        kwargs["exif"] = exif
    img.save(out, format=fmt, **kwargs)
    return out


def add_perturbation_full_size(full_raw: Image.Image, source_t: torch.Tensor, perturbed_t: torch.Tensor) -> Image.Image:
    """Upsample the 512-scale perturbation (resize(perturbed) - resize(source)) onto the full image.

    PIL default resample (bicubic for resize) is used by the original (`.resize(og_size)` with no filter).
    Clip to [0,255] and truncate to uint8.
    """
    og = full_raw.size
    up_src = np.array(tensor2img(source_t).resize(og)).astype(np.float32)
    up_adv = np.array(tensor2img(perturbed_t).resize(og)).astype(np.float32)
    out = np.clip(np.array(full_raw).astype(np.float32) + (up_adv - up_src), 0, 255).astype(np.uint8)
    return Image.fromarray(out)

