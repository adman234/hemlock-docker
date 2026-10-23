"""Target image generation (style transfer of the image towards a target concept).

Mirrors opt.Optimizer.get_target_image (evidence E-045..E-047): contain the image in a 512x512
black canvas, run SD img2img with prompt 'a photo of a {target}', strength 0.8, guidance 7.5 and
30 steps, crop back to the contained region, resize to the working image size.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image

from .imageio import contain_image, img2tensor
from .params import GUIDANCE_SCALE, PROCESS_SIZE, STYLE_STRENGTH, STYLE_TRANSFER_ITER


def pad_to_canvas(img: Image.Image, size: int = PROCESS_SIZE) -> np.ndarray:
    """Place the contained image at the top-left of a black size x size canvas."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    arr = np.array(img)
    canvas[: img.size[1], : img.size[0], :] = arr
    return canvas


def crop_from_canvas(canvas: np.ndarray, size_wh: tuple[int, int]) -> Image.Image:
    w, h = size_wh
    return Image.fromarray(canvas[:h, :w, :])


def generate_target_image(
    cur_img: Image.Image,
    prompt_concept: str,
    run_pipeline: Callable[..., Image.Image],
    steps: int = STYLE_TRANSFER_ITER,
    strength: float = STYLE_STRENGTH,
    guidance: float = GUIDANCE_SCALE,
    device: str = "cpu",
) -> Image.Image:
    """run_pipeline(prompt=[...], image=<[-1,1] tensor>, strength, guidance_scale, num_inference_steps)
    must return a PIL image (the first of pipe(...).images)."""
    contained = contain_image(cur_img.copy())
    canvas = pad_to_canvas(contained)
    tensor = img2tensor(Image.fromarray(canvas), device)
    out = run_pipeline(
        prompt=[f"a photo of a {prompt_concept}"],
        image=tensor,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=steps,
    )
    cropped = crop_from_canvas(np.array(out), contained.size)
    return cropped.resize(cur_img.size)
