"""Model and resource loading.

The original downloads Stable Diffusion v1.5 components (unet, text_encoder, vae) plus
proprietary pickles (CLIP, LPIPS, BLIP, target-concept asset) into ~/.nightshade
(evidence E-060..E-064). This module loads *publicly distributable* equivalents instead and
never touches or redistributes the original assets. Every loader imports its heavy dependency
lazily so the rest of the package works without them.

Nothing here downloads automatically except through the Hugging Face / lpips libraries the
caller chooses to install and run; model ids are parameters.
"""
from __future__ import annotations

from typing import Callable

import torch

DEFAULT_SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # public mirror of SD v1.5 (CreativeML OpenRAIL-M)
DEFAULT_BLIP_ID = "Salesforce/blip-image-captioning-large"      # approximation of blip_model.pt (L-002)
DEFAULT_CLIP_ID = "openai/clip-vit-base-patch32"                # approximation of clip_model.p (L-003)


def load_vae_encoder(model_id: str, device: str, half: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return f(x[-1,1]) -> latent_dist.mean, matching Optimizer.get_latent_with_crop's encode step."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    if half:
        vae = vae.half()
    vae.requires_grad_(False)

    def encode(x: torch.Tensor) -> torch.Tensor:
        if half:
            x = x.half()
        return vae.encode(x).latent_dist.mean

    return encode


def load_img2img_pipeline(model_id: str, device: str, half: bool):
    """StableDiffusionImg2ImgPipeline; cuda/mps use fp16 weights, CPU uses fp32 + attention slicing."""
    from diffusers import StableDiffusionImg2ImgPipeline

    if half:
        return StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id).to(device)
    pipe.enable_attention_slicing()
    return pipe


def load_lpips(device: str, half: bool, net: str = "alex") -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """LPIPS distance. The backbone of the original lpips_fn.p is unknown until the pickle is
    inspected (U-005); 'alex' is the lpips package default and is used as an approximation."""
    import lpips

    fn = lpips.LPIPS(net=net).to(device)
    if half:
        fn = fn.half()
    fn.requires_grad_(False)
    return lambda a, b: fn(a.half() if half else a, b.half() if half else b)


def load_blip(model_id: str, device: str):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    return BlipProcessor.from_pretrained(model_id), BlipForConditionalGeneration.from_pretrained(model_id).to(device)


def load_clip(model_id: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    return CLIPProcessor.from_pretrained(model_id), CLIPModel.from_pretrained(model_id).to(device)
