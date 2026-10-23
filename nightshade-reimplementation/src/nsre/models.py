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

import os
from typing import Callable

import torch

DEFAULT_SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # public mirror of SD v1.5 (CreativeML OpenRAIL-M)
DEFAULT_BLIP_ID = "Salesforce/blip-image-captioning-base"       # blip_model.pt config names this id (U-003 resolved); weight identity unverified
DEFAULT_CLIP_ID = "openai/clip-vit-base-patch32"                # clip_model.p is ViT-B/32, 512-d (U-004 resolved); weight identity unverified


def _variant(model_dir: str, sub: str):
    """"fp16" when a local diffusers folder only holds *.fp16.safetensors for this component."""
    d = os.path.join(model_dir, sub) if os.path.isdir(model_dir) else None
    if d and os.path.isdir(d):
        names = os.listdir(d)
        if any(".fp16." in n for n in names) and not any(n in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin", "model.safetensors", "pytorch_model.bin") for n in names):
            return "fp16"
    return None


def load_vae_encoder(model_id: str, device: str, half: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return f(x[-1,1]) -> latent_dist.mean, matching Optimizer.get_latent_with_crop's encode step."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", **_kw(model_id, "vae")).to(device)
    if half:
        vae = vae.half()
    vae.requires_grad_(False)

    def encode(x: torch.Tensor) -> torch.Tensor:
        if half:
            x = x.half()
        return vae.encode(x).latent_dist.mean

    return encode


def _kw(model_id: str, sub: str) -> dict:
    v = _variant(model_id, sub)
    return {"variant": v} if v else {}


def load_img2img_pipeline(model_id: str, device: str, half: bool):
    """StableDiffusionImg2ImgPipeline; cuda/mps use fp16 weights, CPU uses fp32 + attention slicing."""
    from diffusers import StableDiffusionImg2ImgPipeline

    extra = dict(safety_checker=None, requires_safety_checker=False, **_kw(model_id, "unet"))
    if half:
        return StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, **extra).to(device)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32, **extra).to(device)
    pipe.enable_attention_slicing()
    return pipe


def load_lpips(device: str, half: bool, net: str = "vgg", pickle_path: str | None = None) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """LPIPS distance. The original's lpips_fn.p is an LPIPS module with a vgg16 backbone (U-005,
    resolved by pickletools inspection). `net='vgg'` builds the same architecture from the lpips
    package (ImageNet backbone weights come from torchvision's download). `pickle_path` loads a
    locally held lpips_fn.p instead, for local parity experiments only (unpickles code: trusted
    files only; never redistributed)."""
    import lpips

    if pickle_path:
        import pickle

        with open(pickle_path, "rb") as f:
            fn = pickle.load(f).to(device)
    else:
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


