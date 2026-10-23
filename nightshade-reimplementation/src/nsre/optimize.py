"""Core optimization and perturbation constraints.

Reimplements opt.Optimizer.core_optimize / get_mask / get_latent_with_crop from Nightshade 1.1
(evidence E-040..E-049). Models are injected so the algorithm is testable without weights.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .params import (
    CLAMP_SLACK,
    LR_FINAL_FRACTION,
    MASK_DIFF_THRESHOLD,
    MASK_FRAC_HIGH,
    MASK_FRAC_LOW,
    PROCESS_SIZE,
    Params,
)

LatentFn = Callable[[torch.Tensor], torch.Tensor]  # [1,3,512,512] in [-1,1] -> latent mean
PerceptualFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (adv, src) -> LPIPS distance


def get_mask(source: torch.Tensor, target: torch.Tensor):
    """Return (mask, masked_target). Mask is forced to all ones when the changed fraction is
    outside [0.05, 0.30]. NOTE: the original computes this mask but core_optimize never reads
    it; only masked_target is used downstream. Kept for fidelity."""
    diff = (source - target).sum(1, keepdim=True)
    mask = (diff.abs() > MASK_DIFF_THRESHOLD).int()
    mask = torch.concat([mask] * 3, dim=1)
    masked_target = target * mask + (1 - mask) * source
    frac = torch.mean(mask.float())
    if frac > MASK_FRAC_HIGH or frac < MASK_FRAC_LOW:
        mask = mask * 0 + 1
    return mask, masked_target


def crop_geometry(height: int, width: int, size: int = PROCESS_SIZE):
    """Return (r_range, long_width, eot_override). Mirrors the three accepted shapes."""
    if width > size and height == size:
        return width - size, True, None
    if height > size and width == size:
        return height - size, False, None
    if height == size and width == size:
        return 0, False, 1
    raise ValueError(f"unsupported processed shape {height}x{width}; one side must equal {size}")


def crop_for_latent(x: torch.Tensor, crop_idx: int, long_width: bool, size: int = PROCESS_SIZE) -> torch.Tensor:
    """F.crop(top, left, h, w): along the long axis only."""
    if long_width:
        return TF.crop(x, 0, crop_idx, size, size)
    return TF.crop(x, crop_idx, 0, size, size)


def step_size(lr_initial: float, i: int, total: int) -> float:
    """Linear decay from lr_initial at i=0 towards lr_initial/100 at i=total."""
    return lr_initial - ((lr_initial - lr_initial * LR_FINAL_FRACTION) / total) * i


@dataclass
class OptimizeResult:
    adv: torch.Tensor
    modifier: torch.Tensor
    losses: list
    lpips_losses: list


def core_optimize(
    source: torch.Tensor,
    target: torch.Tensor,
    params: Params,
    encode_latent: LatentFn,
    perceptual: PerceptualFn,
    rng: Optional[random.Random] = None,
    progress: Optional[Callable[[float], None]] = None,
    og_size: Optional[tuple] = None,
) -> OptimizeResult:
    """Minimise ||E(resize(clamp(src+delta))) - E(target)||_2 (latent L2) plus an LPIPS hinge
    penalty, with raw-gradient steps and a +-1.3*max_change clamp on delta.

    source/target: [1,3,H,W] in [-1,1], shorter side 512 (same shape).
    """
    rng = rng or random.Random()
    height, width = source.shape[-2], source.shape[-1]
    r_range, long_width, eot_override = crop_geometry(height, width)
    eot = 1 if eot_override else params.eot

    # The original resizes adv to the FULL image size (height, width) and back to a 512 short side
    # every step (E-052), so downscaling loss is part of the objective. og_size=(H0, W0).
    resize_large = T.Resize(tuple(og_size) if og_size else (height, width))
    resize_proc = T.Resize(PROCESS_SIZE)
    modifier = torch.clone(source) * 0.0
    max_change = params.max_change
    bound = max_change * CLAMP_SLACK
    losses, lp_losses = [], []

    for i in range(params.tot_steps):
        if progress:
            progress(i / params.tot_steps * 100)
        lr = step_size(params.lr_initial, i, params.tot_steps)
        modifier.requires_grad_(True)
        tot_grad = 0.0
        for _ in range(eot):
            crop_idx = rng.randrange(0, r_range) if r_range != 0 else 0
            tgt_lat = encode_latent(crop_for_latent(target, crop_idx, long_width))
            adv = torch.clamp(modifier + source, -1, 1)
            adv = resize_proc(resize_large(adv))
            adv_lat = encode_latent(crop_for_latent(adv, crop_idx, long_width))
            loss = (adv_lat - tgt_lat).norm()
            grad = torch.autograd.grad(loss, modifier)[0]
            tot_grad = tot_grad + grad
            losses.append(float(loss.detach()))
        tot_grad = tot_grad / eot

        adv = torch.clamp(modifier + source, -1, 1)
        lp = perceptual(adv, source)
        penalty = torch.maximum(lp - max_change, lp * 0).sum() * params.penalty_initial
        grad_lp = torch.autograd.grad(penalty, modifier)[0]
        lp_losses.append(float(lp.detach().mean()))
        tot_grad = tot_grad + grad_lp

        modifier = modifier - tot_grad * lr
        modifier = torch.clamp(modifier, min=-bound, max=bound)
        modifier.grad = None

    modifier = modifier.detach()
    adv = torch.clamp(modifier + source, -1, 1)
    return OptimizeResult(adv=adv, modifier=modifier, losses=losses, lpips_losses=lp_losses)
