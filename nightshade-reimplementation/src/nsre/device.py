"""CPU/GPU dispatch, mirroring nightshade.Nightshade.detect_device / check_device_health."""
from __future__ import annotations

import torch

MIN_CUDA_MB = 4500       # original: GPU memory must exceed 4500 MB else CPU fallback
XL_GPU_MB = 10000        # original Optimizer: >10000 MB keeps the SD model resident ("extra_large_gpu")


def check_device_health(device: str) -> bool:
    """Half-precision 9*3 == 27 smoke test on the accelerator."""
    if device == "cpu":
        return False  # original returns False for cpu; it is only ever called for accelerators
    try:
        x = torch.ones(1, device=device).half() * 9
        y = torch.ones(1, device=device).half() * 3
        return int(x * y) == 27
    except Exception:
        return False


def detect_device() -> str:
    if torch.cuda.is_available():
        mb = torch.cuda.get_device_properties(0).total_memory / 1048576
        device = "cuda" if mb > MIN_CUDA_MB else "cpu"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if device != "cpu" and not check_device_health(device):
        device = "cpu"
    return device


def use_half(device: str) -> bool:
    """Original: fp16 for cuda/mps, fp32 on CPU."""
    return device in ("cuda", "mps")


def extra_large_gpu() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / 1048576 > XL_GPU_MB)
