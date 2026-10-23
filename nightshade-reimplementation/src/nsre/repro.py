"""Logging and reproducibility helpers: seeds and per-run manifests."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import random
import time
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("nsre")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed: Optional[int]) -> None:
    """Optional. The original never seeds torch (CHECK_RUN is False) so SD sampling is
    nondeterministic; seeding here is an opt-in reproducibility feature, not parity behaviour."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def write_manifest(path: str, record: dict) -> None:
    record = dict(record, written_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(path, "w", encoding="utf8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
