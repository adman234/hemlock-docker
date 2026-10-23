"""Image-fidelity metrics for reference/reimplementation comparisons.

usage: python -I metrics.py ORIGINAL SHADED [--json]
"""
import json
import math
import sys

import numpy as np
from PIL import Image


def compare(a_path: str, b_path: str) -> dict:
    a, b = Image.open(a_path), Image.open(b_path)
    info = {"a_size": a.size, "b_size": b.size, "a_format": a.format, "b_format": b.format, "a_mode": a.mode, "b_mode": b.mode}
    if a.size != b.size:
        info["error"] = "size mismatch"
        return info
    x = np.array(a.convert("RGB")).astype(np.float64)
    y = np.array(b.convert("RGB")).astype(np.float64)
    d = y - x
    mse = float((d ** 2).mean())
    info.update(
        max_abs_error=float(np.abs(d).max()),
        mean_abs_error=float(np.abs(d).mean()),
        frac_changed=float((np.abs(d).sum(-1) > 0).mean()),
        psnr_db=float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse),
        per_channel_mean_abs=[float(np.abs(d[..., c]).mean()) for c in range(3)],
    )
    return info


if __name__ == "__main__":
    r = compare(sys.argv[1], sys.argv[2])
    print(json.dumps(r, indent=2) if "--json" in sys.argv else r)
