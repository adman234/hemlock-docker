"""Drive the ORIGINAL Nightshade 1.1 modules (unpacked .pyc from the shipped app) once.

usage: python -I ref_run.py REFDIR IMAGE OUTDIR MODE RQ TARGET CONCEPT [SEED]
Run with USERPROFILE pointing at a scratch home that contains .nightshade assets.
MODE = 0/1/2 (intensity), RQ = -1..3. TARGET/CONCEPT may be '-' for auto.
Optional SEED seeds torch and python random BEFORE the call (the original never seeds torch).
Prints one JSON line with timing and peak VRAM.
"""
import json
import os
import random
import sys
import time
import traceback

refdir, image, outdir, mode, rq, target, concept = sys.argv[1:8]
seed = int(sys.argv[8]) if len(sys.argv) > 8 else None
sys.path.insert(0, refdir)
os.makedirs(outdir, exist_ok=True)

import torch  # noqa: E402

t0 = time.time()
rec = {"image": image, "mode": int(mode), "rq": int(rq), "target": target, "concept": concept, "seed": seed}
try:
    import nightshade  # noqa: E402

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    ns = nightshade.Nightshade(outdir, None)
    ns.update_params(outdir, int(mode), int(rq), None)
    rec["device"] = ns.device
    t1 = time.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    out = ns.run_protection_prod(
        [image],
        force_target=None if target == "-" else target,
        overwrite_concept=None if concept == "-" else concept,
    )
    rec.update(output=out, init_s=t1 - t0, run_s=time.time() - t1, ok=True)
    if torch.cuda.is_available():
        rec["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1048576
except Exception as e:  # noqa: BLE001
    rec.update(ok=False, error=repr(e), tb=traceback.format_exc()[-1500:])
print("RESULT " + json.dumps(rec))
