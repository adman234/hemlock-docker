"""Run the nsre reimplementation once with the same local assets as ref_run.py.

usage: python -I reimpl_run.py NSRE_SRC IMAGE OUTDIR MODE RQ TARGET [SEED]
Environment: USERPROFILE must point at the scratch home holding .nightshade assets.
Prints one JSON line prefixed with RESULT.
"""
import json
import os
import sys
import time
import traceback

nsre_src, image, outdir, mode, rq, target = sys.argv[1:7]
seed = int(sys.argv[7]) if len(sys.argv) > 7 else None
sys.path.insert(0, nsre_src)
os.makedirs(outdir, exist_ok=True)
root = os.path.join(os.path.expanduser("~"), ".nightshade")

import torch  # noqa: E402

rec = {"image": image, "mode": int(mode), "rq": int(rq), "target": target, "seed": seed}
t0 = time.time()
try:
    from nsre import device as dev_mod, models  # noqa: E402
    from nsre.params import compute_level_to_opt_setting, get_params, intensity_to_level  # noqa: E402
    from nsre.pipeline import Components, shade_image  # noqa: E402
    from nsre.repro import seed_everything  # noqa: E402
    from nsre.target import generate_target_image  # noqa: E402

    device = dev_mod.detect_device()
    half = dev_mod.use_half(device)
    sd = os.path.join(root, "base", "base")
    seed_everything(seed)
    encode = models.load_vae_encoder(sd, device, half)
    perceptual = models.load_lpips(device, half, pickle_path=os.path.join(root, "lpips_fn.p"))
    pipe = models.load_img2img_pipeline(sd, device, half)

    def run_pipeline(**kw):
        with torch.no_grad():
            return pipe(**kw).images[0]

    comps = Components(
        encode, perceptual,
        lambda small, tgt: generate_target_image(small, tgt, run_pipeline, device=device),
        device=device, half=half,
    )
    params = get_params(intensity_to_level(int(mode)), compute_level_to_opt_setting(int(rq)))
    t1 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    out = shade_image(image, outdir, params, comps, target, detected_concept="-", seed=seed)
    rec.update(output=out, device=device, init_s=t1 - t0, run_s=time.time() - t1, ok=True)
    if torch.cuda.is_available():
        rec["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1048576
except Exception as e:  # noqa: BLE001
    rec.update(ok=False, error=repr(e), tb=traceback.format_exc()[-1500:])
print("RESULT " + json.dumps(rec))
