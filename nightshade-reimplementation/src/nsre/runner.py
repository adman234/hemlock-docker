"""Batch runner for the CLI: wires an Engine into the folder workflow."""
from __future__ import annotations

import os

from . import imageio
from .engine import Engine
from .params import Params
from .repro import log, seed_everything


def run_batch(paths, out_dir, params: Params, args):
    pool = mapping = None
    if args.concepts_file:
        with open(args.concepts_file, encoding="utf8") as f:
            pool = [ln.strip() for ln in f if ln.strip()]
    if args.mapping_file:
        import json

        with open(args.mapping_file, encoding="utf8") as f:
            mapping = json.load(f)
    if not args.target and not (pool or mapping):
        raise SystemExit("provide -t/--target or --concepts-file (the original's concept pool is proprietary)")

    seed_everything(args.seed)
    eng = Engine(
        args.sd_model or "stable-diffusion-v1-5/stable-diffusion-v1-5", device=args.device,
        use_concept_models=not (args.target and args.concept),
        lpips_pickle=args.lpips_pickle, pool=pool, mapping=mapping,
    )
    log.info("device=%s half=%s", eng.device, eng.half)
    lines = []
    for path in paths:
        r = eng.shade(path, out_dir, params, tag=args.concept, target=args.target, seed=args.seed,
                      on_stage=lambda s: log.info("  %s", s))
        log.info("concept=%s target=%s", r.concept, r.target)
        lines.append(f"{r.output}: {r.concept}")
    header = ("To achieve maximal effect, please try to include the poison tags below as part of the "
              "ALT text field when you post corresponding images online. ")
    with open(os.path.join(out_dir, "nightshade-result.txt"), "w+") as f:
        f.write("\n".join([header] + lines))
