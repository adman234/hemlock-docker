"""Command line interface. Flag names follow the original nightshade.main (-d, -od, -m, -rq, -t)."""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

from . import imageio
from .params import compute_level_to_opt_setting, get_params, intensity_to_level


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nsre", description="Nightshade 1.1 reimplementation (research)")
    p.add_argument("-d", "--directory", default="imgs/", help="input directory (all files, like the original)")
    p.add_argument("-od", "--out-dir", default=None, help="output directory (default: input directory)")
    p.add_argument("-m", "--mode", type=int, default=0, choices=[0, 1, 2],
                   help="intensity: 0=LOW 1=DEFAULT 2=HIGH (original CLI default is 0)")
    p.add_argument("-rq", "--rq", type=int, default=0, choices=[-1, 0, 1, 2, 3],
                   help="render quality: 0..3 (-1 is the original's 4-step smoke mode)")
    p.add_argument("-t", "--target", default=None, help="force target concept (skips target selection)")
    p.add_argument("--concept", default=None, help="override detected concept (single image)")
    p.add_argument("--concepts-file", default=None, help="newline-separated target concept pool")
    p.add_argument("--mapping-file", default=None, help="optional JSON concept->target overrides (like the original precompute_mapping)")
    p.add_argument("--sd-model", default=None, help="SD v1.5 diffusers id or local path")
    p.add_argument("--lpips-pickle", default=None, help="local lpips_fn.p for parity experiments (trusted file)")
    p.add_argument("--device", default=None, help="cpu|cuda|mps (default: auto)")
    p.add_argument("--seed", type=int, default=None, help="optional seed for crops/SD sampling")
    p.add_argument("--dry-run", action="store_true", help="validate inputs and print parameters only")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    out_dir = args.out_dir or args.directory
    level = intensity_to_level(args.mode)
    params = get_params(level, compute_level_to_opt_setting(args.rq))

    paths = [
        f for f in sorted(glob.glob(os.path.join(args.directory, "*")))
        if os.path.isfile(f) and imageio.check_is_img(f)
        # NOTE: the original skips names containing '-nightshade-intensity', a string its own
        # outputs never contain ('-protected-intensity-'); we skip the real output marker.
        and "-protected-intensity" not in f
    ]
    print(f"parameters: {params.as_dict()}")
    print(f"{len(paths)} image(s) to process; output -> {out_dir}")
    if args.dry_run:
        return 0
    if not paths:
        print("no images found", file=sys.stderr)
        return 1
    if not os.path.isdir(out_dir):
        print(f"Output folder {out_dir} does not exist.", file=sys.stderr)
        return 2

    from .runner import run_batch  # heavy imports live here

    run_batch(paths, out_dir, params, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

