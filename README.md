# hemlock-docker

Evidence-driven reverse engineering of the Nightshade 1.1 (Windows) desktop application and an independent reimplementation of
its shading pipeline, for authorized research and interoperability.

| Path | Content |
|---|---|
| `research/evidence.md` | evidence ledger (E-xxx), the source of truth for every claim |
| `research/artifact-inventory.md`, `dependency-map.md`, `analysis-plan.md` | Phase 1 |
| `research/architecture.md`, `data-flow.md`, `function-map.md`, `parameter-table.md` | Phase 2 |
| `research/experiment-protocol.md`, `experiments.csv`, `reference-observations.md` | Phase 3 (designed, not yet run) |
| `research/parity-report.md`, `known-limitations.md` | Phase 5 |
| `research/tools/` | scripts that unpack the PyInstaller archive and disassemble the app modules (own code, no vendor code) |
| `nightshade-reimplementation/` | independent implementation (`nsre`) and tests |

No vendor code, disassembly, model weights or proprietary assets are stored here. Re-run `research/tools/` against your own
copy of the application to regenerate the raw material.

## Run the tests

```
cd nightshade-reimplementation
pip install torch torchvision numpy pillow einops pytest
PYTHONPATH=src:tests python -m pytest -q
```

## Headline findings

- The app is a PyInstaller build with plain, readable Python 3.11 bytecode. No model weights ship; nine assets download on first run.
- The attack is a targeted feature-space attack on the **Stable Diffusion v1.5 VAE encoder**: SD img2img makes a target image of a different concept, then a perturbation is optimized so the perturbed image's VAE latent matches the target's, under an LPIPS hinge penalty and a 1.3 x max_change clamp.
- Intensity maps to max_change 0.035 / 0.07 / 0.12; render quality maps to 100 / 150 / 300 / 500 steps with 1 / 2 / 3 / 6 crop samples per step (HIGH adds 200 steps).
- Torch is never seeded, so SD target synthesis is nondeterministic; crop offsets are reseeded per image when the target comes from the pool.
- Open: numerical parity with the reference (needs the downloaded assets and a run of the original), and the proprietary target pool.
