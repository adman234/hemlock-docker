# Analysis plan and status

Decision: recover logic from CPython bytecode (high-level artifacts, rule 3). Native decompilation is not needed.

| Step | Status | Output |
|---|---|---|
| 1 Identify version, packaging, runtime | done | artifact-inventory.md, E-001..E-007 |
| 2 Extract CArchive and PYZ app modules (copies only) | done | `research/tools/carch.py`, `pyz.py` |
| 3 Disassemble `nightshade`, `opt`, `utils`, `downloader`, `app` | done | E-010..E-072 |
| 4 Parameter tables and call graph | done | parameter-table.md, function-map.md |
| 5 Data-flow and architecture write-up | done | architecture.md, data-flow.md |
| 6 Independent implementation with injected models | done | `nightshade-reimplementation/` |
| 7 Offline tests (stand-in encoder and LPIPS) | done, 63 pass, 1 skipped (CUDA) | `tests/` |
| 8 Download assets and run the reference app | done (user ran the downloads, all MD5s match) | experiments.csv, results/ |
| 9 Reference vs reimplementation parity measurements | done | parity-report.md, measured-results.md |
| 10 Inspect pickled assets (`pickletools`) to resolve U-001..U-006 | done | evidence.md E-090..E-095 |

## Remaining high-value experiments

1. Measure whether the protection holds up (fine-tune a small model on shaded vs unshaded sets). This is the main unanswered question (U-013).
2. Compare the Hugging Face BLIP/CLIP checkpoints against the originals numerically (needs those downloads).
3. Isolate the VRAM difference (U-012).
4. Capture the original's runtime network behaviour (U-011).
5. Review licences and terms (U-010).
