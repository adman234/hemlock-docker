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
| 7 Offline tests (stand-in encoder and LPIPS) | done, 55 pass, 1 skipped (CUDA) | `tests/` |
| 8 Download assets and run the reference app | **blocked on approval** (see below) | experiments.csv |
| 9 Reference vs reimplementation parity measurements | blocked on 8 | parity-report.md |
| 10 Inspect unpickled assets without executing them (`pickletools`) to resolve U-003..U-005 | blocked on 8 | evidence.md |

## Why step 8 is blocked

The reference app downloads several GB (sizes are not in the bytecode; the SD v1.5 unet alone is a multi-GB fp32 file) over plain HTTP from `mirror.cs.uchicago.edu` and Hugging Face on first launch
(from general knowledge, not verified here, the `runwayml/stable-diffusion-v1-5` repository was taken down, so the app's own URL may 404, U-009),
and it unpickles several downloaded files. Fetching those files and running the app with them is a download the mission did
not explicitly authorize, so it needs a go-ahead. Proposed isolation: run the app in a Windows Sandbox or a throwaway user
profile with the install directory read-only, `.nightshade` pointed at a scratch location, and a network allow-list limited to
the two hosts above. Equivalent public SD v1.5 weights can be placed in `base/base` by hand to avoid the dead HF link.

## Highest-value next experiments

1. Run the reference at LOW/DEFAULT/HIGH x render quality 0 on a fixed 512x512 and a 768x512 image; record outputs, runtimes, VRAM.
2. Same input twice: quantify run-to-run variability (expected: SD sampling noise only, since torch is unseeded and crops are reseeded).
3. Inspect `lpips_fn.p` with `pickletools.dis` (no unpickling) to identify the backbone (U-005).
4. Inspect decrypted `full_asset3.p` structure with `pickletools` to learn the pool size and embedding model (U-001, U-004).
5. Compare reimplementation output with the reference on the same target image (inject the reference's target image) to isolate the optimizer from SD noise.
