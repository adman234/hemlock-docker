# Data flow

Shapes use N=1, H and W after the stated step. Value ranges: "u8" = 0..255, "[-1,1]" = SD convention.

| # | Stage | Input | Operation | Output | Evidence | Confidence |
|---|---|---|---|---|---|---|
| 1 | Import | path | `os.path.exists`; `Image.open`; errors give `None` | PIL image | E-040 | O |
| 2 | Orientation | PIL | EXIF 3/6/8 via `rotate(180/270/90, expand=True)`; other values ignored; EXIF read failure with `OSError` gives `None` | PIL | E-040 | O |
| 3 | Colour | PIL any mode | `.convert('RGB')` (alpha dropped by PIL, not composited) | RGB u8 | E-040 | O |
| 4 | Size cap | RGB | if long side > 5120: `thumbnail((5120,5120), LANCZOS)` | RGB u8 | E-041 | O |
| 5 | Concept | RGB (full size) | BLIP with prefix; nouns; CLIP tie-break | str | E-060, E-061 | O |
| 6 | Target concept | str | CLIP sim; ranks 4..9; `random.seed(678121)`; `choice` | str | E-062 | O |
| 7 | Working copy | RGB (H0,W0) | `Resize(512, BILINEAR)`: short side 512, other side scaled | RGB (512,Wl) or (Hl,512) | E-042 | O |
| 8 | Target synthesis | working copy | `contain_image` (long side 512) at top-left of black 512x512 canvas; SD img2img; crop; `resize(working size)` | RGB u8, working size | E-045 | O |
| 9 | Tensors | working copy, target | `x/127.5 - 1`, float32, `c h w`, device, fp16 on cuda/mps | `[1,3,H,W]` [-1,1] | E-042 | O |
| 10 | Mask | src, tgt | threshold 0.3 on channel-summed diff; forced all ones outside 5%..30% | masked target `[1,3,H,W]` | E-055 | O |
| 11 | Crop plan | src shape | one side must be 512; `r_range = long - 512` | crop offsets | E-050 | O |
| 12 | Optimize | src, masked target | see architecture; `δ` init 0; `T` steps x `eot` crops | adv `[1,3,H,W]` [-1,1] | E-051..E-054 | O |
| 13 | Lift to full res | full RGB, src, adv | `resize(tensor2img(adv)) - resize(tensor2img(src))` added in float32 | full-size RGB | E-044 | O |
| 14 | Quantize | float | `clip(0,255).astype(uint8)` (truncate) | u8 | E-044 | O |
| 15 | Save | PIL, EXIF | PNG if path ends `.png`, else JPEG `quality=100`, `exif=` original | file `<stem>-protected-intensity-<LEVEL>-V1.<ext>` | E-043 | O |
| 16 | Report | concepts | `nightshade-result.txt` with ALT-text header and `path: concept` lines | text | E-022 | O |

## Sources of nondeterminism

| Source | Original behaviour | Evidence |
|---|---|---|
| SD img2img sampling noise | torch is never seeded (`CHECK_RUN=False`); output varies per run | E-056 |
| Crop offsets | `random.randrange` on the Python global RNG; reseeded to 678121 by target selection, so deterministic per image when the target comes from the pool path | E-052, E-062, E-063 |
| fp16 GPU kernels | cudnn/cublas selection may vary by hardware | I |
| Concept pick | deterministic given models | E-061 |

## Error handling observed

- Unreadable image: `load_img` returns `None`, the image is skipped silently (`continue`).
- Missing output dir: exception `Output folder {} does not exist.`; unselected: `Please select output folder before proceeding.`
- More than 3 images on MPS: exception.
- Processed shape with neither side equal to 512: bare `Exception` with no message (E-050).
- Non-image files in the input directory are filtered by `check_is_img`.
