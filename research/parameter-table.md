# Parameter table

Defaults are the values the shipped app uses. "GUI default" is the slider position at start.

## User-facing controls

| Control | Range | GUI default | CLI flag (default) | Maps to |
|---|---|---|---|---|
| Intensity | 0, 1, 2 | 1 | `-m` (0) | LOW, DEFAULT, HIGH |
| Render quality | 0..3 | 0 | `-rq` (0) | opt_setting `'1'..'4'` (+1) |
| Poison tag | free text, at most 3 words, single image only | auto-detected | `-t` forces the *target* concept; the GUI tag overrides the *detected* concept | E-020 |

## Intensity (E-030)

| Level | max_change | lr_initial | Clamp on δ (1.3 x max_change) | Approx. 8-bit level |
|---|---|---|---|---|
| LOW | 0.035 | 0.03 | 0.0455 | 5.8 |
| DEFAULT | 0.07 | 0.1 | 0.091 | 11.6 |
| HIGH | 0.12 | 0.2 | 0.156 | 19.9 |

(Tensor range is 2 for 255 levels, so 1 tensor unit = 127.5 levels.) `max_change` is also the hinge threshold on the LPIPS distance, so one number sets both the perceptual budget and, via 1.3 x, the L-infinity box.

## Render quality (E-032)

| GUI rq | opt_setting | eot | tot_steps | penalty_initial | UI time labels (set A / set B) |
|---|---|---|---|---|---|
| (-1, CLI smoke) | `'0'` | none (E-034) | 4 | 500 | n/a |
| 0 Faster | `'1'` | 1 | 100 | 1000 | 5 min / ~30 min |
| 1 Medium | `'2'` | 2 | 150 | 500 | ~10 min / ~70 min |
| 2 Slower | `'3'` | 3 | 300 | 500 | ~15 min / ~120 min |
| 3 Slowest | `'4'` | 6 | 500 | 800 | ~20 min / ~180 min |

HIGH intensity adds 200 steps to every row. The two UI time columns are the two label sets in the GUI string table (E-020 region); which one belongs to GPU versus CPU is **not** established (U-008).

## Fixed constants

| Name | Value | Evidence |
|---|---|---|
| PROCESS_SIZE | 512 | E-042 |
| MAX_RES (long side cap) | 5120 | E-041 |
| SD strength / guidance / steps | 0.8 / 7.5 / 30 (`style_transfer_iter`) | E-045 |
| Target prompt | `a photo of a {target}` | E-022 |
| LR schedule | `lr - ((lr - lr/100)/T) * i` | E-051 |
| δ clamp | `+-1.3 * max_change` | E-053 |
| Loss | `||E(adv) - E(tgt)||₂` (VAE latent mean), plus `λ * sum(max(LPIPS - max_change, 0))`, λ = penalty_initial | E-052, E-053 |
| Mask threshold / fraction window | 0.3 / [0.05, 0.30] (mask unused downstream) | E-055 |
| Concept blocklist | image, photo, white, new, black, design, painting | E-061 |
| Fallback concept | painting | E-061 |
| Target rank window / seed | [4:10] / 678121 | E-062 |
| BLIP prefix | `a photography of` | E-060 |
| JPEG quality | 100, no subsampling arg (PIL default) | E-043 |
| Min GPU memory | 4500 MB (cuda); "extra large" > 10000 MB | E-070, E-047 |
| MPS batch limit | 3 images | E-022 |

## Present but unused

`lr_const = 0.005`, `n_runs = 1` (E-033); sign-gradient update branch (dead, E-053).

## Measured against the table (4070 SUPER, 512x512 DEFAULT)

| GUI rq | steps x eot | original time | nsre time | MAE (original / nsre) |
|---|---|---|---|---|
| 0 | 100 x 1 | 27 s | 19 s | 2.87 / 2.72 |
| 1 | 150 x 2 | 33 s | 24 s | 2.75 / 2.80 |
| 2 | 300 x 3 | 56 s | 52 s | 3.52 / 3.31 |
| 3 | 500 x 6 | 92 s | 82 s | 4.09 / 4.36 |

The LPIPS backbone is vgg16 (E-091). Maximum per-pixel change was 6, 12 and 20 grey levels for LOW, DEFAULT and HIGH on 512-scale PNG output (E-086).
