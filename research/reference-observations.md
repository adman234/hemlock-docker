# Reference observations

Measured by running the original modules (see `parity-report.md` for setup). Raw data in `research/results/ref_runs.jsonl`,
tables in `research/measured-results.md`. Static predictions made earlier (E-0xx) are checked in the last column.

| Observation | Value | Prediction | Result |
|---|---|---|---|
| Output size equals input size, name `<stem>-protected-intensity-<LEVEL>-V1.<ext>` | all 12 successful size/format runs | E-043, E-044 | confirmed |
| `.png` gives PNG, `.jpg` gives JPEG, `.PNG` and `.gif` give JPEG data in that extension | X-FMT-4, X-FMT-7 | E-043 | confirmed |
| Alpha dropped, L and P inputs become RGB PNG | X-FMT-1..3 | E-040 | confirmed |
| GIF input | crash `AttributeError('_getexif')` (E-080) | not predicted | new defect |
| EXIF orientation 6 | output pixels already rotated, tag kept (E-081) | not predicted | new defect |
| Max per-pixel change, 512-scale PNG | LOW 6, DEFAULT 12, HIGH 20 levels | about 5.8 / 11.6 / 19.9 | confirmed |
| Max change on larger images / JPEG output | 16 to 17 (1024 and 2000x1500), 20 to 35 (JPEG) | slack for resampling and codec | consistent |
| Mean absolute error and PSNR | LOW 1.1 / 44.0 dB, DEFAULT 2.4 to 2.9 / 36.4 to 37.8 dB, HIGH 5.1 / 31.4 dB | n/a | measured |
| Render quality (512x512, DEFAULT) | MAE 2.9, 2.75, 3.5, 4.1 for rq 0..3 (more passes, stronger and more visible) | monotone | confirmed |
| Run-to-run variability, 5 identical runs | MAE sd 0.077; perturbation cosine 0.096; mean abs difference 3.3 | nondeterministic (E-056) | confirmed, large |
| Square input deterministic apart from SD | not separable here; whole-run variability is the same as for non-square | E-050 | SD dominates |
| `-protected-intensity-` file reprocessed, `-nightshade-intensity` skipped | X-BAD-a-* | E-024 | confirmed |
| No valid input | `UnboundLocalError: out_file_name` | E-082 | new defect |
| render level -1 | `KeyError('eot')` even for square | E-034 | confirmed, stronger |
| Timing (4070 SUPER) | 26 s, 33 s, 56 s, 92 s for rq 0..3 at 512x512 DEFAULT; HIGH rq 0 about 59 s; CPU rq 0 about 640 s | GUI says 5 to 180 min | GUI labels are far too pessimistic on this GPU |
| Peak VRAM | 7.4 to 8.1 GB | needs about 5 GB per the GUI | consistent |
| Auto target for a given concept | comes from `precompute_mapping` for every probed common word | E-062 | mapping dominates the similarity path |
