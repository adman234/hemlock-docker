# Reference observations

No runtime observations of the reference application exist yet. The application has not been executed: its model assets are
not installed (E-007) and downloading them was not authorized. Everything in `evidence.md` is static (class **O/D** = read from
shipped bytecode, not observed executing).

What static analysis already predicts, to be confirmed or refuted by the experiments in `experiment-protocol.md`:

| Prediction | Basis | Experiment |
|---|---|---|
| Output has the same pixel size as the input, name `<stem>-protected-intensity-<LEVEL>-V1.<ext>` | E-043, E-044 | X-DIM, X-FMT |
| `.png` inputs give PNG, everything else (including `.PNG`, `.gif`) gives JPEG data at q=100 | E-043 | X-FMT-6, X-FMT-7 |
| Max per-pixel change at DEFAULT is about 12 levels plus resampling slack, at LOW about 6, at HIGH about 20 | E-053, parameter-table | X-INT |
| Square 512x512 inputs are deterministic apart from the SD target | E-050, E-056 | X-REP |
| Non-square inputs differ run to run through SD noise (unseeded) | E-056 | X-REP |
| `a-protected-intensity-LOW-V1.png` in the input folder is processed again | E-024 | X-SKIP-2 |
| rq `-1` (opt_setting `'0'`) can raise `KeyError: 'eot'` for non-square inputs | E-034 | X-RQ via CLI harness |
| Alpha is dropped, not composited | PIL `convert('RGB')` semantics, E-040 | X-FMT-1 |
