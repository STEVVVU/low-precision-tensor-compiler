# P21 recorded GPU evidence

These records come from the real `tools/p21_verify.py --drive` and `--benchmark` runs on the NVIDIA GeForce RTX 5080 Laptop GPU (SM120), with TileLang 0.1.15 and Torch 2.11.0+cu128. The JSON values and generated CUDA/TIR files are the run's evidence; local absolute paths were converted to relative paths for publication. No timings were recomputed or synthesized for this bundle.

- [`drive-result.json`](drive-result.json) contains the GPU correctness results, rounded-float64 absolute/ULP errors, selected plans, and deterministic exact-value/odd-tail check.
- [`benchmark-result.json`](benchmark-result.json) contains all seven five-call samples, ranges, medians, and wall-clock and CUDA-event measurements for the same medium shape.
- [`source/`](source/) contains the corresponding generated `.cu` and TileLang `.tir` files, including the K=5 case with a poisoned unused high nibble.

After review, the benchmark's categorical faster/slower labels were replaced with **median-only observations** and an explicit no-statistical-conclusion note. This presentation-only correction leaves all recorded samples, ranges, medians, and raw median ratios unchanged. The sample ranges overlap and placement timing differences are inconclusive. The `*_speedup` fields are raw median ratios retained from the measured run and should be read alongside the sample arrays and ranges, not as statistical claims.

The CPU CI validates the evidence schema and required generated-code signatures. It does not rerun or independently reproduce the GPU results. To run the product on supported hardware, follow the repository [setup and verification commands](../../../README.md#reproduce-on-windows-cuda).
