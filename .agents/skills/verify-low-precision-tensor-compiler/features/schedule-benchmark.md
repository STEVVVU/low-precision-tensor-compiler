# Schedule benchmark

## Sub-features

- `--sweep` freezes six MNK cases at seed 21, including decode-heavy GEMV, small and large GEMM, and odd tails. It compares auto, forced fragment, forced shared, eager packed decode plus matmul, and predecoded cuBLAS. The latter excludes decode and retains expanded FP16 weights, so it is a compute-only control.
- Each shape retains seven rotated samples, p95, range, errors, peak Torch allocated and reserved bytes, input hash, source SHA and local diff hash. Compilation or cache lookup is recorded separately before warmup. Every sampled output must exactly match its independently checked warm output.

- Compare the selected fused kernel with Torch unpack plus matmul.
- Compare automatic, forced fragment, and forced shared plans at the same medium shape.
- Warm every path before timing and rotate five-way side order over seven samples. Frozen `--sweep` and `--held-out` use 100 warm calls per side, 20 calls before each sample and 200 calls per sample; the shorter single-shape `--benchmark` retains five calls per sample.
- Keep compilation out of the timed region and verify every sampled output.
- Report host wall-clock operation latency separately from CUDA event stream time.

## How to get to it (user POV)

Run `tools/p21_verify.py --benchmark` from the repository root after configuring the isolated Python and CUDA paths.

## Driving it with the verification driver

Run Doctor and Drive first. Then run Benchmark from [the verification skill](../SKILL.md). The driver writes raw samples, ranges, medians, the selected plan, and all comparison results to `.verification-runs/P21/benchmark-result.json`.

## Gotchas

- Wall-clock timings include Python dispatch, output allocation, all Torch unpack operations, and CUDA work. CUDA event intervals measure elapsed stream time and may include host enqueue gaps; they do not isolate kernel execution. Both exclude input creation and compilation.
- A Torch baseline includes unpacking and dense matmul. It is an end-to-end reference for this input format, not a vendor-tuned low-bit GEMM library.
- Torch controls request FP32 output before conversion to FP16; the original default output policy failed the existing ATOL/RTOL bound at the larger shape. Global backend flags and core ATOL0.0625/RTOL0.001 are unchanged, but the expanded loop disabled the original two-ULP gate. Default128x256x256 benchmark/profile and Drive enforce that original gate. Expanded diagnostics explicitly record their weaker criterion and retain ULP maxima; six of nine shapes fail two ULP and cannot support an original-gate equal-quality speed claim.
- `docs/evidence/P21/metrics-summary.json` audits maximum ULP and original-gate PASS/FAIL for every shape and all five sides. Only1x4096x4096,16x1024x1024,and33x257x513 satisfy that gate in the expanded receipts. The absence of independent pre-observation approval evidence is disclosed; no current numerical kernel fix or rerun is claimed.
- `--graph-diagnostic` measures full-operation graph replay separately from the public path. The candidate selector was reverted after its public comparison remained inconclusive. `docs/evidence/P21/metrics-summary.json` retains all nine clean cases, seven raw samples per side and graph diagnostics with source-receipt hashes.
- Keep the GPU on AC power and avoid concurrent CUDA jobs when comparing runs.
