# Schedule benchmark

## Sub-features

- Compare the selected fused kernel with Torch unpack plus matmul.
- Compare automatic, forced fragment, and forced shared plans at the same medium shape.
- Warm every path before timing and rotate four-way side order over seven samples.
- Keep compilation out of the timed region and verify every sampled output.
- Report host wall-clock operation latency separately from CUDA event stream time.

## How to get to it (user POV)

Run `tools/p21_verify.py --benchmark` from the repository root after configuring the isolated Python and CUDA paths.

## Driving it with the verification driver

Run Doctor and Drive first. Then run Benchmark from [the verification skill](../SKILL.md). The driver writes raw samples, ranges, medians, the selected plan, and all comparison results to `.verification-runs/P21/benchmark-result.json`.

## Gotchas

- Wall-clock timings include Python dispatch, output allocation, all Torch unpack operations, and CUDA work. CUDA event timings measure elapsed stream work and exclude host-only work. Both exclude input creation and compilation.
- A Torch baseline includes unpacking and dense matmul. It is an end-to-end reference for this input format, not a vendor-tuned low-bit GEMM library.
- Keep the GPU on AC power and avoid concurrent CUDA jobs when comparing runs.
