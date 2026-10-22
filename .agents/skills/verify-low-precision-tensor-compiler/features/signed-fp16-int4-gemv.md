# Upstream signed FP16 x INT4 GEMV

## Sub-features

- Pack two signed four-bit values into each byte.
- Compile the upstream TileLang dequantization kernel.
- Compare the upstream fallback result with independent CPU and PyTorch references.

## How to get to it (user POV)

Run `tools/p21_verify.py --drive` from the repository root with the isolated Python environment and CUDA toolkit on `PATH`. The full drive includes this upstream fallback.

## Driving it with the verification driver

Run the Doctor command first. Then run the Drive command from [the verification skill](../SKILL.md). The JSON result and generated CUDA source are written under `.verification-runs/P21/`.

## Gotchas

- The upstream fast LOP3 decoder fails to compile with the installed CUDA 13 NVCC because its injected source uses `uint` without a visible declaration. This path drives the upstream scalar signed-nibble converter instead.
- Do not claim this GEMV example proves the new tiled GEMM or its scheduling policy. Drive the GEMM feature separately after it lands.
