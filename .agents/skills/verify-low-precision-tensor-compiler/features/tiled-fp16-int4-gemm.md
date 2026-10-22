# Tiled signed FP16 x INT4 GEMM

## Sub-features

- Pack signed values -8 through 7, low nibble first.
- Compile a tiled `T.gemm` kernel for any positive M, N, and K, including odd dimensions.
- Choose decoded weights in registers or a swizzled shared-memory tile at compile time.
- Compare the result with CPU float64 and CUDA float32 references.
- Check exact signed, cancellation, and zero outputs with a poisoned unused high nibble for an odd K tail.

## How to get to it (user POV)

Run `examples/dequantize_gemm/p21_quickstart.py` by script path from the repository root. It imports the pinned TileLang wheel and the adjacent P21 module, then calls `pack_signed_int4(weights)` and `fp16xint4_gemm(activations, packed_weights)` on the active SM120 device.

## Driving it with the verification driver

Run Doctor, then Drive from [the verification skill](../SKILL.md). Drive compiles small odd-tail and medium cases, then checks generated CUDA and TIR for the selected compile-time plan.

## Gotchas

- `packed_B` must be contiguous `uint8` with shape `[N, ceil(K/2)]`. The even K value occupies the low nibble.
- This P21 example targets the active SM120 CUDA device and rejects a non-current or different-capability target.
- TileLang 0.1.15's NVCC fast LOP3 GEMV example fails on the installed CUDA 13 toolchain. The upstream fallback remains in the Drive path; the new GEMM uses TileLang integer operations.
- The schedule estimate covers tile storage, not register allocation after lowering. Check the generated code and measured latency before changing its heuristic.
