# Exact FP16 by signed INT4 accumulation

On 24 October 2026, the opt-in exact mode passed 3,174,707 checked outputs with zero finite FP16 ULP error. This includes all 63,488 finite FP16 encodings multiplied by each of the 16 signed INT4 values, a 65,536-row launch-boundary case, targeted cancellation/midpoint/subnormal/IEEE/poisoned-tail fixtures, and one correctness call for each of the nine earlier frozen shapes. The independent oracle is CPU FP64 arithmetic rounded to FP16; nonfinite results are checked separately by exact array equality, including NaNs. This does not change the earlier tensor-core FP32 failures.

## Mechanism and reproduction

The old tensor-core path returns zero for `[65504, 2^-12, -65504]` dotted with `[1,1,1]`, rather than 2^-12. The observed error is 3,072 FP16 ULP. The new mode returns 2^-12 exactly. It accumulates scaled finite products as signed int64, bounded to K <= 1,049,088, then uses CUDA `__double2half` for direct round-to-nearest-even conversion. The [CUDA API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html) specifies this conversion. A generic double-to-half CUTLASS cast may go through FP32, so the direct intrinsic is deliberate. The one-dimensional launch avoids a grid-Y limit for tall matrices. The generated [CUDA source](exact.cu) retains the integer accumulation, bound-independent scalar loop, nonfinite branch and direct conversion.

Use `fp16xint4_gemm(A, packed_B, accumulation="exact")`, or pass the same trailing argument to `compile_fp16xint4`. Exact mode accepts auto placement only. It has no expanded dense weight matrix, no data-dependent scratch and no host scan/readback. It preserves nonfinite arithmetic classes, but not NaN payloads or negative-zero bit identity. The fast FP32 path remains the default.

Run Doctor and strict Drive in the maintained CUDA environment, then `python tools/p21_exact.py --output <unused-directory> --benchmark`. The driver refuses to overwrite a prior directory and retains every result, source/input/reference hash, generated kernel and diagnostic trace. [Raw result](result.json), [retained-file hashes](sha256.json).

## Checked shapes

| M,N,K | Outputs | Maximum finite ULP |
| --- | ---: | ---: |
| 1,4096,4096 | 4,096 | 0 |
| 16,1024,1024 | 16,384 | 0 |
| 128,1024,1024 | 131,072 | 0 |
| 512,1024,1024 | 524,288 | 0 |
| 256,2048,2048 | 524,288 | 0 |
| 33,257,513 | 8,481 | 0 |
| 256,1536,1536 | 393,216 | 0 |
| 384,768,1024 | 294,912 | 0 |
| 192,1024,2048 | 196,608 | 0 |

## Resource use and measured cost

The measured host was NVIDIA GeForce RTX 5080 Laptop GPU / sm_120, Python 3.11.9, Torch 2.11.0+cu128, CUDA runtime 12.8. The complete drive recorded 22.82 seconds after Doctor, including kernel compilation, correctness oracles, the small Timer experiment and separate profiling. Whole interpreter startup is outside that interval. Parent-process peak working set was 1,002,360,832 bytes, including imports and host compiler state but excluding compiler child processes. Maximum recorded Torch device allocation was 8,405,504 bytes; this excludes native compiler/device allocations and is not total VRAM.

Native Torch Timer measured public-call cost, including output allocation after warm compilation, at 128x256x256. Each mode received five alternating Timer runs, with native raw blocks and iterations-per-block retained. The figures below are medians of those five run medians and their full ranges. They are descriptive costs on this small workload, not a throughput or general speedup claim.

| Mode | Public-call ms median [min,max] |
| --- | ---: |
| fp32 | 0.057446 [0.055193, 0.059407] |
| exact | 0.056998 [0.053923, 0.071687] |

Ranges overlap. The separately recorded [diagnostic profile](diagnostic-profile.json) confirms both actual CUDA kernels executed; their traced spans are instrumentation, excluded from timing samples. Exact mode replaces tensor-core GEMM with O(M*N*K) scalar integer work. A small-case timing observation does not establish equal throughput, arbitrary-size cost, fused-library superiority or whole-model quality.

The mathematical finite-input bound is independent of input ordering. GPU execution evidence is limited to the tested active SM120 adapter and inputs. The K-overflow boundary and incompatible accumulation/placement choices were rejected before launch. No numerical tolerance was relaxed. Earlier raw campaigns and strict failures remain retained.

Public text uses LF line endings and one trailing CUDA-source newline. JSON values and CUDA statements are unchanged. The original byte hashes are retained in [raw-sha256.json](raw-sha256.json); all original files remain in the ignored run directory. The published-file hashes in sha256.json describe the normalized copies.
