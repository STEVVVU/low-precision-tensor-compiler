# Large-GEMM numerical diagnosis

The original two-FP16-ULP target still fails for the 256x2048x2048 case. The diagnostic does not change that gate, the kernel, decode placement, input seed, or earlier receipts. Doctor and the strict default Drive passed before this diagnostic on the RTX 5080 Laptop GPU, TileLang 0.1.15, Torch 2.11.0+cu128 and driver 592.01.

All eight arms compared 524,288 finite outputs with the same CPU float64 reference rounded to FP16. The [complete receipt](numerical-diagnostic-2026-10-22.json) retains input/reference/source hashes, observed precision flags and eight worst coordinates per arm. The published JSON normalizes line endings to LF and has SHA256 `27687da7a3b621a71218dcff1092504c691d86dbdf0aae4058fc660ddb525ea6`. The unchanged original Windows receipt has SHA256 `1f5f097e41d9680ec8dae48c2937e1c1edde391834897f125e9aa120b1bafead`.

| Arm | Maximum FP16 ULP | Outputs above two ULP | Original target |
| --- | ---: | ---: | --- |
| torch_unpack_plus_matmul | 188 | 18 | FAIL |
| torch_predecoded_fp16 | 19179 | 25161 | FAIL |
| torch_predecoded_fp32_output | 188 | 18 | FAIL |
| torch_fp32 | 22 | 10 | FAIL |
| torch_cuda_fp64_accum_fp16_output | 1 | 0 | PASS |
| tilelang_auto | 204 | 33 | FAIL |
| tilelang_fragment | 204 | 33 | FAIL |
| tilelang_shared | 204 | 33 | FAIL |

At coordinate (104,1678), the exact-products `math.fsum` control and CPU float64 result both round to 0.00035381317138671875. TileLang produces 0.00030517578125, an absolute difference of 0.00004863739013671875 and 204 FP16 ULP. The sum of absolute products is 6688.402137756348, and the dot-product condition number is about 18.9 million. Packed Torch's FP32-output path also misses this value, by 188 ULP. A separate FP32 dense control misses its own worst coordinate by 22 ULP. The CUDA FP64 control stays within one ULP, but it is a diagnostic, not a replacement kernel or timing baseline.

These observations support accumulation-rounding sensitivity under severe cancellation. They do not prove that every other shape or implementation defect has been excluded. Decode-placement arms have identical error maxima on this fixture, and no nonfinite outputs were observed. Actual reduced-precision flags are retained without modification. In particular, the default FP16-output cuBLAS control is distinct from the FP32-output dense controls used by the accepted benchmark.

The two-ULP target is a strict output-accuracy goal, not a bound derived for arbitrary length-2048 FP32 reductions. A large ULP count near zero is not by itself evidence of wrong packing, indexing or noncompliant arithmetic. Conversely, failing that goal cannot be called an equal-quality win. [NVIDIA's floating-point guide](https://docs.nvidia.com/cuda/floating-point/index.html#verifying-gpu-results) explains why operation order and precision affect comparisons. [PyTorch 2.11 numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html) documents reduction-order differences and FP16 reduction controls. Neither source guarantees this kernel meets a particular application error budget.

No downstream application precision budget has been established. The current gate and failed results remain unchanged. A future acceptance revision must state the accumulation contract and application error needs before new comparisons; it must not convert this observed failure into a pass. Higher-precision or compensated accumulation would require a separately measured implementation and performance tradeoff, not an unmeasured fallback.

Reproduce with the documented CUDA environment, Doctor and Drive, then `python tools/p21_diagnose.py --output <fresh-json-path>` in an exclusive GPU slot. The run's process receipts and stdout/stderr remain under ignored `.verification-runs/P21/numerical-20261022-0907/`. All three processes exited normally. No speed measurement was collected by this diagnostic.
