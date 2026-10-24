---
name: verify-low-precision-tensor-compiler
description: Run the low-precision TileLang compiler on CUDA, check its active environment, and capture generated code, correctness, and timing evidence.
---

# Verify the low-precision tensor compiler

The compiler is a short-lived Python API over TileLang. Run each command from the repository root in a fresh process. The driver uses the pinned TileLang 0.1.15 wheel and the isolated Python environment.

## Launch

Set `CUDA_PATH` to the isolated CUDA toolkit, add its `bin` directory to `PATH`, and run the Python driver. The driver exits after the run and starts no service.

```powershell
$env:CUDA_PATH = 'C:\Users\steve\.codex\tools\coursework-cuda\toolkit'
$env:CUDA_HOME = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$python = 'C:\Users\steve\.codex\tools\coursework-tensor\venv\Scripts\python.exe'
& $python tools\p21_verify.py --doctor
```

## Doctor

Run Doctor before every drive. It checks the imported TileLang version and install path, CUDA availability, and the active GPU compute capability. Stop if the command exits with an error or reports a GPU other than the intended device.

To exercise the documented consumer entry point from the repository root, run the example by its script path after Doctor:

```powershell
& $python examples\dequantize_gemm\p21_quickstart.py
```

This keeps the unbuilt checkout root off the import path for TileLang while the example imports its adjacent P21 module.

## Drive

Run the deterministic upstream fallback and the FP16 by signed INT4 GEMM cases. The cases cover fragment and shared decode placement, odd K and N tails, and a medium matrix. The driver checks signed packing, invalid inputs, a CPU float64 reference, and a CUDA float32 reference.

```powershell
& $python tools\p21_verify.py --drive
```

To measure the selected plan after Drive passes, run:

```powershell
& $python tools\p21_verify.py --benchmark
```

To inspect which CPU and CUDA operations dominate, capture a separate diagnostic trace:

```powershell
& $python tools\p21_verify.py --profile
```

Profile timings are diagnostic only. The benchmark excludes compilation after warmup. It times the public TileLang call, including output allocation, and separately records CUDA event stream time. Five sides compare auto, forced fragment, forced shared, Torch unpack plus matmul, and predecoded cuBLAS. The predecoded control retains expanded weights and excludes decoding. Default128x256x256 benchmark/profile and Drive enforce the original two-ULP gate plus ATOL0.0625/RTOL0.001. Other shapes use explicitly labeled mixed-tolerance diagnostics with reported ULP maxima; passing those checks does not imply passing the original gate. The single-shape `--benchmark` rotates all five sides over seven samples, with five calls per sample; the larger diagnostic sweep uses the longer protocol below. CUDA event intervals can include host enqueue gaps and do not isolate kernel execution.

## Evidence

For the opt-in exact accumulator, run `& $python tools/p21_exact.py --output .verification-runs/P21/<new-name> --benchmark` after Doctor and Drive. This drives the real scalar CUDA path, exhausts every finite FP16 encoding with all signed INT4 values, checks cancellation/direct rounding/subnormal/nonfinite and poisoned odd-tail fixtures, and performs one exact correctness call for each of the nine frozen shapes. It rejects unsafe K and incompatible mode/placement choices, preserves generated CUDA, and requires exact array equality with an independent CPU FP64 oracle. The optional benchmark uses native Torch Timer for five alternating public-call samples at 128x256x256 and captures a separate diagnostic profile. It does not send exact mode through the older 200-call tensor-core sweep. Evidence includes source/input hashes, Torch allocation and parent-process memory with explicit scope. The 24 October run passed all 3,174,707 outputs with zero finite ULP error; see `docs/evidence/P21/exact-20261024/results.md`. This does not change the historical FP32 results or claim cross-device execution.

After Drive, run `& $python tools\p21_verify.py --sweep` in an exclusive GPU slot for the six-shape frozen workload and `--held-out` for the three predeclared additional shapes. Both use100 warm calls per side,20 calls before each sample, and seven rotated samples of200 calls. The sweep writes per-shape results and `sweep-result.json`. Use an external subprocess timeout of30minutes. Do not overlap it with other GPU work or compiler builds. The coordinated22October2026 campaign passed its mixed-tolerance protocol, but six of nine shapes fail the original two-ULP criterion, with up to204ULP. README and `docs/evidence/P21/metrics-summary.json` identify every shape/side's maxima and original-gate status. Only three frozen shapes also satisfy two ULP; other timing results remain diagnostics under the weaker criterion. No independent evidence establishes approval of the ULP override before observations. Earlier noisy runs remain diagnostic.

`--graph-diagnostic` captures the full warmed operations and checks their outputs independently, then records seven rotated samples of 200 graph replays. This is a separate diagnostic scope, not public Python-call latency. Run `tools/p21_summarize.py` to validate retained sample counts, correctness and hashes and produce the curated evidence. The experimental selector was reverted because the public comparison did not establish a reliable improvement.

The driver writes JSON results, TileLang TIR, and generated CUDA source under `.verification-runs/P21/`. It prints each output path and its correctness result. Generated code must show packed-byte loads, signed nibble decoding, and the selected conversion placement. Keep the directory after a drive or benchmark.

## Cleanup

The driver runs synchronously and exits after CUDA work completes. It starts no server or child process. Leave `.verification-runs/P21/` in place because it holds the proof. TileLang's process-local CUDA modules disappear when Python exits.
