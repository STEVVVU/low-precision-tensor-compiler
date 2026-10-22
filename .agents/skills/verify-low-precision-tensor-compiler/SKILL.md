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

Profile timings are diagnostic only. The benchmark excludes compilation after warmup. It times the public TileLang call, including output allocation and the full Torch unpack-plus-matmul baseline, and separately records CUDA event stream time. It compares auto, forced fragment, and forced shared placement at the same medium shape. It rotates four-way side order over seven samples per side, with five calls per sample.

## Evidence

The driver writes JSON results, TileLang TIR, and generated CUDA source under `.verification-runs/P21/`. It prints each output path and its correctness result. Generated code must show packed-byte loads, signed nibble decoding, and the selected conversion placement. Keep the directory after a drive or benchmark.

## Cleanup

The driver runs synchronously and exits after CUDA work completes. It starts no server or child process. Leave `.verification-runs/P21/` in place because it holds the proof. TileLang's process-local CUDA modules disappear when Python exits.
