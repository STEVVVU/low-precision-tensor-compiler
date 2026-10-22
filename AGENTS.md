# P21 project commands and environment

This checkout is the upstream TileLang v0.1.15 source with a bounded P21 Python example. Preserve upstream source history, license, and attribution. P21's public example targets the active SM120 CUDA device; `compile_fp16xint4` rejects a non-current or non-SM120 device.

Use the isolated coursework Python environment and CUDA toolkit documented in `.agents/skills/verify-low-precision-tensor-compiler/SKILL.md`. The shared `coursework-ml` Torch installation is read-only. Do not install, upgrade, or remove packages there. Keep generated outputs in the ignored `.verification-runs/P21/` directory; do not modify upstream source or generated build outputs as part of the P21 Python extension.

From PowerShell at the repository root, configure the isolated environment as shown in the verification skill, then run:

```powershell
& $python tools\p21_verify.py --doctor
& $python examples\dequantize_gemm\p21_quickstart.py
& $python tools\p21_verify.py --drive
& $python tools\p21_verify.py --profile
& $python tools\p21_verify.py --benchmark
& $python -m ruff check examples\dequantize_gemm\p21_fp16xint4.py examples\dequantize_gemm\p21_quickstart.py tools\p21_verify.py
& $python -m ruff format --check examples\dequantize_gemm\p21_fp16xint4.py examples\dequantize_gemm\p21_quickstart.py tools\p21_verify.py
```

Doctor must pass before Drive, and Drive must pass before Profile or Benchmark. Profile data is diagnostic. Benchmark separately reports host wall-clock operation time and CUDA event stream time. Preserve raw evidence for review.
