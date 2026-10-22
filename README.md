# Low-precision tensor compiler

This repository preserves the upstream TileLang compiler source, examples, history, and MIT license at `v0.1.15` (`a35f8ddf45eba16c21211ec8822d56ce5363036f`). The original upstream landing page is preserved verbatim as [README.upstream.md](README.upstream.md). The P21 addition is a focused Python TileLang schedule example and its real-GPU verification driver; it does not modify TileLang's native compiler or runtime.

## Implemented path

[`examples/dequantize_gemm/p21_fp16xint4.py`](examples/dequantize_gemm/p21_fp16xint4.py) packs two signed INT4 values per byte, even K index in the low nibble, and compiles `A[M,K] @ B[N,K].T` for FP16 activations and signed values in `[-8, 7]`. The kernel decodes each packed tile, masks odd K tails, accumulates in FP32, and stores FP16. Its compile-time policy chooses either a register fragment or a swizzled shared-memory tile for decoded weights, alongside M/N/K tiles and pipeline stages.

P21 reuses TileLang's TIR frontend, pipeline, memory allocation/copy, layout inference, GEMM lowering, and native CUDA compiler. The authored extension is the schedule and decode-placement policy in the Python example; it is a bounded placement heuristic, not a reproduction of Ladder's search or compiler transformations. The current public API explicitly supports the active SM120 CUDA device and does not claim multi-GPU dispatch.

```powershell
$python = (Resolve-Path .venv\Scripts\python.exe).Path
& $python examples\dequantize_gemm\p21_quickstart.py
```

Run the example by script path from the repository root. It imports the installed TileLang wheel and the P21 module beside the script; importing `tilelang` from a Python `-c` command at the checkout root instead resolves the unbuilt development tree.

For the exact input contract, resource heuristic, validation, and benchmark limitations, see the [P21 guide](examples/dequantize_gemm/P21.md).

## Reproduce on Windows CUDA

The tested environment used Python 3.11, the official TileLang 0.1.15 Windows wheel, PyTorch 2.11.0 with CUDA 12.8, CUDA Toolkit 13.0.88, and an RTX 5080 Laptop GPU (SM120). Use a clean project-local virtual environment and install CUDA Toolkit separately from NVIDIA. PowerShell commands below use paths relative to the checkout; set `CUDA_PATH` to the installed Toolkit directory.

```powershell
py -3.11 -m venv .venv
$python = (Resolve-Path .venv\Scripts\python.exe).Path
$env:CUDA_PATH = Read-Host 'Path to CUDA Toolkit 13.0.88'
& $python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128'
& $python -m pip install --require-hashes --no-deps -r requirements-p21-win-cu128.lock
& $python -m pip install --index-url https://pypi.org/simple 'ruff==0.16.5'

$env:CUDA_HOME = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
nvcc --version
& $python tools\p21_verify.py --doctor
& $python tools\p21_verify.py --drive
& $python tools\p21_verify.py --profile
& $python tools\p21_verify.py --benchmark
& $python -m ruff check examples\dequantize_gemm\p21_fp16xint4.py examples\dequantize_gemm\p21_quickstart.py tools\p21_verify.py
& $python -m ruff format --check examples\dequantize_gemm\p21_fp16xint4.py examples\dequantize_gemm\p21_quickstart.py tools\p21_verify.py
```

The requirements lock pins the official release wheel and runtime dependency versions with SHA-256 hashes. PyTorch is installed first from its official CUDA 12.8 wheel index; the lock is then installed with dependency resolution disabled so it cannot replace that Torch build. Doctor and Drive run the real TileLang compiler and CUDA kernel. Profile is diagnostic; Benchmark excludes warm compilation and reports both public-call wall time and CUDA-event stream time.

## Verification evidence

The compact [GPU evidence bundle](docs/evidence/P21/README.md) includes the actual SM120 Drive and Benchmark JSON plus generated CUDA and TIR from the measured run. The measured medium 128×256×256 output had maximum absolute error `0.0625`, maximum 2 FP16 ULP, and one of 32,768 outputs above 1 ULP versus CPU float64 rounded to FP16. A deterministic odd-K test poisoned the unused high nibble to `0xF` and produced exact cancellation, signed, and zero outputs.

The timing JSON retains raw samples and ranges. In this run, wall-clock medians were 0.4103 ms for Torch unpack plus matmul, 0.0758 ms for auto, 0.0763 ms for forced fragment, and 0.0750 ms for forced shared. CUDA-event medians were 0.2206, 0.0640, 0.0781, and 0.0774 ms, respectively, with wide overlapping ranges. These are single-run median observations with no statistical conclusion; they do not establish that one decode placement is faster. The Torch comparison includes unpack and dense matmul and is not a vendor low-bit GEMM.

The inherited upstream fallback GEMV passed its reference checks with maximum absolute error 0.125 and maximum 4 FP16 ULP on its recorded input. The upstream fast-LOP3 example still fails NVCC on the tested CUDA 13 toolchain; details and the compiler diagnostic boundary are in the [P21 guide](examples/dequantize_gemm/P21.md).

The active GitHub Actions workflow runs only CPU syntax, lint, and evidence-integrity checks. It does not install TileLang or claim to execute CUDA. Original upstream CI, distribution, bot, and documentation workflow files remain byte-for-byte under [`maint/upstream-workflows/`](maint/upstream-workflows/), outside GitHub's active workflow directory.

## Research context and attribution

P21 is informed by the [CMU FP16xINT4 final report](https://www.andrew.cmu.edu/user/zhiboc/15745-fa24/final_report.pdf) and [Ladder, OSDI 2024](https://www.usenix.org/system/files/osdi24-wang-lei.pdf). The report's kernel implementation was independently written; no student source or published benchmark results are copied or claimed. The underlying compiler is TileLang under the upstream MIT license; see the preserved [LICENSE](LICENSE) and [upstream README](README.upstream.md).
