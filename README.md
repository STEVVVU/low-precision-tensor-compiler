# Low-precision tensor compiler

## Contents

- [Architecture and execution guide](#architecture-and-execution-guide)
- [Implementation techniques](#implementation-techniques)
- [Reproduce on Windows CUDA](#reproduce-on-windows-cuda)
- [Verification evidence](#verification-evidence)
- [Research context and attribution](#research-context-and-attribution)

## Architecture and execution guide

This checkout contains upstream TileLang and a focused Python schedule example. P21 adds a signed INT4 GEMM schedule and its verifier. TileLang still owns TIR construction, lowering, layout inference, CUDA code generation, and launch. The exact accumulation option is separate; the default remains the FP32 tensor-core path.

The diagram shows where the example ends and the inherited compiler begins.

```mermaid
flowchart LR
Input["FP16 A and packed INT4 B"] --> Example["P21 Python schedule"]
Example --> TileLang["TileLang frontend and lowering"]
TileLang --> CUDA["CUDA compiler and runtime"]
CUDA --> Kernel["GPU kernel"]
Kernel --> Output["FP16 output"]
Verify["P21 verifier"] --> Example
Verify --> Compare["CPU and CUDA references"]
Output --> Compare
```

For each output tile, P21 loads [packed INT4](#packed-int4) weights, decodes signed nibbles, and multiplies them by FP16 activations. A compile-time [decode placement](#decode-placement) choice places decoded weights in registers or a swizzled shared-memory tile. The reduction accumulates in FP32, masks an [odd K tail](#odd-k-tail), and stores FP16. Drive checks correctness before profiling or timing. The separate [exact accumulation](#exact-accumulation) option uses integer accumulation and direct FP16 rounding, with a different cost profile.

```mermaid
flowchart TD
subgraph Example["P21 schedule"]
API["Validate SM120 device and input shapes"] --> Policy["Choose tile and decode placement"]
Policy --> Load["Describe activation and packed-weight tile loads"]
Load --> Decode["Describe INT4 decode and K-tail masking"]
Decode --> Reduce["Describe FP32 GEMM reduction"]
Reduce --> Store["Describe FP16 tile store"]
end
subgraph Compiler["Inherited TileLang pipeline"]
TIR["Build TIR"] --> Layout["Infer layouts and shared staging"]
Layout --> Lower["Lower GEMM and pipeline"]
Lower --> Build["Compile and launch CUDA"]
end
Store --> TIR
Build --> Check["Verifier compares output"]
Check -->|"pass"| Report["Record result or timing"]
Check -->|"fail"| Stop["Keep failure; skip benchmark"]
```

| Subsystem | Responsibility and owned invariant | Source |
| --- | --- | --- |
| P21 schedule | Validate device and tensor contract; create tiled GEMM and output | [`p21_fp16xint4.py`](examples/dequantize_gemm/p21_fp16xint4.py) |
| Quickstart | Construct sample input and call the example | [`p21_quickstart.py`](examples/dequantize_gemm/p21_quickstart.py) |
| Verification driver | Check environment, execute GPU checks, retain evidence, and gate measurement | [`p21_verify.py`](tools/p21_verify.py) |
| Exact diagnostic | Check finite FP16-by-INT4 inputs against a rounded independent oracle | [`p21_exact.py`](tools/p21_exact.py) |
| TileLang compiler | Own compiler and runtime behavior inherited upstream | [`tilelang/`](tilelang/) |

### Run one GEMM

1. Create FP16 activation matrix A and packed signed INT4 weight matrix B. Even K occupies the low nibble.
2. Call the schedule. It checks the supported active device and chooses a tile and decode placement.
3. TileLang lowers the schedule and compiles the generated kernel through the installed CUDA toolchain.
4. The GPU decodes weights, masks an odd K tail, reduces, and writes FP16 output.
5. Drive compares output with independent references. A failed correctness check stops measurement and retains failure evidence.

<a id="packed-int4"></a>
<details><summary>What is packed INT4?</summary>

Packed INT4 stores two signed four-bit integers in one byte. P21 places even K in the low nibble and odd K in the high nibble. For example, `-1` and `3` share a byte after two's-complement encoding.
</details>

<a id="decode-placement"></a>
<details><summary>What is decode placement?</summary>

Decode placement selects where the kernel expands packed nibbles. P21 chooses register fragments or a swizzled shared-memory tile at compile time; this bounded heuristic is not a general search system.
</details>

<a id="odd-k-tail"></a>
<details><summary>What is an odd K tail?</summary>

An odd K dimension leaves one unused nibble in the final byte. P21 masks that lane so padding cannot affect the dot product; verification poisons the unused nibble to check the case.
</details>

<a id="tensor-core-accumulation"></a>
<details><summary>What is tensor-core accumulation?</summary>

The default path accumulates in FP32 before converting to FP16. Cancellation can still exceed the original two-ULP gate for larger cases, as the evidence below records.
</details>

<a id="exact-accumulation"></a>
<details><summary>What is exact accumulation?</summary>

The optional mode sums finite FP16-by-INT4 products as scaled signed integers, then rounds once to FP16. It has a separate measured cost and uses scalar rather than tensor-core accumulation.
</details>

<a id="two-ulp-gate"></a>
<details><summary>What is the two-ULP gate?</summary>

The gate limits output error to two representable FP16 steps from the rounded CPU float64 reference. Passing an absolute-relative tolerance does not pass the gate when ULP error is larger.
</details>

<a id="sm120"></a>
<details><summary>What is SM120?</summary>

SM120 is the CUDA compute capability targeted by the P21 entry point. The example rejects a different active device and does not dispatch across GPUs.
</details>

### Design choices and limits

These rationales are inferred from the implementation; behavior and measured limits are documented in the evidence sections below.

| Choice | Current behavior and rationale | Alternative and cost to reconsider |
| --- | --- | --- |
| Python schedule over compiler changes | Keeps the extension focused while reusing TileLang lowering. | A native compiler pass could generalize behavior but expands compiler maintenance. |
| Decode packed input in kernel | Avoids requiring a pre-expanded weight tensor at the call boundary. | Predecode simplifies compute but consumes expanded storage and shifts decode work to callers. |
| Compile-time register/shared choice | Avoids runtime branching between placement policies. | Runtime choice needs dispatch and evidence across shapes and devices. |
| FP32 tensor-core default | Uses existing GEMM lowering and preserves current default behavior. | Exact integer mode improves finite rounding but costs tensor-core throughput. |
| SM120 device restriction | Matches the verified target without implying portability. | Multi-architecture dispatch needs separate builds and verification evidence. |

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

## Implementation techniques

This section covers the active P21 Python example, its quickstart and verification paths, and the TileLang compiler mechanisms those paths directly invoke. Upstream examples unrelated to P21 are outside this scope. Rationale is inferred from current code; the entry-point contract and limitations are stated in the [P21 guide](examples/dequantize_gemm/P21.md).

| Active subsystem | Technique cards | Source |
| --- | --- | --- |
| Signed INT4 data representation | [Low-nibble-first packing](#technique-signed-int4-packing), [signed-nibble extension](#technique-signed-nibble-extension), [poisoned-tail masking](#technique-odd-k-tail-masking) | [schedule](examples/dequantize_gemm/p21_fp16xint4.py#L33), [verifier](tools/p21_verify.py#L186) |
| Tiled FP32 tensor-core schedule | [shared-memory tiling and pipelining](#technique-shared-memory-tiling-and-pipelining), [compile-time decode placement](#technique-decode-placement-policy), [bounded resource estimate](#technique-resource-bound-estimation) | [schedule](examples/dequantize_gemm/p21_fp16xint4.py#L47) |
| TileLang compiler pipeline | [TIR lowering, layout inference, and CUDA generation](#technique-tilelang-compilation-pipeline) | [P21 guide](examples/dequantize_gemm/P21.md), [quickstart](examples/dequantize_gemm/p21_quickstart.py#L1) |
| Opt-in exact arithmetic | [scaled int64 accumulation and direct FP16 rounding](#technique-exact-scaled-integer-accumulation), [K overflow bound](#technique-exact-accumulator-overflow-bound) | [schedule](examples/dequantize_gemm/p21_fp16xint4.py#L164), [exact verifier](tools/p21_exact.py#L30) |
| Verification and measurement | [independent numerical oracles and ULP gate](#technique-independent-oracle-and-ulp-gate), [exhaustive finite-value check](#technique-exhaustive-finite-fp16-check), [separated host and device metrics](#technique-measurement-scope-separation) | [driver](tools/p21_verify.py#L29), [exact verifier](tools/p21_exact.py#L19) |
| Quickstart and device boundary | [active-device validation and installed-wheel loading](#technique-device-and-compiler-boundary) | [quickstart](examples/dequantize_gemm/p21_quickstart.py#L1), [schedule](examples/dequantize_gemm/p21_fp16xint4.py#L197) |

### Signed INT4 data representation

<a id="technique-signed-int4-packing"></a>

<details>
<summary>Low-nibble-first packing</summary>

**Mechanism.** `pack_signed_int4` validates a nonempty rank-two signed integer tensor in `[-8,7]`, pads odd K with zero, masks each value to four-bit two's-complement, then combines even K in the low nibble and odd K in the high nibble.

**Example.** `[-8, -1, 0, 7]` becomes bytes `[0xF8, 0x70]`; a fifth value gets a zero partner in the final byte.

**Why here.** This halves the stored weight width while keeping the public input a simple integer tensor.

**Tradeoffs.** Callers pay packing work and the kernel must decode nibbles. The returned representation is byte-oriented and assumes contiguous packing order.

**Code and ownership.** [pack_signed_int4](examples/dequantize_gemm/p21_fp16xint4.py#L33), P21 extension.

</details>

<a id="technique-signed-nibble-extension"></a>

<details>
<summary>Signed-nibble extension</summary>

**Mechanism.** The tiled kernel loads an unsigned byte, selects a nibble by K parity, masks four bits, and maps codes 8–15 to -8–-1 by subtracting 16 before converting to FP16.

**Example.** A nibble `0xF` becomes `15 - 16 = -1`.

**Why here.** This recovers the signed values represented by the packer without expanding the entire weight matrix on the host.

**Tradeoffs.** Decode instructions are repeated as tiles are consumed; predecoded weights avoid that work but need expanded storage and a different input path.

**Code and ownership.** `_compile.main` [decode loop](examples/dequantize_gemm/p21_fp16xint4.py#L145), P21 extension using TileLang TIR operations.

</details>

<a id="technique-odd-k-tail-masking"></a>

<details>
<summary>Odd-K tail masking</summary>

**Mechanism.** The packer zero-pads odd K, while the kernel independently checks the logical K index before writing each decoded value into the reduction tile. Out-of-range lanes become zero, so the unused high nibble cannot contribute.

**Example.** For K=5, verification overwrites the unused high nibble of the final byte with `0xF`; the result must still match the five-element dot product.

**Why here.** Tile widths are rounded for block scheduling and packed bytes always contain two lanes, so the final logical lane needs an explicit validity condition.

**Tradeoffs.** Mask predicates add tail handling; without them poisoned padding can change the answer.

**Code and ownership.** `_compile.main` [K guard](examples/dequantize_gemm/p21_fp16xint4.py#L147), `_verify_signed_cancellation_and_poisoned_tail` [poison fixture](tools/p21_verify.py#L186), P21 extension.

</details>

### Tiled FP32 tensor-core schedule

<a id="technique-shared-memory-tiling-and-pipelining"></a>

<details>
<summary>Shared-memory tiling and pipelined GEMM</summary>

**Mechanism.** A CUDA block owns an M-by-N output tile. Each K iteration stages bounded activation and packed-weight tiles, decodes the weights, and calls `T.gemm` into an FP32 accumulator. `T.Pipelined` requests two or three stages according to K length; the result is cast to FP16 once at the store.

**Example.** A block reads `A[by*BM:..., ko*BK:...]` and the packed B rows for `bx`, then accumulates the corresponding partial product.

**Why here.** The output decomposition lets TileLang lower the contraction through its GEMM path while staging only a slice of inputs at a time.

**Tradeoffs.** Tile and stage choices affect shared storage and occupancy; FP32 accumulation can still exceed the two-ULP gate on cancellation-heavy larger cases. Pipelining is a request to the compiler, not a measured overlap guarantee.

**Code and ownership.** `_compile.main` [tile and reduction](examples/dequantize_gemm/p21_fp16xint4.py#L115), P21 schedule; `T.Pipelined` and `T.gemm` are inherited TileLang mechanisms.

</details>

<a id="technique-decode-placement-policy"></a>

<details>
<summary>Compile-time decode placement</summary>

**Mechanism.** The schedule selects either a decoded register fragment or a decoded swizzled shared-memory tile. The branch is resolved while building the TileLang kernel; both paths share packed loads and FP32 GEMM, but the shared path copies decoded values before the contraction.

**Example.** `placement="fragment"` calls GEMM with `B_fp16_fragment`; `placement="shared"` copies it into `B_fp16_shared` first.

**Why here.** A single API can compare two storage locations for decoded values without a runtime branch inside each kernel invocation.

**Tradeoffs.** Fragment use can grow per-thread register demand; shared placement consumes staged shared memory and copy work. The heuristic is bounded and does not search all layouts or guarantee a faster choice.

**Code and ownership.** `_schedule` and `_compile.main` [placement branch](examples/dequantize_gemm/p21_fp16xint4.py#L47), P21 extension.

</details>

<a id="technique-resource-bound-estimation"></a>

<details>
<summary>Logical resource-bound estimation</summary>

**Mechanism.** The planner estimates decoded values per thread against one warp's width, and estimates staged logical bytes against the device shared-memory limit. Auto mode may reduce stages to two before falling back from shared to fragment placement; forced shared placement rejects estimates above the limit.

**Example.** Auto chooses shared when decoded values per thread exceed the warp-size threshold, then checks the staged-byte estimate against 85% of the reported block limit.

**Why here.** This provides a deterministic device-aware choice with no tuning database or benchmark search.

**Tradeoffs.** The estimate omits padding and compiler lifetime details, as its source comment says. It cannot prove launch feasibility or actual register/shared use; inspect generated code and compiler resource metadata for that.

**Code and ownership.** `_schedule` [resource heuristic](examples/dequantize_gemm/p21_fp16xint4.py#L47), P21 extension.

</details>

<a id="technique-tilelang-compilation-pipeline"></a>

<details>
<summary>TileLang TIR compilation pipeline</summary>

**Mechanism.** The Python decorator turns the schedule body into TileLang's tensor IR program. The inherited toolchain performs layout inference and lowering, generates CUDA, and launches the compiled kernel; P21 records TIR and generated CUDA for selected cases to inspect the actual lowered path.

**Example.** `kernel.get_kernel_source()` is checked for packed-byte loads, signed nibble operations, and shared decode when selected; `kernel.get_tir(...).script()` is saved with it.

**Why here.** The coursework change supplies a schedule in Python while reusing the installed TileLang compiler instead of modifying its native passes.

**Tradeoffs.** Generated code and resource use are compiler-version and target dependent. P21 pins its documented consumer to TileLang 0.1.15; this is not a promise that another version emits identical code.

**Code and ownership.** [JIT schedule](examples/dequantize_gemm/p21_fp16xint4.py#L106), [code evidence checks](tools/p21_verify.py#L139); compiler passes are inherited TileLang.

</details>

### Opt-in exact arithmetic

<a id="technique-exact-scaled-integer-accumulation"></a>

<details>
<summary>Scaled int64 accumulation with direct FP16 rounding</summary>

**Mechanism.** Exact mode represents each finite FP16 activation as an integer in units of 2^-24, multiplies that integer by the decoded signed INT4 value, and sums in int64. It converts the final sum back to FP64 scale and calls CUDA `__double2half` for direct final rounding. Nonfinite activation products use a separate FP32 accumulator to retain IEEE special-value behavior.

**Example.** For finite `a`, accumulate `int64(a * 2^24) * weight`; only after reduction compute `double(total) * 2^-24` and round to half.

**Why here.** It avoids intermediate FP32 summation error for finite values where exact FP16 rounding is required.

**Tradeoffs.** This scalar path performs work per output and K element and does not use tensor cores. NaN payload identity and negative-zero payload identity are not promised.

**Code and ownership.** `_compile_exact.main` [integer and special accumulation](examples/dequantize_gemm/p21_fp16xint4.py#L164), P21 extension.

</details>

<a id="technique-exact-accumulator-overflow-bound"></a>

<details>
<summary>Exact accumulator overflow bound</summary>

**Mechanism.** Before compiling exact mode, `_schedule` bounds K using the largest finite FP16 magnitude, the 2^24 scale, maximum INT4 magnitude 8, and signed int64 capacity. It rejects larger reductions and rejects forced fragment/shared placement because exact mode uses scalar decoding.

**Example.** The maximum accepted K is 1,049,088; an attempted K=1,049,089 is expected to raise before launch.

**Why here.** This makes the integer accumulation's representable range an explicit input constraint rather than relying on overflow behavior.

**Tradeoffs.** The bound is conservative for typical inputs because it covers worst-case values at every reduction step. Exact mode is therefore constrained by the format's maximum, not observed data.

**Code and ownership.** `_EXACT_K_LIMIT` and `_schedule` [guard](examples/dequantize_gemm/p21_fp16xint4.py#L26), rejection check in [exact verifier](tools/p21_exact.py#L77), P21 extension.

</details>

### Verification and measurement

<a id="technique-independent-oracle-and-ulp-gate"></a>

<details>
<summary>Independent oracles and FP16 ULP gate</summary>

**Mechanism.** The driver computes a CPU float64 matrix product and rounds it to FP16, then separately compares with a CUDA float32 reference. It checks absolute-relative tolerance and maps ordered FP16 bit patterns to integer positions to measure representable-step distance. The original two-ULP condition is a separate hard gate for strict cases.

**Example.** A result can meet ATOL/RTOL and still fail when its maximum distance from the rounded CPU reference exceeds two FP16 ULP.

**Why here.** Different accumulation paths can disagree near cancellation and rounding boundaries; two independent references expose distinct errors.

**Tradeoffs.** CPU float64 reference construction costs host memory and compute. Expanded diagnostic shapes intentionally omit the ULP rejection and are not evidence that the default quality gate passes.

**Code and ownership.** `_check_output` and `_fp16_ulp_distances` [comparisons](tools/p21_verify.py#L59), P21 verification extension; Torch/NumPy arithmetic is dependency-provided.

</details>

<a id="technique-exhaustive-finite-fp16-check"></a>

<details>
<summary>Exhaustive finite-FP16 check</summary>

**Mechanism.** The exact diagnostic enumerates all 65,536 half encodings, filters to finite values, pairs each with all sixteen signed INT4 values, and compares GPU output numerically with a CPU float64 product rounded to FP16, then reports finite-value ULP distances. Numeric equality does not distinguish positive and negative zero. Separate fixtures cover cancellation, midpoint rounding, subnormals, nonfinite arithmetic, and an odd tail whose unused nibble is poisoned.

**Example.** The Cartesian check covers every finite half input against weights -8 through 7, not every possible matrix reduction.

**Why here.** The scalar exact path has a compact input domain for individual products, making exhaustive finite-value validation practical.

**Tradeoffs.** This exhausts scalar value pairs and targeted reductions, not all shapes, concurrent execution, or devices. Results apply to the tested SM120 setup.

**Code and ownership.** `drive` [small oracles](tools/p21_exact.py#L30), P21 verification extension; NumPy supplies the independent host arithmetic.

</details>

<a id="technique-measurement-scope-separation"></a>

<details>
<summary>Separate host and device measurement scopes</summary>

**Mechanism.** The fast-path benchmark warms compiled paths, then reports host wall time for the public call (including output allocation) and CUDA event stream time separately. The profiler is a diagnostic trace, not a benchmark sample. Exact mode uses Torch Timer for the public call and reports Torch allocation separately from parent-process RSS/working set.

**Example.** The exact timing report's parent RSS excludes child compiler processes and device allocations; Torch's peak allocation covers only Torch-accounted device memory.

**Why here.** These scopes distinguish Python/allocation overhead from queued GPU work and avoid treating unlike memory counters as equivalent.

**Tradeoffs.** CUDA event intervals may include host enqueue gaps and do not isolate kernel execution. The exact scalar timing is not a tensor-core speed claim.

**Code and ownership.** `benchmark` and `profile` [fast driver](tools/p21_verify.py#L305), `drive` [exact measurements](tools/p21_exact.py#L30), P21 extension using Torch profiler/Timer.

</details>

### Quickstart and device boundary

<a id="technique-device-and-compiler-boundary"></a>

<details>
<summary>Active-device and installed-compiler boundary</summary>

**Mechanism.** The public API requires rank-two contiguous CUDA tensors on the same active device, checks dtype and packed width, and rejects a device other than the current SM120 GPU. Doctor also verifies TileLang 0.1.15 came from the installed wheel rather than the unbuilt checkout. The quickstart imports its adjacent example module and compares outputs with CPU/GPU references.

**Example.** Invoking the quickstart by repository-relative script path keeps the checkout root from shadowing the installed TileLang package.

**Why here.** The implementation is verified for one local CUDA/compiler setup and needs a reproducible consumer entry point.

**Tradeoffs.** There is no device dispatch or portability claim. A different architecture or compiler install is rejected or unverified and needs its own validation.

**Code and ownership.** `compile_fp16xint4` [device check](examples/dequantize_gemm/p21_fp16xint4.py#L197), [quickstart import and version guard](examples/dequantize_gemm/p21_quickstart.py#L1), `doctor` [environment check](tools/p21_verify.py#L29), P21 extension.

</details>

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

An opt-in `accumulation="exact"` path now addresses cancellation without loosening the numerical gate. It sums finite FP16-by-INT4 products as scaled signed integers and rounds directly to FP16 once. [The exact-mode report](docs/evidence/P21/exact-20261024/results.md) retains **3,174,707 checked outputs**, including all finite FP16 encodings with all 16 signed INT4 values and all nine earlier shapes. Finite results matched the independent rounded FP64 oracle with zero ULP error. The original FP32 tensor-core path remains the default and retains its documented cancellation limits. Exact mode supports K up to 1,049,088, uses scalar rather than tensor-core accumulation, and has separately measured cost.

This stronger correctness run used at most 8,405,504 bytes of Torch device allocation and a 1,002,360,832-byte parent-process peak working set. Its recorded drive took 22.82 seconds after Doctor, including kernel compilation, the oracles, a small timing experiment and a separate diagnostic profile. Those memory measures exclude compiler children and non-Torch device allocations. Reproduce with `python tools/p21_exact.py --output <new-evidence-directory> --benchmark` in the documented CUDA environment. The small timing ranges overlap; no speedup follows.

The coordinated campaign on 22 October 2026 measured six frozen shapes and three held-out shapes at seed 21. Each side received 100 warmup calls and 20 more before each sample, then seven rotated samples of 200 calls. All outputs passed independent CPU float64 and CUDA float32 references at ATOL0.0625/RTOL0.001. However the expanded benchmark disabled the original two-FP16-ULP hard gate; it did not retain all original acceptance checks. There is no independent timestamped evidence that this broader numerical boundary was approved before observations. The original default128×256×256 benchmark/profile now enforce two ULP again, and Drive stays strict. No GPU rerun or numerical kernel fix accompanied that restoration.

The original-gate audit below reports every side's maximum ULP against CPU float64 rounded to FP16. Only three frozen cases also satisfy the original two-ULP criterion. Timings for the other six cases are diagnostic under the weaker absolute-plus-relative criterion and cannot support an equal-quality speed claim under the original gate. Raw receipts are preserved unchanged.

| M,N,K | Auto | Fragment | Shared | Packed Torch | Predecoded Torch | Original 2-ULP gate |
| --- | --- | --- | --- | --- | --- | --- |
| 1,4096,4096 | 1 | 1 | 1 | 0 | 0 | PASS |
| 16,1024,1024 | 1 | 1 | 1 | 1 | 1 | PASS |
| 128,1024,1024 | 14 | 14 | 14 | 14 | 14 | FAIL |
| 512,1024,1024 | 29 | 29 | 29 | 29 | 29 | FAIL |
| 256,2048,2048 | 204 | 204 | 204 | 188 | 188 | FAIL |
| 33,257,513 | 1 | 1 | 1 | 1 | 1 | PASS |
| 256,1536,1536 | 23 | 23 | 23 | 23 | 23 | FAIL |
| 384,768,1024 | 32 | 32 | 32 | 32 | 32 | FAIL |
| 192,1024,2048 | 38 | 38 | 38 | 38 | 38 | FAIL |

The automatic placement heuristic remains unchanged. A tested long-K fragment override did not establish a public-call improvement across held-out shapes and was reverted. Larger-case numerical diagnosis remains open; the existing mixed tolerance does not make a204-ULP result satisfy the original gate.

The following timing table includes only the three cases whose recorded errors satisfy the original gate. It reports host wall milliseconds as median [minimum,maximum]. The packed Torch baseline includes eager signed decoding and optimized dense matmul. Predecoded cuBLAS retains expanded FP16 weights and excludes their decode, so it is a compute-only control with a different storage cost. Both dense controls request FP32 output before final FP16 rounding; default FP16 output failed the frozen large-case ATOL/RTOL checks.

| M,N,K | Auto | Packed decode + matmul | Predecoded cuBLAS |
| --- | --- | --- | --- |
| 1,4096,4096 | .15950 [.05967,.38268] | 1.59914 [1.17676,1.76736] | .20879 [.05477,.59314] |
| 16,1024,1024 | .02896 [.02684,.03182] | .16342 [.15191,.17614] | .02249 [.02048,.02465] |
| 33,257,513 | .03066 [.02745,.03384] | .16618 [.15485,.18206] | .02311 [.01984,.02551] |

These three cases support lower latency than the scoped eager packed-input baseline. They do not establish superiority over a tuned fused INT4 library; the predecoded control is faster at16×1024×1024. Raw wall and CUDA-event samples, batch p95, every shape/side's original-gate PASS/FAIL, numerical errors, compilation-or-cache-lookup time, memory and held-out diagnostics are in [the curated metrics](docs/evidence/P21/metrics-summary.json), with SHA256 hashes of immutable raw receipts. Maximum observed Torch peak allocation was167,968,768 bytes and peak reservation333,447,168 bytes. Larger diagnostic outputs have maximum absolute error0.5 and up to204ULP; passing ATOL/RTOL does not override their failed two-ULP criterion.

CUDA events measure elapsed stream spans and can include gaps while Python enqueues work. They do not establish continuous GPU busy time. The separately retained CUDA-graph diagnostic removes the public-operation dispatch and allocation path, with a maximum reservation of 254MiB, but replay enqueue gaps can remain. It is not public-call latency. A separate Torch profiler trace identified the extra decoded shared staging in the candidate comparison; timings from that trace are diagnostic. Earlier runs with unconfirmed resource isolation remain diagnostic and are excluded from this table.

Reproduce after Doctor and Drive in an exclusive GPU slot:

```powershell
& $python tools/p21_verify.py --sweep --evidence-dir .verification-runs/P21-metrics-clean-sweep
& $python tools/p21_verify.py --held-out --evidence-dir .verification-runs/P21-metrics-clean-held-out
& $python tools/p21_verify.py --graph-diagnostic --evidence-dir .verification-runs/P21-metrics-clean-graph-diagnostic
& $python tools/p21_summarize.py
```

The maintained numerical diagnostic is `tools/p21_diagnose.py`. A subsequent [large-GEMM diagnosis](docs/evidence/P21/numerical-diagnosis.md) passed Doctor and strict Drive, then reproduced the large-case failure. TileLang had 33 of 524,288 finite outputs above two FP16 ULP (maximum 204); an ordinary FP32 dense control also failed, while the CUDA FP64 control stayed within one ULP. The worst dot product had severe cancellation. The original gate remains unchanged, and no larger-case equal-quality speedup is claimed.

A fresh [strict Drive recheck](docs/evidence/P21/drive-recheck-20261022.json) after host memory became available passed all four default cases, special-value/poisoned-tail checks and three invalid-input checks on the same SM120 environment. It does not supersede the expanded-shape failures or supply new timing results. The separate exact mode added on 24 October has its own implementation, correctness checks and measured cost above; the FP64 diagnostic is not substituted for the tensor-core path.

The compact [GPU evidence bundle](docs/evidence/P21/README.md) includes the actual SM120 Drive and Benchmark JSON plus generated CUDA and TIR from the measured run. The measured medium 128×256×256 output had maximum absolute error `0.0625`, maximum 2 FP16 ULP, and one of 32,768 outputs above 1 ULP versus CPU float64 rounded to FP16. A deterministic odd-K test poisoned the unused high nibble to `0xF` and produced exact cancellation, signed, and zero outputs.

The earlier small-shape timing JSON remains historical evidence under its original two-ULP check. Its broad ranges did not establish a decode-placement advantage. The coordinated expanded campaign has the explicitly limited scope audited above.

The inherited upstream fallback GEMV passed its reference checks with maximum absolute error 0.125 and maximum 4 FP16 ULP on its recorded input. The upstream fast-LOP3 example still fails NVCC on the tested CUDA 13 toolchain; details and the compiler diagnostic boundary are in the [P21 guide](examples/dequantize_gemm/P21.md).

The active GitHub Actions workflow runs only CPU syntax, lint, and evidence-integrity checks. It does not install TileLang or claim to execute CUDA. Original upstream CI, distribution, bot, and documentation workflow files remain byte-for-byte under [`maint/upstream-workflows/`](maint/upstream-workflows/), outside GitHub's active workflow directory.

## Research context and attribution

P21 is informed by the [CMU FP16xINT4 final report](https://www.andrew.cmu.edu/user/zhiboc/15745-fa24/final_report.pdf) and [Ladder, OSDI 2024](https://www.usenix.org/system/files/osdi24-wang-lei.pdf). The report's kernel implementation was independently written; no student source or published benchmark results are copied or claimed. The underlying compiler is TileLang under the upstream MIT license; see the preserved [LICENSE](LICENSE) and [upstream README](README.upstream.md).
