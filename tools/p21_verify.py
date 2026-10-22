"""Drive P21's real TileLang CUDA paths and preserve generated-code evidence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tilelang
from tilelang import language as T


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / ".verification-runs" / "P21"
ATOL = 0.0625
RTOL = 0.001


def doctor() -> tuple[torch.device, str, str]:
    if tilelang.__version__ != "0.1.15":
        raise RuntimeError(f"Expected TileLang 0.1.15, got {tilelang.__version__}")
    imported_from = Path(tilelang.__file__).resolve()
    if ROOT in imported_from.parents:
        raise RuntimeError(f"Expected the isolated TileLang wheel, imported {imported_from}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise RuntimeError(f"Expected SM120, got SM{capability[0]}{capability[1]}")
    return device, torch.cuda.get_device_name(device), f"sm_{capability[0]}{capability[1]}"


def _load_p21_module():
    example_dir = ROOT / "examples" / "dequantize_gemm"
    sys.path.insert(0, str(example_dir))
    from p21_fp16xint4 import compile_fp16xint4, fp16xint4_gemm, pack_signed_int4

    return compile_fp16xint4, fp16xint4_gemm, pack_signed_int4


def _inputs(m: int, n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k)).astype(np.float16)
    q = rng.integers(-8, 8, size=(n, k), dtype=np.int8)
    return a, q


def _check_output(
    output: torch.Tensor,
    a_cpu: np.ndarray,
    q_cpu: np.ndarray,
    a_cuda: torch.Tensor,
    device: torch.device,
    max_ulp_limit: int | None = 2,
) -> dict[str, float | int]:
    cpu_ref64 = a_cpu.astype(np.float64) @ q_cpu.astype(np.float64).T
    cpu_ref16 = cpu_ref64.astype(np.float16)
    torch.testing.assert_close(output.float().cpu(), torch.from_numpy(cpu_ref16).float(), atol=ATOL, rtol=RTOL)
    q_cuda = torch.from_numpy(q_cpu.copy()).to(device=device, dtype=torch.float32)
    torch_ref = a_cuda.float() @ q_cuda.T
    torch.testing.assert_close(output.float(), torch_ref, atol=ATOL, rtol=RTOL)
    actual = output.float().cpu().numpy()
    expected = cpu_ref16.astype(np.float32)
    actual_bits = output.cpu().numpy().view(np.uint16)
    expected_bits = cpu_ref16.view(np.uint16)
    actual_ordered = np.where(actual_bits & 0x8000, np.bitwise_not(actual_bits), actual_bits | 0x8000).astype(np.int32)
    expected_ordered = np.where(expected_bits & 0x8000, np.bitwise_not(expected_bits), expected_bits | 0x8000).astype(np.int32)
    ulp_distances = np.abs(actual_ordered - expected_ordered)
    max_ulp = int(ulp_distances.max())
    if max_ulp_limit is not None and max_ulp > max_ulp_limit:
        raise AssertionError(f"output differs from rounded CPU float64 by {max_ulp} FP16 ULPs")
    return {
        "max_abs": float(np.abs(actual - expected).max()),
        "max_ulp": max_ulp,
        "outputs_over_1_ulp": int(np.count_nonzero(ulp_distances > 1)),
        "output_count": int(ulp_distances.size),
    }


def _assert_bad_input(call, exception: type[Exception]) -> None:
    try:
        call()
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}")


def _verify_upstream_fallback(device: torch.device, evidence_dir: Path) -> dict[str, object]:
    example_dir = ROOT / "examples" / "dequantize_gemm"
    sys.path.insert(0, str(example_dir))
    from example_dequant_gemv_fp16xint4 import dequantize_gemv

    m, n, k = 2, 64, 256
    a_cpu, q_cpu = _inputs(m, n, k, 21)
    codes = (q_cpu.astype(np.int16) & 15).astype(np.uint8)
    packed_cpu = codes[:, 0::2] | (codes[:, 1::2] << 4)
    a = torch.from_numpy(a_cpu.copy()).to(device)
    packed = torch.from_numpy(packed_cpu.copy()).to(device)
    output = torch.empty((m, n), dtype=torch.float16, device=device)
    kernel = dequantize_gemv(
        m,
        n,
        k,
        T.float16,
        T.float16,
        T.float32,
        num_bits=4,
        storage_dtype=T.uint8,
        source_format="uint",
        n_partition=4,
        reduce_thread=32,
        fast_decoding=False,
    )
    kernel(a, packed, output)
    error = _check_output(output, a_cpu, q_cpu, a, device, max_ulp_limit=None)
    source = kernel.get_kernel_source()
    if "__global__" not in source:
        raise RuntimeError("upstream fallback produced no generated CUDA kernel source")
    source_path = evidence_dir / "upstream-fallback-gemv.cu"
    source_path.write_text(source, encoding="utf-8")
    return {"shape_mnk": [m, n, k], "error_vs_cpu_float64_rounded_fp16": error, "cuda_source": str(source_path)}


def _verify_case(
    name: str,
    shape: tuple[int, int, int],
    requested: str,
    device: torch.device,
    evidence_dir: Path,
) -> dict[str, object]:
    compile_kernel, gemm, pack = _load_p21_module()
    m, n, k = shape
    a_cpu, q_cpu = _inputs(m, n, k, 100 + m + n + k)
    a = torch.from_numpy(a_cpu.copy()).to(device)
    packed = pack(torch.from_numpy(q_cpu.copy()).to(device))
    output, plan = gemm(a, packed, requested)
    error = _check_output(output, a_cpu, q_cpu, a, device)

    kernel, _ = compile_kernel(m, n, k, device, requested)
    source = kernel.get_kernel_source()
    if "__global__" not in source or "uchar* __restrict__ B" not in source:
        raise RuntimeError(f"{name}: source does not show the packed-byte CUDA kernel")
    if "& 15" not in source or "- 16" not in source:
        raise RuntimeError(f"{name}: source does not show signed nibble decoding")
    has_shared_decode = "B_fp16_shared" in source
    if has_shared_decode != (plan.placement == "shared"):
        raise RuntimeError(f"{name}: generated source does not match the selected decode placement")

    stem = f"case-{name}-{plan.placement}"
    source_path = evidence_dir / f"{stem}.cu"
    tir_path = evidence_dir / f"{stem}.tir"
    source_path.write_text(source, encoding="utf-8")
    tir_path.write_text(kernel.get_tir(a, packed, output).script(), encoding="utf-8")
    return {
        "name": name,
        "shape_mnk": [m, n, k],
        "requested_placement": requested,
        "selected_plan": plan.__dict__,
        "error_vs_cpu_float64_rounded_fp16": error,
        "generated_cuda_source": str(source_path),
        "source_checks": {
            "cuda_kernel": True,
            "packed_byte_input": True,
            "signed_nibble_decode": True,
            "selected_shared_decode_present": has_shared_decode,
        },
        "tir": str(tir_path),
    }


def _verify_signed_cancellation_and_poisoned_tail(device: torch.device, evidence_dir: Path) -> dict[str, object]:
    compile_kernel, gemm, pack = _load_p21_module()
    a_cpu = np.array([[1, -1, 0, 1, 0]], dtype=np.float16)
    q_cpu = np.zeros((16, 5), dtype=np.int8)
    q_cpu[0, :2] = [-8, -8]  # exact cancellation under [1, -1]
    q_cpu[1, :2] = [-8, 7]  # exact negative signed result
    q_cpu[3, :2] = [7, -8]  # exact positive signed result
    q_cpu[4:] = np.resize(np.arange(-8, 8, dtype=np.int8), (12, 5))
    q_cpu[2] = 0  # exact zero output

    a = torch.from_numpy(a_cpu.copy()).to(device)
    packed = pack(torch.from_numpy(q_cpu.copy()).to(device))
    packed[:, -1] = (packed[:, -1] & 0x0F) | 0xF0  # poison unused K=5 high nibble
    output, plan = gemm(a, packed, "auto")
    error = _check_output(output, a_cpu, q_cpu, a, device)
    expected_special = torch.tensor([[0.0, -15.0, 0.0, 15.0]], dtype=torch.float16, device=device)
    torch.testing.assert_close(output[:, :4], expected_special, atol=0, rtol=0)

    kernel, _ = compile_kernel(1, 16, 5, device, "auto")
    source = kernel.get_kernel_source()
    tir = kernel.get_tir(a, packed, output).script()
    if "< 5" not in source and "< 5" not in tir:
        raise RuntimeError("odd-tail generated code does not expose the K=5 mask")
    source_path = evidence_dir / "special-values.cu"
    tir_path = evidence_dir / "special-values.tir"
    source_path.write_text(source, encoding="utf-8")
    tir_path.write_text(tir, encoding="utf-8")
    return {
        "shape_mnk": [1, 16, 5],
        "exact_outputs": {"cancellation": 0, "negative": -15, "zero": 0, "positive": 15},
        "unused_high_nibble": "poisoned to 0xF for the odd K=5 tail",
        "error_vs_cpu_float64_rounded_fp16": error,
        "generated_code_k_mask": True,
        "cuda_source": str(source_path),
        "tir": str(tir_path),
    }


def drive(evidence_dir: Path) -> dict[str, object]:
    device, gpu_name, target = doctor()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    compile_kernel, gemm, pack = _load_p21_module()

    known = torch.tensor([[-8, -1, 0, 7]], dtype=torch.int8)
    if pack(known).tolist() != [[0xF8, 0x70]]:
        raise AssertionError("signed INT4 packing must place the even K value in the low nibble")
    _assert_bad_input(lambda: pack(torch.tensor([[-9, 0]], dtype=torch.int8)), ValueError)

    a_check = torch.zeros((1, 2), dtype=torch.float16, device=device)
    b_check = pack(torch.zeros((1, 2), dtype=torch.int8, device=device))
    _assert_bad_input(lambda: gemm(a_check.float(), b_check), TypeError)
    _assert_bad_input(lambda: gemm(a_check, b_check[:, :0]), ValueError)

    upstream = _verify_upstream_fallback(device, evidence_dir)
    special_values = _verify_signed_cancellation_and_poisoned_tail(device, evidence_dir)
    cases = [
        _verify_case("small-edge-fragment", (3, 17, 35), "auto", device, evidence_dir),
        _verify_case("small-edge-shared-control", (3, 17, 35), "shared", device, evidence_dir),
        _verify_case("rectangular-odd-tails", (48, 129, 65), "auto", device, evidence_dir),
        _verify_case("medium", (128, 256, 256), "auto", device, evidence_dir),
    ]
    result = {
        "tilelang": tilelang.__version__,
        "torch": torch.__version__,
        "gpu": gpu_name,
        "target": target,
        "upstream_fallback": upstream,
        "special_values_and_poisoned_odd_tail": special_values,
        "cases": cases,
        "signed_pack_check": "-8,-1,0,7 -> f8,70",
        "invalid_input_checks": ["out-of-range weights", "wrong activation dtype", "packed-weight shape"],
        "tolerance": {
            "atol": ATOL,
            "rtol": RTOL,
            "output_dtype": "float16",
            "fp16_ulp_limit": 2,
            "rationale": "FP32 tile accumulation is rounded once to FP16; the deterministic medium run has at most two half-ULP steps and reports absolute error separately.",
        },
        "pass": True,
    }
    (evidence_dir / "drive-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _torch_unpack_matmul(a: torch.Tensor, packed: torch.Tensor, k: int) -> torch.Tensor:
    values = packed.to(torch.int16)
    low = values & 15
    high = (values >> 4) & 15
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    weights = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)[:, :k].to(torch.float16)
    return a @ weights.T


def _median_observation(auto: list[float], control: list[float]) -> str:
    auto_median = float(np.median(auto))
    control_median = float(np.median(control))
    if auto_median < control_median:
        observation = "auto median lower"
    elif auto_median > control_median:
        observation = "auto median higher"
    else:
        observation = "medians equal"
    return f"median-only observation: {observation}; no statistical conclusion"


def benchmark(evidence_dir: Path, samples: int = 7, calls_per_sample: int = 5) -> dict[str, object]:
    device, gpu_name, target = doctor()
    compile_kernel, gemm, pack = _load_p21_module()
    m, n, k = 128, 256, 256
    a_cpu, q_cpu = _inputs(m, n, k, 21)
    a = torch.from_numpy(a_cpu.copy()).to(device)
    packed = pack(torch.from_numpy(q_cpu.copy()).to(device))
    kernel, plan = compile_kernel(m, n, k, device, "auto")
    output = torch.empty((m, n), dtype=torch.float16, device=device)
    kernel(a, packed, output)
    torch.cuda.synchronize(device)

    functions = {
        "torch_unpack_plus_matmul": lambda: _torch_unpack_matmul(a, packed, k),
        "tilelang_auto": lambda: gemm(a, packed, "auto")[0],
        "tilelang_fragment_control": lambda: gemm(a, packed, "fragment")[0],
        "tilelang_shared_control": lambda: gemm(a, packed, "shared")[0],
    }
    for fn in functions.values():
        for _ in range(5):
            result = fn()
        torch.cuda.synchronize(device)
        _check_output(result, a_cpu, q_cpu, a, device)

    wall_latencies: dict[str, list[float]] = {name: [] for name in functions}
    event_latencies: dict[str, list[float]] = {name: [] for name in functions}
    order = tuple(functions)
    for sample in range(samples):
        rotated_order = order[sample % len(order) :] + order[: sample % len(order)]
        for name in rotated_order:
            fn = functions[name]
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(calls_per_sample):
                result = fn()
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - start) * 1000 / calls_per_sample
            _check_output(result, a_cpu, q_cpu, a, device)
            wall_latencies[name].append(elapsed)

            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(calls_per_sample):
                result = fn()
            end_event.record()
            end_event.synchronize()
            event_elapsed = start_event.elapsed_time(end_event) / calls_per_sample
            _check_output(result, a_cpu, q_cpu, a, device)
            event_latencies[name].append(event_elapsed)

    baseline = wall_latencies["torch_unpack_plus_matmul"]
    selected = wall_latencies["tilelang_auto"]
    fragment = wall_latencies["tilelang_fragment_control"]
    shared = wall_latencies["tilelang_shared_control"]
    event_baseline = event_latencies["torch_unpack_plus_matmul"]
    event_selected = event_latencies["tilelang_auto"]
    event_fragment = event_latencies["tilelang_fragment_control"]
    event_shared = event_latencies["tilelang_shared_control"]
    baseline_over_selected = float(np.median(baseline) / np.median(selected))
    fragment_over_auto = float(np.median(fragment) / np.median(selected))
    shared_over_auto = float(np.median(shared) / np.median(selected))
    event_baseline_over_selected = float(np.median(event_baseline) / np.median(event_selected))
    event_fragment_over_auto = float(np.median(event_fragment) / np.median(event_selected))
    event_shared_over_auto = float(np.median(event_shared) / np.median(event_selected))
    record = {
        "tilelang": tilelang.__version__,
        "torch": torch.__version__,
        "gpu": gpu_name,
        "target": target,
        "shape_mnk": [m, n, k],
        "plan": plan.__dict__,
        "claim_tested": "the selected fused kernel lowers end-to-end latency versus Torch unpack plus matmul",
        "timing": {
            "wall_clock": "around the public operation, including output allocation and decode; compile excluded after warmup",
            "cuda_events": "elapsed CUDA stream time around the same warmed operations; excludes host-only work and compile, includes enqueued CUDA work",
        },
        "samples_per_side": samples,
        "calls_per_sample": calls_per_sample,
        "wall_clock_latency_ms": {
            name: {
                "samples": values,
                "median": float(np.median(values)),
                "range": [float(min(values)), float(max(values))],
            }
            for name, values in wall_latencies.items()
        },
        "cuda_event_latency_ms": {
            name: {
                "samples": values,
                "median": float(np.median(values)),
                "range": [float(min(values)), float(max(values))],
            }
            for name, values in event_latencies.items()
        },
        "baseline_over_selected_speedup": baseline_over_selected,
        "fragment_over_auto_speedup": fragment_over_auto,
        "shared_over_auto_speedup": shared_over_auto,
        "cuda_event_baseline_over_selected_speedup": event_baseline_over_selected,
        "cuda_event_fragment_over_auto_speedup": event_fragment_over_auto,
        "cuda_event_shared_over_auto_speedup": event_shared_over_auto,
        "result": {
            "auto_vs_torch_unpack_plus_matmul": _median_observation(selected, baseline),
            "auto_vs_fragment_control": _median_observation(selected, fragment),
            "auto_vs_shared_control": _median_observation(selected, shared),
            "event_auto_vs_torch_unpack_plus_matmul": _median_observation(event_selected, event_baseline),
            "event_auto_vs_fragment_control": _median_observation(event_selected, event_fragment),
            "event_auto_vs_shared_control": _median_observation(event_selected, event_shared),
        },
        "pass": True,
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "benchmark-result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def profile(evidence_dir: Path) -> dict[str, object]:
    device, gpu_name, target = doctor()
    compile_kernel, gemm, pack = _load_p21_module()
    m, n, k = 128, 256, 256
    a_cpu, q_cpu = _inputs(m, n, k, 21)
    a = torch.from_numpy(a_cpu.copy()).to(device)
    packed = pack(torch.from_numpy(q_cpu.copy()).to(device))
    kernel, plan = compile_kernel(m, n, k, device, "auto")
    output = torch.empty((m, n), dtype=torch.float16, device=device)
    kernel(a, packed, output)
    functions = {
        "torch_unpack_plus_matmul": lambda: _torch_unpack_matmul(a, packed, k),
        "tilelang_selected_plan": lambda: gemm(a, packed, "auto")[0],
    }
    for fn in functions.values():
        result = fn()
        torch.cuda.synchronize(device)
        _check_output(result, a_cpu, q_cpu, a, device)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    evidence_dir.mkdir(parents=True, exist_ok=True)
    trace_path = evidence_dir / "profile-trace.json"
    outputs = {}
    with torch.profiler.profile(activities=activities, record_shapes=False) as profiler:
        for name, fn in functions.items():
            with torch.profiler.record_function(name):
                outputs[name] = fn()
            torch.cuda.synchronize(device)
    for result in outputs.values():
        _check_output(result, a_cpu, q_cpu, a, device)
    summary = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=24)
    profiler.export_chrome_trace(str(trace_path))
    record = {
        "tilelang": tilelang.__version__,
        "torch": torch.__version__,
        "gpu": gpu_name,
        "target": target,
        "shape_mnk": [m, n, k],
        "plan": plan.__dict__,
        "trace": str(trace_path),
        "summary": summary,
        "purpose": "diagnostic only; profiler timings are not benchmark results",
        "pass": True,
    }
    (evidence_dir / "profile-result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--doctor", action="store_true")
    mode.add_argument("--drive", action="store_true")
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--profile", action="store_true")
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    if args.doctor:
        device, gpu_name, target = doctor()
        print(
            json.dumps(
                {"tilelang": tilelang.__version__, "torch": torch.__version__, "gpu": gpu_name, "target": target, "device": str(device)},
                indent=2,
            )
        )
    elif args.drive:
        print(json.dumps(drive(args.evidence_dir), indent=2))
    elif args.benchmark:
        print(json.dumps(benchmark(args.evidence_dir), indent=2))
    else:
        print(json.dumps(profile(args.evidence_dir), indent=2))


if __name__ == "__main__":
    main()
