"""Drive exact signed FP16/INT4 arithmetic with small independent oracles."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import psutil
import torch
from torch.utils.benchmark import Timer

from p21_verify import BENCHMARK_SHAPES, HELD_OUT_SHAPES, ROOT, _fp16_ulp_distances, _inputs, _load_p21_module, doctor


def check(actual, expected):
    actual = actual.cpu().numpy()
    np.testing.assert_array_equal(actual, expected)
    finite = np.isfinite(expected)
    return {
        "outputs": expected.size,
        "finite_outputs": int(finite.sum()),
        "max_ulp": int(_fp16_ulp_distances(actual[finite], expected[finite]).max(initial=0)),
    }


def drive(output_dir, benchmark=False):
    output_dir.mkdir(parents=True, exist_ok=False)
    device, gpu, target = doctor()
    compile_kernel, gemm, pack = _load_p21_module()
    started = time.perf_counter()
    result = {
        "schema": 1,
        "gpu": gpu,
        "target": target,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "accumulation": "exact",
        "seed": 21,
        "small_oracles": {},
        "shapes": [],
        "timing": None,
    }

    values = np.arange(65536, dtype=np.uint16).view(np.float16)
    values = values[np.isfinite(values)].reshape(-1, 1)
    weights = np.arange(-8, 8, dtype=np.int8).reshape(-1, 1)
    a = torch.from_numpy(values.copy()).to(device)
    packed = pack(torch.from_numpy(weights).to(device))
    with np.errstate(over="ignore", invalid="ignore"):
        reference = (values.astype(np.float64) @ weights.astype(np.float64).T).astype(np.float16)
    output, _ = gemm(a, packed, accumulation="exact")
    result["small_oracles"]["all_finite_fp16_times_all_int4"] = check(output, reference)
    del a, packed, output

    a = torch.ones((65536, 1), dtype=torch.float16, device=device)
    packed = pack(torch.ones((1, 1), dtype=torch.int8, device=device))
    output, _ = gemm(a, packed, accumulation="exact")
    result["small_oracles"]["rows_beyond_cuda_grid_y_limit"] = check(output, np.ones((65536, 1), dtype=np.float16))
    del a, packed, output

    values = np.array(
        [[65504, 2**-12, -65504], [1, 2**-11, 2**-24], [2**-24, 0, 0], [np.inf, 1, -np.inf], [np.inf, 1, 3], [np.nan, 1, 3]],
        dtype=np.float16,
    )
    weights = np.array([[1, 1, 1], [-8, 7, -1], [0, 0, 0]], dtype=np.int8)
    a = torch.from_numpy(values).to(device)
    packed = pack(torch.from_numpy(weights).to(device))
    packed[:, -1] |= 0xF0
    with np.errstate(over="ignore", invalid="ignore"):
        reference = (values.astype(np.float64) @ weights.astype(np.float64).T).astype(np.float16)
    output, plan = gemm(a, packed, accumulation="exact")
    result["small_oracles"]["cancellation_midpoint_subnormal_nonfinite_poisoned_tail"] = check(output, reference)
    assert float(output[0, 0]) == 2**-12
    assert float(output[1, 0]) == 1 + 2**-10
    kernel, _ = compile_kernel(6, 3, 3, device, accumulation="exact")
    source = kernel.get_kernel_source()
    assert "__double2half" in source and "int64_t" in source and "isfinite" in source
    (output_dir / "exact.cu").write_text(source, encoding="utf-8")
    result["exact_plan"] = asdict(plan)
    for kwargs in ({"accumulation": "bad"}, {"accumulation": "exact", "placement": "shared"}):
        try:
            compile_kernel(1, 1, 1, device, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid configuration {kwargs}")
    try:
        compile_kernel(1, 1, 1049089, device, accumulation="exact")
    except ValueError:
        result["overflow_bound_rejected"] = True
    else:
        raise AssertionError("Accepted an unsafe int64 accumulation length")

    for m, n, k in (*BENCHMARK_SHAPES, *HELD_OUT_SHAPES):
        values, weights = _inputs(m, n, k, 21)
        reference = (values.astype(np.float64) @ weights.astype(np.float64).T).astype(np.float16)
        a = torch.from_numpy(values).to(device)
        packed = pack(torch.from_numpy(weights).to(device))
        torch.cuda.reset_peak_memory_stats()
        output, _ = gemm(a, packed, accumulation="exact")
        record = {
            "shape_mnk": [m, n, k],
            **check(output, reference),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "input_sha256": hashlib.sha256(values.tobytes() + weights.tobytes()).hexdigest(),
            "rounded_reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        }
        result["shapes"].append(record)
        del a, packed, output

    if benchmark:
        shape = (128, 256, 256)
        values, weights = _inputs(*shape, 21)
        a = torch.from_numpy(values).to(device)
        packed = pack(torch.from_numpy(weights).to(device))
        functions = {mode: lambda mode=mode: gemm(a, packed, accumulation=mode)[0] for mode in ("fp32", "exact")}
        reference = (values.astype(np.float64) @ weights.astype(np.float64).T).astype(np.float16)
        for function in functions.values():
            output = function().cpu().numpy()
            assert int(_fp16_ulp_distances(output, reference).max()) <= 2
        measurements = {mode: [] for mode in functions}
        for repeat in range(5):
            for mode in tuple(functions) if repeat % 2 == 0 else tuple(reversed(functions)):
                measurement = Timer(stmt="function()", globals={"function": functions[mode]}, num_threads=1).blocked_autorange(
                    min_run_time=0.05
                )
                measurements[mode].append(
                    {
                        "median_ms": measurement.median * 1000,
                        "number_per_run": measurement.number_per_run,
                        "raw_seconds": measurement.raw_times,
                    }
                )
        result["timing"] = {
            "shape_mnk": shape,
            "measurements": measurements,
            "scope": "native Torch Timer public-call cost, including allocation, after warm compilation; five alternating samples per mode; exact scalar arithmetic versus FP32 tensor cores, not equal general accuracy",
        }
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
            for mode, function in functions.items():
                with torch.profiler.record_function(f"p21_{mode}"):
                    function()
                    torch.cuda.synchronize()
        profile.export_chrome_trace(str(output_dir / "diagnostic-profile.json"))
        result["diagnostic_profile"] = "diagnostic-profile.json; separately collected after timing, not a timing sample"
    result["elapsed_seconds_including_compile_and_oracles"] = time.perf_counter() - started
    memory = psutil.Process().memory_info()
    result["parent_process_memory"] = {
        "rss_at_end_bytes": memory.rss,
        "peak_working_set_bytes": getattr(memory, "peak_wset", None),
        "scope": "parent process only, including imports/compiler host state; excludes child compiler processes and device memory",
    }
    result["source_sha256"] = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), ROOT / "tools/p21_verify.py", ROOT / "examples/dequantize_gemm/p21_fp16xint4.py")
    }
    result["pass"] = True
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    print(json.dumps(drive(args.output, args.benchmark), indent=2))
