"""Compare accumulation policies on the frozen large-GEMM oracle case."""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from p21_verify import ATOL, EVIDENCE, ROOT, RTOL, _fp16_ulp_distances, _inputs, _load_p21_module, _torch_unpack_matmul, doctor


SHAPE = (256, 2048, 2048)
SEED = 21
WORST_LIMIT = 8
DEFAULT_OUTPUT = EVIDENCE / "large-gemm-numerical-diagnostic-expanded.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _worst_examples(actual: np.ndarray, reference: np.ndarray, a_cpu: np.ndarray, q_cpu: np.ndarray) -> tuple[dict[str, object], ...]:
    finite = np.isfinite(actual) & np.isfinite(reference)
    flat_finite = np.flatnonzero(finite)
    ulp = _fp16_ulp_distances(actual, reference)
    worst = flat_finite[np.lexsort((flat_finite, ulp.ravel()[flat_finite]))[-WORST_LIMIT:]][::-1]
    examples = []
    for flat_index in worst:
        row, col = np.unravel_index(int(flat_index), actual.shape)
        products = a_cpu[row].astype(np.float64) * q_cpu[col].astype(np.float64)
        exact_sum = math.fsum(float(value) for value in products)
        sum_abs_products = math.fsum(abs(float(value)) for value in products)
        abs_sum = abs(exact_sum)
        condition = None if sum_abs_products == 0 and abs_sum == 0 else ("infinite" if abs_sum == 0 else sum_abs_products / abs_sum)
        numpy_fp32_sum = np.sum(a_cpu[row].astype(np.float32) * q_cpu[col].astype(np.float32), dtype=np.float32)
        examples.append(
            {
                "coordinate_m_n": [int(row), int(col)],
                "actual_fp16": float(actual[row, col]),
                "reference_fp16": float(reference[row, col]),
                "absolute_error": float(abs(float(actual[row, col]) - float(reference[row, col]))),
                "ulp_distance": int(ulp[row, col]),
                "dot_condition_number": condition,
                "sum_abs_products_float64": sum_abs_products,
                "numpy_fp32_sum": float(numpy_fp32_sum),
                "numpy_fp32_sum_rounded_fp16": float(np.float16(numpy_fp32_sum)),
                "math_fsum_products_float64": exact_sum,
                "math_fsum_rounded_fp16": float(np.float16(exact_sum)),
            }
        )
    return tuple(examples)


def _arm_result(output: torch.Tensor, reference: np.ndarray, a_cpu: np.ndarray, q_cpu: np.ndarray) -> dict[str, object]:
    actual = output.detach().to(torch.float16).cpu().numpy()
    finite = np.isfinite(actual) & np.isfinite(reference)
    difference = np.abs(actual.astype(np.float32) - reference.astype(np.float32))
    ulp = _fp16_ulp_distances(actual, reference)
    finite_ulp = ulp[finite]
    finite_count = int(np.count_nonzero(finite))
    max_ulp = int(finite_ulp.max()) if finite_count else None
    nonfinite_actual = int(np.count_nonzero(~np.isfinite(actual)))
    nonfinite_reference = int(np.count_nonzero(~np.isfinite(reference)))
    finite_difference = difference[finite]
    return {
        "max_abs": float(finite_difference.max()) if finite_count else None,
        "violations_at_existing_tolerance": int(
            np.count_nonzero(~finite | (difference > ATOL + RTOL * np.abs(reference.astype(np.float32))))
        ),
        "finite_ordered_fp16_ulp": {
            "max": max_ulp,
            "count_over_1": int(np.count_nonzero(finite_ulp > 1)),
            "count_over_2": int(np.count_nonzero(finite_ulp > 2)),
            "finite_comparisons": finite_count,
            "nonfinite_actual": nonfinite_actual,
            "nonfinite_reference": nonfinite_reference,
        },
        "passes_original_two_ulp_gate": (max_ulp is not None and max_ulp <= 2 and nonfinite_actual == 0 and nonfinite_reference == 0),
        "worst_examples": _worst_examples(actual, reference, a_cpu, q_cpu),
    }


def _runtime_metadata(device: torch.device, gpu: str, target: str) -> dict[str, object]:
    try:
        driver = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = None
    matmul = torch.backends.cuda.matmul
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver,
        "gpu": gpu,
        "target": target,
        "device": str(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "input_dtype": {"activation": "float16", "weights": "int8 signed values", "output": "float16"},
        "torch_precision_flags_observed_without_changes": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": bool(matmul.allow_tf32),
            "allow_fp16_reduced_precision_reduction": bool(matmul.allow_fp16_reduced_precision_reduction),
            "allow_bf16_reduced_precision_reduction": bool(matmul.allow_bf16_reduced_precision_reduction),
        },
    }


def diagnose(output_path: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    device, gpu, target = doctor()
    _, gemm, pack = _load_p21_module()
    m, n, k = SHAPE
    a_cpu, q_cpu = _inputs(m, n, k, SEED)
    cpu_ref64 = a_cpu.astype(np.float64) @ q_cpu.astype(np.float64).T
    reference = cpu_ref64.astype(np.float16)
    a = torch.from_numpy(a_cpu.copy()).to(device)
    packed = pack(torch.from_numpy(q_cpu.copy()).to(device))
    decoded = torch.from_numpy(q_cpu.copy()).to(device=device, dtype=torch.float16)
    decoded64 = torch.from_numpy(q_cpu.copy()).to(device=device, dtype=torch.float64)
    a64 = a.to(dtype=torch.float64)
    functions = {
        "torch_unpack_plus_matmul": lambda: _torch_unpack_matmul(a, packed, k),
        "torch_predecoded_fp16": lambda: a @ decoded.T,
        "torch_predecoded_fp32_output": lambda: torch.mm(a, decoded.T, out_dtype=torch.float32).half(),
        "torch_fp32": lambda: (a.float() @ decoded.float().T).half(),
        "torch_cuda_fp64_accum_fp16_output": lambda: (a64 @ decoded64.T).to(torch.float16),
        "tilelang_auto": lambda: gemm(a, packed, "auto")[0],
        "tilelang_fragment": lambda: gemm(a, packed, "fragment")[0],
        "tilelang_shared": lambda: gemm(a, packed, "shared")[0],
    }
    input_bytes = a_cpu.tobytes(order="C") + q_cpu.tobytes(order="C")
    source_files = (
        ROOT / "tools" / "p21_diagnose.py",
        ROOT / "tools" / "p21_verify.py",
        ROOT / "examples" / "dequantize_gemm" / "p21_fp16xint4.py",
    )
    record = {
        "purpose": "diagnostic only; no runtime fallback or optimization",
        "frozen_input_identity": {
            "shape_mnk": [m, n, k],
            "seed": SEED,
            "generator": "numpy.random.default_rng(seed=21), standard_normal cast to float16 activations, integers(-8, 8) int8 weights",
            "activation_dtype": str(a_cpu.dtype),
            "weight_dtype": str(q_cpu.dtype),
            "activation_sha256": _sha256(a_cpu.tobytes(order="C")),
            "weight_sha256": _sha256(q_cpu.tobytes(order="C")),
            "source_inputs_sha256": _sha256(input_bytes),
        },
        "reference": {
            "method": "CPU NumPy float64 matmul on the frozen float16/int8 input values, then round to float16",
            "cpu_float64_reference_sha256": _sha256(cpu_ref64.tobytes(order="C")),
            "cpu_float64_reference_rounded_fp16_sha256": _sha256(reference.tobytes(order="C")),
        },
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path.read_bytes()) for path in source_files},
        "model_sha256": None,
        "runtime": _runtime_metadata(device, gpu, target),
        "results": {},
    }
    for name, function in functions.items():
        record["results"][name] = _arm_result(function(), reference, a_cpu, q_cpu)
    record["passes_original_two_ulp_gate"] = all(result["passes_original_two_ulp_gate"] for result in record["results"].values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
