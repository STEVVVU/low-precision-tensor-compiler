import ast
from pathlib import Path

import numpy as np


def _load_ulp_helper_without_importing_torch():
    source = Path(__file__).resolve().parents[3] / "tools" / "p21_verify.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    helper = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_fp16_ulp_distances")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_fp16_ulp_distances"]


def test_fp16_ulp_ordering_across_zero_signs_adjacent_values_and_subnormals():
    distance = _load_ulp_helper_without_importing_torch()
    expected_bits = np.array([0x8000, 0x3C00, 0xBC00, 0x0001, 0x8001, 0x0000], dtype=np.uint16)
    actual_bits = np.array([0x0000, 0x3C01, 0xBC01, 0x0002, 0x8002, 0x8000], dtype=np.uint16)

    assert distance(actual_bits.view(np.float16), expected_bits.view(np.float16)).tolist() == [1] * 6


if __name__ == "__main__":
    test_fp16_ulp_ordering_across_zero_signs_adjacent_values_and_subnormals()
