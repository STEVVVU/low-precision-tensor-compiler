"""Compare output accumulation policies on the frozen large-GEMM oracle case."""

import json
import numpy as np
import torch

from p21_verify import EVIDENCE, _inputs, _load_p21_module, _torch_unpack_matmul, doctor


def main():
    device, gpu, target = doctor()
    _, gemm, pack = _load_p21_module()
    m, n, k = 256, 2048, 2048
    a_cpu, q_cpu = _inputs(m, n, k, 21)
    reference = (a_cpu.astype(np.float64) @ q_cpu.astype(np.float64).T).astype(np.float16).astype(np.float32)
    a = torch.from_numpy(a_cpu.copy()).to(device)
    packed = pack(torch.from_numpy(q_cpu.copy()).to(device))
    decoded = torch.from_numpy(q_cpu.copy()).to(device=device, dtype=torch.float16)
    functions = {
        "torch_unpack_plus_matmul": lambda: _torch_unpack_matmul(a, packed, k),
        "torch_predecoded_fp16": lambda: a @ decoded.T,
        "torch_predecoded_fp32_output": lambda: torch.mm(a, decoded.T, out_dtype=torch.float32).half(),
        "torch_fp32": lambda: (a.float() @ decoded.float().T).half(),
        "tilelang_auto": lambda: gemm(a, packed, "auto")[0],
        "tilelang_fragment": lambda: gemm(a, packed, "fragment")[0],
        "tilelang_shared": lambda: gemm(a, packed, "shared")[0],
    }
    record = {
        "gpu": gpu,
        "target": target,
        "shape": [m, n, k],
        "seed": 21,
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "results": {},
    }
    for name, function in functions.items():
        actual = function().float().cpu().numpy()
        difference = np.abs(actual - reference)
        record["results"][name] = {
            "max_abs": float(difference.max()),
            "violations_at_existing_tolerance": int(np.count_nonzero(difference > 0.0625 + 0.001 * np.abs(reference))),
            "outputs": int(actual.size),
        }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "large-gemm-numerical-diagnostic.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
