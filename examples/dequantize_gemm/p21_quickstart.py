"""Run a small real-GPU P21 example from the repository root."""

from pathlib import Path

import tilelang
import torch

from p21_fp16xint4 import fp16xint4_gemm, pack_signed_int4


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
tilelang_path = Path(tilelang.__file__).resolve()
if REPOSITORY_ROOT in tilelang_path.parents:
    raise RuntimeError("P21 requires the installed TileLang wheel; run this file by path so the checkout does not shadow it")
if tilelang.__version__ != "0.1.15":
    raise RuntimeError(f"P21 is verified with TileLang 0.1.15, got {tilelang.__version__}")


device = torch.device("cuda", torch.cuda.current_device())
generator = torch.Generator().manual_seed(21)
weights = torch.randint(-8, 8, (17, 35), dtype=torch.int8, generator=generator)
activations = torch.randn((3, 35), dtype=torch.float16, generator=generator)
packed_weights = pack_signed_int4(weights).to(device)
activations = activations.to(device)
output, schedule = fp16xint4_gemm(activations, packed_weights)
reference = activations.float() @ weights.to(device=device, dtype=torch.float32).T
torch.testing.assert_close(output.float(), reference, atol=0.0625, rtol=0.001)

print(f"TileLang {tilelang.__version__} from {tilelang_path}")
print(f"P21 passed on {torch.cuda.get_device_name(device)}: {tuple(output.shape)}, {schedule}")
