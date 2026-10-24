"""FP16 by signed INT4 GEMM with compile-time decode placement selection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from typing import Literal

import tilelang
from tilelang import language as T
import torch


Placement = Literal["auto", "fragment", "shared"]
Accumulation = Literal["fp32", "exact"]
_EXACT_K_LIMIT = ((1 << 63) - 1) // (65504 * (1 << 24) * 8)


@dataclass(frozen=True)
class Schedule:
    block_m: int
    block_n: int
    block_k: int
    stages: int
    threads: int
    placement: Literal["fragment", "shared"]
    estimated_shared_bytes: int
    decoded_values_per_thread: int
    accumulation: Accumulation = "fp32"


def pack_signed_int4(weights: torch.Tensor) -> torch.Tensor:
    """Pack an [N, K] int8 tensor as [N, ceil(K/2)] low-nibble-first uint8."""
    if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[1] == 0:
        raise ValueError("weights must have non-empty shape [N, K]")
    if weights.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("weights must use a signed integer dtype")
    if weights.min().item() < -8 or weights.max().item() > 7:
        raise ValueError("signed INT4 weights must be in [-8, 7]")
    if weights.shape[1] % 2:
        weights = torch.cat((weights, torch.zeros_like(weights[:, :1])), dim=1)
    codes = weights.to(torch.int16).bitwise_and(15).to(torch.uint8)
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def _schedule(
    m: int,
    n: int,
    k: int,
    device: torch.device,
    requested: Placement,
    accumulation: Accumulation = "fp32",
) -> Schedule:
    if min(m, n, k) < 1:
        raise ValueError("M, N, and K must be positive")
    if requested not in ("auto", "fragment", "shared"):
        raise ValueError(f"unsupported decode placement: {requested}")
    if accumulation not in ("fp32", "exact"):
        raise ValueError(f"unsupported accumulation: {accumulation}")
    if accumulation == "exact":
        if k > _EXACT_K_LIMIT:
            raise ValueError(f"exact accumulation requires K <= {_EXACT_K_LIMIT}")
        if requested != "auto":
            raise ValueError("exact accumulation requires auto placement; it decodes scalar weights")
        return Schedule(1, 128, 1, 1, 128, "fragment", 0, 1, "exact")

    properties = torch.cuda.get_device_properties(device)
    block_m = 16 if m < 64 else 64
    block_n = 64 if n < 128 else 128
    block_k = min(64, max(16, ceil(k / 16) * 16))
    threads = 4 * int(properties.warp_size)
    stages = 3 if k >= 3 * block_k else 2
    decoded_values_per_thread = ceil(block_n * block_k / threads)
    register_value_budget = int(properties.warp_size)
    # shortcut: budget logical staged bytes; use compiler resource metadata before adding padding-aware scheduling.
    estimated_shared_bytes = stages * (2 * block_m * block_k + block_n * block_k // 2 + 2 * block_n * block_k)
    shared_limit = int(getattr(properties, "shared_memory_per_block_optin", 0) or properties.shared_memory_per_block)

    if requested == "auto":
        placement: Literal["fragment", "shared"] = "shared" if decoded_values_per_thread > register_value_budget else "fragment"
        if placement == "shared" and estimated_shared_bytes > shared_limit * 0.85:
            stages = 2
            estimated_shared_bytes = stages * (2 * block_m * block_k + block_n * block_k // 2 + 2 * block_n * block_k)
        if placement == "shared" and estimated_shared_bytes > shared_limit * 0.85:
            placement = "fragment"
    else:
        placement = requested
        if placement == "shared" and estimated_shared_bytes > shared_limit:
            raise ValueError(f"shared decode needs about {estimated_shared_bytes} bytes, above the device limit {shared_limit}")

    if placement == "fragment":
        estimated_shared_bytes = stages * (2 * block_m * block_k + block_n * block_k // 2)
    return Schedule(
        block_m,
        block_n,
        block_k,
        stages,
        threads,
        placement,
        estimated_shared_bytes,
        decoded_values_per_thread,
    )


@lru_cache(maxsize=64)
def _compile(m: int, n: int, k: int, schedule: Schedule):
    if schedule.accumulation == "exact":
        return _compile_exact(m, n, k)
    bm, bn, bk = schedule.block_m, schedule.block_n, schedule.block_k
    stages, threads = schedule.stages, schedule.threads
    decode_to_shared = schedule.placement == "shared"

    @tilelang.jit
    def main(
        A: T.Tensor((m, k), T.float16),
        B: T.Tensor((n, (k + 1) // 2), T.uint8),
        C: T.Tensor((m, n), T.float16),
    ):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((bm, bk), T.float16)
            B_packed_shared = T.alloc_shared((bn, bk // 2), T.uint8)
            B_packed_fragment = T.alloc_fragment((bn, bk // 2), T.uint8)
            B_fp16_fragment = T.alloc_fragment((bn, bk), T.float16)
            C_fragment = T.alloc_fragment((bm, bn), T.float32)
            if decode_to_shared:
                B_fp16_shared = T.alloc_shared((bn, bk), T.float16)
                T.annotate_layout({B_fp16_shared: tilelang.layout.make_swizzled_layout(B_fp16_shared, k_major=True)})

            T.clear(C_fragment)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=stages):
                for i, j in T.Parallel(bm, bk):
                    row = by * bm + i
                    col = ko * bk + j
                    A_shared[i, j] = T.if_then_else((row < m) & (col < k), A[row, col], T.cast(0, T.float16))

                for i, j in T.Parallel(bn, bk // 2):
                    row = bx * bn + i
                    col = ko * (bk // 2) + j
                    B_packed_shared[i, j] = T.if_then_else((row < n) & (col < (k + 1) // 2), B[row, col], T.cast(0, T.uint8))
                T.copy(B_packed_shared, B_packed_fragment)

                for i, j in T.Parallel(bn, bk):
                    packed = T.cast(B_packed_fragment[i, j // 2], T.int32)
                    nibble = (packed >> ((j % 2) * 4)) & 15
                    signed = T.if_then_else(nibble >= 8, nibble - 16, nibble)
                    value = T.if_then_else(ko * bk + j < k, T.cast(signed, T.float16), T.cast(0, T.float16))
                    B_fp16_fragment[i, j] = value

                if decode_to_shared:
                    T.copy(B_fp16_fragment, B_fp16_shared)
                    T.gemm(A_shared, B_fp16_shared, C_fragment, transpose_B=True)
                else:
                    T.gemm(A_shared, B_fp16_fragment, C_fragment, transpose_B=True)

            for i, j in T.Parallel(bm, bn):
                row, col = by * bm + i, bx * bn + j
                if (row < m) & (col < n):
                    C[row, col] = T.cast(C_fragment[i, j], T.float16)

    return main


def _compile_exact(m: int, n: int, k: int):
    @tilelang.jit
    def main(
        A: T.Tensor((m, k), T.float16),
        B: T.Tensor((n, (k + 1) // 2), T.uint8),
        C: T.Tensor((m, n), T.float16),
    ):
        with T.Kernel(T.ceildiv(m * n, 128), threads=128) as bx:
            total = T.alloc_local((1,), T.int64)
            special = T.alloc_local((1,), T.float32)
            index = bx * 128 + T.get_thread_binding()
            row, col = index // n, index % n
            total[0] = T.int64(0)
            special[0] = T.float32(0)
            if index < m * n:
                for ki in T.serial(k):
                    code = (T.cast(B[col, ki // 2], T.int32) >> ((ki % 2) * 4)) & 15
                    weight = T.if_then_else(code >= 8, code - 16, code)
                    value = T.cast(A[row, ki], T.float32)
                    if T.isfinite(value):
                        # FP16 values are integral in units of 2^-24; the compile-time K bound prevents overflow.
                        total[0] += T.cast(value * 16777216.0, T.int64) * T.cast(weight, T.int64)
                    else:
                        special[0] += value * T.cast(weight, T.float32)
                C[row, col] = T.if_then_else(
                    T.isfinite(special[0]),
                    T.call_pure_extern(T.float16, "__double2half", T.cast(total[0], T.float64) * T.float64(2**-24)),
                    T.cast(special[0], T.float16),
                )

    return main


def compile_fp16xint4(
    m: int,
    n: int,
    k: int,
    device: torch.device,
    placement: Placement = "auto",
    accumulation: Accumulation = "fp32",
):
    """Return a kernel for the current SM120 CUDA device and its hardware-aware plan."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("P21 requires the active CUDA device; use the current SM120 device")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    if device_index != torch.cuda.current_device():
        raise ValueError("P21 requires the active CUDA device; use the current SM120 device")
    device = torch.device("cuda", device_index)
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError("P21 currently supports the active SM120 CUDA device")
    plan = _schedule(m, n, k, device, placement, accumulation)
    return _compile(m, n, k, plan), plan


def fp16xint4_gemm(
    A: torch.Tensor,
    packed_B: torch.Tensor,
    placement: Placement = "auto",
    accumulation: Accumulation = "fp32",
) -> tuple[torch.Tensor, Schedule]:
    if A.ndim != 2 or packed_B.ndim != 2:
        raise ValueError("A and packed_B must be rank-two tensors")
    if A.dtype != torch.float16 or packed_B.dtype != torch.uint8:
        raise TypeError("A must be float16 and packed_B must be uint8")
    if not A.is_cuda or A.device != packed_B.device:
        raise ValueError("A and packed_B must be CUDA tensors on the same device")
    if not A.is_contiguous() or not packed_B.is_contiguous():
        raise ValueError("A and packed_B must be contiguous")
    m, k = A.shape
    n = packed_B.shape[0]
    if packed_B.shape[1] != (k + 1) // 2:
        raise ValueError(f"packed_B must have shape [N, {(k + 1) // 2}]")
    kernel, plan = compile_fp16xint4(m, n, k, A.device, placement, accumulation)
    output = torch.empty((m, n), dtype=torch.float16, device=A.device)
    kernel(A, packed_B, output)
    return output, plan
