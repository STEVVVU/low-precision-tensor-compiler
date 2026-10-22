#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const half_t* __restrict__ A, const uchar* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const half_t* __restrict__ A, const uchar* __restrict__ B, half_t* __restrict__ C) {
  float accum_res[1];
  half_t A_local[8];
  uchar B_quant_local[4];
  half_t B_dequantize_local[8];
  float reduced_accum_res[1];
  accum_res[0] = 0x0p+0f/*0.000000e+00*/;
  *(uint4*)(A_local + 0) = *(uint4*)(A + ((((int)blockIdx.y) * 256) + (((int)threadIdx.x) * 8)));
  *(uint*)(B_quant_local + 0) = *(uint*)(B + (((((int)blockIdx.x) * 512) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)));
  for (int ki = 0; ki < 8; ++ki) {
    B_dequantize_local[ki] = ((half_t)((((((int)B_quant_local[(ki >> 1)]) >> ((ki & 1) * 4)) & 15) << 28) >> 28));
  }
  for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
    accum_res[0] = (accum_res[0] + ((float)(A_local[ki_1] * B_dequantize_local[ki_1])));
  }
  float red_buf0[1];
  float t0[1];
  uint mask[1];
  red_buf0[0] = accum_res[0];
  mask[0] = __activemask();
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
  if (((int)threadIdx.x) == 0) {
    C[(((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 4)) + ((int)threadIdx.y))] = ((half_t)red_buf0[0]);
  }
}
