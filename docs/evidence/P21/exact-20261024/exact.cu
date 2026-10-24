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
  int64_t total[1];
  float special[1];
  total[0] = (int64_t)0;
  special[0] = 0x0p+0f/*0.000000e+00*/;
  if (((int)threadIdx.x) < 18) {
    for (int ki = 0; ki < 3; ++ki) {
      int code = ((((int)B[(((((int)threadIdx.x) % 3) * 2) + (ki >> 1))]) >> ((ki & 1) * 4)) & 15);
      float value = ((float)A[(((((int)threadIdx.x) / 3) * 3) + ki)]);
      if (isfinite(value)) {
        int condval;
        if ((8 <= code)) {
          condval = (code - 16);
        } else {
          condval = code;
        }
        total[0] = (total[0] + (((int64_t)(value * 0x1p+24f/*1.677722e+07*/)) * ((int64_t)condval)));
      } else {
        int condval_1;
        if ((8 <= code)) {
          condval_1 = (code - 16);
        } else {
          condval_1 = code;
        }
        special[0] = (special[0] + (value * ((float)condval_1)));
      }
    }
    half_t condval_2;
    if (isfinite(special[0])) {
      condval_2 = __double2half((((double)total[0]) * 0x1p-24/*5.960464e-08*/));
    } else {
      condval_2 = ((half_t)special[0]);
    }
    C[((int)threadIdx.x)] = condval_2;
  }
}
