#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
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
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_packed_shared = ((void*)((char*)buf_dyn_shmem + 512));
  float C_fragment[8];
  uchar B_packed_fragment[4];
  half_t B_fp16_fragment[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_fragment + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    half_t condval;
    if (((((i_1 * 8) + (((int)threadIdx.x) >> 4)) < 1) & ((((int)threadIdx.x) & 15) < 5))) {
      condval = A[(((i_1 * 40) + ((((int)threadIdx.x) >> 4) * 5)) + (((int)threadIdx.x) & 15))];
    } else {
      condval = half_t(0x0p+0f/*0.000000e+00*/);
    }
    ((half_t*)A_shared)[((((i_1 * 128) + ((((int)threadIdx.x) >> 4) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((int)threadIdx.x) & 7))] = condval;
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    uchar condval_1;
    if (((i_2 < 1) & ((((int)threadIdx.x) & 7) < 3))) {
      condval_1 = B[(((i_2 * 48) + ((((int)threadIdx.x) >> 3) * 3)) + (((int)threadIdx.x) & 7))];
    } else {
      condval_1 = (uchar)0;
    }
    ((uchar*)B_packed_shared)[((i_2 * 128) + ((int)threadIdx.x))] = condval_1;
  }
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    B_packed_fragment[i_3] = ((uchar*)B_packed_shared)[((((((((int)threadIdx.x) >> 5) * 128) + ((i_3 >> 1) * 64)) + (((((int)threadIdx.x) & 31) >> 2) * 8)) + ((i_3 & 1) * 4)) + (((int)threadIdx.x) & 3))];
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    int packed = ((int)B_packed_fragment[(i_4 >> 1)]);
    half_t condval_2;
    if (((((((i_4 & 3) >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1)) < 5)) {
      int condval_3;
      if ((8 <= ((packed >> ((i_4 & 1) * 4)) & 15))) {
        condval_3 = (((packed >> ((i_4 & 1) * 4)) & 15) - 16);
      } else {
        condval_3 = ((packed >> ((i_4 & 1) * 4)) & 15);
      }
      condval_2 = ((half_t)condval_3);
    } else {
      condval_2 = half_t(0x0p+0f/*0.000000e+00*/);
    }
    B_fp16_fragment[i_4] = condval_2;
  }
  {
    half_t A_local[8];
    tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[(((((int)threadIdx.x) & 15) * 16) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local[0])));
    tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_fragment + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_fp16_fragment + 0));
    tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_fragment + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_fp16_fragment + 4));
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    half_t C_local_cast[2];
    if (((((i_5 & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) < 1) & (((int)threadIdx.x) < 32)) {
      uint1 __1;
      float2 v_ = *(float2*)(C_fragment + (i_5 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(C_local_cast + 0) = __1;
      *(uint1*)(C + ((((((i_5 & 1) * 128) + ((((int)threadIdx.x) >> 5) * 16)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + ((i_5 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast + 0);
    }
  }
}
