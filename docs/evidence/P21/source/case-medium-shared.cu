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
  void* B_fp16_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* B_packed_shared = ((void*)((char*)buf_dyn_shmem + 24576));
  float C_fragment[64];
  uchar B_packed_fragment[32];
  half_t B_fp16_fragment[64];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_fragment + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int ko = 0; ko < 4; ++ko) {
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      half_t broadcast_var_1 = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval;
      if (((bool)1 & (bool)1)) {
        condval = *(uint4*)(A + (((((((int)blockIdx.y) * 16384) + (i_1 * 4096)) + ((((int)threadIdx.x) >> 3) * 256)) + (ko * 64)) + ((((int)threadIdx.x) & 7) * 8)));
      } else {
        condval = make_uint4(__pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1));
      }
      *(uint4*)(((half_t*)A_shared) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      uchar broadcast_var_2 = (uchar)0;
      uint4 condval_1;
      if (((bool)1 & (bool)1)) {
        condval_1 = *(uint4*)(B + (((((((int)blockIdx.x) * 16384) + (i_2 * 8192)) + ((((int)threadIdx.x) >> 1) * 128)) + (ko * 32)) + ((((int)threadIdx.x) & 1) * 16)));
      } else {
        condval_1 = tl::broadcast<uint4>(broadcast_var_2);
      }
      *(uint4*)(((uchar*)B_packed_shared) + ((i_2 * 2048) + (((int)threadIdx.x) * 16))) = condval_1;
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      *(uint4*)(B_packed_fragment + (i_3 * 16)) = *(uint4*)(((uchar*)B_packed_shared) + ((i_3 * 2048) + (((int)threadIdx.x) * 16)));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      int packed = ((int)B_packed_fragment[(i_4 >> 1)]);
      int condval_2;
      if ((8 <= ((packed >> ((i_4 & 1) * 4)) & 15))) {
        condval_2 = (((packed >> ((i_4 & 1) * 4)) & 15) - 16);
      } else {
        condval_2 = ((packed >> ((i_4 & 1) * 4)) & 15);
      }
      B_fp16_fragment[i_4] = ((half_t)condval_2);
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      *(uint4*)(((half_t*)B_fp16_shared) + ((((((i_5 >> 2) * 4096) + ((((int)threadIdx.x) >> 1) * 64)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_5 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_5 & 1)) & 1) * 8))) = *(uint4*)(B_fp16_fragment + (i_5 * 8));
    }
    {
      half_t A_local[16];
      half_t B_local[32];
      __syncthreads();
      for (int ki = 0; ki < 4; ++ki) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[((((((((((int)threadIdx.x) & 63) >> 5) * 2048) + (i_6 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[(i_6 * 8)])));
        }
        #pragma unroll
        for (int i_7 = 0; i_7 < 4; ++i_7) {
          tl::ptx_ldmatrix_x4((&(((half_t*)B_fp16_shared)[((((((((((int)threadIdx.x) >> 6) * 4096) + (i_7 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_7 * 8)])));
        }
        for (int i_8 = 0; i_8 < 2; ++i_8) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_fragment + ((i_8 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_8 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_fragment + (((i_8 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_8 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 32; ++i_9) {
    half_t C_local_cast[2];
    uint1 __1;
    float2 v_ = *(float2*)(C_fragment + (i_9 * 2));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    *(uint1*)(C_local_cast + 0) = __1;
    *(uint1*)(C + (((((((((((int)blockIdx.y) * 16384) + (((((int)threadIdx.x) & 63) >> 5) * 8192)) + ((i_9 >> 4) * 4096)) + ((i_9 & 1) * 2048)) + (((((int)threadIdx.x) & 31) >> 2) * 256)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_9 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local_cast + 0);
  }
}
