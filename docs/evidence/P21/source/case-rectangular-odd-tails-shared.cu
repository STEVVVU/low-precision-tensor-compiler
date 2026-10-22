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
  void* B_packed_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 20480));
  float C_fragment[16];
  uchar B_packed_fragment[32];
  half_t B_fp16_fragment[64];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_fragment + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int ko = 0; ko < 2; ++ko) {
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      half_t condval;
      if (((bool)1 & (((ko * 64) + (((int)threadIdx.x) & 63)) < 65))) {
        condval = A[(((((((int)blockIdx.y) * 1040) + (i_1 * 130)) + ((((int)threadIdx.x) >> 6) * 65)) + (ko * 64)) + (((int)threadIdx.x) & 63))];
      } else {
        condval = half_t(0x0p+0f/*0.000000e+00*/);
      }
      ((half_t*)A_shared)[((((((i_1 * 128) + ((((int)threadIdx.x) >> 6) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((i_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_1 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((int)threadIdx.x) & 7))] = condval;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 32; ++i_2) {
      uchar condval_1;
      if ((((((((int)blockIdx.x) * 128) + (i_2 * 4)) + (((int)threadIdx.x) >> 5)) < 129) & (((ko * 32) + (((int)threadIdx.x) & 31)) < 33))) {
        condval_1 = B[(((((((int)blockIdx.x) * 4224) + (i_2 * 132)) + ((((int)threadIdx.x) >> 5) * 33)) + (ko * 32)) + (((int)threadIdx.x) & 31))];
      } else {
        condval_1 = (uchar)0;
      }
      ((uchar*)B_packed_shared)[((i_2 * 128) + ((int)threadIdx.x))] = condval_1;
    }
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      *(uint4*)(B_packed_fragment + (i_3 * 16)) = *(uint4*)(((uchar*)B_packed_shared) + ((i_3 * 2048) + (((int)threadIdx.x) * 16)));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      int packed = ((int)B_packed_fragment[(i_4 >> 1)]);
      half_t condval_2;
      if (((((ko * 64) + ((((int)threadIdx.x) & 1) * 32)) + (i_4 & 31)) < 65)) {
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
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      *(uint4*)(((half_t*)B_fp16_shared) + ((((((i_5 >> 2) * 4096) + ((((int)threadIdx.x) >> 1) * 64)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_5 & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_5 & 1)) & 1) * 8))) = *(uint4*)(B_fp16_fragment + (i_5 * 8));
    }
    {
      half_t A_local[8];
      half_t B_local[16];
      __syncthreads();
      for (int ki = 0; ki < 4; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)A_shared)[(((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        #pragma unroll
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          tl::ptx_ldmatrix_x4((&(((half_t*)B_fp16_shared)[((((((((((int)threadIdx.x) >> 5) * 2048) + (i_6 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_6 * 8)])));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_fragment + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_fragment + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 16; ++i_7) {
    if ((bool)1 & ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 32)) + ((i_7 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1)) < 129)) {
      C[((((((((((int)blockIdx.y) * 2064) + (((i_7 & 3) >> 1) * 1032)) + (((((int)threadIdx.x) & 31) >> 2) * 129)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + ((i_7 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1))] = ((half_t)C_fragment[i_7]);
    }
  }
}
