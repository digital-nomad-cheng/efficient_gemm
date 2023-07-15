#include "common.h"
#include "timer.h"

#include <iostream>

#define CEIL_DIV(M, N) (M + N - 1) / N

void mm_gpu_navie(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
    dim3 block_dim{32, 32};
    dim3 grid_dim{CEIL_DIV(M, block_dim.x), CEIL_DIV(N, block_dim.y)};
    navie_gemm_kernel<<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

void mm_gpu_coalesing(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
    dim3 block_dim{32 * 32, 1, 1};
    dim3 grid_dim{CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1};
    coalesing_gemm_kernel<<<grid_dim, block_dim>>>(A, B, C, M, N, K, 32);
}

void mm_gpu_shared_tiling(float *A, float *B, float *C, unsigned int M, unsigned int N, unsigned int K) {
    const int block_size = 16;
    dim3 block_dim{block_size, block_size, 1};
    dim3 grid_dim{CEIL_DIV(N, block_size), CEIL_DIV(M, block_size), 1};
    shared_mem_tiling_gemm_kernel<<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

void mm_gpu_shared_coalesing(float *A, float *B, float *C, unsigned int M, unsigned int N, unsigned int K) {
    const int block_size = 16;
    dim3 block_dim{block_size * block_size, 1, 1};
    dim3 grid_dim{CEIL_DIV(N, block_size), CEIL_DIV(M, block_size), 1};
    shared_mem_coalesing_gemm_kernel<<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}
