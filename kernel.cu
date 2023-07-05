
#include "common.h"
#include "timer.h"
#include <iostream>
// #include "00_navie_gemm.cu"

#define CEIL_DIV(M, N) (M + N - 1) / N

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
    dim3 block_dim{32, 32};
    dim3 grid_dim{CEIL_DIV(M, block_dim.x), CEIL_DIV(N, block_dim.y)};
    navie_gemm_kernel<<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

