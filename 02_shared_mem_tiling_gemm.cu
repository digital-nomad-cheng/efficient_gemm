#include <cuda_runtime.h>
#include <stdio.h>
#define BLOCKSIZE 16
__global__ void shared_mem_tiling_gemm_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{

  // shared memory for loading blockks of A and B
  __shared__ float As[BLOCKSIZE][BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

  const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  float result = 0.0f;
  for (unsigned int bkIdx = 0; bkIdx < (K - 1) / BLOCKSIZE + 1; bkIdx++) {
    // Loading into shared memory and do boundary checking
    if (row < M && (bkIdx * BLOCKSIZE + threadIdx.x) < K) {
      As[threadIdx.y][threadIdx.x] = A[row * K + (bkIdx * BLOCKSIZE + threadIdx.x)];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }
    if (col < N && (bkIdx * BLOCKSIZE + threadIdx.y) < K) {
      Bs[threadIdx.y][threadIdx.x] = B[(bkIdx * BLOCKSIZE + threadIdx.y) * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    for (unsigned int k = 0; k < BLOCKSIZE; k++) {
        result += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = result;
  }
} 
