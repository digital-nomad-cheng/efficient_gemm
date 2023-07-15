#include <cuda_runtime.h>
#include <stdio.h>
#define BLOCKSIZE 16
__global__ void shared_mem_coalesing_gemm_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE][BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;
  const uint row = cRow * BLOCKSIZE + threadRow;
  const uint col = cCol * BLOCKSIZE + threadCol;

  float result = 0.0f;
  for (unsigned int bkIdx = 0; bkIdx < (K - 1) / BLOCKSIZE + 1; bkIdx++) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    if (row < M && (bkIdx * BLOCKSIZE + threadCol) < K) { 
      As[threadRow][threadCol] = A[row * K + (bkIdx * BLOCKSIZE + threadCol)];
    } else {
      As[threadRow][threadCol] = 0.0f;
    }
    if (col < N && (BLOCKSIZE * bkIdx + threadRow) < K) {
      Bs[threadRow][threadCol] = B[(bkIdx * BLOCKSIZE + threadRow) * N + col];
    } else {
      Bs[threadRow][threadCol] = 0.0f;
    }

    // block threads in this block until cache is fully populated
    __syncthreads();

    // execute the dotproduct on the currently cached block
    
    for (unsigned int k = 0; k < BLOCKSIZE; k++) {
    result += As[threadRow][k] *
            Bs[k][threadCol];
    }
    
    __syncthreads();

  }

  if (row < M && col < N) {
    C[row * N + col] = result;
  }

} 
