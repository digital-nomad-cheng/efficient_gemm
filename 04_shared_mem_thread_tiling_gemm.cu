#include <cuda_runtime.h>
#include <stdio.h>

#define BM 128 // acutal number of threads we have for each block
              // we want these to be large as possible to have more warps
              // which means more scheduling opportunity
#define BN 32 // register coarsensing in N dimension
              // limited by the number of available registers
#define BK 4  // use BK * BN = BM to maintain load balance 
// register tiling size in M dimension, each M is reused 16 times within one thread

__global__ void shared_mem_thread_tiling_gemm_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K)
{
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint tidx = threadIdx.x;

  // allocate space for the current blocktile in SMEM
  __shared__ float Bs[BK][BN];

  // allocate thread-local cache for results in registerfile
  float threadResults[BN] = {0.0};
  float regA[BK] = {0.0};
  
  // thread index for incooperating loading B into smem
  uint tidxRowB = tidx / BN; // 0-3 here
  uint tidxColB = tidx % BN; // 0-15 here
  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // each thread load BK A elements into registers
    for (uint idx = 0; idx < BK; idx++) {
      regA[idx] = A[(cRow * BM + tidx) * K + bkIdx + idx];
    }
    // each thread load one B element into shared memory
    Bs[tidxRowB][tidxColB] = B[(bkIdx + tidxRowB) * N + cCol * BN + tidxColB]; 
    __syncthreads();
     
    // each perform BK steps inner product
    for (uint kIdx = 0; kIdx < BK; ++kIdx) {
      // each thread need to perform BN calculations/thread coarsensing
      // for each step, A is stored in register file
      // B is stored in shared memory
      for (uint nIdx = 0; nIdx < BN; ++nIdx) {
        threadResults[nIdx] += regA[kIdx] * Bs[kIdx][nIdx];
      }
    }
    __syncthreads();
  }

  // write out the results, each thread need to write back BN values in register
  for (uint nIdx = 0; nIdx < BN; ++nIdx) {
    C[(cRow * BM + tidx) * N + cCol * BN + nIdx] = threadResults[nIdx];
  }
} 
