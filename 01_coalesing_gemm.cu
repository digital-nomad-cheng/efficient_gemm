#include <cuda_runtime.h>

__global__ void coalesing_gemm_kernel(const float *A, const float *B, float *C, unsigned M, unsigned N, unsigned K, unsigned BLOCKSIZE)
{
  const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (x < M && y < N) {
    float result = 0.0;
    for (int i = 0; i < K; ++i) {
      result += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = result;
  }
}
