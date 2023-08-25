#include <cuda_runtime.h>

__global__ void navie_gemm_kernel(const float * A, const float * B, float * C, 
  unsigned M, unsigned N, unsigned K) {
  
  unsigned int m = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int n = threadIdx.y + blockDim.y * blockIdx.y;

  if (m >= M || n >= N)
    return;
  float result = 0.0f;
  for (unsigned k = 0; k < K; ++k) {
    result += A[m*K + k] * B[k * N + n];
  }
  
  C[m * N + n] = result;
}
