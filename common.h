
void mm_gpu_navie(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K);
void mm_gpu_coalesing(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K);

__global__ void navie_gemm_kernel(const float * A, const float * B, float * C, 
  unsigned M, unsigned N, unsigned K);
__global__ void coalesing_gemm_kernel(const float * A, const float * B, float * C, 
  unsigned M, unsigned N, unsigned K, unsigned BLOCKSIZE);

