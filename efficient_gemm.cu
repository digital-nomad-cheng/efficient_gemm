
#include "common.h"
#include <iostream>

#define TILE_SIZE 32
#define M_BTILE_DIM 128
#define N_BTILE_DIM 128
#define K_BTILE_DIM 8
#define NUM_THREADS 256 // maximum num threads depend on number of registers
#define NUM_THREADS_PER_WARP 32

#define M_WTILE_DIM 64
#define N_WTILE_DIM 32

#define TTILE_PER_THREAD_Y 2
#define TTILE_PER_THREAD_X 2
#define M_TTILE_DIM 4
#define N_TTILE_DIM 4


__global__ void mm_tiled_kernel(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    // Declare input tiles in shared memory

    // TODO
    __shared__ float A_s[M_BTILE_DIM][K_BTILE_DIM];
    __shared__ float B_s[K_BTILE_DIM][N_BTILE_DIM];
    // Declare and initialize output tiles in registers
    float sums[M_BTILE_DIM * N_BTILE_DIM / NUM_THREADS];
    // TODO initializations
    int warpIdx = threadIdx.x / NUM_THREADS_PER_WARP;  
    int laneIdx = threadIdx.x % NUM_THREADS_PER_WARP;
    int A_ratio = N_BTILE_DIM / N_WTILE_DIM; // 4
    int B_ratio = M_BTILE_DIM / M_WTILE_DIM; // 2
    
    // TODO
    float sum = 0.0f; 
    // Iterate over input tiles
    for(unsigned int tile = 0; tile < (K + K_BTILE_DIM - 1)/K_BTILE_DIM; ++tile) {

        // Load A tile to shared memory
        // TODO
        unsigned int rowsPerSubTile = NUM_THREADS / K_BTILE_DIM;    
        // unsigned int blockStartRow = blockIdx.y * M_BTILE_DIM; // ?
        // blockStartRow index blockIdx.x / num_K_blocks
        unsigned int blockStartRow = blockIdx.x / ((K + K_BTILE_DIM - 1)/ K_BTILE_DIM) * M_BTILE_DIM;
        #pragma unroll
        for (unsigned int subTile = 0; subTile < M_BTILE_DIM/rowsPerSubTile; ++subTile) {
            unsigned int threadRow = subTile*rowsPerSubTile + threadIdx.x/K_BTILE_DIM;
            unsigned int threadCol = threadIdx.x % K_BTILE_DIM;
            unsigned int row = blockStartRow + threadRow;
            unsigned int col = tile*K_BTILE_DIM + threadCol;
            if(row < M && col < K) {
                A_s[threadRow][threadCol] = A[row*K + col];
            } else {
                A_s[threadRow][threadCol] = 0.0f;
            }
        }

        // Load B tile to shared memory
        // TODO
        unsigned int colsPerSubTile = NUM_THREADS / K_BTILE_DIM;
        // unsigned int blockStartCol = blockIdx.x * N_BTILE_DIM; // ?
        unsigned int blockStartCol = blockIdx.x % ((K + K_BTILE_DIM - 1) / K_BTILE_DIM) * N_BTILE_DIM;
        #pragma unroll
        for (unsigned int subTile = 0; subTile < N_BTILE_DIM/colsPerSubTile; ++subTile) {
            unsigned int threadCol = subTile*colsPerSubTile + threadIdx.x/K_BTILE_DIM;
            unsigned int threadRow = threadIdx.x % K_BTILE_DIM;
            unsigned int col = blockStartCol + threadCol;
            unsigned int row = tile*K_BTILE_DIM + threadRow;
            if(row < K && col < N) {
                B_s[threadRow][threadCol] = B[row*N + col];
            } else {
                B_s[threadRow][threadCol] = 0.0f;
            }
        }

        __syncthreads();

        // Compute with shared memory tiles
        #pragma unroll
        for(unsigned int k = 0; k < K_BTILE_DIM; ++k) { // Iterate over input strips

            // Load A strip to registers

            // TODO
            const unsigned int numSubStripsA = M_BTILE_DIM/M_WTILE_DIM;
            float A_wr[numSubStripsA];
            unsigned int warpStartRow = warpIdx / A_ratio * M_WTILE_DIM;
            #pragma unroll
            for (unsigned int subStrip = 0; subStrip < numSubStripsA; ++subStrip) {
                A_wr[subStrip] = A_s[warpStartRow + subStrip*NUM_THREADS_PER_WARP + laneIdx][k];
            }
            // Load B strip to registers

            // TODO
            const unsigned int numSubStripsB = N_BTILE_DIM/N_WTILE_DIM; 
            float B_wr[numSubStripsB];
            unsigned int warpStartCol = warpIdx % A_ratio * N_WTILE_DIM;
            #pragma unroll
            for (unsigned int subStrip = 0; subStrip < numSubStripsB; ++subStrip) {
                B_wr[subStrip] = B_s[k][warpStartCol + subStrip*NUM_THREADS_PER_WARP + laneIdx];
            }
            
            // Compute with register strips

            // TODO
            #pragma unroll
            for(unsigned int threadTileY = 0; threadTileY < TTILE_PER_THREAD_Y; ++threadTileY) {
                // Shuffle A tile to thread registers
                unsigned int threadStartRow = threadTileY * M_WTILE_DIM / TTILE_PER_THREAD_Y;
                float A_r[M_TTILE_DIM];
                #pragma unroll
                for(unsigned int threadRow = 0; threadRow < M_TTILE_DIM; ++threadRow) {
                    unsigned int warpRow = threadStartRow + threadRow;
                    unsigned int subStripA = warpRow / NUM_THREADS_PER_WARP;
                    unsigned int srcThreadA = warpRow % NUM_THREADS_PER_WARP;
                    A_r[threadRow] = __shfl_sync(0xffffffff, A_wr[subStripA], srcThreadA);
                }
                #pragma unroll
                for(unsigned int threadTileX = 0; threadTileX < TTILE_PER_THREAD_X; ++threadTileX) {
                    // Shuffle B tile to thread registers
                    unsigned int threadStartCol = threadTileX * N_WTILE_DIM / TTILE_PER_THREAD_X;
                    float B_r[N_TTILE_DIM];
                    #pragma unroll
                    for(unsigned int threadCol = 0; threadCol < N_TTILE_DIM; ++threadCol) {
                        unsigned int warpCol = threadStartRow + threadCol;
                        unsigned int subStripB = warpCol / NUM_THREADS_PER_WARP;
                        unsigned int srcThreadB = warpCol % NUM_THREADS_PER_WARP;
                        B_r[threadCol] = __shfl_sync(0xffffffff, B_wr[subStripB], srcThreadB);
                    }

                    // Compute with register tiles
                    #pragma unroll
                    for(unsigned int threadRow = 0; threadRow < M_TTILE_DIM; ++threadRow) {
                        #pragma unroll
                        for(unsigned int threadCol = 0; threadCol < N_TTILE_DIM; ++threadCol) {
                            sum[warpStartRow + warpRow] += A_r[threadRow] * B_r[threadCol]; // ?
                        }
                    }
                }
            }

        }

        __syncthreads();

    }

    // Write output tiles to global memory

    // TODO
    // unsigned int row = blockIdx.x / ((K + K_BTILE_DIM - 1)/ K_BTILE_DIM) * M_BTILE_DIM + threadIdx.x / 
    // unsigned int col = blockIdx.x % ((K + K_BTILE_DIM - 1)/ K_BTILE_DIM) * N_BTILE_DIM + threadIdx.x % 

}

/*
__global__ __launch_bounds__(1024) void navie_gemm_kernel(const float* A, const float* B, float* C, const unsigned int M, const unsigned int N, const unsigned int K) {
    unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    float sum = 0.0f;
    if (col < N && row < M) {
        for (unsigned int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


__global__ void tiled_gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;

  for (int tile = 0; tile < K / TILE_SIZE; ++tile) {
    // Load tiles into shared memory
    As[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];

    __syncthreads();

    // compute
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    
    __syncthreads();
  }

  C[row * N + col] = sum;
}
*/

void mm_gpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

    // Call matrix multiply kernel
    // dim3 block_dim{TILE_SIZE, TILE_SIZE, 1};
    // std::cout << M << N << K << std::endl;
    // dim3 grid_dim{(N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1};
    // printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
    // printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    // mm_tiled_kernel<<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    
    // TODO
    dim3 block_dim{NUM_THREADS};
    int max_size = M > N ? M : N;
    dim3 grid_dim{(K + K_BTILE_DIM - 1) / K_BTILE_DIM};
    mm_tiled_kernel<<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}

