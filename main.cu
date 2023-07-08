#include "timer.h"
#include "common.h"
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

void verify_result(const float *C_cpu, const float *C_gpu, const uint M, const uint N) {
    for (unsigned int row = 0; row < M; ++row) {
            for (unsigned int col = 0; col < N; ++col) {
                float diff = (C_cpu[row*N + col] - C_gpu[row*N + col])/C_cpu[row*N + col];
                const float tolerance = 0.00001;
                if(diff > tolerance || diff < -tolerance || isnan(diff)) {
                    printf("Mismatch at row %u, col %u (CPU result = %e, GPU result = %e)\n", row, col, C_cpu[row*N + col], C_gpu[row*N + col]);
                    exit(0);
                }
            }
        }
}

void mm_cpu(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
    unsigned int TILE_DIM = 32;
    #define MIN(x, y) (((x) < (y))?(x):(y))
    for(unsigned int rowTile = 0; rowTile < (M + TILE_DIM - 1)/TILE_DIM; ++rowTile) {
        for(unsigned int colTile = 0; colTile < (N + TILE_DIM - 1)/TILE_DIM; ++colTile) {
            for(unsigned int iTile = 0; iTile < (K + TILE_DIM - 1)/TILE_DIM; ++iTile) {
                for (unsigned int row = rowTile*TILE_DIM; row < MIN((rowTile + 1)*TILE_DIM, M); ++row) {
                    for (unsigned int col = colTile*TILE_DIM; col < MIN((colTile + 1)*TILE_DIM, N); ++col) {
                        float sum = 0.0f;
                        for(unsigned int i = iTile*TILE_DIM; i < MIN((iTile + 1)*TILE_DIM, K); ++i) {
                            sum += A[row*K + i]*B[i*N + col];
                        }
                        if(iTile == 0) {
                            C[row*N + col] = sum;
                        } else {
                            C[row*N + col] += sum;
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char**argv) {

    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int M = (argc > 1)?(atoi(argv[1])):1000;
    unsigned int N = (argc > 2)?(atoi(argv[2])):1200;
    unsigned int K = (argc > 3)?(atoi(argv[3])):1100;
    unsigned int verify = (argc > 4)?(atoi(argv[4])):1;
    float* A = (float*) malloc(M*K*sizeof(float));
    float* B = (float*) malloc(K*N*sizeof(float));
    float* C_cpu = (float*) malloc(M*N*sizeof(float));
    float* C_gpu = (float*) malloc(M*N*sizeof(float));
    for (unsigned int row = 0; row < M; ++row) {
        for (unsigned int col = 0; col < K; ++col) {
            A[row*K + col] = 1.0*rand()/RAND_MAX;
        }
    }
    for (unsigned int row = 0; row < K; ++row) {
        for (unsigned int col = 0; col < N; ++col) {
            B[row*N + col] = 1.0*rand()/RAND_MAX;
        }
    }

    // Compute on CPU
    if(verify) {
        startTime(&timer);
        mm_cpu(A, B, C_cpu, M, N, K);
        stopTime(&timer);
        printElapsedTime(timer, "CPU time", CYAN);
    }

    // Allocate GPU memory
    startTime(&timer);
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, M*K*sizeof(float));
    cudaMalloc((void**) &B_d, K*N*sizeof(float));
    cudaMalloc((void**) &C_d, M*N*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(A_d, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Compute on GPU
    startTime(&timer);
    mm_gpu_navie(A_d, B_d, C_d, M, N, K);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Navie kernel time", GREEN);
    
    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(C_gpu, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    if (verify) {
        verify_result(C_cpu, C_gpu, M, N);
    }

    // Compute on GPU
    startTime(&timer);
    mm_gpu_coalesing(A_d, B_d, C_d, M, N, K);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Coalesing kernel time", GREEN);
    
    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(C_gpu, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    if (verify) {
        verify_result(C_cpu, C_gpu, M, N);
    }

    // Free GPU memory
    startTime(&timer);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

    // Free memory
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);

    return 0;

}

