#include "common.h"
#include "util.hpp"
#include "timer.hpp"
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
        // std::exit(EXIT_FAILURE);
        std::exit(err);
    }
}

void get_cuda_info() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
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
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    get_cuda_info();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int M = (argc > 1)?(atoi(argv[1])):1024;
    unsigned int N = (argc > 2)?(atoi(argv[2])):1024;
    unsigned int K = (argc > 3)?(atoi(argv[3])):1024;
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
        // timer.start_gpu();
        timer.start_cpu();
        mm_cpu(A, B, C_cpu, M, N, K);
        // timer.stop_gpu();
        timer.stop_cpu();
        // printElapsedTime(timer, "CPU time", CYAN);
        timer.duration_cpu<Timer::ms>("CPU time");
    }

    // Allocate GPU memory
    // timer.start_gpu();
    timer.start_gpu();
    float *A_d, *B_d, *C_d;
    CHECK_CUDA_ERROR(cudaMalloc((void**) &A_d, M*K*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**) &B_d, K*N*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**) &C_d, M*N*sizeof(float)));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // timer.stop_gpu();
    timer.stop_gpu();
    // printElapsedTime(timer, "Allocation time");
    timer.duration_gpu("Allocation time");

    // Copy data to GPU
    timer.start_gpu();
    CHECK_CUDA_ERROR(cudaMemcpy(A_d, A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(B_d, B, K*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    timer.stop_gpu();
    // printElapsedTime(timer, "Copy to GPU time");
    timer.duration_gpu("Copy to GPU time");
    
    // Compute on GPU
    timer.start_gpu();
    mm_gpu_navie(A_d, B_d, C_d, M, N, K);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    timer.stop_gpu();
    // printElapsedTime(timer, "Navie kernel time", GREEN);
    timer.duration_gpu("Navie kernel time");

    // Copy data from GPU
    timer.start_gpu();
    CHECK_CUDA_ERROR(cudaMemcpy(C_gpu, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    timer.stop_gpu();
    // printElapsedTime(timer, "Copy from GPU time");
    timer.duration_gpu("Copy from GPU time");

    if (verify) {
        verify_result(C_cpu, C_gpu, M, N);
    }

    // Compute on GPU
    CHECK_CUDA_ERROR(cudaMemset(C_d, 0.0, M*N*sizeof(float)));
    timer.start_gpu();
    mm_gpu_coalesing(A_d, B_d, C_d, M, N, K);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    timer.stop_gpu();
    // printElapsedTime(timer, "Coalesing kernel time", GREEN);
    timer.duration_gpu("Coalesing kernel time");

    // Copy data from GPU
    timer.start_gpu();
    CHECK_CUDA_ERROR(cudaMemcpy(C_gpu, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    timer.stop_gpu();
    // printElapsedTime(timer, "Copy from GPU time");
    timer.duration_gpu("Copy from GPU time");

    if (verify) {
        verify_result(C_cpu, C_gpu, M, N);
    }

    // Compute on GPU
    CHECK_CUDA_ERROR(cudaMemset(C_d, 0.0, M*N*sizeof(float)));
    timer.start_gpu();
    mm_gpu_shared_tiling(A_d, B_d, C_d, M, N, K);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    timer.stop_gpu();
    // printElapsedTime(timer, "Shared tiling kernel time", GREEN);
    timer.duration_gpu("Shared tiling kernl time");

    // Copy data from GPU
    timer.start_gpu();
    cudaMemcpy(C_gpu, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    timer.stop_gpu();
    // printElapsedTime(timer, "Copy from GPU time");
    timer.duration_gpu("Copy from GPU time");

    if (verify) {
        verify_result(C_cpu, C_gpu, M, N);
    }

    // Compute on GPU
    CHECK_CUDA_ERROR(cudaMemset(C_d, 0.0, M*N*sizeof(float)));
    timer.start_gpu();
    mm_gpu_shared_coalesing(A_d, B_d, C_d, M, N, K);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    timer.stop_gpu();
    // printElapsedTime(timer, "Shared coalesing kernel time", GREEN);
    timer.duration_gpu("Shared coalesing kernel time");

    // Copy data from GPU
    timer.start_gpu();
    cudaMemcpy(C_gpu, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    timer.stop_gpu();
    // printElapsedTime(timer, "Copy from GPU time");
    timer.duration_gpu("Copy from GPU time");

    if (verify) {
        verify_result(C_cpu, C_gpu, M, N);
    }
    
    // Compute on GPU
    CHECK_CUDA_ERROR(cudaMemset(C_d, 0.0, M*N*sizeof(float)));
    timer.start_gpu();
    mm_gpu_thread_tiling(A_d, B_d, C_d, M, N, K);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    timer.stop_gpu();
    // printElapsedTime(timer, "Thread tiling kernel time", GREEN);
    timer.duration_gpu("Thread tiling kernel time");

    // Copy data from GPU
    timer.start_gpu();
    cudaMemcpy(C_gpu, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    timer.stop_gpu();
    // printElapsedTime(timer, "Copy from GPU time");
    timer.duration_gpu("Copy from GPU time");

    if (verify) {
        verify_result(C_cpu, C_gpu, M, N);
    }

    // Free GPU memory
    timer.start_gpu();
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaDeviceSynchronize();
    timer.stop_gpu();
    // printElapsedTime(timer, "Deallocation time");
    timer.duration_gpu("Deallocation time on GPU");
    // Free memory
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);

    return 0;

}

