#include "timer.hpp"

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

Timer::Timer() {
  _cStart = std::chrono::high_resolution_clock::now();
  _cStop = std::chrono::high_resolution_clock::now();
  cudaEventCreate(&_gStart);
  cudaEventCreate(&_gStop);
  _timeElapsed = 0;
}

Timer::~Timer() {
  cudaFree(_gStart);
  cudaFree(_gStop);
}

void Timer::start_cpu() {
  _cStart = std::chrono::high_resolution_clock::now();
}

void Timer::stop_cpu() {
  _cStop = std::chrono::high_resolution_clock::now();
}

void Timer::start_gpu() {
  cudaEventRecord(_gStart);
}

void Timer::stop_gpu() {
  cudaEventRecord(_gStop);
}

void Timer::duration_gpu(std::string msg) {
  CHECK_CUDA_ERROR(cudaEventSynchronize(_gStart));
  CHECK_CUDA_ERROR(cudaEventSynchronize(_gStop));
  cudaEventElapsedTime(&_timeElapsed, _gStart, _gStop);
  
  std::cout << msg << " uses: " << _timeElapsed << "ms" << std::endl;
}
