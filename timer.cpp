#include "timer.hpp"

Timer::Timer() {
  _cStart = std::chrono::high_resolution_clock::now();
  _cStop = std::chrono::high_resolution_clock::now();
  cudaEventCreate(_gStart);
  cudaEventCreate(_gStop);
  _timeElapsed = 0;
}

Timer::~Timer() {
  cudaFree(_gStart);
  cudaFree(_gStop);
}

Timer::start_cpu() {
  _cStart = std::chrono::high_resolution_clock::now();
}

Timer::_cStop_cpu() {
  _cStop = std::chrono::high_resolution_clock::now();
}

Timer::start_gpu() {
  cudaEventRecord(_gStart);
}

Timer::stop_gpu() {
  cudaEventRecord(_gStop);
}

Timer::duration_gpu(std::string msg) {
  CHECK_CUDA_ERROR(cudaEventSynchronize(_gStart));
  CHECK_CUDA_ERROR(cudaEventSynchronize(_gStop));
  cudaEventElapsedTime(&_timeElapsed, _gStart, _gStop);
  
  std::cout << msg << " uses: " << _timeElapsed << "ms" << std::endl;
}
