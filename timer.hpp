#ifndef __TIMER_HPP___
#define __TIMER_HPP___

#include <chrono>
#include <ratio>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "util.hpp"

class Timer {
public:
  using s = std::ratio<1, 1>;
  using ms = std::ratio<1, 1000>;
  using us = std::ratio<1, 1000000>;
  using ns = std::ratio<1, 1000000000>;

public:
  Timer();
  ~Timer();

public:
  void start_cpu();
  void stop_cpu();
  void start_gpu();
  void stop_gpu();
  
  template<typename span>
  void duration_cpu(std::string msg);

  void duration_gpu(std::string msg);

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> _cStart;
  std::chrono::time_point<std::chrono::high_resolution_clock> _cStop;
  cudaEvent_t _gStart;
  cudaEvent_t _gStop;
  float _timeElapsed;
};

template<typename span>
void Timer::duration_cpu(std::string msg) {
  std::string str;
  if (std::is_same<s, span>::value) { str = "s"; }
  if (std::is_same<ms, span>::value) { str = "ms"; }
  if (std::is_same<us, span>::value) { str = "us"; }
  if (std::is_same<ns, span>::value) { str = "ns"; }
  std::chrono::duration<double, span> time = _cStop - _cStart;
  std::cout << msg << " uses: " << time.count() << str << std::endl; 
}

#endif // __TIMER_HPP__
