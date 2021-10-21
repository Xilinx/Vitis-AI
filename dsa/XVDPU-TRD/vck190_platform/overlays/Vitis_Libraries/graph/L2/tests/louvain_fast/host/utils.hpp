/*
 * Copyright 2019 Xilinx, Inc.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef XF_GRAPH_UTILS_H
#define XF_GRAPH_UTILS_H

#include "ap_int.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <vector>
#include <limits>

#define XCL_BANK(n) (((unsigned int)(n)) | XCL_MEM_TOPOLOGY)
#define XCL_BANK0 XCL_BANK(0)
#define XCL_BANK1 XCL_BANK(1)
#define XCL_BANK2 XCL_BANK(2)
#define XCL_BANK3 XCL_BANK(3)
#define XCL_BANK4 XCL_BANK(4)
#define XCL_BANK5 XCL_BANK(5)
#define XCL_BANK6 XCL_BANK(6)
#define XCL_BANK7 XCL_BANK(7)
#define XCL_BANK8 XCL_BANK(8)
#define XCL_BANK9 XCL_BANK(9)
#define XCL_BANK10 XCL_BANK(10)
#define XCL_BANK11 XCL_BANK(11)
#define XCL_BANK12 XCL_BANK(12)
#define XCL_BANK13 XCL_BANK(13)
#define XCL_BANK14 XCL_BANK(14)
#define XCL_BANK15 XCL_BANK(15)
#define XCL_BANK16 XCL_BANK(16)
#define XCL_BANK17 XCL_BANK(17)
#define XCL_BANK18 XCL_BANK(18)
#define XCL_BANK19 XCL_BANK(19)
#define XCL_BANK20 XCL_BANK(20)
#define XCL_BANK21 XCL_BANK(21)
#define XCL_BANK22 XCL_BANK(22)
#define XCL_BANK23 XCL_BANK(23)
#define XCL_BANK24 XCL_BANK(24)
#define XCL_BANK25 XCL_BANK(25)
#define XCL_BANK26 XCL_BANK(26)
#define XCL_BANK27 XCL_BANK(27)
#define XCL_BANK28 XCL_BANK(28)
#define XCL_BANK29 XCL_BANK(29)
#define XCL_BANK30 XCL_BANK(30)
#define XCL_BANK31 XCL_BANK(31)

template <typename T>
T* aligned_alloc(std::size_t num) {
  void* ptr = nullptr;
  if (posix_memalign(&ptr, 4096, num * sizeof(T))) {
    throw std::bad_alloc();
  }
  return reinterpret_cast<T*>(ptr);
}
// Compute time difference
unsigned long diff(const struct timeval* newTime, const struct timeval* oldTime) {
      return (newTime->tv_sec - oldTime->tv_sec) * 1000000 + (newTime->tv_usec - oldTime->tv_usec);
}
#endif
