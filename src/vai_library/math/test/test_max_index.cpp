/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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
#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>

#include "vitis/ai/max_index.hpp"
// #include "../include/vitis/max_index.hpp"
using std::cout;
using std::endl;
using std::printf;
using std::sqrt;
using Clock = std::chrono::high_resolution_clock;
extern
    // static
    __attribute__((noinline)) void
    max_index_c_local(int8_t *d, int c, int g, uint8_t *results) {
  // auto it = std::max_element(d, d+c);
  //__TIC__(max_c)
  for (int i = 0; i < g; ++i) {
    auto it = std::max_element(d, d + c);
    results[i] = it - d;
    // cout << (int)results[i] << endl;
    d += c;
  }
  //__TOC__(max_c)
  // cout << (int)(it-d) << endl;
}
#define __TIC__(tag) auto __##tag##_start_time = Clock::now();

#define __TOC__(tag)                                             \
  auto __##tag##_end_time = Clock::now();                        \
  cout << #tag << " : "                                          \
       << std::chrono::duration_cast<std::chrono::microseconds>( \
              __##tag##_end_time - __##tag##_start_time)         \
              .count()                                           \
       << endl;
// void max_index_neon_c8(int8_t* d, int g, uint8_t* results);
void max_index_neon_c8(int8_t *d, int g, uint8_t *results) {
  // Internal result register
  uint8x8_t d0;
  // Initilize index registers
  uint8x8_t d1 = vcreate_u8(0x0706050403020100);
  uint8x8_t d2 = vdup_n_u8(2);
  uint8x8_t d3 = vdup_n_u8(3);
  uint8x8_t d13 = vcreate_u8(0x0000000006040200);
  uint8x8_t d14 = vcreate_u8(0x0400040004000400);
  uint8x8_t d15 = vcreate_u8(0x0000000000000000);
  uint8x8_t d16 = vcreate_u8(0x0b0a090803020100);
  int j = 0;

  for (int i = 0; i < g / 4; ++i) {
    // Use d4 to d7 to store data
    int8x8x4_t d4x4 = vld4_s8(d);

    // Compare first two columns (column 0 and 1)
    d0 = vclt_s8(d4x4.val[0], d4x4.val[1]);
    d0 = vshr_n_u8(d0, 7);

    // d8 to store table index
    uint8x8_t d8 = vshl_n_u8(d0, 3);
    d8 = vadd_u8(d1, d8);

    // d9 to store max value of column 0 and 1
    int8x8x2_t tab_d{d4x4.val[0], d4x4.val[1]};
    int8x8_t d9 = vtbl2_s8(tab_d, vreinterpret_s8_u8(d8));

    // Compare column 2
    d8 = vclt_s8(d9, d4x4.val[2]);
    d8 = vshr_n_u8(d8, 7);
    d8 = vshl_n_u8(d8, 3);
    d8 = vadd_u8(d1, d8);
    uint8x8x2_t tab_i{d0, d2};
    d0 = vtbl2_u8(tab_i, d8);
    tab_d.val[0] = d9;
    tab_d.val[1] = d4x4.val[2];
    d9 = vtbl2_s8(tab_d, vreinterpret_s8_u8(d8));

    // Compare column 3
    d8 = vclt_s8(d9, d4x4.val[3]);
    d8 = vshr_n_u8(d8, 7);
    d8 = vshl_n_u8(d8, 3);
    d8 = vadd_u8(d1, d8);
    tab_i.val[0] = d0;
    tab_i.val[1] = d3;
    d0 = vtbl2_u8(tab_i, d8);
    tab_d.val[0] = d9;
    tab_d.val[1] = d4x4.val[3];
    d9 = vtbl2_s8(tab_d, vreinterpret_s8_u8(d8));

    // Final compare, only lower 4 bytes are valid
    int8x8_t d10 = vrev16_s8(d9);
    uint8x8_t d11 = vclt_s8(d9, d10);
    d11 = vshr_n_u8(d11, 7);
    uint8x8_t d12 = vadd_u8(vtbl1_u8(d11, d13), d13);
    // print_u8x8(d12);
    d0 = vadd_u8(d0, d14);
    d0 = vtbl1_u8(d0, d12);

    if (j++ == 0) {
      d15 = d0;
    } else {
      uint8x8x2_t temp{d15, d0};
      d15 = vtbl2_u8(temp, d16);
      vst1_u8(results, d15);
      results += 8;
      j = 0;
    }

    // print_u8x8(d0);
    // Next 4 groups
    d += 32;
  }
}

int test_main(const int width, const int height, const int c) {
  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(101);
  std::uniform_int_distribution<> dis(1, 127);
  int g = width * height;
  int total = g * c;
  int8_t *d = new int8_t[total];
  for (int i = 0; i < total; ++i) {
    d[i] = dis(gen);
  }
  for (int i = 0; i < 16; ++i) {
    d[i] = i;
  }
  uint8_t *results_c = new uint8_t[g];
  uint8_t *results_neon = new uint8_t[g];
  __TIC__(c)
  max_index_c_local(d, c, g, results_c);
  __TOC__(c)
  __TIC__(neon)
  // auto results_neon = vitis::ai::max_index(d, width, height, c);
  // uint8_t* results_neon = new uint8_t[g];
  // max_index_neon_c8(d, g, results_neon);
  vitis::ai::max_index_void(d, width, height, c, results_neon);
  __TOC__(neon)
  __TIC__(neon1)
  auto rr = vitis::ai::max_index(d, width, height, c);  (void)rr;
  __TOC__(neon1)
  auto x = std::memcmp(&results_neon[0], &results_c[0], g);
  if (x != 0) {
    std::cerr << "result is not correct, please fix bugs. size=" << g
              << std::endl;
    for (auto i = 0; i < g; ++i) {
      if (results_neon[i] != results_c[i]) {
        std::cerr << "error at [" << i << "], " << (int)results_c[i]
                  << " is expected, but " << (int)results_neon[i] << " is given"
                  << std::endl;
        std::cerr << "input data: ";
        for (auto n = 0; n < c; ++n) {
          std::cerr << ' ' << (int)d[c * i + n];
        }
        std::cerr << std::endl;
        break;
      }
    }
    delete[] results_c;
    delete[] d;
    abort();
  }
  delete[] results_c;
  delete[] d;
  return x == 0 ? 0 : 1;
}

int main(int argc, char *argv[]) {
  bool ok = true;
  // the baseline for performance
  ok = test_main(512, 320, 16) && ok;
  // this will triggered a bug, 8x8x2
  ok = test_main(521, 19 * 16 + 15, 16) && ok;
  // the baseline for performance
  // 2503 us -> 1173 us
  ok = test_main(512, 320, 8) && ok;
  // this will triggered a bug, 8x8x2
  ok = test_main(521, 19 * 16 + 15, 8) && ok;
  // the baseline for performance
  // 4203 us -> 2748 us
  ok = test_main(512, 320 * 4, 4) && ok;
  return ok ? 0 : 1;
}
