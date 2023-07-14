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
using namespace std;
using std::cout;
using std::endl;
using std::printf;
using std::sqrt;
using Clock = std::chrono::high_resolution_clock;
void print_s8x8(int8x8_t d) {
  cout << (int)vget_lane_s8(d, 7) << ", " << (int)vget_lane_s8(d, 6) << ", "
       << (int)vget_lane_s8(d, 5) << ", " << (int)vget_lane_s8(d, 4) << ", "
       << (int)vget_lane_s8(d, 3) << ", " << (int)vget_lane_s8(d, 2) << ", "
       << (int)vget_lane_s8(d, 1) << ", " << (int)vget_lane_s8(d, 0) << endl;
}

void print_u8x8(uint8x8_t d) {
  cout << (int)vget_lane_u8(d, 7) << ", " << (int)vget_lane_u8(d, 6) << ", "
       << (int)vget_lane_u8(d, 5) << ", " << (int)vget_lane_u8(d, 4) << ", "
       << (int)vget_lane_u8(d, 3) << ", " << (int)vget_lane_u8(d, 2) << ", "
       << (int)vget_lane_u8(d, 1) << ", " << (int)vget_lane_u8(d, 0) << endl;
}

void print_s8x16(int8x16_t d) {
  cout << (int)vgetq_lane_s8(d, 0) << ", " << (int)vgetq_lane_s8(d, 1) << ", "
       << (int)vgetq_lane_s8(d, 2) << ", " << (int)vgetq_lane_s8(d, 3) << ", "
       << (int)vgetq_lane_s8(d, 4) << ", " << (int)vgetq_lane_s8(d, 5) << ", "
       << (int)vgetq_lane_s8(d, 6) << ", " << (int)vgetq_lane_s8(d, 7) << ", "
       << (int)vgetq_lane_s8(d, 8) << ", " << (int)vgetq_lane_s8(d, 9) << ", "
       << (int)vgetq_lane_s8(d, 10) << ", " << (int)vgetq_lane_s8(d, 11) << ", "
       << (int)vgetq_lane_s8(d, 12) << ", " << (int)vgetq_lane_s8(d, 13) << ", "
       << (int)vgetq_lane_s8(d, 14) << ", " << (int)vgetq_lane_s8(d, 15)
       << endl;
}

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(101);
  std::uniform_int_distribution<> dis(1, 127);
  int total = 128;
  int8_t* d = new int8_t[total];
  for (int i = 0; i < total; ++i) {
    d[i] = dis(gen);
  }
  for (int i = 0; i < 16; ++i) {
    d[i] = i;
  }
  int8x16_t data;
  data = vld1q_s8(d);
  /*auto h = vget_high_s8(data);
  auto l = vget_low_s8(data);
  auto data2 = vzipq_s8(h, l);*/
  print_s8x16(data);
}
