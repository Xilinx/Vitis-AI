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
#include "vitis/ai/max_index.hpp"

#include <algorithm>
#if ENABLE_NEON
#include <arm_neon.h>
#endif
#include <stdint.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::printf;
using std::sqrt;
using Clock = std::chrono::high_resolution_clock;

#define UNUSED(expr) \
  do {               \
    (void)(expr);    \
  } while (0)

#define __TIC__(tag) auto __##tag##_start_time = Clock::now();

#define __TOC__(tag)                                             \
  auto __##tag##_end_time = Clock::now();                        \
  cout << #tag << " : "                                          \
       << std::chrono::duration_cast<std::chrono::microseconds>( \
              __##tag##_end_time - __##tag##_start_time)         \
              .count()                                           \
       << endl;

namespace vitis {
namespace ai {
void max_index_c(int8_t *d, int c, int g, uint8_t *results) {
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
}  // namespace ai
}  // namespace vitis
#ifdef ENABLE_NEON
static void print_s8x8(int8x8_t d) {
  UNUSED(print_s8x8);  // unused for debugging
  cout << (int)vget_lane_s8(d, 7) << ", " << (int)vget_lane_s8(d, 6) << ", "
       << (int)vget_lane_s8(d, 5) << ", " << (int)vget_lane_s8(d, 4) << ", "
       << (int)vget_lane_s8(d, 3) << ", " << (int)vget_lane_s8(d, 2) << ", "
       << (int)vget_lane_s8(d, 1) << ", " << (int)vget_lane_s8(d, 0) << endl;
}

static void print_u8x8(uint8x8_t d) {
  UNUSED(print_u8x8);  // unused for debugging
  cout << (int)vget_lane_u8(d, 7) << ", " << (int)vget_lane_u8(d, 6) << ", "
       << (int)vget_lane_u8(d, 5) << ", " << (int)vget_lane_u8(d, 4) << ", "
       << (int)vget_lane_u8(d, 3) << ", " << (int)vget_lane_u8(d, 2) << ", "
       << (int)vget_lane_u8(d, 1) << ", " << (int)vget_lane_u8(d, 0) << endl;
}

// if mask is FF, select b, otherwise select a.
static int8x8_t select(const int8x8_t a, const int8x8_t b,
                       const int8x8_t mask) {
  const int8x8_t mask_b = mask;
  const int8x8_t mask_a = vmvn_s8(mask_b);
  const int8x8_t selected_from_a = vand_s8(a, mask_a);
  const int8x8_t selected_from_b = vand_s8(b, mask_b);
  const int8x8_t selected = vorr_s8(selected_from_a, selected_from_b);
  return selected;
}

static int8x8x2_t compare(const int8x8_t a,        //
                          const int8x8_t b,        //
                          const int8x8_t index_a,  //
                          const int8x8_t index_b   //
) {
  (void)compare;
  const int8x8_t cmp_a_less_than_b = vreinterpret_s8_u8(vclt_s8(a, b));
  const int8x8_t max_value = select(a, b, cmp_a_less_than_b);
  const int8x8_t max_index = select(index_a, index_b, cmp_a_less_than_b);
  return int8x8x2_t{max_value, max_index};
}

static void max_index_neon_c4(int8_t *d, int g, uint8_t *results) {
  const auto g_to = g / 8 * 8;
  for (int i = 0; i < g / 8; ++i) {
    volatile int8x8x4_t d4x4 = vld4_s8(d);
    const int8x8x2_t cmp_c0_c1 = compare(d4x4.val[0], d4x4.val[1],        //
                                         vcreate_s8(0x0000000000000000),  //
                                         vcreate_s8(0x0101010101010101));
    const int8x8x2_t cmp_c0_c1_c2 =
        compare(cmp_c0_c1.val[0], d4x4.val[2],  //
                cmp_c0_c1.val[1], vcreate_s8(0x0202020202020202));
    const int8x8x2_t cmp_c0_c1_c2_c3 =
        compare(cmp_c0_c1_c2.val[0], d4x4.val[3],  //
                cmp_c0_c1_c2.val[1], vcreate_s8(0x0303030303030303));
    d += 32;
    vst1_u8(results, vreinterpret_u8_s8(cmp_c0_c1_c2_c3.val[1]));
    results += 8;
  }
  if (g_to != g) {
    const auto c = 4;
    vitis::ai::max_index_c(d, c, g - g_to, results);
  }
}

static void max_index_neon_c8(int8_t *d, int g, uint8_t *results) {
  const auto g_to = g / 16 * 16;
  for (int i = 0; i < g / 16; ++i) {
    uint8x16_t temp;
    for (int j = 0; j < 16; j = j + 4) {
      volatile int8x8x4_t d4x4 = vld4_s8(d);
      const int8x8x2_t cmp_c0_c1 = compare(d4x4.val[0], d4x4.val[1],        //
                                           vcreate_s8(0x0400040004000400),  //
                                           vcreate_s8(0x0501050105010501));
      const int8x8x2_t cmp_c0_c1_c2 =
          compare(cmp_c0_c1.val[0], d4x4.val[2],  //
                  cmp_c0_c1.val[1], vcreate_s8(0x0602060206020602));
      const int8x8x2_t cmp_c0_c1_c2_c3 =
          compare(cmp_c0_c1_c2.val[0], d4x4.val[3],  //
                  cmp_c0_c1_c2.val[1], vcreate_s8(0x0703070307030703));

      // Final compare, only lower 4 bytes are valid
      const int8x8x2_t cmp_r0_vs_r1_and_r2_vs_r3 =
          compare(cmp_c0_c1_c2_c3.val[0], vrev16_s8(cmp_c0_c1_c2_c3.val[0]),
                  cmp_c0_c1_c2_c3.val[1], vrev16_s8(cmp_c0_c1_c2_c3.val[1]));
      temp[j + 0] = cmp_r0_vs_r1_and_r2_vs_r3.val[1][0];
      temp[j + 1] = cmp_r0_vs_r1_and_r2_vs_r3.val[1][2];
      temp[j + 2] = cmp_r0_vs_r1_and_r2_vs_r3.val[1][4];
      temp[j + 3] = cmp_r0_vs_r1_and_r2_vs_r3.val[1][6];
      d += 32;
    }
    vst1q_u8(results, temp);
    results += 16;
  }
  if (g_to != g) {
    const auto c = 8;
    vitis::ai::max_index_c(d, c, g - g_to, results);
  }
}

static void max_index_neon_c12(int8_t *d, int g, uint8_t *results) {
  const auto g_to = g / 16 * 16;
  for (int i = 0; i < g / 16; ++i) {
    uint8x16_t temp;
    for (int j = 0; j < 16; j = j + 2) {
      volatile int8x8x4_t d4x4 = vld4_s8(d);
      const int8x8x2_t cmp_c0_c1 = compare(d4x4.val[0], d4x4.val[1],        //
                                           vcreate_s8(0xFFFF080400080400),  //
                                           vcreate_s8(0xFFFF090501090501));
      const int8x8x2_t cmp_c0_c1_c2 =
          compare(cmp_c0_c1.val[0], d4x4.val[2],  //
                  cmp_c0_c1.val[1], vcreate_s8(0xFFFF0A06020A0602));
      int8x8x2_t cmp_c0_c1_c2_c3 =
          compare(cmp_c0_c1_c2.val[0], d4x4.val[3],  //
                  cmp_c0_c1_c2.val[1], vcreate_s8(0xFFFF0B07030B0703));

      cmp_c0_c1_c2_c3.val[0][6] = cmp_c0_c1_c2_c3.val[0][3];
      cmp_c0_c1_c2_c3.val[1][6] = cmp_c0_c1_c2_c3.val[1][3];
      cmp_c0_c1_c2_c3.val[0][3] = 0xFF;
      cmp_c0_c1_c2_c3.val[1][3] = 0xFF;
      cmp_c0_c1_c2_c3.val[0][7] = 0xFF;
      // Final compare, only lower 4 bytes are valid
      const int8x8x2_t cmp_r0_vs_r1_and_r2_vs_r3 =
          compare(cmp_c0_c1_c2_c3.val[0], vrev16_s8(cmp_c0_c1_c2_c3.val[0]),
                  cmp_c0_c1_c2_c3.val[1], vrev16_s8(cmp_c0_c1_c2_c3.val[1]));
      const int8x8x2_t cmp_r01_vs_r23 =
          compare(cmp_r0_vs_r1_and_r2_vs_r3.val[0],
                  vreinterpret_s8_s16(vrev32_s16(
                      vreinterpret_s16_s8(cmp_r0_vs_r1_and_r2_vs_r3.val[0]))),
                  cmp_r0_vs_r1_and_r2_vs_r3.val[1],
                  vreinterpret_s8_s16(vrev32_s16(
                      vreinterpret_s16_s8(cmp_r0_vs_r1_and_r2_vs_r3.val[1]))));

      temp[j] = cmp_r01_vs_r23.val[1][0];
      temp[j + 1] = cmp_r01_vs_r23.val[1][4];
      d += 24;
    }
    vst1q_u8(results, temp);
    results += 16;
  }
  if (g_to != g) {
    const auto c = 12;
    vitis::ai::max_index_c(d, c, g - g_to, results);
  }
}

static void max_index_neon_c16(int8_t *d, int g, uint8_t *results) {
  const auto g_to = g / 16 * 16;
  for (int i = 0; i < g / 16; ++i) {
    uint8x16_t temp;
    for (int j = 0; j < 16; j = j + 2) {
      volatile int8x8x4_t d4x4 = vld4_s8(d);
      const int8x8x2_t cmp_c0_c1 = compare(d4x4.val[0], d4x4.val[1],        //
                                           vcreate_s8(0x0C0804000C080400),  //
                                           vcreate_s8(0x0D0905010D090501));
      const int8x8x2_t cmp_c0_c1_c2 =
          compare(cmp_c0_c1.val[0], d4x4.val[2],  //
                  cmp_c0_c1.val[1], vcreate_s8(0x0E0A06020E0A0602));
      const int8x8x2_t cmp_c0_c1_c2_c3 =
          compare(cmp_c0_c1_c2.val[0], d4x4.val[3],  //
                  cmp_c0_c1_c2.val[1], vcreate_s8(0x0F0B07030F0B0703));

      // Final compare, only lower 4 bytes are valid
      const int8x8x2_t cmp_r0_vs_r1_and_r2_vs_r3 =
          compare(cmp_c0_c1_c2_c3.val[0], vrev16_s8(cmp_c0_c1_c2_c3.val[0]),
                  cmp_c0_c1_c2_c3.val[1], vrev16_s8(cmp_c0_c1_c2_c3.val[1]));
      const int8x8x2_t cmp_r01_vs_r23 =
          compare(cmp_r0_vs_r1_and_r2_vs_r3.val[0],
                  vreinterpret_s8_s16(vrev32_s16(
                      vreinterpret_s16_s8(cmp_r0_vs_r1_and_r2_vs_r3.val[0]))),
                  cmp_r0_vs_r1_and_r2_vs_r3.val[1],
                  vreinterpret_s8_s16(vrev32_s16(
                      vreinterpret_s16_s8(cmp_r0_vs_r1_and_r2_vs_r3.val[1]))));

      temp[j] = cmp_r01_vs_r23.val[1][0];
      temp[j + 1] = cmp_r01_vs_r23.val[1][4];
      d += 32;
    }
    vst1q_u8(results, temp);
    results += 16;
  }
  if (g_to != g) {
    const auto c = 16;
    vitis::ai::max_index_c(d, c, g - g_to, results);
  }
}

/*
static void max_index_neon_c19(int8_t *d, int g, uint8_t *results) {
  const auto g_to = g / 16 * 16;
  for (int i = 0; i < g / 16; ++i) {
    uint8x16_t temp;
    for (int j = 0; j < 16; ++j) {
      volatile int8x8x4_t d4x4 = vld4_s8(d);
      d4x4.val[3][4] = 0xff;
      const int8x8x2_t cmp_c0_c1 = compare(d4x4.val[0], d4x4.val[1],        //
                                           vcreate_s8(0xFFFFFF100C080400),  //
                                           vcreate_s8(0xFFFFFF110D090501));
      const int8x8x2_t cmp_c0_c1_c2 =
          compare(cmp_c0_c1.val[0], d4x4.val[2],  //
                  cmp_c0_c1.val[1], vcreate_s8(0xFFFFFF120E0A0602));
      int8x8x2_t cmp_c0_c1_c2_c3 =
          compare(cmp_c0_c1_c2.val[0], d4x4.val[3],  //
                  cmp_c0_c1_c2.val[1], vcreate_s8(0xFFFFFF130F0B0703));
      cmp_c0_c1_c2_c3.val[0][5] = 0xff;
      cmp_c0_c1_c2_c3.val[0][6] = 0xff;
      cmp_c0_c1_c2_c3.val[0][7] = 0xff;

      // Final compare, only lower 4 bytes are valid
      const int8x8x2_t cmp_r0_vs_r1_and_r2_vs_r3 =
          compare(cmp_c0_c1_c2_c3.val[0], vrev16_s8(cmp_c0_c1_c2_c3.val[0]),
                  cmp_c0_c1_c2_c3.val[1], vrev16_s8(cmp_c0_c1_c2_c3.val[1]));
      const int8x8x2_t cmp_r01_vs_r23 =
          compare(cmp_r0_vs_r1_and_r2_vs_r3.val[0],
                  vreinterpret_s8_s16(vrev32_s16(
                      vreinterpret_s16_s8(cmp_r0_vs_r1_and_r2_vs_r3.val[0]))),
                  cmp_r0_vs_r1_and_r2_vs_r3.val[1],
                  vreinterpret_s8_s16(vrev32_s16(
                      vreinterpret_s16_s8(cmp_r0_vs_r1_and_r2_vs_r3.val[1]))));

      temp[j] = cmp_r01_vs_r23.val[0][0] >= cmp_r01_vs_r23.val[0][4] ? cmp_r01_vs_r23.val[1][0]: cmp_r01_vs_r23.val[1][4];
      d += 19;
    }
    vst1q_u8(results, temp);
    results += 16;
  }
  if (g_to != g) {
    const auto c = 19;
    vitis::ai::max_index_c(d, c, g - g_to, results);
  }
}
*/

#endif

namespace vitis {
namespace ai {
// it has been tested on ZU9. overhead of initializing a
// vector<uint_8t> is < ~0.1ms for 640x360 feature map
// std::vector<uint8_t>
// uint8_t*
void max_index_void(int8_t *feature_map, int width, int height, int channel,
                    uint8_t *ret) {
  const auto g = width * height;
#ifdef ENABLE_NEON
  // std::vector<uint8_t> ret;  //(g);
  // ret.reserve(g);
  // auto ret = new uint8_t[g];
  switch (channel) {
    case 4:
      max_index_neon_c4(feature_map, g, &ret[0]);
      break;
    case 8:
      max_index_neon_c8(feature_map, g, &ret[0]);
      break;
    case 12:
      max_index_neon_c12(feature_map, g, &ret[0]);
      break;
    case 16:
      max_index_neon_c16(feature_map, g, &ret[0]);
      break;
    //case 19:
    //  max_index_neon_c19(feature_map, g, &ret[0]);
    //  break;
    default:
      max_index_c(feature_map, channel, g, &ret[0]);
  }
#else
  max_index_c(feature_map, channel, g, &ret[0]);
#endif
}
std::vector<uint8_t> max_index(int8_t *feature_map, int width, int height,
                               int channel) {
  const auto g = width * height;
  std::vector<uint8_t> ret(g);
#ifdef ENABLE_NEON
  switch (channel) {
    case 4:
      max_index_neon_c4(feature_map, g, &ret[0]);
      break;
    case 8:
      max_index_neon_c8(feature_map, g, &ret[0]);
      break;
    case 12:
      max_index_neon_c12(feature_map, g, &ret[0]);
      break;
    case 16:
      max_index_neon_c16(feature_map, g, &ret[0]);
      break;
    //case 19:
    //  max_index_neon_c19(feature_map, g, &ret[0]);
    //  break;
    default:
      max_index_c(feature_map, channel, g, &ret[0]);
  }
#else
  max_index_c(feature_map, channel, g, &ret[0]);
#endif
  return ret;
}
}  // namespace ai
}  // namespace vitis
// int main() {
//   std::random_device rd;
//   std::mt19937 gen(rd());
//   std::uniform_int_distribution<> dis(1, 127);

//   int c = 8;
//   int g = 640*480;
//   int total = g * c;
//   int8_t *d = new int8_t[total];

//   for (int i = 0; i < total; ++i) {
//     d[i] = dis(gen);
//     //cout << (int)d[i] << ' ';
//     //if (i % c == (c-1)) cout << endl;
//   }

//   uint8_t* results_c = new uint8_t[g];
//   uint8_t* results_neon = new uint8_t[g];
// __TIC__(c)
//   max_index_c(d, c, g, results_c);
// __TOC__(c)
// __TIC__(neon)
//   max_index_neon_c8(d, g, results_neon);
// __TOC__(neon)

// /*
//   for (int i = 0; i < g; ++i) {
//     cout << (int)results_c[i] << endl;
//   }

//   for (int i = 0; i < g; ++i) {
//     cout << (int)results_neon[i] << endl;
//   }
// */

//   delete[] results_neon;
//   delete[] results_c;
//   delete[] d;

// /*
//   float32x4_t q0 = vld1q_f32(d);
//   float32x4_t q1 = vrndq_f32(q0);
//   cout << vgetq_lane_f32(q1, 0) << ", " << vgetq_lane_f32(q1, 1) << ", "
//        << vgetq_lane_f32(q1, 2) << ", " << vgetq_lane_f32(q1, 3) << endl;

//   q1 = vrndaq_f32(q0);
//   cout << vgetq_lane_f32(q1, 0) << ", " << vgetq_lane_f32(q1, 1) << ", "
//        << vgetq_lane_f32(q1, 2) << ", " << vgetq_lane_f32(q1, 3) << endl;

//   q1 = vrndiq_f32(q0);
//   cout << vgetq_lane_f32(q1, 0) << ", " << vgetq_lane_f32(q1, 1) << ", "
//        << vgetq_lane_f32(q1, 2) << ", " << vgetq_lane_f32(q1, 3) << endl;

//   q1 = vrndmq_f32(q0);
//   cout << vgetq_lane_f32(q1, 0) << ", " << vgetq_lane_f32(q1, 1) << ", "
//        << vgetq_lane_f32(q1, 2) << ", " << vgetq_lane_f32(q1, 3) << endl;

//   q1 = vrndnq_f32(q0);
//   cout << vgetq_lane_f32(q1, 0) << ", " << vgetq_lane_f32(q1, 1) << ", "
//        << vgetq_lane_f32(q1, 2) << ", " << vgetq_lane_f32(q1, 3) << endl;

//   q1 = vrndpq_f32(q0);
//   cout << vgetq_lane_f32(q1, 0) << ", " << vgetq_lane_f32(q1, 1) << ", "
//        << vgetq_lane_f32(q1, 2) << ", " << vgetq_lane_f32(q1, 3) << endl;

//   q1 = vrndxq_f32(q0);
//   cout << vgetq_lane_f32(q1, 0) << ", " << vgetq_lane_f32(q1, 1) << ", "
//        << vgetq_lane_f32(q1, 2) << ", " << vgetq_lane_f32(q1, 3) << endl;

//   cout << 1.f/sqrt(d[0]) << ", " << 1.f/sqrt(d[1]) << ", "
//        << 1.f/sqrt(d[2]) << ", " << 1.f/sqrt(d[3]) << endl;
//   float32x4_t q0 = vld1q_f32(d);
//   float32x4_t q1 = vrsqrteq_f32(q0);
//   cout << vgetq_lane_f32(q1, 0) << ", " << vgetq_lane_f32(q1, 1) << ", "
//        << vgetq_lane_f32(q1, 2) << ", " << vgetq_lane_f32(q1, 3) << endl;
//   q1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(q0, q1), q1), q1);
//   cout << vgetq_lane_f32(q1, 0) << ", " << vgetq_lane_f32(q1, 1) << ", "
//        << vgetq_lane_f32(q1, 2) << ", " << vgetq_lane_f32(q1, 3) << endl;
//   q1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(q0, q1), q1), q1);
//   cout << vgetq_lane_f32(q1, 0) << ", " << vgetq_lane_f32(q1, 1) << ", "
//        << vgetq_lane_f32(q1, 2) << ", " << vgetq_lane_f32(q1, 3) << endl;

// */
// }
