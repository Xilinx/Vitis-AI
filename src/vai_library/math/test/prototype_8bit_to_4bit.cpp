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
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <array>
#include <type_traits>
#include <vitis/ai/env_config.hpp>

DEF_ENV_PARAM(DEBUG_TEST, "0");
using Clock = std::chrono::high_resolution_clock;

template <typename N>
std::string to_string(N d) {
  std::ostringstream str;
  const bool  is_signed = std::is_signed<
      typename std::remove_reference<decltype(d[0])>::type>::value;
  str << (is_signed ? "s" : "u") << ("dq"[(sizeof(d) / 8u) - 1u]) << "[";  //
  unsigned char* p = (unsigned char*)&d;
  for (auto i = 0u; i < sizeof(d); ++i) {
    str << ((i == 0u) || (i % sizeof(d[0]) != 0) ? "" : "_");
    str << std::hex << std::setfill('0') << std::setw(2) << ((unsigned int)p[i])
        << std::dec;
  }
  str << ":";
  for (auto i = 0u; i < sizeof(d) / sizeof(d[0]); ++i) {
    str << ((i == 0u) ? "" : ",");
    if (is_signed) {
      str << (signed int)d[i];
    } else {
      str << (unsigned int)d[i];
    }
  };
  str << "]";
  return str.str();
}

int8x8_t normalize(const int16x8_t data2, float mean_f, int fix_pos) {
  /* create mean value vector */
  const int16x8_t mean = vdupq_n_s16(128);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST)) << "mean = " << to_string(mean);
  /* y = x - mean, use i16 to represent u8 or i8, so no overflow. */
  const int16x8_t data3 = vsubq_s16(data2, mean);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << "data3 = vsubq_s16(data2, mean) " << to_string(data3) << std::endl;
  /* shift to right, rounded. convert float point to fix point.
   * still represented as i16, but actually range should be [-128, 127];
   * */
  const int16x8_t data4 =
      fix_pos > 0 ? vrshrq_n_s16(data3, 1) : vshlq_n_s16(data3, fix_pos);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << "data4 = shift(data3, fixpos) " << to_string(data4) << std::endl;
  /* make sure it is between [-128, 127], saturate truncate
   * still represented as i16
   * */
  const int16x8_t data5 =
      vminq_s16(vmaxq_s16(data4, vdupq_n_s16(-128)), vdupq_n_s16(127));
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << "data5 = saturate(data4, -128,127) " << to_string(data5) << std::endl;
  /* convert i16 to i8 */
  const int8x8_t data6 = vmovn_s16(data5);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << "data6 = i16_to_i8(data5) " << to_string(data6) << std::endl;
  return data6;
}

int8x8_t conv_8bit_to_4bit(int8x8_t even, int8x8_t odd) {
  const int8x8_t e1 = vmin_s8(vmax_s8(even, vdup_n_s8(-8)), vdup_n_s8(7));
  const int8x8_t e2 = vand_s8(e1, vdup_n_s8(0xf));
  const int8x8_t o1 = vmin_s8(vmax_s8(odd, vdup_n_s8(-8)), vdup_n_s8(7));
  const int8x8_t o2 = vshl_n_s8(o1, 4);
  const int8x8_t data1 = vadd_s8(e2, o2);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << " e1 " << to_string(e1) << " o1 " << to_string(o1) << " o2 "
      << to_string(o2) << " data " << to_string(data1);
  return data1;
}

void convert_8bit_to_4bitx(const uint8_t* input, int8_t* output) {
  const uint8x16_t data1 = vld1q_u8(input);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << "data1 = " << to_string(data1) << std::endl;
  const uint16x8_t data_even1 =
      vandq_u16(vreinterpretq_u16_u8(data1), vdupq_n_u16(0x00ff));
  const uint16x8_t data_odd1 =
      vandq_u16(vreinterpretq_u16_u8(vrev16q_u8(data1)), vdupq_n_u16(0x00ff));
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << "data_even1 = " << to_string(data_even1) << std::endl;
  const int8x8_t data_even2 =
      normalize(vreinterpretq_s16_u16(data_even1), 128, 1);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << "data_even2 = " << to_string(data_even2) << std::endl;

  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << "data_odd1 = " << to_string(data_odd1) << std::endl;
  const int8x8_t data_odd2 =
      normalize(vreinterpretq_s16_u16(data_odd1), 128, 1);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << "data_odd2 = " << to_string(data_odd2) << std::endl;
  const int8x8_t data2 = conv_8bit_to_4bit(data_even2, data_odd2);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << "data2 = " << to_string(data2) << std::endl;
  vst1_s8(output, data2);
}
int main() {
  int total = 256;
  auto from = std::array<uint8_t, 256>();
  for (int i = 0; i < total; ++i) {
    from[i] = (uint8_t)i;
  }
  auto to = std::array<int8_t, 128>();
  for (int i = 0; i < total; i = i + 16) {
    convert_8bit_to_4bitx(&from[i], &to[i / 2]);
  }
  for (auto i = 0u; i < to.size(); i = i + 1) {
    if (i % 16 == 0) {
      std::cout << std::endl;
    }
    std::cout << ' ' << std::hex << std::setfill('0') << std::setw(2)
              << ((unsigned int)(to[i] & 0xff));
  }
  std::cout << std::endl;
  return 0;
}
