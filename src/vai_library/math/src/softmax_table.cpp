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
#include "../src/softmax_table.hpp"

#include <arm_neon.h>
#include <glog/logging.h>
#include <UniLog/UniLog.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace vitis {
namespace ai {

// const int8x8_t dup_n_s8_64 = vdup_n_s8(64);
// const int8x8_t dup_n_s8_96 = vdup_n_s8(96);
// const int8x8_t dup_n_s8_112 = vdup_n_s8(112);

static void softmax2_neon_table(const int8_t* input, int position,
                                unsigned int group, float* output);
static void softmax4_neon_table(const int8_t* input, int fixpos,
                                unsigned int group, float* output);
static void softmax8_neon_table(const int8_t* input, int fixpos,
                                unsigned int group, float* output);
static void softmax16_neon_table(const int8_t* input, int fixpos,
                                 unsigned int group, float* output);
static void softmax32_neon_table(const int8_t* input, int fixpos,
                                 unsigned int group, float* output);

static void softmax3_neon_table(const int8_t* input, int fixpos,
                                unsigned int group, float* output);
static void softmax6_neon_table(const int8_t* input, int fixpos,
                                unsigned int group, float* output);
static void softmax12_neon_table(const int8_t* input, int fixpos,
                                 unsigned int group, float* output);
static void softmax24_neon_table(const int8_t* input, int fixpos,
                                 unsigned int group, float* output);

//调用前必须检查cls(2,3,4,6,8,12,16) 和 fixpos(定点位置仅支持5,6,7,8) 是否合法
void softmax_neon_table(const int8_t* input, int fixpos, unsigned int cls,
                        unsigned int group, float* output) {
  switch (cls) {
    case 2:
      softmax2_neon_table(input, fixpos, group, output);
      break;
    case 4:
      softmax4_neon_table(input, fixpos, group, output);
      break;
    case 8:
      softmax8_neon_table(input, fixpos, group, output);
      break;
    case 16:
      softmax16_neon_table(input, fixpos, group, output);
      break;
    case 32:
      softmax32_neon_table(input, fixpos, group, output);
      break;
    case 3:
      softmax3_neon_table(input, fixpos, group, output);
      break;
    case 6:
      softmax6_neon_table(input, fixpos, group, output);
      break;
    case 12:
      softmax12_neon_table(input, fixpos, group, output);
      break;
    case 24:
      softmax24_neon_table(input, fixpos, group, output);
      break;
    default:
      exit(-1);
  }
}

struct Table {
  const float miny_;
  const float step_;
  const uint8x8x4_t char_table_;
};

static inline Table getTable(int point) {
  switch (point) {
    case 8:
      return {0.606531f,   //
              0.0038881f,  //
              {vcreate_u8(0xA49A90877E756D65), vcreate_u8(0xFFF2E6DACEC3B8AE),
               vcreate_u8(0x26201A140F0A0400), vcreate_u8(0x5D554E463F39322C)}};
    case 7:
      return {0.367879f,    //
              0.00909863f,  //
              {vcreate_u8(0x897E746A61595149), vcreate_u8(0xFFECDBCCBDAFA195),
               vcreate_u8(0x17130F0C08050200), vcreate_u8(0x423C352F2A25201B)}};
    case 6:
      return {0.135335f,   //
              0.0250411f,  //
              {vcreate_u8(0x5A4F453C342D2722), vcreate_u8(0xFFE0C5AD98857567),
               vcreate_u8(0x0706040302010000), vcreate_u8(0x1D1916120F0D0B09)}};

    case 5:
      // 1: 查表顺序为正序， 会将-128-127
      //   通过饱和加减变为{0,0,0...0,1,2,3...127,127...127}
      //    <-64 -> 0 && >64 ->127
      /// 2 : 饱和左移动
      return {0.135335f,   //
              0.0250411f,  //
              {
                  vcreate_u8(0x5A4F453C342D2722),
                  vcreate_u8(0xFFE0C5AD98857567),
                  vcreate_u8(0x0706040302010000),
                  vcreate_u8(0x1D1916120F0D0B09),
              }};
    case 4:
      // 查表顺序为正序， 会将-128-127
      // 通过饱和加减变为{0,0,0...0,1,2,3...127,127...127}
      //  <-64 -> 0 && >64 ->127
      return {0.0183156f,  //
              0.166678f,   //
              {vcreate_u8(0x221A14100C090705), vcreate_u8(0xFEC69A785D48382C),
               vcreate_u8(0x0000000000000000), vcreate_u8(0x0403020201010000)}};

    case 3:
      // 查表顺序为正序， 会将-128-127
      // 通过饱和加减变为{0,0,0...0,1,2,3...63,63...63}
      //  <-32 -> 0 && >32 ->63
      return {0.0183156f,  //
              0.166678f,   //
              {vcreate_u8(0x221A14100C090705), vcreate_u8(0xFEC69A785D48382C),
               vcreate_u8(0x0000000000000000), vcreate_u8(0x0403020201010000)}};

    case 2:
      // 查表顺序为正序， 会将-128-127
      // 通过饱和加减变为{0,0,0...0,1,2,3...63,63...63}
      //  <-32 -> 0 && >32 ->127
      return {0.000335463f,  //
              7.09036f,      //
              {vcreate_u8(0x0402010100000000), vcreate_u8(0xFF9A5D3822140C07),
               vcreate_u8(0x0000000000000000), vcreate_u8(0x0000000000000000)}};
    case 1:
      // 查表顺序为正序， 会将-128-127
      // 通过饱和加减变为{0,0,0...0,1,2,3...15,15...15}
      //  <-16 -> 0 && >15 ->127
      return {0.000335463f,  //
              7.09036f,      //
              {vcreate_u8(0x0402010100000000), vcreate_u8(0xFF9A5D3822140C07),
               vcreate_u8(0x0000000000000000), vcreate_u8(0x0000000000000000)}};
    default:
      // LOG(FATAL) << "wrong fixpos! point=" << point;
      UNI_LOG_FATAL(VAILIB_MATH_FIX_POS_ERROR)
          << "wrong fixpos! point=" << point;
  }
}

static inline float getScale(int fixpos) {
  switch (fixpos) {
    case 8:
      return 0.00390625f;
    case 7:
      return 0.0078125f;
    case 6:
      return 0.015625f;
    case 5:
      return 0.03125f;
    case 4:
      return 0.0625f;
    case 3:
      return 0.125f;
    case 2:
      return 0.25f;
    case 1:
      return 0.5f;
    default:
      return 0.0f;
  }
}

static void inline softmax_c_table(const int8_t* input, float scale,
                                   unsigned int cls, float* output) {
  float sum = 0.f;
  for (unsigned int i = 0; i < cls; ++i) {
    auto x = input[i] * scale;
    output[i] = exp(x);
    sum += output[i];
  }
  for (unsigned int i = 0; i < cls; ++i) {
    output[i] /= sum;
  }
}

static void softmax_c_table(const int8_t* input, float scale, unsigned int cls,
                            unsigned int group, float* output) {
  for (unsigned int i = 0; i < group; ++i) {
    softmax_c_table(input, scale, cls, output);
    input += cls;
    output += cls;
  }
}
/*static inline uint8x8_t tbl4_u8(const uint8x8x4_t q_table_, const uint8x8_t
   index){ return vtbl4_u8(q_table_, index);
    }*/
static inline uint8x8_t value2char(const uint8x8x4_t q_table_, const int8x8_t q,
                                   const int step) {
  int8x8_t index;
  // int8x8_t q1;
  switch (step) {
    case 4:
      /*// (x- 64 + 64 + 64) >> 2
    q1 = vqsub_s8(q, dup_n_s8_64);
    q1 = vqadd_s8(q1, dup_n_s8_64);
    q1 = vqadd_s8(q1, dup_n_s8_64);
    index = vshr_n_u8(vreinterpret_u8_s8(q1), 2);*/
      //饱和左移一位
      index = vqshl_n_s8(q, 1);
      break;
    case 2:
      /*// (x-96 + 96 + 96 - 64) >> 1
    q1 = vqsub_s8(q, dup_n_s8_96);
    q1 = vqadd_s8(q1, dup_n_s8_96);
    q1 = vqadd_s8(q1, dup_n_s8_96);
    q1 = vqsub_s8(q1, dup_n_s8_64);
    index = vshr_n_u8(vreinterpret_u8_s8(q1), 1);*/
      //饱和左移 2 位
      index = vqshl_n_s8(q, 2);
      break;
    case 1:
      // (x - 112 + 112 + 112 - 96)
      /*q1 = vqsub_s8(q, dup_n_s8_112);
    q1 = vqadd_s8(q1, dup_n_s8_112);
    q1 = vqadd_s8(q1, dup_n_s8_112);
    q1 = vqsub_s8(q1, dup_n_s8_96);
    index = vreinterpret_u8_s8(q1);*/
      index = vqshl_n_s8(q, 3);
      break;
    default:
      // index = vshr_n_u8(vreinterpret_u8_s8(q), 3);
      index = q;
      break;
  }
  return vtbl4_u8(q_table_, vshr_n_u8(vreinterpret_u8_s8(index), 3));
}

static inline uint8x8x4_t value2char(const uint8x8x4_t q_table_,
                                     const int8x8x4_t q, const int step) {
  return {value2char(q_table_, q.val[0], step),
          value2char(q_table_, q.val[1], step),
          value2char(q_table_, q.val[2], step),
          value2char(q_table_, q.val[3], step)};
}
static inline uint8x8x3_t value2char(const uint8x8x4_t q_table_,
                                     const int8x8x3_t q, const int step) {
  return {value2char(q_table_, q.val[0], step),
          value2char(q_table_, q.val[1], step),
          value2char(q_table_, q.val[2], step)};
}

static inline float32x4_t char2float(const uint16x4_t d0_l, const float stepy,
                                     const float32x4_t f_miny) {
  const float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(d0_l));
  return vmlaq_n_f32(f_miny, f0, stepy);  // ai + bi * c;
}

static inline float32x4x4_t char2float(const uint16x4x4_t d, const float stepy,
                                       const float32x4_t f_miny) {
  return {
      char2float(d.val[0], stepy, f_miny), char2float(d.val[1], stepy, f_miny),
      char2float(d.val[2], stepy, f_miny), char2float(d.val[3], stepy, f_miny)};
}

static inline float32x4x3_t char2float(const uint16x4x3_t d, const float stepy,
                                       const float32x4_t f_miny) {
  return {char2float(d.val[0], stepy, f_miny),
          char2float(d.val[1], stepy, f_miny),
          char2float(d.val[2], stepy, f_miny)};
}

static inline uint16x4x4_t vget_low_u16x4(const uint8x8x4_t x) {
  return {vget_low_u16(vmovl_u8(x.val[0])), vget_low_u16(vmovl_u8(x.val[1])),
          vget_low_u16(vmovl_u8(x.val[2])), vget_low_u16(vmovl_u8(x.val[3]))};
}

static inline uint16x4x3_t vget_low_u16x3(const uint8x8x3_t x) {
  return {vget_low_u16(vmovl_u8(x.val[0])), vget_low_u16(vmovl_u8(x.val[1])),
          vget_low_u16(vmovl_u8(x.val[2]))};
}

static inline uint16x4x4_t vget_high_u16x4(const uint8x8x4_t x) {
  return {vget_high_u16(vmovl_u8(x.val[0])), vget_high_u16(vmovl_u8(x.val[1])),
          vget_high_u16(vmovl_u8(x.val[2])), vget_high_u16(vmovl_u8(x.val[3]))};
}
static inline uint16x4x3_t vget_high_u16x3(const uint8x8x3_t x) {
  return {vget_high_u16(vmovl_u8(x.val[0])), vget_high_u16(vmovl_u8(x.val[1])),
          vget_high_u16(vmovl_u8(x.val[2]))};
}

static inline float32x4x4_t float2result_4(float32x4x4_t f) {
  //求和
  float32x4_t q_sum = vaddq_f32(f.val[0], f.val[1]);
  q_sum = vaddq_f32(q_sum, f.val[2]);
  q_sum = vaddq_f32(q_sum, f.val[3]);
  //求倒数
  q_sum = vrecpeq_f32(q_sum);
  return {vmulq_f32(f.val[0], q_sum), vmulq_f32(f.val[1], q_sum),
          vmulq_f32(f.val[2], q_sum), vmulq_f32(f.val[3], q_sum)};
}
static inline int fixpos2step(int fixpos) {
  int step = 8;
  switch (fixpos) {
    case 5:
    case 4:
      step = 4;
      break;
    case 3:
    case 2:
      step = 2;
      break;
    case 1:
      step = 1;
      break;
    default:
      break;
  }
  return step;
}

static void softmax4_internal_table(const int8_t* input, int fixpos,
                                    unsigned int group, float* output) {
  unsigned int batch = group / 8;
  int step = fixpos2step(fixpos);
  // __TIC__(expTable)
  // ExpTable expTable(scale);
  Table table = getTable(fixpos);
  float miny = table.miny_;
  float stepy = table.step_;
  uint8x8x4_t q_table_ = table.char_table_;

  float32x4_t f_miny = vdupq_n_f32(miny);
  // __TOC__(expTable)

  for (unsigned int i = 0; i < batch; ++i) {
    const int8x8x4_t q = vld4_s8(input);
    //查表获取对应的uint8_t
    const uint8x8x4_t q_tbl = value2char(q_table_, q, step);

    //扩展至16x8x4
    // const uint16x8x4_t d = vmovl_u8x4(q_tbl);

    //获取低位进行计算
    const uint16x4x4_t d_l = vget_low_u16x4(q_tbl);
    //将16位的无符号int 转为 32位float (ai * step + f_miny)
    const float32x4x4_t f_l = char2float(d_l, stepy, f_miny);
    const float32x4x4_t r_l = float2result_4(f_l);
    vst4q_f32(output, r_l);
    output += 16;

    // 取高位进行计算
    const uint16x4x4_t d_h = vget_high_u16x4(q_tbl);
    //将16位的无符号int 转为 32位float
    const float32x4x4_t f_h = char2float(d_h, stepy, f_miny);
    const float32x4x4_t r_h = float2result_4(f_h);
    vst4q_f32(output, r_h);
    output += 16;

    input += 32;
  }
}

static void softmax4_neon_table(const int8_t* input, int fixpos,
                                unsigned int group, float* output) {
  unsigned int aligned = group & (-8);
  softmax4_internal_table(input, fixpos, aligned, output);
  unsigned int remain = group - aligned;
  input += (4 * aligned);
  output += (4 * aligned);
  softmax_c_table(input, getScale(fixpos), 4, remain, output);
}

static inline float32x4x4_t float2result_8(float32x4x4_t f) {
  //求和
  float32x4_t q_sum = vaddq_f32(f.val[0], f.val[1]);
  q_sum = vaddq_f32(q_sum, f.val[2]);
  q_sum = vaddq_f32(q_sum, f.val[3]);
  float32x4_t q_sum_rev = vrev64q_f32(q_sum);
  q_sum = vaddq_f32(q_sum, q_sum_rev);
  //求倒数
  q_sum = vrecpeq_f32(q_sum);
  return {vmulq_f32(f.val[0], q_sum), vmulq_f32(f.val[1], q_sum),
          vmulq_f32(f.val[2], q_sum), vmulq_f32(f.val[3], q_sum)};
}
static void softmax8_internal_table(const int8_t* input, int fixpos,
                                    unsigned int group, float* output) {
  unsigned int batch = group / 4;
  int step = fixpos2step(fixpos);

  Table table = getTable(fixpos);
  float miny = table.miny_;
  float stepy = table.step_;
  uint8x8x4_t q_table_ = table.char_table_;

  float32x4_t f_miny = vdupq_n_f32(miny);

  for (unsigned int i = 0; i < batch; ++i) {
    const int8x8x4_t q = vld4_s8(input);

    const uint8x8x4_t q_tbl = value2char(q_table_, q, step);

    // const uint16x8x4_t d = vmovl_u8x4(q_tbl);

    //=====
    // 先取低位进行计算
    const uint16x4x4_t d_l = vget_low_u16x4(q_tbl);
    //将16位的无符号int 转为 32位float
    const float32x4x4_t f_l = char2float(d_l, stepy, f_miny);

    const float32x4x4_t r_l = float2result_8(f_l);
    vst4q_f32(output, r_l);
    output += 16;
    // print_s8x8(d_16);
    //=====
    // 取高位进行计算
    const uint16x4x4_t d_h = vget_high_u16x4(q_tbl);

    //将16位的无符号int 转为 32位float
    const float32x4x4_t f_h = char2float(d_h, stepy, f_miny);
    const float32x4x4_t r_h = float2result_8(f_h);
    vst4q_f32(output, r_h);
    output += 16;

    input += 32;
  }
}

static void softmax8_neon_table(const int8_t* input, int fixpos,
                                unsigned int group, float* output) {
  unsigned int aligned = group & (-4);
  softmax8_internal_table(input, fixpos, aligned, output);
  unsigned int remain = group - aligned;
  input += (8 * aligned);
  output += (8 * aligned);
  softmax_c_table(input, getScale(fixpos), 8, remain, output);
}

static inline float32x4x4_t float2result_16(float32x4x4_t f) {
  //求和
  float32x4_t q_sum = vaddq_f32(f.val[0], f.val[1]);
  q_sum = vaddq_f32(q_sum, f.val[2]);
  q_sum = vaddq_f32(q_sum, f.val[3]);
  float32x4_t q_sum_rev = vrev64q_f32(q_sum);
  q_sum = vaddq_f32(q_sum, q_sum_rev);

  q_sum = vdupq_n_f32(vgetq_lane_f32(q_sum, 1) + vgetq_lane_f32(q_sum, 2));
  //求倒数
  q_sum = vrecpeq_f32(q_sum);
  return {vmulq_f32(f.val[0], q_sum), vmulq_f32(f.val[1], q_sum),
          vmulq_f32(f.val[2], q_sum), vmulq_f32(f.val[3], q_sum)};
}
static void softmax16_internal_table(const int8_t* input, int fixpos,
                                     unsigned int group, float* output) {
  unsigned int batch = group / 2;

  int step = fixpos2step(fixpos);

  Table table = getTable(fixpos);
  float miny = table.miny_;
  float stepy = table.step_;
  uint8x8x4_t q_table_ = table.char_table_;

  float32x4_t f_miny = vdupq_n_f32(miny);

  for (unsigned int i = 0; i < batch; ++i) {
    const int8x8x4_t q = vld4_s8(input);

    const uint8x8x4_t q_tbl = value2char(q_table_, q, step);

    // const uint16x8x4_t d = vmovl_u8x4(q_tbl);

    //=====
    // 先取低位进行计算
    const uint16x4x4_t d_l = vget_low_u16x4(q_tbl);
    //将16位的无符号int 转为 32位float
    const float32x4x4_t f_l = char2float(d_l, stepy, f_miny);

    const float32x4x4_t r_l = float2result_16(f_l);
    vst4q_f32(output, r_l);
    output += 16;
    // print_s8x8(d_16);
    //=====
    // 取高位进行计算
    const uint16x4x4_t d_h = vget_high_u16x4(q_tbl);

    //将16位的无符号int 转为 32位float
    const float32x4x4_t f_h = char2float(d_h, stepy, f_miny);
    const float32x4x4_t r_h = float2result_16(f_h);
    vst4q_f32(output, r_h);
    output += 16;

    input += 32;
  }
}

static void softmax16_neon_table(const int8_t* input, int fixpos,
                                 unsigned int group, float* output) {
  unsigned int aligned = group & (-2);
  softmax16_internal_table(input, fixpos, aligned, output);
  unsigned int remain = group - aligned;
  input += (16 * aligned);
  output += (16 * aligned);
  softmax_c_table(input, getScale(fixpos), 16, remain, output);
}

static inline float32x4_t sum_float32x4x4(float32x4x4_t f) {
  //求和
  float32x4_t q_sum = vaddq_f32(f.val[0], f.val[1]);
  q_sum = vaddq_f32(q_sum, f.val[2]);
  q_sum = vaddq_f32(q_sum, f.val[3]);
  float32x4_t q_sum_rev = vrev64q_f32(q_sum);
  q_sum = vaddq_f32(q_sum, q_sum_rev);
  return vdupq_n_f32(vgetq_lane_f32(q_sum, 1) + vgetq_lane_f32(q_sum, 2));
}
static inline float32x4_t sum_and_recp_32(float32x4x4_t f_l,
                                          float32x4x4_t f_h) {
  //求和
  float32x4_t q_sum = sum_float32x4x4(f_l);
  q_sum = vaddq_f32(q_sum, sum_float32x4x4(f_h));
  return vrecpeq_f32(q_sum);
}

static inline float32x4x4_t float2result_32(float32x4x4_t f,
                                            float32x4_t q_sum) {
  return {vmulq_f32(f.val[0], q_sum), vmulq_f32(f.val[1], q_sum),
          vmulq_f32(f.val[2], q_sum), vmulq_f32(f.val[3], q_sum)};
}

static void softmax32_internal_table(const int8_t* input, int fixpos,
                                     unsigned int group, float* output) {
  int step = fixpos2step(fixpos);

  Table table = getTable(fixpos);
  float miny = table.miny_;
  float stepy = table.step_;
  uint8x8x4_t q_table_ = table.char_table_;

  float32x4_t f_miny = vdupq_n_f32(miny);

  for (unsigned int i = 0; i < group; ++i) {
    const int8x8x4_t q = vld4_s8(input);

    const uint8x8x4_t q_tbl = value2char(q_table_, q, step);

    // const uint16x8x4_t d = vmovl_u8x4(q_tbl);

    //=====
    // 先取低位进行计算
    const uint16x4x4_t d_l = vget_low_u16x4(q_tbl);
    //将16位的无符号int 转为 32位float
    const float32x4x4_t f_l = char2float(d_l, stepy, f_miny);
    // const float32x4x4_t r_l = float2result_16(f_l);
    // vst4q_f32(output, r_l);
    // output += 16;
    // print_s8x8(d_16);
    //=====
    // 取高位进行计算
    const uint16x4x4_t d_h = vget_high_u16x4(q_tbl);

    //将16位的无符号int 转为 32位float
    const float32x4x4_t f_h = char2float(d_h, stepy, f_miny);
    // const float32x4x4_t r_h = float2result_16(f_h);
    const float32x4_t q_sum = sum_and_recp_32(f_l, f_h);

    const float32x4x4_t r_l = float2result_32(f_l, q_sum);
    vst4q_f32(output, r_l);
    output += 16;

    const float32x4x4_t r_h = float2result_32(f_h, q_sum);
    vst4q_f32(output, r_h);
    output += 16;

    input += 32;
  }
}

static void softmax32_neon_table(const int8_t* input, int fixpos,
                                 unsigned int group, float* output) {
  softmax32_internal_table(input, fixpos, group, output);
}

static inline float32x4x4_t float2result_2(float32x4x4_t f) {
  // 求和
  float32x4_t q_sum = vaddq_f32(f.val[0], f.val[1]);
  // 取倒数
  q_sum = vrecpeq_f32(q_sum);

  float32x4_t q_sum_2 = vaddq_f32(f.val[2], f.val[3]);
  // 取倒数
  q_sum_2 = vrecpeq_f32(q_sum_2);

  return {
      vmulq_f32(f.val[0], q_sum),
      vmulq_f32(f.val[1], q_sum),
      vmulq_f32(f.val[2], q_sum_2),
      vmulq_f32(f.val[3], q_sum_2),
  };
}

static void softmax2_internal_table(const int8_t* input, int fixpos,
                                    unsigned int group, float* output) {
  unsigned int batch = group / 16;
  int step = fixpos2step(fixpos);

  Table table = getTable(fixpos);
  float miny = table.miny_;
  float stepy = table.step_;
  uint8x8x4_t q_table_ = table.char_table_;

  float32x4_t f_miny = vdupq_n_f32(miny);

  for (unsigned int i = 0; i < batch; ++i) {
    const int8x8x4_t q = vld4_s8(input);

    const uint8x8x4_t q_tbl = value2char(q_table_, q, step);
    // 先取低位进行计算

    const uint16x4x4_t d_l = vget_low_u16x4(q_tbl);

    //将16位的无符号int 转为 32位float
    const float32x4x4_t f_l = char2float(d_l, stepy, f_miny);

    const float32x4x4_t r_l = float2result_2(f_l);

    vst4q_f32(output, r_l);
    output += 16;

    // 取高位进行计算
    const uint16x4x4_t d_h = vget_high_u16x4(q_tbl);

    //将16位的无符号int 转为 32位float
    const float32x4x4_t f_h = char2float(d_h, stepy, f_miny);
    const float32x4x4_t r_h = float2result_2(f_h);

    vst4q_f32(output, r_h);
    output += 16;

    input += 32;
  }
}

static void softmax2_neon_table(const int8_t* input, int fixpos,
                                unsigned int group, float* output) {
  unsigned int aligned = group & (-16);
  softmax2_internal_table(input, fixpos, aligned, output);
  unsigned int remain = group - aligned;
  input += (2 * aligned);
  output += (2 * aligned);
  softmax_c_table(input, getScale(fixpos), 2, remain, output);
}

static inline float32x4x3_t float2result_3(float32x4x3_t f) {
  // 求和
  float32x4_t q_sum = vaddq_f32(f.val[0], f.val[1]);
  q_sum = vaddq_f32(q_sum, f.val[2]);
  // 取倒数
  q_sum = vrecpeq_f32(q_sum);
  return {vmulq_f32(f.val[0], q_sum), vmulq_f32(f.val[1], q_sum),
          vmulq_f32(f.val[2], q_sum)};
}
static void softmax3_internal_table(const int8_t* input, int fixpos,
                                    unsigned int group, float* output) {
  unsigned int batch = group / 8;
  int step = fixpos2step(fixpos);

  Table table = getTable(fixpos);
  float miny = table.miny_;
  float stepy = table.step_;
  uint8x8x4_t q_table_ = table.char_table_;

  float32x4_t f_miny = vdupq_n_f32(miny);

  for (unsigned int i = 0; i < batch; ++i) {
    const int8x8x3_t q = vld3_s8(input);

    const uint8x8x3_t q_tbl = value2char(q_table_, q, step);
    // 先取低位进行计算

    const uint16x4x3_t d_l = vget_low_u16x3(q_tbl);

    //将16位的无符号int 转为 32位float
    const float32x4x3_t f_l = char2float(d_l, stepy, f_miny);

    const float32x4x3_t r_l = float2result_3(f_l);

    vst3q_f32(output, r_l);
    output += 12;

    // 取高位进行计算
    const uint16x4x3_t d_h = vget_high_u16x3(q_tbl);

    //将16位的无符号int 转为 32位float
    const float32x4x3_t f_h = char2float(d_h, stepy, f_miny);
    const float32x4x3_t r_h = float2result_3(f_h);

    vst3q_f32(output, r_h);
    output += 12;

    input += 24;
  }
}

static void softmax3_neon_table(const int8_t* input, int fixpos,
                                unsigned int group, float* output) {
  unsigned int aligned = group & (-8);
  softmax3_internal_table(input, fixpos, aligned, output);
  unsigned int remain = group - aligned;
  input += (3 * aligned);
  output += (3 * aligned);
  softmax_c_table(input, getScale(fixpos), 3, remain, output);
}

static inline float32x4x3_t float2result_6(float32x4x3_t f) {
  // 求和
  float32x4_t q_sum = vaddq_f32(f.val[0], f.val[1]);
  q_sum = vaddq_f32(q_sum, f.val[2]);
  float32x4_t q_sum_rev = vrev64q_f32(q_sum);
  q_sum = vaddq_f32(q_sum, q_sum_rev);
  // 取倒数
  q_sum = vrecpeq_f32(q_sum);
  return {vmulq_f32(f.val[0], q_sum), vmulq_f32(f.val[1], q_sum),
          vmulq_f32(f.val[2], q_sum)};
}
static void softmax6_internal_table(const int8_t* input, int fixpos,
                                    unsigned int group, float* output) {
  unsigned int batch = group / 4;
  int step = fixpos2step(fixpos);

  Table table = getTable(fixpos);
  float miny = table.miny_;
  float stepy = table.step_;
  uint8x8x4_t q_table_ = table.char_table_;

  float32x4_t f_miny = vdupq_n_f32(miny);

  for (unsigned int i = 0; i < batch; ++i) {
    const int8x8x3_t q = vld3_s8(input);

    const uint8x8x3_t q_tbl = value2char(q_table_, q, step);
    // 先取低位进行计算

    const uint16x4x3_t d_l = vget_low_u16x3(q_tbl);

    //将16位的无符号int 转为 32位float
    const float32x4x3_t f_l = char2float(d_l, stepy, f_miny);

    const float32x4x3_t r_l = float2result_6(f_l);

    vst3q_f32(output, r_l);
    output += 12;

    // 取高位进行计算
    const uint16x4x3_t d_h = vget_high_u16x3(q_tbl);

    //将16位的无符号int 转为 32位float
    const float32x4x3_t f_h = char2float(d_h, stepy, f_miny);
    const float32x4x3_t r_h = float2result_6(f_h);

    vst3q_f32(output, r_h);
    output += 12;

    input += 24;
  }
}

static void softmax6_neon_table(const int8_t* input, int fixpos,
                                unsigned int group, float* output) {
  unsigned int aligned = group & (-4);
  softmax6_internal_table(input, fixpos, aligned, output);
  unsigned int remain = group - aligned;
  input += (6 * aligned);
  output += (6 * aligned);
  softmax_c_table(input, getScale(fixpos), 6, remain, output);
}

static inline float32x4x3_t float2result_12(float32x4x3_t f) {
  // 求和
  float32x4_t q_sum = vaddq_f32(f.val[0], f.val[1]);
  q_sum = vaddq_f32(q_sum, f.val[2]);
  float32x4_t q_sum_rev = vrev64q_f32(q_sum);
  q_sum = vaddq_f32(q_sum, q_sum_rev);

  q_sum = vdupq_n_f32(vgetq_lane_f32(q_sum, 1) + vgetq_lane_f32(q_sum, 2));
  // 取倒数
  q_sum = vrecpeq_f32(q_sum);
  return {vmulq_f32(f.val[0], q_sum), vmulq_f32(f.val[1], q_sum),
          vmulq_f32(f.val[2], q_sum)};
}
static void softmax12_internal_table(const int8_t* input, int fixpos,
                                     unsigned int group, float* output) {
  unsigned int batch = group / 2;
  int step = fixpos2step(fixpos);

  Table table = getTable(fixpos);
  float miny = table.miny_;
  float stepy = table.step_;
  uint8x8x4_t q_table_ = table.char_table_;

  float32x4_t f_miny = vdupq_n_f32(miny);

  for (unsigned int i = 0; i < batch; ++i) {
    const int8x8x3_t q = vld3_s8(input);

    const uint8x8x3_t q_tbl = value2char(q_table_, q, step);
    // 先取低位进行计算

    const uint16x4x3_t d_l = vget_low_u16x3(q_tbl);

    //将16位的无符号int 转为 32位float
    const float32x4x3_t f_l = char2float(d_l, stepy, f_miny);

    const float32x4x3_t r_l = float2result_12(f_l);

    vst3q_f32(output, r_l);
    output += 12;

    // 取高位进行计算
    const uint16x4x3_t d_h = vget_high_u16x3(q_tbl);

    //将16位的无符号int 转为 32位float
    const float32x4x3_t f_h = char2float(d_h, stepy, f_miny);
    const float32x4x3_t r_h = float2result_12(f_h);

    vst3q_f32(output, r_h);
    output += 12;

    input += 24;
  }
}

static void softmax12_neon_table(const int8_t* input, int fixpos,
                                 unsigned int group, float* output) {
  unsigned int aligned = group & (-2);
  softmax12_internal_table(input, fixpos, aligned, output);
  unsigned int remain = group - aligned;
  input += (12 * aligned);
  output += (12 * aligned);
  softmax_c_table(input, getScale(fixpos), 6, remain, output);
}

static inline float32x4_t sum_float32x4x3(float32x4x3_t f) {
  //求和
  float32x4_t q_sum = vaddq_f32(f.val[0], f.val[1]);
  q_sum = vaddq_f32(q_sum, f.val[2]);
  float32x4_t q_sum_rev = vrev64q_f32(q_sum);
  q_sum = vaddq_f32(q_sum, q_sum_rev);
  return vdupq_n_f32(vgetq_lane_f32(q_sum, 1) + vgetq_lane_f32(q_sum, 2));
}
static inline float32x4_t sum_and_recp_24(float32x4x3_t f_l,
                                          float32x4x3_t f_h) {
  //求和
  float32x4_t q_sum = sum_float32x4x3(f_l);
  q_sum = vaddq_f32(q_sum, sum_float32x4x3(f_h));
  return vrecpeq_f32(q_sum);
}

static inline float32x4x3_t float2result_24(float32x4x3_t f,
                                            float32x4_t q_sum) {
  return {vmulq_f32(f.val[0], q_sum), vmulq_f32(f.val[1], q_sum),
          vmulq_f32(f.val[2], q_sum)};
}
static void softmax24_internal_table(const int8_t* input, int fixpos,
                                     unsigned int group, float* output) {
  int step = fixpos2step(fixpos);

  Table table = getTable(fixpos);
  float miny = table.miny_;
  float stepy = table.step_;
  uint8x8x4_t q_table_ = table.char_table_;

  float32x4_t f_miny = vdupq_n_f32(miny);

  for (unsigned int i = 0; i < group; ++i) {
    const int8x8x3_t q = vld3_s8(input);

    const uint8x8x3_t q_tbl = value2char(q_table_, q, step);
    // 先取低位进行计算

    const uint16x4x3_t d_l = vget_low_u16x3(q_tbl);
    //将16位的无符号int 转为 32位float
    const float32x4x3_t f_l = char2float(d_l, stepy, f_miny);

    // 取高位进行计算ok.
    const uint16x4x3_t d_h = vget_high_u16x3(q_tbl);
    //将16位的无符号int 转为 32位float
    const float32x4x3_t f_h = char2float(d_h, stepy, f_miny);

    const float32x4_t q_sum = sum_and_recp_24(f_l, f_h);

    const float32x4x3_t r_l = float2result_24(f_l, q_sum);
    vst3q_f32(output, r_l);
    output += 12;

    const float32x4x3_t r_h = float2result_24(f_h, q_sum);
    vst3q_f32(output, r_h);
    output += 12;

    input += 24;
  }
}

static void softmax24_neon_table(const int8_t* input, int fixpos,
                                 unsigned int group, float* output) {
  softmax24_internal_table(input, fixpos, group, output);
}

}  // namespace ai
}  // namespace vitis
