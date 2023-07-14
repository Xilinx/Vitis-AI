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
#ifdef ENABLE_NEON

#include <arm_neon.h>

#include <cmath>
#include <cstring>

#include "vitis/ai/ssd_normalizer.hpp"

// Hack: vrndaq_f32 is not defined in armv7.
// We can use identity function to replace rounding.
#ifdef __ARM_ARCH_7A__
#define vrndaq_f32(X) X
#endif

using std::memset;
using std::round;

float sum_neon(const float *input, unsigned int size) {
  unsigned int aligned = size & (-4);
  float sum = 0.f;
  if (aligned > 0) {
    float32x4_t q0 = vld1q_f32(input);
    input += 4;
    for (unsigned int i = 0; i < aligned / 4 - 1; ++i) {
      float32x4_t q1 = vld1q_f32(input);
      q0 = vaddq_f32(q0, q1);
      input += 4;
    }
    float32x2_t d4 = vadd_f32(vget_high_f32(q0), vget_low_f32(q0));
    sum = vget_lane_f32(vpadd_f32(d4, d4), 0);
  }
  for (unsigned int i = 0; i < size - aligned; ++i) {
    sum += input[i];
  }
  return sum;
}

float square_sum_neon(const int8_t *input, unsigned int size) {
  unsigned int aligned = size & (-8);
  int32x4_t q0 = vdupq_n_s32(0);
  for (unsigned int i = 0; i < aligned / 8; ++i) {
    int8x8_t d2 = vld1_s8(input);
    int16x8_t q2 = vmovl_s8(d2);
    int16x4_t d4 = vget_low_s16(q2);
    q0 = vmlal_s16(q0, d4, d4);
    int16x4_t d5 = vget_high_s16(q2);
    q0 = vmlal_s16(q0, d5, d5);
    input += 8;
  }
  int32x2_t d3 = vadd_s32(vget_high_s32(q0), vget_low_s32(q0));
  float sum = vget_lane_s32(vpadd_s32(d3, d3), 0);
  for (unsigned int i = 0; i < size - aligned; ++i) {
    sum += input[i] * input[i];
  }
  return sum;
}

void inv_sqrt_inplace_neon(float *data, unsigned int size, float eps) {
  unsigned int aligned = size & (-4);
  float32x4_t veps = vdupq_n_f32(eps);
  for (unsigned int i = 0; i < aligned / 4; ++i) {
    float32x4_t q0 = vld1q_f32(data);
    q0 = vaddq_f32(q0, veps);
    float32x4_t q1 = vrsqrteq_f32(q0);
    q1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(q0, q1), q1), q1);
    // q1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(q0, q1), q1), q1);
    vst1q_f32(data, q1);
    data += 4;
  }
  for (unsigned int i = 0; i < size - aligned; ++i) {
    data[i] = 1.f / sqrt(data[i] + eps);
  }
}
void scalar_prod_neon(float scalar, const float *input, unsigned int size,
                      float *output) {
  unsigned int aligned = size & (-4);
  for (unsigned int i = 0; i < aligned / 4; ++i) {
    float32x4_t q0 = vld1q_f32(input);
    q0 = vmulq_n_f32(q0, scalar);
    vst1q_f32(output, q0);
    input += 4;
    output += 4;
  }
  for (unsigned int i = 0; i < size - aligned; ++i) {
    output[i] = scalar * input[i];
  }
}

void scalar_prod_inplace_neon(float scalar, float *data, unsigned int size) {
  unsigned int aligned = size & (-4);
  for (unsigned int i = 0; i < aligned / 4; ++i) {
    float32x4_t q0 = vld1q_f32(data);
    q0 = vmulq_n_f32(q0, scalar);
    vst1q_f32(data, q0);
    data += 4;
  }
  for (unsigned int i = 0; i < size - aligned; ++i) {
    data[i] = scalar * data[i];
  }
}

void scalar_prod_neon(float scalar, const int8_t *input, unsigned int size,
                      int8_t *output) {
  unsigned int aligned = size & (-8);
  for (unsigned int i = 0; i < aligned / 8; ++i) {
    int8x8_t d0 = vld1_s8(input);
    int16x8_t q1 = vmovl_s8(d0);
    int16x4_t d2 = vget_low_s16(q1);
    int16x4_t d3 = vget_high_s16(q1);

    int32x4_t q3 = vmovl_s16(d2);
    float32x4_t q4 = vcvtq_f32_s32(q3);
    q4 = vrndaq_f32(vmulq_n_f32(q4, scalar));
    q3 = vcvtq_s32_f32(q4);
    d2 = vmovn_s32(q3);

    q3 = vmovl_s16(d3);
    q4 = vcvtq_f32_s32(q3);
    q4 = vrndaq_f32(vmulq_n_f32(q4, scalar));
    q3 = vcvtq_s32_f32(q4);
    d3 = vmovn_s32(q3);

    int16x8_t q2 = vcombine_s16(d2, d3);
    int8x8_t d1 = vmovn_s16(q2);
    vst1_s8(output, d1);

    input += 8;
    output += 8;
  }

  for (unsigned int i = 0; i < size - aligned; ++i) {
    output[i] = round(scalar * input[i]);
  }
}

void dot_prod_neon(const float *scale, const int8_t *input, unsigned int size,
                   int8_t *output) {
  unsigned int aligned = size & (-8);
  for (unsigned int i = 0; i < aligned / 8; ++i) {
    int8x8_t d0 = vld1_s8(input);
    int16x8_t q1 = vmovl_s8(d0);
    int16x4_t d2 = vget_low_s16(q1);
    int16x4_t d3 = vget_high_s16(q1);

    int32x4_t q3 = vmovl_s16(d2);
    float32x4_t q4 = vcvtq_f32_s32(q3);
    float32x4_t q5 = vld1q_f32(scale);
    q4 = vrndaq_f32(vmulq_f32(q4, q5));
    q3 = vcvtq_s32_f32(q4);
    d2 = vmovn_s32(q3);

    q3 = vmovl_s16(d3);
    q4 = vcvtq_f32_s32(q3);
    q5 = vld1q_f32(scale + 4);
    q4 = vrndaq_f32(vmulq_f32(q4, q5));
    q3 = vcvtq_s32_f32(q4);
    d3 = vmovn_s32(q3);

    int16x8_t q2 = vcombine_s16(d2, d3);
    int8x8_t d1 = vmovn_s16(q2);
    vst1_s8(output, d1);

    input += 8;
    output += 8;
    scale += 8;
  }

  for (unsigned int i = 0; i < size - aligned; ++i) {
    output[i] = round(scale[i] * input[i]);
  }
}

void scalar_dot_prod_neon(float scalar, const float *scale, const int8_t *input,
                          unsigned int size, int8_t *output) {
  unsigned int aligned = size & (-8);
  for (unsigned int i = 0; i < aligned / 8; ++i) {
    int8x8_t d0 = vld1_s8(input);
    int16x8_t q1 = vmovl_s8(d0);

    int32x4_t q3 = vmovl_s16(vget_low_s16(q1));
    float32x4_t q4 = vcvtq_f32_s32(q3);
    float32x4_t q5 = vld1q_f32(scale);
    q4 = vrndaq_f32(vmulq_n_f32(vmulq_f32(q4, q5), scalar));
    q3 = vcvtq_s32_f32(q4);
    int16x4_t d2 = vmovn_s32(q3);

    q3 = vmovl_s16(vget_high_s16(q1));
    q4 = vcvtq_f32_s32(q3);
    q5 = vld1q_f32(scale + 4);
    q4 = vrndaq_f32(vmulq_n_f32(vmulq_f32(q4, q5), scalar));
    q3 = vcvtq_s32_f32(q4);
    int16x4_t d3 = vmovn_s32(q3);

    int16x8_t q2 = vcombine_s16(d2, d3);
    int8x8_t d1 = vmovn_s16(q2);
    vst1_s8(output, d1);

    input += 8;
    output += 8;
    scale += 8;
  }

  for (unsigned int i = 0; i < size - aligned; ++i) {
    output[i] = round(scalar * scale[i] * input[i]);
  }
}

int dot_2d_prod_neon(const float *scale_w, const float *scale_h,
                     const int8_t *input, unsigned int width,
                     unsigned int height, int8_t *output) {
  unsigned int aligned_w = width & (-8);
  if (aligned_w < width) return -1;
  const int8_t *pi = input;
  int8_t *po = output;

  for (auto w = 0u; w < aligned_w / 8; ++w) {
    float32x4_t q0 = vld1q_f32(scale_w);
    float32x4_t q1 = vld1q_f32(scale_w + 4);
    for (auto h = 0u; h < height; ++h) {
      int8x8_t d4 = vld1_s8(pi);
      int16x8_t q2 = vmovl_s8(d4);
      int16x4_t d6 = vget_low_s16(q2);
      int16x4_t d7 = vget_high_s16(q2);

      int32x4_t q4 = vmovl_s16(d6);
      float32x4_t q5 = vcvtq_f32_s32(q4);
      q5 = vmulq_n_f32(q5, scale_h[h]);
      q5 = vrndaq_f32(vmulq_f32(q5, q0));
      q4 = vcvtq_s32_f32(q5);
      d6 = vmovn_s32(q4);

      q4 = vmovl_s16(d7);
      q5 = vcvtq_f32_s32(q4);
      q5 = vmulq_n_f32(q5, scale_h[h]);
      q5 = vrndaq_f32(vmulq_f32(q5, q1));
      q4 = vcvtq_s32_f32(q5);
      d7 = vmovn_s32(q4);

      int16x8_t q3 = vcombine_s16(d6, d7);
      int8x8_t d5 = vmovn_s16(q3);
      vst1_s8(po, d5);

      pi += width;
      po += width;
    }
    input += 8;
    pi = input;
    output += 8;
    po = output;
    scale_w += 8;
  }

  return 0;
}

namespace vitis {
namespace ai {

int SSDNormalizer::normalize_neon(const int8_t *input, int8_t *output) {
  if (channel_ % 8 != 0) return -1;

  if (across_spatial_) {
    float norm = square_sum_neon(input, num_);
    norm = sqrt(norm + eps_);
    if (channel_shared_) {
      norm = scale_[0] / norm;
      scalar_prod_neon(norm, input, num_, output);
    } else {
      scalar_prod_neon(1. / norm, scale_, channel_, scale_buf_);
      for (int i = 0; i < spatial_dim_; ++i) {
        dot_prod_neon(scale_buf_, input, channel_, output);
        input += channel_;
        output += channel_;
      }
    }
  } else {
    const int8_t *pi = input;
    for (int i = 0; i < spatial_dim_; ++i) {
      norm_buf_[i] = square_sum_neon(pi, channel_);
      pi += channel_;
    }
    inv_sqrt_inplace_neon(norm_buf_, spatial_dim_, eps_);
    if (channel_shared_) {
      scalar_prod_inplace_neon(scale_[0], norm_buf_, spatial_dim_);
      for (int i = 0; i < spatial_dim_; ++i) {
        scalar_prod_neon(norm_buf_[i], input, channel_, output);
        input += channel_;
        output += channel_;
      }
    } else {
      pi = input;
      for (int i = 0; i < spatial_dim_; ++i) {
        scalar_dot_prod_neon(norm_buf_[i], scale_, input, channel_, output);
        input += channel_;
        output += channel_;
      }
    }
  }
  return 0;
}

}  // namespace ai
}  // namespace vitis

#endif
