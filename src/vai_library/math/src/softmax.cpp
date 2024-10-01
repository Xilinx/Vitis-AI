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
#include <fcntl.h>
#include <glog/logging.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <mutex>
#include <vitis/ai/env_config.hpp>

#include "./softmax_table.hpp"
#include "vitis/ai/math.hpp"
#include "xir/sfm_controller.hpp"
#ifdef ENABLE_NEON
#include <arm_neon.h>
extern "C" float32x4_t exp_ps(float32x4_t);
#endif
DEF_ENV_PARAM(DEBUG_DPMATH, "0");
DEF_ENV_PARAM(XLNX_ENABLE_C_SOFTMAX, "0");

using std::exp;

int GLOBAL_ENABLE_C_SOFTMAX = 0;

namespace vitis {
namespace ai {

struct req_softmax_t {
  uint32_t width;  /* width dimention of Tensor */
  uint32_t height; /* height dimention of Tensor */
  uint32_t input;  /* physical address of input Tensor */
  uint32_t output; /* physical address of output Tensor */
  uint32_t scale;  /* quantization info of input Tensor */
  uint32_t offset; /* offset value for input Tensor */
};

//# Templatized input datatype
template <typename T>
static void softmax_c(T* input, float scale, unsigned int cls,
                      unsigned int group, float* output);

//# Templatized softmax_c
template <typename T>
static void softmax_c(T* input, float scale, unsigned int cls, float* output);
#ifdef ENABLE_NEON
static void softmax2_neon(const int8_t* input, float scale, unsigned int group,
                          float* output);
static void softmax4_neon(const int8_t* input, float scale, unsigned int group,
                          float* output);
#endif
/*
 static void softmax_my(const int8_t *input, float scale, unsigned int cls,
                       unsigned int group, float *output);

void softmax(const int8_t *input, float scale, unsigned int cls,
             unsigned int group, float *output) {
  if (GLOBAL_ENABLE_C_SOFTMAX == 0) {
    dpuRunSoftmax((int8_t *)input, output, cls, group, scale);
  } else {
    softmax_my(input, scale, cls, group, output);
  }
  }*/

//# Softmax method with float input data, used for DPUV1
void softmax(const float* input, float scale, unsigned int cls,
             unsigned int group, float* output) {
  softmax_c(input, scale, cls, group, output);
}

void softmax(const int8_t* input, float scale, unsigned int cls,
             unsigned int group, float* output) {
  if (ENV_PARAM(XLNX_ENABLE_C_SOFTMAX)) {
    GLOBAL_ENABLE_C_SOFTMAX = 2;
  }

#ifdef ENABLE_NEON
  if (GLOBAL_ENABLE_C_SOFTMAX == 1) {
    if (cls == 2) {
      softmax2_neon(input, scale, group, output);
    } else if (cls == 4) {
      softmax4_neon(input, scale, group, output);
    } else {
      softmax_c(input, scale, cls, group, output);
    }
  } else if (GLOBAL_ENABLE_C_SOFTMAX == 0) {
    //判断scale
    auto scale2fixpos = [](float scale1) { return std::abs((int)log2(scale1)); };
    int fixpos = scale2fixpos(scale);
    bool fixpoint_supported = (fixpos < 9 && fixpos > 0);
    bool cls_supported =
        (cls == 2 || cls == 4 || cls == 8 || cls == 16 || cls == 32 ||
         cls == 3 || cls == 6 || cls == 12 || cls == 24);
    bool neon_opt_supported = cls_supported && fixpoint_supported;
    if (neon_opt_supported) {
      softmax_neon_table(input, fixpos, cls, group, output);
      return;
    }
    static auto hw_smfc = xir::SfmController::get_instance();
    if (hw_smfc && hw_smfc->supported(scale, cls, group)) {
      hw_smfc->run(input, scale, cls, group, output);
    } else {
      softmax_c(input, scale, cls, group, output);
    }
  } else {
    softmax_c(input, scale, cls, group, output);
  }
#else
  if (GLOBAL_ENABLE_C_SOFTMAX == 0) {
    static auto hw_smfc = xir::SfmController::get_instance();
    if (hw_smfc && hw_smfc->supported(scale, cls, group)) {
      hw_smfc->run(input, scale, cls, group, output);
    } else {
      softmax_c(input, scale, cls, group, output);
    }
  } else {
    softmax_c(input, scale, cls, group, output);
  }
#endif
}

template <typename T>
static void softmax_c(T* input, float scale, unsigned int cls,
                      unsigned int group, float* output) {
  for (unsigned int i = 0; i < group; ++i) {
    softmax_c(input, scale, cls, output);
    input += cls;
    output += cls;
  }
}
template <typename T>
static void softmax_c(T* input, float scale, unsigned int cls, float* output) {
  if (ENV_PARAM(DEBUG_DPMATH) >= 5) {
    auto mode =
        std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
    CHECK(std::ofstream("softmax_c_input.bin", mode)
              .write((char*)(input), sizeof(T) * cls)
              .good())
        << " faild to write to "
        << "softmax_c_input.bin";
  }
  float sum = 0.f;
  for (unsigned int i = 0; i < cls; ++i) {
    output[i] = exp(input[i] * scale);
    sum += output[i];
  }
  for (unsigned int i = 0; i < cls; ++i) output[i] /= sum;
  if (ENV_PARAM(DEBUG_DPMATH) >= 5) {
    auto mode =
        std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
    CHECK(std::ofstream("softmax_c_output.bin", mode)
              .write((char*)(output), sizeof(float) * cls)
              .good())
        << " faild to write to "
        << "softmax_c_output.bin";
  }
}

/*
 * 4-class softmax
 */
#ifdef ENABLE_NEON
static void softmax4_internal(const int8_t*, float, unsigned int, float*);
static void softmax4_neon(const int8_t* input, float scale, unsigned int group,
                          float* output) {
  unsigned int aligned = group & (-8);
  softmax4_internal(input, scale, aligned, output);
  unsigned int remain = group - aligned;
  input += (4 * aligned);
  output += (4 * aligned);
  softmax_c(input, scale, 4, remain, output);
}

/*
 * 2-class softmax
 */
static void softmax2_internal(const int8_t*, float, unsigned int, float*);
static void softmax2_neon(const int8_t* input, float scale, unsigned int group,
                          float* output) {
  unsigned int aligned = group & (-8);
  softmax2_internal(input, scale, aligned, output);
  unsigned int remain = group - aligned;
  input += (2 * aligned);
  output += (2 * aligned);
  softmax_c(input, scale, 2, remain, output);
}

/*
 * 2-class softmax, but only output class 1 (class 0 is considerred as
 * background)
 */
void softmax2_o1_internal(const int8_t*, float, unsigned int, float*);
void softmax2_o1_neon(const int8_t* input, float scale, unsigned int group,
                      float* output) {
  unsigned int aligned = group & (-8);
  softmax2_o1_internal(input, scale, aligned, output);

  unsigned int remain = group - aligned;
  input += (2 * aligned);
  output += aligned;

  for (unsigned int i = 0; i < remain; ++i) {
    float e0 = exp(input[0] * scale);
    float e1 = exp(input[1] * scale);
    output[i] = e1 / (e0 + e1);
    input += 2;
  }
}

/*
 * Assume group is divided by 8
 */

static void softmax4_internal(const int8_t* input, float scale,
                              unsigned int group, float* output) {
  unsigned int batch = group / 8;

  for (unsigned int i = 0; i < batch; ++i) {
    /* Interleaved load 32 bytes into 4 NEON registers */
    int8x8x4_t q01 = vld4_s8(input);
    /* Convert to 16-bit integers */
    int16x8_t q2 = vmovl_s8(q01.val[0]);
    int16x8_t q3 = vmovl_s8(q01.val[1]);
    int16x8_t q4 = vmovl_s8(q01.val[2]);
    int16x8_t q5 = vmovl_s8(q01.val[3]);

    /* Process first 4 groups */
    int16x4_t d10 = vget_low_s16(q2);
    int16x4_t d11 = vget_low_s16(q3);
    int16x4_t d12 = vget_low_s16(q4);
    int16x4_t d13 = vget_low_s16(q5);

    float32x4_t q8 = vcvtq_f32_s32(vmovl_s16(d10));
    float32x4_t q9 = vcvtq_f32_s32(vmovl_s16(d11));
    float32x4_t q10 = vcvtq_f32_s32(vmovl_s16(d12));
    float32x4_t q11 = vcvtq_f32_s32(vmovl_s16(d13));

    q8 = exp_ps(vmulq_n_f32(q8, scale));
    q9 = exp_ps(vmulq_n_f32(q9, scale));
    q10 = exp_ps(vmulq_n_f32(q10, scale));
    q11 = exp_ps(vmulq_n_f32(q11, scale));

    float32x4_t q12 = vaddq_f32(q8, q9);
    q12 = vaddq_f32(q12, q10);
    q12 = vaddq_f32(q12, q11);
    q12 = vrecpeq_f32(q12);

    q8 = vmulq_f32(q12, q8);
    q9 = vmulq_f32(q12, q9);
    q10 = vmulq_f32(q12, q10);
    q11 = vmulq_f32(q12, q11);

    float32x4x4_t b0 = {q8, q9, q10, q11};
    vst4q_f32(output, b0);
    output += 16;

    /* Process last 4 groups */
    d10 = vget_high_s16(q2);
    d11 = vget_high_s16(q3);
    d12 = vget_high_s16(q4);
    d13 = vget_high_s16(q5);

    q8 = vcvtq_f32_s32(vmovl_s16(d10));
    q9 = vcvtq_f32_s32(vmovl_s16(d11));
    q10 = vcvtq_f32_s32(vmovl_s16(d12));
    q11 = vcvtq_f32_s32(vmovl_s16(d13));

    q8 = exp_ps(vmulq_n_f32(q8, scale));
    q9 = exp_ps(vmulq_n_f32(q9, scale));
    q10 = exp_ps(vmulq_n_f32(q10, scale));
    q11 = exp_ps(vmulq_n_f32(q11, scale));

    q12 = vaddq_f32(q8, q9);
    q12 = vaddq_f32(q12, q10);
    q12 = vaddq_f32(q12, q11);
    q12 = vrecpeq_f32(q12);

    q8 = vmulq_f32(q12, q8);
    q9 = vmulq_f32(q12, q9);
    q10 = vmulq_f32(q12, q10);
    q11 = vmulq_f32(q12, q11);

    float32x4x4_t b1 = {q8, q9, q10, q11};
    vst4q_f32(output, b1);
    output += 16;

    input += 32;
  }
}

/*
 * Assume group is divided by 8
 */
static void softmax2_internal(const int8_t* input, float scale,
                              unsigned int group, float* output) {
  unsigned int batch = group / 8;

  for (unsigned int i = 0; i < batch; ++i) {
    /* Interleaved load 16 bytes into 2 NEON registers */
    int8x8x2_t q0 = vld2_s8(input);
    /* Convert to 16-bit integers */
    int16x8_t q1 = vmovl_s8(q0.val[0]);
    int16x8_t q2 = vmovl_s8(q0.val[1]);

    int16x4_t d2 = vget_low_s16(q1);
    int16x4_t d3 = vget_high_s16(q1);
    int16x4_t d4 = vget_low_s16(q2);
    int16x4_t d5 = vget_high_s16(q2);

    /* Process first 4 groups */
    float32x4_t q3 = vcvtq_f32_s32(vmovl_s16(d2));
    float32x4_t q4 = vcvtq_f32_s32(vmovl_s16(d4));
    q3 = exp_ps(vmulq_n_f32(q3, scale));
    q4 = exp_ps(vmulq_n_f32(q4, scale));

    float32x4_t q7 = vaddq_f32(q3, q4);
    q7 = vrecpeq_f32(q7);
    q3 = vmulq_f32(q7, q3);
    q4 = vmulq_f32(q7, q4);

    /* Process last 4 groups */
    float32x4_t q5 = vcvtq_f32_s32(vmovl_s16(d3));
    float32x4_t q6 = vcvtq_f32_s32(vmovl_s16(d5));
    q5 = exp_ps(vmulq_n_f32(q5, scale));
    q6 = exp_ps(vmulq_n_f32(q6, scale));

    float32x4_t q8 = vaddq_f32(q5, q6);
    q8 = vrecpeq_f32(q8);
    q5 = vmulq_f32(q8, q5);
    q6 = vmulq_f32(q8, q6);

    /* Save to memory */
    float32x4x2_t b0 = {q3, q4};
    vst2q_f32(output, b0);
    output += 8;
    float32x4x2_t b1 = {q5, q6};
    vst2q_f32(output, b1);
    output += 8;

    input += 16;
  }
}

float32x4x2_t softmax2_vector_o1_neon(int8x8_t d0, int8x8_t d1, float scale) {
  /* Convert to 16-bit integers */
  int16x8_t q1 = vmovl_s8(d0);
  int16x8_t q2 = vmovl_s8(d1);

  int16x4_t d2 = vget_low_s16(q1);
  int16x4_t d3 = vget_high_s16(q1);
  int16x4_t d4 = vget_low_s16(q2);
  int16x4_t d5 = vget_high_s16(q2);

  /* Process first 4 groups */
  float32x4_t q3 = vcvtq_f32_s32(vmovl_s16(d2));
  float32x4_t q4 = vcvtq_f32_s32(vmovl_s16(d4));
  q3 = exp_ps(vmulq_n_f32(q3, scale));
  q4 = exp_ps(vmulq_n_f32(q4, scale));

  q3 = vaddq_f32(q3, q4);
  q3 = vrecpeq_f32(q3);
  q4 = vmulq_f32(q3, q4);

  /* Process last 4 groups */
  float32x4_t q5 = vcvtq_f32_s32(vmovl_s16(d3));
  float32x4_t q6 = vcvtq_f32_s32(vmovl_s16(d5));
  q5 = exp_ps(vmulq_n_f32(q5, scale));
  q6 = exp_ps(vmulq_n_f32(q6, scale));

  q5 = vaddq_f32(q5, q6);
  q5 = vrecpeq_f32(q5);
  q6 = vmulq_f32(q5, q6);

  return {q4, q6};
}

/*
 * Assume group is divided by 8
 */
void softmax2_o1_internal(const int8_t* input, float scale, unsigned int group,
                          float* output) {
  unsigned int batch = group / 8;

  for (unsigned int i = 0; i < batch; ++i) {
    /* Interleaved load 16 bytes into 2 NEON registers */
    int8x8x2_t q0 = vld2_s8(input);
    /* Convert to 16-bit integers */
    int16x8_t q1 = vmovl_s8(q0.val[0]);
    int16x8_t q2 = vmovl_s8(q0.val[1]);

    int16x4_t d2 = vget_low_s16(q1);
    int16x4_t d3 = vget_high_s16(q1);
    int16x4_t d4 = vget_low_s16(q2);
    int16x4_t d5 = vget_high_s16(q2);

    /* Process first 4 groups */
    float32x4_t q3 = vcvtq_f32_s32(vmovl_s16(d2));
    float32x4_t q4 = vcvtq_f32_s32(vmovl_s16(d4));
    q3 = exp_ps(vmulq_n_f32(q3, scale));
    q4 = exp_ps(vmulq_n_f32(q4, scale));

    q3 = vaddq_f32(q3, q4);
    q3 = vrecpeq_f32(q3);
    q4 = vmulq_f32(q3, q4);

    /* Process last 4 groups */
    float32x4_t q5 = vcvtq_f32_s32(vmovl_s16(d3));
    float32x4_t q6 = vcvtq_f32_s32(vmovl_s16(d5));
    q5 = exp_ps(vmulq_n_f32(q5, scale));
    q6 = exp_ps(vmulq_n_f32(q6, scale));

    q5 = vaddq_f32(q5, q6);
    q5 = vrecpeq_f32(q5);
    q6 = vmulq_f32(q5, q6);

    /* Save to memory */
    vst1q_f32(output, q4);
    output += 4;
    vst1q_f32(output, q6);
    output += 4;

    input += 16;
  }
}
#endif
}  // namespace ai
}  // namespace vitis
