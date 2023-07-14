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
#include "vitis/ai/image_util.hpp"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include <math.h>

#include <UniLog/UniLog.hpp>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <tuple>
#include <utility>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
DEF_ENV_PARAM(USING_OLD_NEON, "1");
DEF_ENV_PARAM(DEBUG_IMAGE_UTIL, "0");
DEF_ENV_PARAM(XLNX_ENABLE_ROUND_SETINPUT, "0");

using std::get;
using std::make_tuple;
using std::pair;
using std::tuple;
using std::vector;
int GLOBAL_ENABLE_ROUND_SETINPUT = 0;

namespace vitis {
namespace ai {

#ifdef ENABLE_NEON
void transform_bgr(int w, int h, unsigned char* src, signed char* dst,
                   float var_shift_B, float var_scale_B, float var_shift_G,
                   float var_scale_G, float var_shift_R, float var_scale_R);

void transform_rgb(int w, int h, unsigned char* src, signed char* dst,
                   float var_shift_B, float var_scale_B, float var_shift_G,
                   float var_scale_G, float var_shift_R, float var_scale_R);

bool calc_scale(const std::vector<float>& src, std::vector<int>& dst);

void transform_mean_scale(int w, int h, const uint8_t* src, int8_t* dst,
                          const std::vector<float>& mean,
                          const std::vector<int>& scale, bool is_dst_bgr);
#endif
void NormalizeInputData(const float* input, int rows, int cols, int channels,
                        int stride, const std::vector<float>& mean,
                        const std::vector<float>& scale, int8_t* data) {
  // std::ofstream output_file("final.txt");
  for (auto h = 0; h < rows; ++h) {
    for (auto w = 0; w < cols; ++w) {
      for (auto c = 0; c < channels; ++c) {
        auto value = (int)round(
            (input[h * stride + w * channels + c] * 1.0f - mean[c]) * scale[c]);
        value = std::max(-128, value);
        value = std::min(127, value);
        //     output_file << ((input[h * stride + w * channels + c] * 1.0f -
        //     mean[c]) * scale[c]) << std::endl;
        //(int)((input[h * cols * channels + w * channels + c] - mean[c]) *
        // scale[c]);
        data[h * cols * channels + w * channels + c] = (char)value;
      }
    }
  }
}

void NormalizeInputData(const uint8_t* input, int rows, int cols, int channels,
                        int stride, const std::vector<float>& mean,
                        const std::vector<float>& scale, int8_t* data) {
  if (ENV_PARAM(XLNX_ENABLE_ROUND_SETINPUT) == 1) {
    GLOBAL_ENABLE_ROUND_SETINPUT = 1;
  }
#ifndef ENABLE_NEON
  if (GLOBAL_ENABLE_ROUND_SETINPUT == 0) {
    for (auto h = 0; h < rows; ++h) {
      for (auto w = 0; w < cols; ++w) {
        for (auto c = 0; c < channels; ++c) {
          auto value =
              (int)((input[h * stride + w * channels + c] * 1.0f - mean[c]) *
                    scale[c]);
          data[h * cols * channels + w * channels + c] = (char)value;
          // IMPORTANT: we must keep this loop as simple as possible so
          // that gcc can optimize it in the release version
#ifndef NDEBUG
          LOG_IF(INFO, ENV_PARAM(DEBUG_IMAGE_UTIL) &&
                           h * cols * channels + w * channels + c < 10)  //
              << "value " << value << " "                                //
              << "mean[c] " << mean[c] << " "                            //
              << "scale[c] " << scale[c] << " "                          //
              << "input: " << (input[h * stride + w * channels + c] * 1.0f)
              << " ";
#endif
        }
      }
    }
  } else {
    for (auto h = 0; h < rows; ++h) {
      for (auto w = 0; w < cols; ++w) {
        for (auto c = 0; c < channels; ++c) {
          auto value = (int)round(
              (input[h * stride + w * channels + c] * 1.0f - mean[c]) *
              scale[c]);
          value = std::max(-128, value);
          value = std::min(127, value);
          data[h * cols * channels + w * channels + c] = (char)value;
          LOG_IF(INFO, ENV_PARAM(DEBUG_IMAGE_UTIL) &&
                           h * cols * channels + w * channels + c < 10)  //
              << "value " << value << " "                                //
              << "mean[c] " << mean[c] << " "                            //
              << "scale[c] " << scale[c] << " "                          //
              << "input: " << (input[h * stride + w * channels + c] * 1.0f)
              << " ";
        }
      }
    }
  }
#endif
#ifdef ENABLE_NEON
  if (GLOBAL_ENABLE_ROUND_SETINPUT == 0) {
    std::vector<int> scale_int;
    if (!ENV_PARAM(USING_OLD_NEON) && cols >= 16 &&
        calc_scale(scale, scale_int)) {
      transform_mean_scale(cols, rows, input, data, mean, scale_int, true);
    } else {
      for (auto i = 0; i < rows; ++i) {
        transform_bgr(cols, 1, const_cast<uint8_t*>(input) + i * stride,
                      data + i * cols * 3, mean[0], scale[0], mean[1], scale[1],
                      mean[2], scale[2]);
      }
    }
  } else {
    for (auto h = 0; h < rows; ++h) {
      for (auto w = 0; w < cols; ++w) {
        for (auto c = 0; c < channels; ++c) {
          auto value = (int)round(
              (input[h * stride + w * channels + c] * 1.0f - mean[c]) *
              scale[c]);
          value = std::max(-128, value);
          value = std::min(127, value);
          data[h * cols * channels + w * channels + c] = (char)value;
          LOG_IF(INFO, ENV_PARAM(DEBUG_IMAGE_UTIL) &&
                           h * cols * channels + w * channels + c < 10)  //
              << "value " << value << " "                                //
              << "mean[c] " << mean[c] << " "                            //
              << "scale[c] " << scale[c] << " "                          //
              << "input: " << (input[h * stride + w * channels + c] * 1.0f)
              << " ";
        }
      }
    }
  }
#endif
}

//# Method for float input data used for DPUV1 and NCHW format
void NormalizeInputData(const uint8_t* input, int rows, int cols, int channels,
                        int stride, const std::vector<float>& mean,
                        const std::vector<float>& scale, float* data) {
  for (auto c = 0; c < channels; c++) {
    for (auto h = 0; h < rows; h++) {
      for (auto w = 0; w < cols; w++) {
        float value =
            (float)((input[h * cols * channels + w * channels + c] * 1.0f -
                     mean[c]) *
                    scale[c]);
        data[(c * rows * cols) + (h * cols) + w] = (float)value;
      }
    }
  }
}

#ifdef ENABLE_NEON
// caculate:
// b = (b-shiftB)*scaleB
// g = (g-shiftG)*scaleG
// r = (r-shiftR)*scaleR
void transform_bgr(int w, int h, unsigned char* src, signed char* dst,
                   float var_shift_B, float var_scale_B, float var_shift_G,
                   float var_scale_G, float var_shift_R, float var_scale_R) {
  float32x4_t shiftB = vdupq_n_f32(var_shift_B);
  float32x4_t shiftG = vdupq_n_f32(var_shift_G);
  float32x4_t shiftR = vdupq_n_f32(var_shift_R);

  float32x4_t scaleB = vdupq_n_f32(var_scale_B);
  float32x4_t scaleG = vdupq_n_f32(var_scale_G);
  float32x4_t scaleR = vdupq_n_f32(var_scale_R);

  for (int i = 0; i < h; i++) {
    int idx_base = i * w * 3;
    for (int j = 0; j < w; j += 8) {
      int idx = idx_base + j * 3;

      // init
      uint8x8x3_t sbgr_u8;
      uint16x8x3_t sbgr_u16;
      sbgr_u8 = vld3_u8(src + idx);
      sbgr_u16.val[0] = vmovl_u8(sbgr_u8.val[0]);
      sbgr_u16.val[1] = vmovl_u8(sbgr_u8.val[1]);
      sbgr_u16.val[2] = vmovl_u8(sbgr_u8.val[2]);

      // get low part u32
      uint32x4_t sb_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[0]));
      uint32x4_t sg_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[1]));
      uint32x4_t sr_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[2]));

      // get high part u32
      uint32x4_t sb_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[0]));
      uint32x4_t sg_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[1]));
      uint32x4_t sr_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[2]));

      // get low part float
      float32x4_t sb_low_f32 = vcvtq_f32_u32(sb_low_u32);
      float32x4_t sg_low_f32 = vcvtq_f32_u32(sg_low_u32);
      float32x4_t sr_low_f32 = vcvtq_f32_u32(sr_low_u32);

      // get high part float
      float32x4_t sb_high_f32 = vcvtq_f32_u32(sb_high_u32);
      float32x4_t sg_high_f32 = vcvtq_f32_u32(sg_high_u32);
      float32x4_t sr_high_f32 = vcvtq_f32_u32(sr_high_u32);

      // calculate low part float
      sb_low_f32 = vmulq_f32(vsubq_f32(sb_low_f32, shiftB), scaleB);
      sg_low_f32 = vmulq_f32(vsubq_f32(sg_low_f32, shiftG), scaleG);
      sr_low_f32 = vmulq_f32(vsubq_f32(sr_low_f32, shiftR), scaleR);

      // calculate low part float
      sb_high_f32 = vmulq_f32(vsubq_f32(sb_high_f32, shiftB), scaleB);
      sg_high_f32 = vmulq_f32(vsubq_f32(sg_high_f32, shiftG), scaleG);
      sr_high_f32 = vmulq_f32(vsubq_f32(sr_high_f32, shiftR), scaleR);

      // get the result low part int32
      int32x4_t db_low_s32 = vcvtq_s32_f32(sb_low_f32);
      int32x4_t dg_low_s32 = vcvtq_s32_f32(sg_low_f32);
      int32x4_t dr_low_s32 = vcvtq_s32_f32(sr_low_f32);

      // get the result high part int32
      int32x4_t db_high_s32 = vcvtq_s32_f32(sb_high_f32);
      int32x4_t dg_high_s32 = vcvtq_s32_f32(sg_high_f32);
      int32x4_t dr_high_s32 = vcvtq_s32_f32(sr_high_f32);

      // get the result low part int16
      int16x4_t db_low_s16 = vmovn_s32(db_low_s32);
      int16x4_t dg_low_s16 = vmovn_s32(dg_low_s32);
      int16x4_t dr_low_s16 = vmovn_s32(dr_low_s32);

      // get the result high part int16
      int16x4_t db_high_s16 = vmovn_s32(db_high_s32);
      int16x4_t dg_high_s16 = vmovn_s32(dg_high_s32);
      int16x4_t dr_high_s16 = vmovn_s32(dr_high_s32);

      // combine low and high into int16x8
      int16x8_t db_s16 = vcombine_s16(db_low_s16, db_high_s16);
      int16x8_t dg_s16 = vcombine_s16(dg_low_s16, dg_high_s16);
      int16x8_t dr_s16 = vcombine_s16(dr_low_s16, dr_high_s16);

      // combine low and high into int16x8
      int8x8x3_t dbgr;
      dbgr.val[0] = vmovn_s16(db_s16);
      dbgr.val[1] = vmovn_s16(dg_s16);
      dbgr.val[2] = vmovn_s16(dr_s16);

      // store...
      vst3_s8(dst + idx, dbgr);
    }
  }
}
#endif

void NormalizeInputData(const cv::Mat& img, const std::vector<float>& mean,
                        const std::vector<float>& scale, int8_t* data) {
  NormalizeInputData(img.data, img.rows, img.cols, img.channels(), img.step,
                     mean, scale, data);
}

void NormalizeInputDataRGB(const cv::Mat& img, const std::vector<float>& mean,
                           const std::vector<float>& scale, int8_t* data) {
  NormalizeInputDataRGB(img.data, img.rows, img.cols, img.channels(), img.step,
                        mean, scale, data);
}

void NormalizeInputDataRGB(const cv::Mat& img, const std::vector<float>& mean,
                           const std::vector<float>& scale, float* data) {
  NormalizeInputDataRGB(img.data, img.rows, img.cols, img.channels(), img.step,
                        mean, scale, data);
}

//# Method for float input data used for DPUV1 and NCHW format
void NormalizeInputDataRGB(const uint8_t* input, int rows, int cols,
                           int channels, int stride,
                           const std::vector<float>& mean,
                           const std::vector<float>& scale, float* data) {
  for (auto c = 0; c < channels; c++) {
    for (auto h = 0; h < rows; h++) {
      for (auto w = 0; w < cols; w++) {
        float value =
            (float)((input[h * cols * channels + w * channels + c] * 1.0f -
                     mean[c]) *
                    scale[c]);
        data[(abs(c - 2) * rows * cols) + (h * cols) + w] = (float)value;
      }
    }
  }
}

void NormalizeInputDataRGB(const uint8_t* input, int rows, int cols,
                           int channels, int stride,
                           const std::vector<float>& mean,
                           const std::vector<float>& scale, int8_t* data) {
  if (ENV_PARAM(XLNX_ENABLE_ROUND_SETINPUT) == 1) {
    GLOBAL_ENABLE_ROUND_SETINPUT = 1;
  }
#ifndef ENABLE_NEON
  for (auto h = 0; h < rows; ++h) {
    for (auto w = 0; w < cols; ++w) {
      for (auto c = 0; c < channels; ++c) {
        // Warning : Only support channels = 3
        // HarDNet-MSEG model failed to compare golden data, adjust this
        // to rounding
        // The code in the comment will bring performance loss, do not use
        // round when publishing
        // auto value = (int)round(
        //    (input[h * stride + w * channels + c] * 1.0f - mean[c]) *
        //    scale[c]);
        // value = std::max(-128, value);
        // value = std::min(127, value);
        int value = 0;
        if (GLOBAL_ENABLE_ROUND_SETINPUT != 0) {
          value = (int)std::round(
              ((input[h * stride + w * channels + c] * 1.0f - mean[c]) *
               scale[c]));
        } else {
          value =
              (int)((input[h * stride + w * channels + c] * 1.0f - mean[c]) *
                    scale[c]);
        }
        data[h * cols * channels + w * channels + abs(c - 2)] = (char)value;
      }
    }
  }
#else
  if (GLOBAL_ENABLE_ROUND_SETINPUT == 0) {
    std::vector<int> scale_int;
    if (!ENV_PARAM(USING_OLD_NEON) && cols >= 16 &&
        calc_scale(scale, scale_int)) {
      transform_mean_scale(cols, rows, input, data, mean, scale_int, false);
    } else {
      for (auto i = 0; i < rows; ++i) {
        transform_rgb(cols, 1, const_cast<uint8_t*>(input) + i * stride,
                      data + i * cols * 3, mean[0], scale[0], mean[1], scale[1],
                      mean[2], scale[2]);
      }
    }
  } else {
    for (auto h = 0; h < rows; ++h) {
      for (auto w = 0; w < cols; ++w) {
        for (auto c = 0; c < channels; ++c) {
          // Warning : Only support channels = 3
          // HarDNet-MSEG model failed to compare golden data, adjust this
          // to rounding
          // The code in the comment will bring performance loss, do not use
          // round when publishing
          // auto value = (int)round(
          //    (input[h * stride + w * channels + c] * 1.0f - mean[c]) *
          //    scale[c]);
          // value = std::max(-128, value);
          // value = std::min(127, value);
          auto value = (int)std::round(
              ((input[h * stride + w * channels + c] * 1.0f - mean[c]) *
               scale[c]));
          data[h * cols * channels + w * channels + abs(c - 2)] = (char)value;
        }
      }
    }
  }
#endif
}

#ifdef ENABLE_NEON
// caculate:
// b = (b-shiftB)*scaleB
// g = (g-shiftG)*scaleG
// r = (r-shiftR)*scaleR
void transform_rgb(int w, int h, unsigned char* src, signed char* dst,
                   float var_shift_B, float var_scale_B, float var_shift_G,
                   float var_scale_G, float var_shift_R, float var_scale_R) {
  float32x4_t shiftB = vdupq_n_f32(var_shift_B);
  float32x4_t shiftG = vdupq_n_f32(var_shift_G);
  float32x4_t shiftR = vdupq_n_f32(var_shift_R);

  float32x4_t scaleB = vdupq_n_f32(var_scale_B);
  float32x4_t scaleG = vdupq_n_f32(var_scale_G);
  float32x4_t scaleR = vdupq_n_f32(var_scale_R);

  for (int i = 0; i < h; i++) {
    int idx_base = i * w * 3;
    for (int j = 0; j < w; j += 8) {
      int idx = idx_base + j * 3;

      // init
      uint8x8x3_t sbgr_u8;
      uint16x8x3_t sbgr_u16;
      sbgr_u8 = vld3_u8(src + idx);
      sbgr_u16.val[0] = vmovl_u8(sbgr_u8.val[0]);
      sbgr_u16.val[1] = vmovl_u8(sbgr_u8.val[1]);
      sbgr_u16.val[2] = vmovl_u8(sbgr_u8.val[2]);

      // get low part u32
      uint32x4_t sb_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[0]));
      uint32x4_t sg_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[1]));
      uint32x4_t sr_low_u32 = vmovl_u16(vget_low_u16(sbgr_u16.val[2]));

      // get high part u32
      uint32x4_t sb_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[0]));
      uint32x4_t sg_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[1]));
      uint32x4_t sr_high_u32 = vmovl_u16(vget_high_u16(sbgr_u16.val[2]));

      // get low part float
      float32x4_t sb_low_f32 = vcvtq_f32_u32(sb_low_u32);
      float32x4_t sg_low_f32 = vcvtq_f32_u32(sg_low_u32);
      float32x4_t sr_low_f32 = vcvtq_f32_u32(sr_low_u32);

      // get high part float
      float32x4_t sb_high_f32 = vcvtq_f32_u32(sb_high_u32);
      float32x4_t sg_high_f32 = vcvtq_f32_u32(sg_high_u32);
      float32x4_t sr_high_f32 = vcvtq_f32_u32(sr_high_u32);

      // calculate low part float
      sb_low_f32 = vmulq_f32(vsubq_f32(sb_low_f32, shiftB), scaleB);
      sg_low_f32 = vmulq_f32(vsubq_f32(sg_low_f32, shiftG), scaleG);
      sr_low_f32 = vmulq_f32(vsubq_f32(sr_low_f32, shiftR), scaleR);

      // calculate low part float
      sb_high_f32 = vmulq_f32(vsubq_f32(sb_high_f32, shiftB), scaleB);
      sg_high_f32 = vmulq_f32(vsubq_f32(sg_high_f32, shiftG), scaleG);
      sr_high_f32 = vmulq_f32(vsubq_f32(sr_high_f32, shiftR), scaleR);

      // get the result low part int32
      int32x4_t db_low_s32 = vcvtq_s32_f32(sb_low_f32);
      int32x4_t dg_low_s32 = vcvtq_s32_f32(sg_low_f32);
      int32x4_t dr_low_s32 = vcvtq_s32_f32(sr_low_f32);

      // get the result high part int32
      int32x4_t db_high_s32 = vcvtq_s32_f32(sb_high_f32);
      int32x4_t dg_high_s32 = vcvtq_s32_f32(sg_high_f32);
      int32x4_t dr_high_s32 = vcvtq_s32_f32(sr_high_f32);

      // get the result low part int16
      int16x4_t db_low_s16 = vmovn_s32(db_low_s32);
      int16x4_t dg_low_s16 = vmovn_s32(dg_low_s32);
      int16x4_t dr_low_s16 = vmovn_s32(dr_low_s32);

      // get the result high part int16
      int16x4_t db_high_s16 = vmovn_s32(db_high_s32);
      int16x4_t dg_high_s16 = vmovn_s32(dg_high_s32);
      int16x4_t dr_high_s16 = vmovn_s32(dr_high_s32);

      // combine low and high into int16x8
      int16x8_t db_s16 = vcombine_s16(db_low_s16, db_high_s16);
      int16x8_t dg_s16 = vcombine_s16(dg_low_s16, dg_high_s16);
      int16x8_t dr_s16 = vcombine_s16(dr_low_s16, dr_high_s16);

      // combine low and high into int16x8
      int8x8x3_t dbgr;
      dbgr.val[2] = vmovn_s16(db_s16);
      dbgr.val[1] = vmovn_s16(dg_s16);
      dbgr.val[0] = vmovn_s16(dr_s16);

      // store...
      vst3_s8(dst + idx, dbgr);
    }
  }
}
#endif

#ifdef ENABLE_NEON

bool calc_scale(const std::vector<float>& src, std::vector<int>& dst) {
  for (auto s : src) {
    if (s == 1) {
      dst.push_back(0);
    } else if (s == 0.5) {
      dst.push_back(-1);
    } else if (s == 0.25) {
      dst.push_back(-2);
    } else if (s == 0.125) {
      dst.push_back(-3);
    } else if (s == 0.0625) {
      dst.push_back(-4);
    } else if (s == 0.03125) {
      dst.push_back(-5);
    } else if (s == 0.015625) {
      dst.push_back(-6);
    } else if (s == 0.0078125) {
      dst.push_back(-7);
    } else if (s == 0.00390625) {
      dst.push_back(-8);
    } else {
      return false;
    }
  }
  return true;
}

// caculate:
// dst = (src-mean)*scale
void transform_mean_scale(int w, int h, const uint8_t* src, int8_t* dst,
                          const std::vector<float>& mean,
                          const std::vector<int>& scale,
                          bool is_dst_bgr = true) {
  auto& scale_B = scale[0];
  auto& scale_G = scale[1];
  auto& scale_R = scale[2];

  unsigned char low_shift_B = mean[0];
  unsigned char high_shift_B = mean[0] + 0.999;
  uint8x16_t low_mean_B = vdupq_n_u8(low_shift_B);
  uint8x16_t high_mean_B = vdupq_n_u8(high_shift_B);

  unsigned char low_shift_G = mean[1];
  unsigned char high_shift_G = mean[1] + 0.999;
  uint8x16_t low_mean_G = vdupq_n_u8(low_shift_G);
  uint8x16_t high_mean_G = vdupq_n_u8(high_shift_G);

  unsigned char low_shift_R = mean[2];
  unsigned char high_shift_R = mean[2] + 0.999;
  uint8x16_t low_mean_R = vdupq_n_u8(low_shift_R);
  uint8x16_t high_mean_R = vdupq_n_u8(high_shift_R);
  auto calc_float_mean = [](const uint8x16_t& in_u8, int scale_,
                            const uint8x16_t& high_mean,
                            const uint8x16_t& low_mean) -> int8x16_t {
    if (scale_ > 0) {
      // scale > 0
      // LOG(FATAL) << "scale > 0 not implementation";
      UNI_LOG_FATAL(VAILIB_MATH_NOT_SUPPORT) << "scale_ > 0 not implementation";
      // temp = vshlq_n_u8(vminq_u8(vabdq_u8(in_u8, high_mean),
      // vabdq_u8(in_u8, low_mean)), scale);
      return int8x16_t();
    } else if (scale_ < 0) {
      // scale < 0
      return vreinterpretq_s8_u8(
          vsubq_u8(vshrq_n_u8(vqsubq_u8(in_u8, high_mean), -scale_),
                   vshrq_n_u8(vqsubq_u8(low_mean, in_u8), -scale_)));
    }
    // scale =0
    return vreinterpretq_s8_u8(
        vsubq_u8(vqsubq_u8(in_u8, high_mean), vqsubq_u8(low_mean, in_u8)));
  };
  auto calc_int_mean = [](const uint8x16_t& in_u8, int scale_,
                          const uint8x16_t& mean_) -> int8x16_t {
    if (scale_ > 0) {
      // scale >0
      // LOG(FATAL) << "scale > 0 not implementation";
      UNI_LOG_FATAL(VAILIB_MATH_NOT_SUPPORT) << "scale_ > 0 not implementation";
      return int8x16_t();
      // temp = vshlq_n_u8(vabdq_u8(in_u8, mean), scale);
    } else if (scale_ < 0) {
      // scale < 0
      return vreinterpretq_s8_u8(
          vsubq_u8(vshrq_n_u8(vqsubq_u8(in_u8, mean_), -scale_),
                   vshrq_n_u8(vqsubq_u8(mean_, in_u8), -scale_)));
    }
    // scale =0
    return vreinterpretq_s8_u8(
        vsubq_u8(vqsubq_u8(in_u8, mean_), vqsubq_u8(mean_, in_u8)));
  };
  uint8x16x3_t sbgr_u8;
  uint8x16_t& sb_u8 = sbgr_u8.val[0];
  uint8x16_t& sg_u8 = sbgr_u8.val[1];
  uint8x16_t& sr_u8 = sbgr_u8.val[2];
  int8x16_t res_b;
  int8x16_t res_g;
  int8x16_t res_r;
  int8x16x3_t res;
  std::vector<int> num_i;
  for (int i = 0; i < w / 16; i++) {
    num_i.push_back(i * 16);
  }
  if (w % 16 != 0) {
    num_i.push_back(w - 16);
  }
  int idx;
  for (auto j = 0; j < h; ++j) {
    for (auto& i : num_i) {
      idx = j * 3 * w + i * 3;
      sbgr_u8 = vld3q_u8(src + idx);
      if (high_shift_B == low_shift_B) {
        res_b = calc_int_mean(sb_u8, scale_B, low_mean_B);
      } else {
        res_b = calc_float_mean(sb_u8, scale_B, high_mean_B, low_mean_B);
      }
      if (high_shift_G == low_shift_G) {
        res_g = calc_int_mean(sg_u8, scale_G, low_mean_G);
      } else {
        res_g = calc_float_mean(sg_u8, scale_G, high_mean_G, low_mean_G);
      }
      if (high_shift_R == low_shift_R) {
        res_r = calc_int_mean(sr_u8, scale_R, low_mean_R);
      } else {
        res_r = calc_float_mean(sr_u8, scale_R, high_mean_R, low_mean_R);
      }
      if (is_dst_bgr) {
        // dst is bgr
        res.val[0] = res_b;
        res.val[1] = res_g;
        res.val[2] = res_r;
      } else {
        // dst is rgb
        res.val[0] = res_r;
        res.val[1] = res_g;
        res.val[2] = res_b;
      }
      vst3q_s8(dst + idx, res);
    }
  }
}
#endif

}  // namespace ai
}  // namespace vitis
