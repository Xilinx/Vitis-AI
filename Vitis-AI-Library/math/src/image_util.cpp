/*
 * Copyright 2019 Xilinx Inc.
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
#include <algorithm>
#include <cstdint>
#include <math.h>
#include <tuple>
#include <utility>
#include <vector>
using std::get;
using std::make_tuple;
using std::pair;
using std::tuple;
using std::vector;

namespace vitis {
namespace ai {

#ifdef ENABLE_NEON
void transform_bgr(int w, int h, unsigned char *src, signed char *dst,
                   float var_shift_B, float var_scale_B, float var_shift_G,
                   float var_scale_G, float var_shift_R, float var_scale_R);

void transform_rgb(int w, int h, unsigned char *src, signed char *dst,
                   float var_shift_B, float var_scale_B, float var_shift_G,
                   float var_scale_G, float var_shift_R, float var_scale_R);
#endif

void NormalizeInputData(const uint8_t *input, int rows, int cols, int channels,
                        int stride, const std::vector<float> &mean,
                        const std::vector<float> &scale, int8_t *data) {
#ifndef ENABLE_NEON
  //#if 1
  for (auto h = 0; h < rows; ++h) {
    for (auto w = 0; w < cols; ++w) {
      for (auto c = 0; c < channels; ++c) {
        auto value =
            (int)((input[h * stride + w * channels + c] * 1.0f - mean[c]) *
                  scale[c]);
        //(int)((input[h * cols * channels + w * channels + c] - mean[c]) *
        // scale[c]);
        data[h * cols * channels + w * channels + c] = (char)value;
      }
    }
  }
#else
  // for (auto i = 0; i < img.rows; ++i) {
  //  transform_bgr(cols, 1, img.data + i*img.step.buf[0], data + i*img.cols*3,
  //      mean[0], scale[0], mean[1], scale[1], mean[2], scale[2]);
  //}y
  for (auto i = 0; i < rows; ++i) {
    transform_bgr(cols, 1, const_cast<uint8_t *>(input) + i * stride,
                  data + i * cols * 3, mean[0], scale[0], mean[1], scale[1],
                  mean[2], scale[2]);
  }
#endif
}

#ifdef ENABLE_NEON
// caculate:
// b = (b-shiftB)*scaleB
// g = (g-shiftG)*scaleG
// r = (r-shiftR)*scaleR
void transform_bgr(int w, int h, unsigned char *src, signed char *dst,
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

void NormalizeInputData(const cv::Mat &img, const std::vector<float> &mean,
                        const std::vector<float> &scale, int8_t *data) {
  NormalizeInputData(img.data, img.rows, img.cols, img.channels(), img.step,
                     mean, scale, data);
}

void NormalizeInputDataRGB(const cv::Mat &img, const std::vector<float> &mean,
                           const std::vector<float> &scale, int8_t *data) {
  NormalizeInputDataRGB(img.data, img.rows, img.cols, img.channels(), img.step,
                        mean, scale, data);
}

void NormalizeInputDataRGB(const uint8_t *input, int rows, int cols,
                           int channels, int stride,
                           const std::vector<float> &mean,
                           const std::vector<float> &scale, int8_t *data) {

#ifndef ENABLE_NEON
  for (auto h = 0; h < rows; ++h) {
    for (auto w = 0; w < cols; ++w) {
      for (auto c = 0; c < channels; ++c) {
        auto value =
            (int)((input[h * stride + w * channels + c] * 1.0f - mean[c]) *
                  scale[c]);
        // Warning : Only support channels = 3
        data[h * cols * channels + w * channels + abs(c - 2)] = (char)value;
      }
    }
  }
#else
  for (auto i = 0; i < rows; ++i) {
    transform_rgb(cols, 1, const_cast<uint8_t *>(input) + i * stride,
                  data + i * cols * 3, mean[0], scale[0], mean[1], scale[1],
                  mean[2], scale[2]);
  }
#endif
}

#ifdef ENABLE_NEON
// caculate:
// b = (b-shiftB)*scaleB
// g = (g-shiftG)*scaleG
// r = (r-shiftR)*scaleR
void transform_rgb(int w, int h, unsigned char *src, signed char *dst,
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
}
} // namespace vitis
