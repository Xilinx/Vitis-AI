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

#include <cmath>
#include <iostream>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"

using namespace std;

namespace {
struct MyOp : public vart::experimental::OpImpBase {
  MyOp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    auto input_op = op->get_input_op("input", 0);
    auto input_shape = input_op->get_output_tensor()->get_shape();
    auto output_shape = op->get_output_tensor()->get_shape();
    CHECK_EQ(input_shape.size(), 4u);
    CHECK_EQ(input_shape.size(), output_shape.size());
    CHECK_EQ(input_shape[0], output_shape[0]);
    batch_ = input_shape[0];
    // CHECK_EQ(input_shape[3], 3);
    channel_ = input_shape[3];
    CHECK_EQ(input_shape[3], output_shape[3]);
    input_h_ = input_shape[1];
    input_w_ = input_shape[2];
    output_h_ = output_shape[1];
    output_w_ = output_shape[2];
    mode_ = op->get_attr<std::string>("mode");
    align_corners_ = op->get_attr<bool>("align_corners");
    half_pixel_centers_ = op->get_attr<bool>("half_pixel_centers");
    auto input_fix_pos =
        input_op->get_output_tensor()->get_attr<int>("fix_point");
    auto output_fix_pos = op->get_output_tensor()->get_attr<int>("fix_point");
    int shift_fix_pos_ = output_fix_pos - input_fix_pos;
    shift_scale_ = pow(2, shift_fix_pos_);
  };

  struct CachedInterpolation {
    int lower;
    int upper;
    float lerp;
  };
  void upsample() {
    auto cal_scale = [](std::int32_t in, std::int32_t out,
                        bool align_corners) -> float {
      return (align_corners && out > 1) ? (in - 1) / static_cast<float>(out - 1)
                                        : in / static_cast<float>(out);
    };
    auto scaler = [](std::int32_t out, float scale,
                     bool half_pixel_centers) -> float {
      return (half_pixel_centers)
                 ? (static_cast<float>(out) + 0.5f) * scale - 0.5f
                 : static_cast<float>(out) * scale;
    };
    auto compute_interpolation_weights =
        [&](const int out_size, const int in_size, const float scale,
            CachedInterpolation* interpolation) {
          interpolation[out_size].lower = 0;
          interpolation[out_size].upper = 0;
          for (int i = out_size - 1; i >= 0; --i) {
            const float in = scaler(i, scale, half_pixel_centers_);
            interpolation[i].lower =
                std::max(static_cast<int>(std::floor(in)), static_cast<int>(0));
            interpolation[i].upper = std::min(static_cast<int>(std::ceil(in)),
                                              static_cast<int>(in_size - 1));
            interpolation[i].lerp = in - interpolation[i].lower;
          }
        };
    if (mode_ == "NEAREST") {
      auto h_scale = cal_scale(input_h_, output_h_, align_corners_);
      auto w_scale = cal_scale(input_w_, output_w_, align_corners_);
      for (int n = 0; n < batch_; n++)
        for (int h = 0; h < output_h_; h++)
          for (int w = 0; w < output_w_; w++)
            for (int c = 0; c < channel_; c++) {
              auto idx = ((n * output_h_ + h) * output_w_ + w) * channel_ + c;
              auto h_scaler = scaler(h, h_scale, half_pixel_centers_);
              auto w_scaler = scaler(w, w_scale, half_pixel_centers_);
              auto h_idx = std::min(
                  (align_corners_) ? static_cast<int>(std::round(h_scaler))
                                   : static_cast<int>(std::floor(h_scaler)),
                  input_h_ - 1);
              auto w_idx = std::min(
                  (align_corners_) ? static_cast<int>(std::round(w_scaler))
                                   : static_cast<int>(std::floor(w_scaler)),
                  input_w_ - 1);
              auto i_idx =
                  ((n * input_h_ + h_idx) * input_w_ + w_idx) * channel_ + c;
              output_f_[idx] = data_in_ptr_[i_idx];
            }
    } else if (mode_ == "BILINEAR") {
      std::vector<CachedInterpolation> xs(output_w_ + 1);
      std::vector<CachedInterpolation> ys(output_h_ + 1);
      auto h_scale = cal_scale(input_h_, output_h_, align_corners_);
      auto w_scale = cal_scale(input_w_, output_w_, align_corners_);
      compute_interpolation_weights(output_h_, input_h_, h_scale, ys.data());
      compute_interpolation_weights(output_w_, input_w_, w_scale, xs.data());
      for (uint i = 0; i < xs.size(); ++i) {
        xs[i].lower *= channel_;
        xs[i].upper *= channel_;
      }
      const int in_row_size = input_w_ * channel_;
      const int in_batch_num_values = input_h_ * in_row_size;
      const int out_row_size = output_w_ * channel_;

      auto compute_lerp = [&](const float top_left, const float top_right,
                              const float bottom_left, const float bottom_right,
                              const float x_lerp, const float y_lerp) -> float {
        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom =
            bottom_left + (bottom_right - bottom_left) * x_lerp;
        return top + (bottom - top) * y_lerp;
      };
      int8_t* in_ptr = data_in_ptr_;
      int idx_start = 0;
      for (int b = 0; b < batch_; ++b) {
        for (int y = 0; y < output_h_; ++y) {
          const int8_t* ys_input_lower_ptr = in_ptr + ys[y].lower * in_row_size;
          const int8_t* ys_input_upper_ptr = in_ptr + ys[y].upper * in_row_size;
          const float ys_lerp = ys[y].lerp;
          for (int x = 0; x < output_w_; ++x) {
            auto xs_lower = xs[x].lower;
            auto xs_upper = xs[x].upper;
            auto xs_lerp = xs[x].lerp;
            for (int c = 0; c < channel_; ++c) {
              const float top_left(ys_input_lower_ptr[xs_lower + c]);
              const float top_right(ys_input_lower_ptr[xs_upper + c]);
              const float bottom_left(ys_input_upper_ptr[xs_lower + c]);
              const float bottom_right(ys_input_upper_ptr[xs_upper + c]);
              output_f_[x * channel_ + c + idx_start] =
                  compute_lerp(top_left, top_right, bottom_left, bottom_right,
                               xs_lerp, ys_lerp);
            }
          }
          idx_start += out_row_size;
        }
        in_ptr += in_batch_num_values;
      }
    } else {
      LOG(FATAL) << "unkonwn mode = " << mode_;
    }
  };
  int8_t fix(float a, float b) {
    auto data = a * b;
    auto data_max = 127.0;
    auto data_min = -128.0;
    if (data > data_max) {
      data = data_max;
    } else if (data < data_min) {
      data = data_min;
    } else if (data < 0 && (data - floor(data)) == 0.5) {
      data = static_cast<float>(ceil(data));
    } else {
      data = static_cast<float>(round(data));
    }
    return (int8_t)data;
  };

  int calculate(vart::simple_tensor_buffer_t<int8_t> output,
                vart::simple_tensor_buffer_t<int8_t> input,
                std::unique_ptr<vart::simple_tensor_buffer_t<int8_t>> size) {
    CHECK(size == nullptr) << "not supported yet";
    for (auto b = 0; b < batch_; b++) {
      data_in_ptr_ = &input.data[b * input_h_ * input_w_ * channel_];
      data_out_ptr_ = &output.data[b * output_h_ * output_w_ * channel_];
      size_t output_size = output_h_ * output_w_ * channel_;
      output_f_.resize(output_size);
      upsample();
      for (auto i = 0u; i < output_size; ++i) {
        data_out_ptr_[i] = fix(output_f_[i], shift_scale_);
      }
    }
    return 0;
  };

  int batch_;
  int input_w_;
  int input_h_;
  int output_w_;
  int output_h_;
  int channel_;
  std::string mode_;
  bool align_corners_;
  bool half_pixel_centers_;
  int8_t* data_in_ptr_{nullptr};
  std::vector<float> output_f_;
  int8_t* data_out_ptr_{nullptr};
  float shift_scale_;
};  // namespace

}  // namespace

DEF_XIR_OP_IMP(MyOp)
