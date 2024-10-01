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

#include "../common/util.hpp"
#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"

using namespace std;

namespace {
typedef struct Pad {
  int pad_l;
  int pad_r;
  int pad_t;
  int pad_b;

  Pad(int l = 0, int r = 0, int t = 0, int b = 0)
      : pad_l(l), pad_r(r), pad_t(t), pad_b(b) {}
} pad_t;
typedef struct Dilation {
  int dilation_w;
  int dilation_h;

  Dilation(int w = 1, int h = 1) : dilation_w(w), dilation_h(h) {}
} dilation_t;
typedef struct Strides {
  int stride_w;
  int stride_h;

  Strides(int w = 1, int h = 1) : stride_w(w), stride_h(h) {}
} stride_t;

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    // pad attr
    pad_mode_ = op->get_attr<string>("pad_mode");
    if (op->has_attr("pad")) {
      auto pad = op->get_attr<std::vector<int>>("pad");
      CHECK_EQ(pad.size(), 4u);
      pad_ = pad_t{pad[0], pad[1], pad[2], pad[3]};
    }

    // dilation attr
    if (op->has_attr("dilation")) {
      auto dilation = op->get_attr<std::vector<int>>("dilation");
      CHECK_EQ(dilation.size(), 2u);
      dilation_ = dilation_t{dilation[0], dilation[1]};
    }

    // nonlinear attr
    nonlinear_ = vitis::ai::cpu_task::util::get_nonlinear(
        op->get_attr<string>("nonlinear"));

    // stride attr
    auto stride = op->get_attr<std::vector<int>>("stride");
    CHECK_EQ(stride.size(), 2u);
    stride_ = stride_t{stride[0], stride[1]};
  };
  int calculate(vart::simple_tensor_buffer_t<int8_t> output,
                vart::simple_tensor_buffer_t<int8_t> input,
                vart::simple_tensor_buffer_t<int8_t> weight,
                std::unique_ptr<vart::simple_tensor_buffer_t<int8_t>> bias) {
    // input info
    auto input_input_tensor = input.tensor;
    auto input_data_ptr = input.data;
    input_shape_ = input.tensor->get_shape();
    input_shape_size_vec_ =
        vitis::ai::cpu_task::util::get_dim_stride_vec(input_shape_);

    // weight info
    auto input_weight_tensor = weight.tensor;
    auto weight_data_ptr = weight.data;
    weight_shape_ = weight.tensor->get_shape();
    weight_shape_size_vec_ =
        vitis::ai::cpu_task::util::get_dim_stride_vec(weight_shape_);

    // output info
    auto output_data_ptr = output.data;
    output_shape_ = output.tensor->get_shape();
    output_shape_size_vec_ =
        vitis::ai::cpu_task::util::get_dim_stride_vec(output_shape_);

    CHECK_EQ(input_shape_.size(), 4u);
    CHECK_EQ(input_shape_.size(), output_shape_.size());
    CHECK_EQ(input_shape_.size(), weight_shape_.size());
    CHECK_EQ(input_shape_[3], weight_shape_[3]);
    CHECK_EQ(output_shape_[3], weight_shape_[0]);

    // fix point
    auto input_fp = input_input_tensor->get_attr<int>("fix_point");
    auto weight_fp = input_weight_tensor->get_attr<int>("fix_point");
    auto output_fp = output.tensor->get_attr<int>("fix_point");

    // scale
    auto shift_cut = (input_fp + weight_fp - output_fp);
    shift_cut_scale_ = std::pow(2.0f, shift_cut);
    six_as_output_ = 6.0f * std::pow(2.0f, output_fp);

    // bias info
    int8_t* bias_data_ptr = nullptr;
    if (bias != nullptr) {
      auto input_bias_tensor = bias->tensor;
      auto bias_shape = input_bias_tensor->get_shape();
      CHECK_EQ(bias_shape[0], weight_shape_[0]);
      bias_data_ptr = bias->data;

      // bias fix point
      auto bias_fp = input_bias_tensor->get_attr<int>("fix_point");
      auto shift_bias = (input_fp + weight_fp - bias_fp);
      shift_bias_scale_ = std::pow(2.0f, shift_bias);
    }

    conv2d_fix(input_data_ptr, weight_data_ptr, bias_data_ptr, output_data_ptr);

    return 0;
  }
  void conv2d_fix(int8_t* input, int8_t* weight, int8_t* bias, int8_t* output) {
    for (auto n = 0; n < output_shape_[0]; ++n) {
      auto in_offset = n * input_shape_size_vec_[0];
      auto n_offset = n * output_shape_size_vec_[0];

      for (auto h = 0; h < output_shape_[1]; ++h) {
        auto h_offset = h * output_shape_size_vec_[1];
        for (auto w = 0; w < output_shape_[2]; ++w) {
          auto w_offset = w * output_shape_size_vec_[2];
          for (auto c = 0; c < output_shape_[3]; ++c) {
            auto out_offset = n_offset + h_offset + w_offset + c;
            auto weight_offset = c * weight_shape_size_vec_[0];
            float filout =
                filter(&input[in_offset], &weight[weight_offset], h, w);
            if (nullptr != bias) {
              filout += bias[c] * shift_bias_scale_;
            }
            output[out_offset] = vitis::ai::cpu_task::util::fix(
                nonlinear(filout / shift_cut_scale_));
          }
        }
      }
    }
    return;
  }
  int filter(int8_t* input, int8_t* weight, int o_h, int o_w) {
    int ret = 0;
    auto src_h = o_h * stride_.stride_h - pad_.pad_t;
    auto src_w = o_w * stride_.stride_w - pad_.pad_l;

    for (auto h = 0; h < weight_shape_[1]; ++h) {
      auto i_h = src_h + h * dilation_.dilation_h;
      if (i_h < 0 || i_h >= input_shape_[1]) continue;

      auto i_h_offset = i_h * input_shape_size_vec_[1];
      auto w_h_offset = h * weight_shape_size_vec_[1];

      for (auto w = 0; w < weight_shape_[2]; ++w) {
        auto i_w = src_w + w * dilation_.dilation_w;
        if (i_w < 0 || i_w >= input_shape_[2]) continue;

        auto i_w_offset = i_w * input_shape_size_vec_[2];
        auto w_w_offset = w * weight_shape_size_vec_[2];

        for (auto c = 0; c < weight_shape_[3]; ++c) {
          int in = input[i_h_offset + i_w_offset + c];
          int wei = weight[w_h_offset + w_w_offset + c];
          ret += wei * in;
        }
      }
    }
    return ret;
  }
  float nonlinear(float in) {
    auto ret = in;
    switch (nonlinear_) {
      case vitis::ai::cpu_task::util::NONLINEAR::RELU:
        if (ret < 0) ret = 0;
        break;
      case vitis::ai::cpu_task::util::NONLINEAR::PRELU:
        if (ret < 0) ret *= prelu;
        break;
      case vitis::ai::cpu_task::util::NONLINEAR::LEAKYRELU:
        if (ret < 0) ret *= 0.01;
        break;
      case vitis::ai::cpu_task::util::NONLINEAR::RELU6:
        ret = max(0.0f, min(six_as_output_, 6.0f));
        break;
      case vitis::ai::cpu_task::util::NONLINEAR::NONE:
      default:
        break;
    }
    return ret;
  }

 private:
  stride_t stride_;
  string pad_mode_;
  pad_t pad_;
  dilation_t dilation_;
  enum vitis::ai::cpu_task::util::NONLINEAR nonlinear_;

  float shift_bias_scale_;
  float shift_cut_scale_;
  float six_as_output_;
  float prelu = 0.0;

  std::vector<int> input_shape_;
  std::vector<int> weight_shape_;
  std::vector<int> output_shape_;
  std::vector<int> input_shape_size_vec_;
  std::vector<int> weight_shape_size_vec_;
  std::vector<int> output_shape_size_vec_;
};
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
