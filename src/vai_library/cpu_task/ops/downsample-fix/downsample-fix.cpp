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

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    auto scale = op->get_attr<std::vector<float>>("scale");
    CHECK_EQ(scale.size(), 2);
    scale_w_ = scale[0];
    scale_h_ = scale[1];

    mode_ = "NEAREST";
    if (op->has_attr("mode")) {
      mode_ = op->get_attr<string>("mode");
    }
  };
  int calculate(vart::simple_tensor_buffer_t<int8_t> output,
                vart::simple_tensor_buffer_t<int8_t> input) {
    // input info
    auto input_tensor = input.tensor;
    input_shape_ = input_tensor->get_shape();
    input_dim_size_vec_ =
        vitis::ai::cpu_task::util::get_dim_stride_vec(input_shape_);
    auto input_data_ptr = input.data;

    // output info
    auto output_tensor = output.tensor;
    output_shape_ = output_tensor->get_shape();
    output_dim_size_vec_ =
        vitis::ai::cpu_task::util::get_dim_stride_vec(output_shape_);
    auto output_data_ptr = output.data;

    CHECK_EQ(input_shape_.size(), output_shape_.size());

    // fix point
    auto input_fp = input_tensor->get_attr<int>("fix_point");
    auto output_fp = output_tensor->get_attr<int>("fix_point");
    shift_scale_ = pow(2.0f, (output_fp - input_fp));

    // NHWC, get offset in each dim loop
    for (auto n = 0; n < output_shape_[0]; ++n) {
      auto in_n_offset = n * input_dim_size_vec_[0];
      auto out_n_offset = n * output_dim_size_vec_[0];

      for (auto h = 0; h < output_shape_[1]; ++h) {
        auto in_h_offset = int(h * scale_h_ * input_dim_size_vec_[1]);
        auto out_h_offset = h * output_dim_size_vec_[1];

        for (auto w = 0; w < output_shape_[2]; ++w) {
          auto in_w_offset = int(w * scale_w_ * input_dim_size_vec_[2]);
          auto out_w_offset = w * output_dim_size_vec_[2];

          for (auto c = 0; c < output_shape_[3]; ++c) {
            auto data =
                input_data_ptr[in_n_offset + in_h_offset + in_w_offset + c];
            output_data_ptr[out_n_offset + out_h_offset + out_w_offset + c] =
                vitis::ai::cpu_task::util::fix(data * shift_scale_);
          }
        }
      }
    }
    return 0;
  }

 private:
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
  std::vector<int> input_dim_size_vec_;
  std::vector<int> output_dim_size_vec_;
  string mode_;
  float scale_w_;
  float scale_h_;
  float shift_scale_;
};
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
