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
    CHECK(op->has_attr("stride")) << "tile-fix op must have stride attr";
    CHECK(op->has_attr("reverse")) << "tile-fix op must have reverse attr";
    stride_ = op->get_attr<int>("stride");
    reverse_ = op->get_attr<bool>("reverse");
  }
  int calculate(vart::simple_tensor_buffer_t<int8_t> output,
                vart::simple_tensor_buffer_t<int8_t> input) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = output.tensor->get_shape();
    CHECK_EQ(input_shape.size(), 4);
    CHECK_EQ(input_shape.size(), output_shape.size());

    CHECK_EQ(input_shape[0], output_shape[0]);
    if (reverse_) {
      CHECK_EQ(input_shape[1] * stride_, output_shape[1]);
      CHECK_EQ(input_shape[2] * stride_, output_shape[2]);
      CHECK_EQ(input_shape[3], output_shape[3] * stride_ * stride_);

      tile(output.data, output_shape, input.data, input_shape);
    } else {
      CHECK_EQ(input_shape[1], output_shape[1] * stride_);
      CHECK_EQ(input_shape[2], output_shape[2] * stride_);
      CHECK_EQ(input_shape[3] * stride_ * stride_, output_shape[3]);

      tile(input.data, input_shape, output.data, output_shape);
    }

    return 0;
  }

  void tile(int8_t* input, const std::vector<int>& input_shape, int8_t* output,
            const std::vector<int>& output_shape) {
    auto input_stride_vec =
        vitis::ai::cpu_task::util::get_dim_stride_vec(input_shape);
    auto size = input_shape[3] * stride_;

    for (auto o_n = 0; o_n < output_shape[0]; ++o_n) {
      auto in_offset_n = o_n * input_stride_vec[0];

      for (auto o_h = 0; o_h < output_shape[1]; ++o_h) {
        auto i_h_start = o_h * stride_;
        for (auto o_w = 0; o_w < output_shape[2]; ++o_w) {
          auto i_w_start = o_w * stride_;

          for (auto s_h = 0; s_h < stride_; ++s_h) {
            auto in_offset = in_offset_n +
                             (i_h_start + s_h) * input_stride_vec[1] +
                             i_w_start * input_stride_vec[2];

            if (reverse_) {
              memcpy(&input[in_offset], output, size);
            } else {
              memcpy(output, &input[in_offset], size);
            }
            output += size;
          }
        }
      }
    }

    return;
  }

 public:
  int stride_;
  bool reverse_;
};  // namespace
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
