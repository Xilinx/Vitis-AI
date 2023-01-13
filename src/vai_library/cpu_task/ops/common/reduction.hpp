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
#pragma once
#include <algorithm>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"

using namespace std;

namespace {
template <typename Op>
struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    axis_ = op->get_attr<decltype(axis_)>("axis");
    is_continuous_ = true;
    for (auto i = 1u; i < axis_.size(); ++i) {
      is_continuous_ = is_continuous_ && (axis_[i - 1] == axis_[i] + 1);
    }
    CHECK(is_continuous_)
        << "TODO: only support continuous axis yet, for performanc";
    auto input_op = op->get_input_op("input", 0);
    CHECK(input_op != nullptr);
    auto input_tensor = input_op->get_output_tensor();
    auto input_shape = input_tensor->get_shape();
    stride_ = 1;
    auto shape_size = (int)input_shape.size();
    for (auto i = 0u; i < axis_.size(); ++i) {
      stride_ = stride_ * input_shape[(shape_size + axis_[i]) % shape_size];
    }
    CHECK_GT(stride_, 0);
    num_of_elements_ = input_tensor->get_element_num();
  }
  int calculate(vart::simple_tensor_buffer_t<float> output,
                vart::simple_tensor_buffer_t<float> input) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = output.tensor->get_shape();
    CHECK_EQ(input_shape.size(), output_shape.size());
    int out_idx = 0;
    for (auto n = 0; n < num_of_elements_; n = n + stride_) {
      output.data[out_idx++] = op_(&input.data[n], &input.data[n + stride_]);
    }
    return 0;
  }

 public:
  vector<int> axis_;
  bool is_continuous_;
  size_t stride_;
  int num_of_elements_;
  Op op_;
};
}  // namespace
