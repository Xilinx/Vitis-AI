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
#pragma once
#include <cmath>

#include "./broadcast_op_index.hpp"
#include "vart/op_imp.h"
#include "vitis/ai/env_config.hpp"
using namespace std;
namespace {
template <typename Op>
struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    auto input_ops = op->get_input_ops("input");
    CHECK_EQ(input_ops.size(), 2u);
    auto input_a = input_ops[0];
    auto input_b = input_ops[1];
    op_index_ = BroadcastOpIndex{input_a->get_output_tensor()->get_shape(),
                                 input_b->get_output_tensor()->get_shape()};
  }

  int calculate(vart::simple_tensor_buffer_t<float> output,
                std::vector<vart::simple_tensor_buffer_t<float>> inputs) {
    CHECK_EQ(inputs.size(), 2u);
    auto input_a = inputs[0];
    auto input_b = inputs[1];
    op_index_.reset_to_zero();
    for (; op_index_.is_end(); op_index_.tick()) {
      auto index_a = op_index_.get_a();
      auto index_b = op_index_.get_b();
      auto index_c = op_index_.get_c();
      auto a = input_a.data[index_a];
      auto b = input_b.data[index_b];
      output.data[index_c] = op_(a, b);
    }
    return 0;
  }

 private:
  BroadcastOpIndex op_index_;
  Op op_;
};
}  // namespace
