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
#include <numeric>

#include "../common/broadcast_op_index.hpp"
#include "vart/op_imp.h"

using namespace std;

namespace {

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    mode_ = op->get_attr<std::string>("type");

    auto input_tensors = op->get_input_tensors();
    for (auto tensor : input_tensors) {
      auto fp = tensor->get_attr<int>("fix_point");
      fix_point_inputs_.push_back(fp);
    }
    auto output_tensor = op->get_output_tensor();
    // check output_tensor has attr for fix_point
    fix_point_output_ = output_tensor->get_attr<int>("fix_point");
    fmap_o_ = output_tensor->get_shape();

    if (mode_ == "ADD") {
      auto max_element = *std::max_element(std::begin(fix_point_inputs_),
                                           std::end(fix_point_inputs_));
      for (auto i : fix_point_inputs_) shift_read_.push_back(max_element - i);
      shift_write_ = fix_point_output_ - max_element;
    } else if (mode_ == "MUL") {
      shift_write_ =
          fix_point_output_ -
          std::accumulate(fix_point_inputs_.begin(), fix_point_inputs_.end(),
                          decltype(fix_point_inputs_)::value_type(0));
    }
  }
  int calculate(vart::simple_tensor_buffer_t<int8_t> result,
                std::vector<vart::simple_tensor_buffer_t<int8_t>> inputs) {
    // check inputs size == 2
    CHECK_EQ(inputs.size(), 2);
    auto input_shape_0 = inputs[0].tensor->get_shape();
    auto input_shape_1 = inputs[1].tensor->get_shape();
    auto output_shape = result.tensor->get_shape();
    auto num_of_dims = input_shape_0.size();
    for (auto dim = 0; dim < (int)num_of_dims; ++dim) {
      CHECK_EQ(output_shape[dim],
               std::max(input_shape_0[dim], input_shape_1[dim]));
      CHECK_EQ(output_shape[dim], fmap_o_[dim]);
    }
    auto op_index_ = BroadcastOpIndex{input_shape_0, input_shape_1};
    for (; op_index_.is_end(); op_index_.tick()) {
      auto index_a = op_index_.get_a();
      auto index_b = op_index_.get_b();
      auto index_c = op_index_.get_c();
      auto a = inputs[0].data[index_a];
      auto b = inputs[1].data[index_b];
      result.data[index_c] = depthwise_fix(a, b);
    }
    return 0;
  }

  int8_t depthwise_fix(int8_t input0, int8_t input1) {
    float tmp = 0.0f;
    if (mode_ == "ADD") {
      tmp = (input0 * std::pow(2, shift_read_[0]) +
             input1 * std::pow(2, shift_read_[1])) *
            pow(2, shift_write_);
    } else if (mode_ == "MUL") {
      tmp = input0 * input1 * std::pow(2, shift_write_);
    }
    return fix(tmp);
  }

  int8_t fix(float data) {
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
  }

 private:
  std::string mode_;
  std::vector<int> fix_point_inputs_;
  int fix_point_output_=0;
  std::vector<int> shift_read_;
  int shift_write_=0;
  std::vector<int> fmap_o_;
};
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
