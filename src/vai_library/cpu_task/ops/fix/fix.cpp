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
    fix_point_ = op->get_attr<int>("fix_point");
    auto round_mode = op->get_attr<std::string>("round_mode");
    CHECK_EQ(round_mode, "DPU_ROUND") << "TODO:";
    // auto bit_width = op->get_attr<int>("quant_in_bit_width");
    // CHECK_EQ(bit_width, 8) << "TODO:";
    scale_ = std::pow(2.0f, fix_point_);
  };
  int calculate(vart::simple_tensor_buffer_t<float> output,
                vart::simple_tensor_buffer_t<float> input) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = output.tensor->get_shape();
    auto num_of_dims = input_shape.size();
    CHECK_EQ(input_shape.size(), output_shape.size());
    CHECK_EQ(num_of_dims, output_shape.size());
    auto num_of_elements = input.tensor->get_element_num();
    for (auto i = 0; i < num_of_elements; ++i) {
      output.data[i] = fix(input.data[i] * scale_) / scale_;
    }
    return 0;
  }
  float fix(float data) {
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
    return data;
  }

 private:
  int fix_point_;
  float scale_;
};
}  // namespace

DEF_XIR_OP_IMP(MyOp)
