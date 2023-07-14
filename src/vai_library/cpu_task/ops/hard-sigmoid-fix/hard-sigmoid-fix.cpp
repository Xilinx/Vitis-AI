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

using namespace std;

namespace {

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    hsigmoid_in_ = op->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = op->get_attr<int>("shift_hsigmoid");
  }
  int calculate(vart::simple_tensor_buffer_t<int8_t> result,
                vart::simple_tensor_buffer_t<int8_t> input) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = result.tensor->get_shape();
    auto num_of_dims = input_shape.size();
    CHECK_EQ(num_of_dims, output_shape.size());
    for (auto dim = 0u; dim < num_of_dims; ++dim) {
      CHECK_EQ(input_shape[dim], output_shape[dim]);
    }
    auto num_of_elements = input.tensor->get_element_num();
    auto channels = input_shape[num_of_dims - 1];
    for (auto i = 0; i < num_of_elements; i = i + channels) {
      hard_sigmoid_fix(&input.data[i], &result.data[i], channels);
    }
    return 0;
  }

  void hard_sigmoid_fix(int8_t* input, int8_t* output, size_t cls) {
    for (auto i = 0u; i < cls; i++) {
      double tmp =
          std::min(pow(2, 32),
                   std::max(0.0, (input[i] * 2731.0f +
                                  3.0f * 2731 * std::pow(2, hsigmoid_in_)))) /
          std::pow(2, shift_hsigmoid_);
      output[i] = fix(tmp);
    }
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
  int hsigmoid_in_;
  int shift_hsigmoid_;
};
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
