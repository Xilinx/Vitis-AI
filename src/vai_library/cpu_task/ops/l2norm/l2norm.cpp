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
inline void L2_normalization(vart::simple_tensor_buffer_t<float> input,vart::simple_tensor_buffer_t<float> output,  int channel, int group) {
  for (int i = 0; i < group; ++i) {
    float sum = 0.0;
    for (int j = 0; j < channel; ++j) {
      int pos = i*channel + j;
      float temp = static_cast<float>(input.data[pos]);
      sum += temp*temp;
    }
    float var = sqrt(sum);
    for (int j = 0; j < channel; ++j) {
      int pos = i*channel + j;
      output.data[pos] = static_cast<float>(input.data[pos]) / var;
    } 
  }
}

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {}
  int calculate(vart::simple_tensor_buffer_t<float> result,
                vart::simple_tensor_buffer_t<float> input) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = result.tensor->get_shape();
    auto num_of_dims = input_shape.size();
    CHECK_EQ(num_of_dims, output_shape.size());
    for (auto dim = 0u; dim < num_of_dims; ++dim) {
      CHECK_EQ(input_shape[dim], output_shape[dim]);
    }
    auto channels = output_shape[3];
    auto height = output_shape[1];
    auto width = output_shape[2];
    L2_normalization(input, result, channels, height * width);
    
    return 0;
  }
};
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
