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
static void sigmoid(float* input, float* output, size_t cls) {
  for (auto i = 0u; i < cls; i++) {
    output[i] = 1.0 / (1.0 + std::exp(-input[i]));
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
    auto num_of_elements = input.tensor->get_element_num();
    auto channels = input_shape[num_of_dims - 1];
    for (auto i = 0; i < num_of_elements; i = i + channels) {
      sigmoid(&input.data[i], &result.data[i], channels);
    }
    return 0;
  }
};
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
