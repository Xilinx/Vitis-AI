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
#include <fstream>
#include <iostream>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
#include "vitis/ai/env_config.hpp"
using namespace std;
DEF_ENV_PARAM(DEBUG_OP_SOFTMAX, "0")
DEF_ENV_PARAM(SOFTMAX_OP_DOUBLE, "0")
namespace {

static void softmax(float* input, float* output, size_t cls) {
  if (!ENV_PARAM(SOFTMAX_OP_DOUBLE)) {
    float sum = 0.f;
    for (auto i = 0u; i < cls; ++i) {
      output[i] = std::exp(input[i]);
      sum += output[i];
    }
    for (unsigned int i = 0u; i < cls; ++i) {
      output[i] /= sum;
    }
  } else {
    double sum = 0.0;
    vector<float> tmp(cls);
    for (auto i = 0u; i < cls; ++i) {
      float f = std::exp((double)input[i]);
      tmp[i] = f;
      sum += f;
    }
    for (unsigned int i = 0u; i < cls; ++i) {
      output[i] = tmp[i] / sum;
    }
  }
}

struct SoftmaxOpImp : public vart::experimental::OpImpBase {
  SoftmaxOpImp(const xir::Op* op, xir::Attrs* attrs)
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
      softmax(&input.data[i], &result.data[i], channels);
    }
    return 0;
  }
};
}  // namespace

DEF_XIR_OP_IMP(SoftmaxOpImp)
