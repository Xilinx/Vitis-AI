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
#include <algorithm>
#include <iostream>
#include <numeric>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
using namespace std;

namespace {

static std::vector<std::pair<int, float>> topk(float* begin, float* output,
                                               int size, int K) {
  auto indices = std::vector<int>(size);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + K, indices.end(),
                    [begin](int a, int b) { return begin[a] > begin[b]; });
  auto ret = std::vector<std::pair<int, float>>(K);
  for (auto i = 0; i < K; ++i) {
    output[i * 2] = (float)indices[i];
    output[i * 2 + 1] = begin[indices[i]];
  }
  return ret;
}

struct TopK_OpImp : public vart::experimental::OpImpBase {
  TopK_OpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {}
  int calculate(vart::simple_tensor_buffer_t<float> result,
                vart::simple_tensor_buffer_t<float> input) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = result.tensor->get_shape();
    auto num_of_dims = input_shape.size();
    CHECK_EQ(num_of_dims, output_shape.size());
    for (auto dim = 0u; dim < num_of_dims - 1; ++dim) {
      CHECK_EQ(input_shape[dim], output_shape[dim]);
    }
    auto channels = input_shape[num_of_dims - 1];
    auto k = output_shape[num_of_dims - 1] / 2;
    auto num_of_elements = input.tensor->get_element_num();
    for (auto i = 0, j = 0; i < num_of_elements; i = i + channels, j = j + k) {
      topk(&input.data[i], &result.data[j], channels, k);
    }
    return 0;
  }
};
}  // namespace

DEF_XIR_OP_IMP(TopK_OpImp)
