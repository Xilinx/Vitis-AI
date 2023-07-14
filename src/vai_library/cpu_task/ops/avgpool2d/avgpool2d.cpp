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
using namespace std;

namespace {

void globalAvePool(float* src, int height, int width, int channel, float* dst) {
  memcpy(dst, src, channel * sizeof(float));
  for (int j = 1; j < width * height; j++) {
    for (int i = 0; i < channel; i++) {
      dst[i] += src[channel * j + i];
    }
  }
  float factor = ((float)width) * ((float)height);
  for (int j = 0; j < channel; j++) {
    dst[j] = dst[j] / factor;
  };
}

struct AvgPool2d_OpImp : public vart::experimental::OpImpBase {
  AvgPool2d_OpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {}
  int calculate(vart::simple_tensor_buffer_t<float> result,
                vart::simple_tensor_buffer_t<float> input) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = result.tensor->get_shape();
    auto num_of_dims = input_shape.size();
    CHECK_EQ(num_of_dims, 4u);
    CHECK_EQ(num_of_dims, output_shape.size());
    CHECK_EQ(input_shape[input_shape.size() - 1],
             output_shape[output_shape.size() - 1]);
    CHECK_EQ(output_shape[0], input_shape[0]);
    CHECK_EQ(output_shape[1], 1);
    CHECK_EQ(output_shape[2], 1);
    auto batch = output_shape[0];
    auto height = input_shape[1];
    auto width = input_shape[2];
    auto channel = input_shape[3];
    for (auto b = 0; b < batch; b++) {
      globalAvePool(input.data + height * width * channel * b, height, width,
                    channel, result.data + channel * b);
    }
    return 0;
  }
};

}  // namespace

DEF_XIR_OP_IMP(AvgPool2d_OpImp)
