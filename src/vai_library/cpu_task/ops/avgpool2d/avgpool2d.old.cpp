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
class AvgPool2d_OpImp : public vart::OpImp {
 public:
  explicit AvgPool2d_OpImp(const xir::Op* op, xir::Attrs * attrs);
  virtual ~AvgPool2d_OpImp();
  AvgPool2d_OpImp(const AvgPool2d_OpImp& other) = delete;
  AvgPool2d_OpImp& operator=(const AvgPool2d_OpImp& rhs) = delete;

 public:
  virtual int calculate(const std::vector<vart::OpImpArg>& inputs,
                        vart::TensorBuffer* output) override;
};

AvgPool2d_OpImp::AvgPool2d_OpImp(const xir::Op* op, xir::Attrs * attrs) : vart::OpImp(op){};
AvgPool2d_OpImp::~AvgPool2d_OpImp() {}

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

int AvgPool2d_OpImp::calculate(const std::vector<vart::OpImpArg>& inputs,
                               vart::TensorBuffer* output_tensor_buffer) {
  CHECK_EQ(inputs.size(), 1u);
  auto& input = inputs[0];
  CHECK_EQ(input.args.size(), 1u);
  auto& input_tensor_buffer = input.args[0];
  auto input_shape = input_tensor_buffer->get_tensor()->get_shape();
  auto output_shape = output_tensor_buffer->get_tensor()->get_shape();
  auto num_of_dims = input_shape.size();
  CHECK_EQ(num_of_dims, 4u);
  CHECK_EQ(num_of_dims, output_shape.size());
  CHECK_EQ(input_shape[input_shape.size() - 1],
           output_shape[output_shape.size() - 1]);
  CHECK_EQ(output_shape[0], input_shape[0]);
  CHECK_EQ(output_shape[1], 1);
  CHECK_EQ(output_shape[2], 1);
  auto dim_index = vart::get_index_zeros(input_tensor_buffer->get_tensor());
  auto buf_input = vart::get_tensor_buffer_data(input_tensor_buffer, dim_index);
  auto buf_output =
      vart::get_tensor_buffer_data(output_tensor_buffer, dim_index);
  CHECK_EQ(buf_input.size / sizeof(float),
           input_tensor_buffer->get_tensor()->get_element_num())
      << "only support float to float, continuous region";
  CHECK_EQ(buf_output.size / sizeof(float),
           output_tensor_buffer->get_tensor()->get_element_num())
      << "only support float to float, continuous region";
  auto batch = output_shape[0];
  auto height = input_shape[1];
  auto width = input_shape[2];
  auto channel = input_shape[3];
  for (auto b = 0; b < batch; b++) {
    float* input = (float*)buf_input.data;
    float* output = (float*)buf_output.data;
    input = input + height * width * channel * b;
    output = output + channel * b;
    globalAvePool(input, height, width, channel, output);
  }
  return 0;
}

}  // namespace
extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t op) {
  return vart::make_vart_opt_imp<AvgPool2d_OpImp>();
}
