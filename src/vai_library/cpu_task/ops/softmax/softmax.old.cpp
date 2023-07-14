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
namespace {
class SoftmaxOpImp : public vart::OpImp {
 public:
  explicit SoftmaxOpImp(const xir::Op* op, xir::Attrs * attrs);
  virtual ~SoftmaxOpImp();
  SoftmaxOpImp(const SoftmaxOpImp& other) = delete;
  SoftmaxOpImp& operator=(const SoftmaxOpImp& rhs) = delete;

 public:
  virtual int calculate(const std::vector<vart::OpImpArg>& inputs,
                        vart::TensorBuffer* output) override;
};

SoftmaxOpImp::SoftmaxOpImp(const xir::Op* op, xir::Attrs * attrs) : vart::OpImp(op){};
SoftmaxOpImp::~SoftmaxOpImp() {}

static void softmax(float* input, float* output, size_t cls) {
  if (ENV_PARAM(DEBUG_OP_SOFTMAX) >= 5) {
    auto mode =
        std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
    CHECK(std::ofstream("softmax_op_input.bin", mode)
              .write((char*)(input), sizeof(float) * cls)
              .good())
        << " faild to write to "
        << "softmax_c_input.bin";
  }
  float sum = 0.f;
  for (auto i = 0u; i < cls; ++i) {
    output[i] = std::exp(input[i]);
    sum += output[i];
  }
  for (unsigned int i = 0u; i < cls; ++i) {
    output[i] /= sum;
  }
  if (ENV_PARAM(DEBUG_OP_SOFTMAX) >= 5) {
    auto mode =
        std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
    CHECK(std::ofstream("softmax_op_output.bin", mode)
              .write((char*)(output), sizeof(float) * cls)
              .good())
        << " faild to write to "
        << "softmax_c_output.bin";
  }
}

int SoftmaxOpImp::calculate(const std::vector<vart::OpImpArg>& inputs,
                            vart::TensorBuffer* output_tensor_buffer) {
  CHECK_EQ(inputs.size(), 1u);
  auto& input = inputs[0];
  CHECK_EQ(input.args.size(), 1u);
  auto& input_tensor_buffer = input.args[0];
  auto input_shape = input_tensor_buffer->get_tensor()->get_shape();
  auto output_shape = input_tensor_buffer->get_tensor()->get_shape();
  auto num_of_dims = input_shape.size();
  CHECK_EQ(num_of_dims, output_shape.size());
  for (auto dim = 0u; dim < num_of_dims; ++dim) {
    CHECK_EQ(input_shape[dim], output_shape[dim]);
  }
  auto dim_index = vart::get_index_zeros(input_tensor_buffer->get_tensor());
  auto buf_input = vart::get_tensor_buffer_data(input_tensor_buffer, dim_index);
  auto buf_output =
      vart::get_tensor_buffer_data(output_tensor_buffer, dim_index);
  CHECK_EQ(buf_input.size / sizeof(float), buf_output.size / sizeof(float))
      << "only support float to float, continuous region";
  auto channels = input_shape[num_of_dims - 1];
  for (auto i = 0u; i < buf_input.size / sizeof(float); i = i + channels) {
    auto input = (float*)buf_input.data;
    auto output = (float*)buf_output.data;
    softmax(&input[i], &output[i], channels);
  }
  return 0;
}

}  // namespace
extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t op) {
  return vart::make_vart_opt_imp<SoftmaxOpImp>();
}
