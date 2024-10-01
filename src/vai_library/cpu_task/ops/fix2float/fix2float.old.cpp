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
#include "vitis/ai/dim_calc.hpp"

using namespace std;

namespace {
class Fix2FloatOpImp : public vart::OpImp {
 public:
  explicit Fix2FloatOpImp(const xir::Op* op, xir::Attrs* attrs);
  virtual ~Fix2FloatOpImp();
  Fix2FloatOpImp(const Fix2FloatOpImp& other) = delete;
  Fix2FloatOpImp& operator=(const Fix2FloatOpImp& rhs) = delete;

 public:
  virtual int calculate(const std::vector<vart::OpImpArg>& inputs,
                        vart::TensorBuffer* output) override;
};

Fix2FloatOpImp::Fix2FloatOpImp(const xir::Op* op, xir::Attrs* attrs)
    : vart::OpImp(op){};
Fix2FloatOpImp::~Fix2FloatOpImp() {}

static int get_fix_point(const xir::Tensor* tensor) {
  CHECK(tensor->has_attr("fix_point"))
      << "get tensor fix_point error! has no fix_point attr, tensor name is "
      << tensor->get_name();
  return tensor->template get_attr<int>("fix_point");
}

int Fix2FloatOpImp::calculate(const std::vector<vart::OpImpArg>& inputs,
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
  int fixpos = get_fix_point(input_tensor_buffer->get_tensor());
  float scale = std::exp2f(-1.0f * (float)fixpos);
  //  auto dim_calc = vitis::ai::DimCalc(input_shape);
  auto dim_index = vart::get_index_zeros(input_tensor_buffer->get_tensor());
  LOG_IF(INFO, false) << dim_index.size() << scale << fixpos;

  auto buf_input = vart::get_tensor_buffer_data(input_tensor_buffer, dim_index);
  auto buf_output =
      vart::get_tensor_buffer_data(output_tensor_buffer, dim_index);
  CHECK_EQ(buf_input.size / sizeof(char), buf_output.size / sizeof(float))
      << "only support xint to float, continuous region";
  for (auto i = 0u; i < buf_input.size; ++i) {
    auto input = (int8_t*)buf_input.data;
    auto output = (float*)buf_output.data;
    output[i] = ((float)(input[i])) * scale;
  }
  return 0;
}

}  // namespace
extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t op) {
  return vart::make_vart_opt_imp<Fix2FloatOpImp>();
}
