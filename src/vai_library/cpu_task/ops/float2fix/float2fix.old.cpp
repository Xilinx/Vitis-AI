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
#include <limits>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
#include "vitis/ai/dim_calc.hpp"

using namespace std;

namespace {
class Fix2FloatOpImp : public vart::OpImp {
 public:
  explicit Fix2FloatOpImp(const xir::Op* op, xir::Attrs * attrs);
  virtual ~Fix2FloatOpImp();
  Fix2FloatOpImp(const Fix2FloatOpImp& other) = delete;
  Fix2FloatOpImp& operator=(const Fix2FloatOpImp& rhs) = delete;

 public:
  virtual int calculate(const std::vector<vart::OpImpArg>& inputs,
                        vart::TensorBuffer* output) override;
};

Fix2FloatOpImp::Fix2FloatOpImp(const xir::Op* op, xir::Attrs * attrs) : vart::OpImp(op){};
Fix2FloatOpImp::~Fix2FloatOpImp() {}

static int get_fix_point(const xir::Tensor* tensor) {
  CHECK(tensor->has_attr("fix_point"))
      << "get tensor fix_point error! has no fix_point attr, tensor name is "
      << tensor->get_name();
  return tensor->template get_attr<int>("fix_point");
}

static inline int8_t float2fix(float data) {
  int data_max = std::numeric_limits<int8_t>::max();
  int data_min = std::numeric_limits<int8_t>::min();
  auto rlt = 0.0f;
  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else if (data < 0 && (data - floor(data)) == 0.5) {
    rlt = std::ceil(data);
  } else {
    rlt = std::round(data);
  }
  return rlt;
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
  int fixpos = get_fix_point(output_tensor_buffer->get_tensor());
  float scale = std::exp2f(1.0f * (float)fixpos);
  auto dim_index = vart::get_index_zeros(input_tensor_buffer->get_tensor());
  LOG_IF(INFO, false) << dim_index.size() << scale << fixpos;

  auto buf_input = vart::get_tensor_buffer_data(input_tensor_buffer, dim_index);
  auto buf_output =
      vart::get_tensor_buffer_data(output_tensor_buffer, dim_index);
  auto num_of_elements = input_tensor_buffer->get_tensor()->get_element_num();
  auto input_data_size =
      input_tensor_buffer->get_tensor()->get_data_type().bit_width / 8;
  auto output_data_size =
      output_tensor_buffer->get_tensor()->get_data_type().bit_width / 8;
  CHECK_EQ(buf_input.size / input_data_size, num_of_elements)
      << "only support float to float, continuous region";
  CHECK_EQ(buf_output.size / output_data_size, num_of_elements)
      << "only support float to float, continuous region";
  auto input_ptr = (float*)buf_input.data;
  auto output_ptr = (int8_t*)buf_output.data;
  for (auto i = 0; i < num_of_elements; ++i) {
    output_ptr[i] = float2fix(input_ptr[i] * scale);
  }
  return 0;
}

}  // namespace
extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t op) {
  return vart::make_vart_opt_imp<Fix2FloatOpImp>();
}
