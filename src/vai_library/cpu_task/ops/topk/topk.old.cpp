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
class TopK_OpImp : public vart::OpImp {
 public:
  explicit TopK_OpImp(const xir::Op* op, xir::Attrs * attrs);
  virtual ~TopK_OpImp();
  TopK_OpImp(const TopK_OpImp& other) = delete;
  TopK_OpImp& operator=(const TopK_OpImp& rhs) = delete;

 public:
  virtual int calculate(const std::vector<vart::OpImpArg>& inputs,
                        vart::TensorBuffer* output) override;
};

TopK_OpImp::TopK_OpImp(const xir::Op* op, xir::Attrs * attrs) : vart::OpImp(op){};
TopK_OpImp::~TopK_OpImp() {}

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

int TopK_OpImp::calculate(const std::vector<vart::OpImpArg>& inputs,
                          vart::TensorBuffer* output_tensor_buffer) {
  CHECK_EQ(inputs.size(), 1u);
  auto& input = inputs[0];
  CHECK_EQ(input.args.size(), 1u);
  auto& input_tensor_buffer = input.args[0];
  auto input_shape = input_tensor_buffer->get_tensor()->get_shape();
  auto output_shape = output_tensor_buffer->get_tensor()->get_shape();
  auto num_of_dims = input_shape.size();
  CHECK_EQ(num_of_dims, output_shape.size());
  for (auto dim = 0u; dim < num_of_dims - 1; ++dim) {
    CHECK_EQ(input_shape[dim], output_shape[dim]);
  }
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
  auto channels = input_shape[num_of_dims - 1];
  auto k = output_shape[num_of_dims - 1] / 2;
  for (auto i = 0u, j = 0u; i < buf_input.size / sizeof(float);
       i = i + channels, j = j + k) {
    auto input = (float*)buf_input.data;
    auto output = (float*)buf_output.data;
    topk(&input[i], &output[j], channels, k);
  }
  return 0;
}

}  // namespace
extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t op) {
  return vart::make_vart_opt_imp<TopK_OpImp>();
}
