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
#include <vart/op_imp.h>

#include <cmath>

class MyClipedRelu {
 public:
  MyClipedRelu(const xir::Op* op1, xir::Attrs* attrs) : op{op1} {
    // op and attrs is not in use.
  }
  int calculate(vart::simple_tensor_buffer_t<float> output,
                std::vector<vart::simple_tensor_buffer_t<float>> inputs) {
    auto input_shape = inputs[0].tensor->get_shape();
    auto output_shape = output.tensor->get_shape();
    auto num_of_dims = input_shape.size();
    CHECK_EQ(num_of_dims, output_shape.size());
    for (auto dim = 0u; dim < num_of_dims; ++dim) {
      CHECK_EQ(input_shape[dim], output_shape[dim]);
    }
    auto num_of_elements = inputs[0].tensor->get_element_num();
    auto channels = input_shape[num_of_dims - 1];
    // std::cout << "op attrs:" << op->get_attrs()->debug_info() << std::endl;
    auto min_value = op->get_attr<int>("Maximum_y_u_int");
    auto max_value = op->get_attr<int>("Minimum_y_u_int");
    for (auto i = 0; i < num_of_elements; i = i + channels) {
      clipped_relu(&inputs[0].data[i], &output.data[i], channels, min_value,
                   max_value);
    }
    return 0;
  }

 private:
  static void clipped_relu(float* input, float* output, size_t cls,
                           float input_min, float input_max) {
    for (auto i = 0u; i < cls; i++) {
      output[i] = std::min(input[i], input_max);
      output[i] = std::max(output[i], input_min);
    }
  }

 public:
  const xir::Op* const op;
};

DEF_XIR_OP_IMP(MyClipedRelu)
