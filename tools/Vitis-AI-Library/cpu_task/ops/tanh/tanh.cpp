/*
 * Copyright 2019 Xilinx Inc.
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

class MyTanhOp {
 public:
  MyTanhOp(const xir::Op* op1, xir::Attrs* attrs) : op{op1} {
    // op and attrs is not in use.
  }
  int calculate(vart::simple_tensor_buffer_t<float> output,
                vart::simple_tensor_buffer_t<float> input) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = output.tensor->get_shape();
    auto num_of_dims = input_shape.size();
    CHECK_EQ(num_of_dims, output_shape.size());
    for (auto dim = 0u; dim < num_of_dims; ++dim) {
      CHECK_EQ(input_shape[dim], output_shape[dim]);
    }
    auto num_of_elements = input.tensor->get_element_num();
    auto channels = input_shape[num_of_dims - 1];
    for (auto i = 0; i < num_of_elements; i = i + channels) {
      tanh(&input.data[i], &output.data[i], channels);
    }
    return 0;
  }

 private:
  static void tanh(float* input, float* output, size_t cls) {
    for (auto i = 0u; i < cls; i++) {
      auto exp1 = std::exp(input[i]);
      auto exp2 = 1.0 / exp1;
      output[i] = (exp1 - exp2) / (exp1 + exp2);
    }
  }

 public:
  const xir::Op* const op;
};

DEF_XIR_OP_IMP(MyTanhOp)
