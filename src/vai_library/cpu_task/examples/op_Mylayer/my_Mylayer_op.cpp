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
class MylayerOp {
 public:
  MylayerOp(const xir::Op* op1, xir::Attrs* attrs) : op{op1} {}
  int calculate(vart::simple_tensor_buffer_t<float> output,
                std::vector<vart::simple_tensor_buffer_t<float>> inputs) {
    CHECK_EQ(inputs.size(), 2);
    auto input_data_shape = inputs[0].tensor->get_shape();
    auto input_alpha_shape = inputs[1].tensor->get_shape();
    auto output_shape = output.tensor->get_shape();
    auto dims = output_shape.size();

    CHECK_EQ(input_data_shape.size(), 4);
    CHECK_EQ(input_alpha_shape.size(), 3);
    for (auto i = 1u; i < dims; i++)
      CHECK_EQ(input_data_shape[i], input_alpha_shape[i - 1]);

    auto element_num = inputs[0].tensor->get_element_num();
    auto alpha_size = inputs[1].tensor->get_element_num();
    for (auto i = 0; i < element_num; i++) {
      if (inputs[0].data[i] < 0) {
        output.data[i] = inputs[0].data[i] * inputs[1].data[i % alpha_size];
      } else {
        output.data[i] = inputs[0].data[i];
      }
    }

    return 0;
  }

 public:
  const xir::Op* const op;
};

DEF_XIR_OP_IMP(MylayerOp)
