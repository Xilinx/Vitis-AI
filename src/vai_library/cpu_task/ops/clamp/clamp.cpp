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

#include <cmath>
#include <iostream>

#include "vart/op_imp.h"

using namespace std;

namespace {

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    min_ = op->get_attr<float>("min");
    max_ = op->get_attr<float>("max");
  }
  int calculate(vart::simple_tensor_buffer_t<float> result,
                std::vector<vart::simple_tensor_buffer_t<float>> inputs) {
    auto input_num_of_elements = inputs[0].tensor->get_element_num();
    auto output_num_of_elements = result.tensor->get_element_num();
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(input_num_of_elements, output_num_of_elements);

    for (auto i = 0; i < input_num_of_elements; i++) {
      result.data[i] = std::min(std::max(inputs[0].data[i], min_), max_);
    }
    return 0;
  }

 private:
  float min_;
  float max_;
};
}  // namespace
DEF_XIR_OP_IMP(MyOpImp)
