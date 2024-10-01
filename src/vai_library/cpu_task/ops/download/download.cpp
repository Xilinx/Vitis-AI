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

using namespace std;

namespace {
struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    auto input_tensors = op->get_input_tensors();
    CHECK_EQ(input_tensors.size(), 1);
    auto output_tensor = op->get_output_tensor();
    CHECK_EQ(output_tensor->get_shape().size(),
             input_tensors[0]->get_shape().size());
  }
  int calculate(vart::simple_tensor_buffer_t<int8_t> output,
                vart::simple_tensor_buffer_t<int8_t> input) {
    auto element_num = output.tensor->get_element_num();
    memcpy(output.data, input.data, element_num * sizeof(int8_t));
    return 0;
  }

};  // namespace
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
