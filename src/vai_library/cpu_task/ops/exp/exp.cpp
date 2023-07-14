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
namespace {
struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {}

  int calculate(vart::simple_tensor_buffer_t<float> output,
                vart::simple_tensor_buffer_t<float> input) {
    auto num_of_elements = input.tensor->get_element_num();
    for (auto i = 0; i < num_of_elements; ++i) {
      output.data[i] = std::exp(input.data[i]);
    }
    return 0;
  }
};
}  // namespace
DEF_XIR_OP_IMP(MyOpImp)
