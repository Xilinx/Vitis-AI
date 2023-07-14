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

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
namespace {

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    groups_ = op->get_attr<int32_t>("groups");
  }
  int calculate(vart::simple_tensor_buffer_t<float> output,
                std::vector<vart::simple_tensor_buffer_t<float>> inputs) {
    auto& input = inputs[0];
    auto input_shape = input.tensor->get_shape();
    auto output_shape = output.tensor->get_shape();
    CHECK_EQ(input_shape.size(), output_shape.size());
    auto channel = *input_shape.rbegin();
    int32_t per_group = channel / groups_;
    for (auto i = 0; i < input.tensor->get_element_num(); i += channel) {
      for (auto j = 0; j < channel; j++) {
        output.data[i + j] =
            input.data[i + j % groups_ * per_group + j / groups_];
      }
    }
    return 0;
  }

 private:
  int32_t groups_;
};

}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
