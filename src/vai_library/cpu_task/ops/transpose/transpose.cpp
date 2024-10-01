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

#include <iostream>
#include <vitis/ai/dim_calc.hpp>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
namespace {

std::vector<int32_t> trans_idx(std::vector<int32_t> input_idx,
                               std::vector<int32_t> order) {
  auto sz = input_idx.size();
  auto output_idx = std::vector<int32_t>(sz, 0);
  for (auto i = 0u; i < sz; ++i) {
    output_idx[i] = input_idx[order[i]];
  }
  return output_idx;
}

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    order_ = op->get_attr<std::vector<int32_t>>("order");
  }
  int calculate(vart::simple_tensor_buffer_t<float> output,
                vart::simple_tensor_buffer_t<float> input) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = output.tensor->get_shape();
    CHECK_EQ(input_shape.size(), order_.size());
    CHECK_EQ(input_shape.size(), output_shape.size());
    auto input_dim_calc = std::make_unique<vitis::ai::DimCalc>(input_shape);
    auto output_dim_calc = std::make_unique<vitis::ai::DimCalc>(output_shape);

    for (auto i = 0; i < input.tensor->get_element_num(); i++) {
      auto input_idx = input_dim_calc->index(i);
      auto output_idx = trans_idx(input_idx, order_);
      auto output_offset = output_dim_calc->offset(output_idx);
      output.data[output_offset] = input.data[i];
    }
    return 0;
  }

 private:
  std::vector<int32_t> order_;
};

}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
