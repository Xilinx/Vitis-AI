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

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    begin_ = op->get_attr<std::vector<int32_t>>("begin");
    end_ = op->get_attr<std::vector<int32_t>>("end");
    strides_ = op->get_attr<std::vector<int32_t>>("strides");
  }
  int calculate(vart::simple_tensor_buffer_t<float> output,
                vart::simple_tensor_buffer_t<float> input) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = output.tensor->get_shape();
    CHECK_EQ(input_shape.size(), begin_.size());
    CHECK_EQ(input_shape.size(), end_.size());
    CHECK_EQ(input_shape.size(), strides_.size());
    CHECK_EQ(input_shape.size(), output_shape.size());
    auto input_dim_calc = std::make_unique<vitis::ai::DimCalc>(input_shape);
    auto output_dim_calc = std::make_unique<vitis::ai::DimCalc>(output_shape);

    for (auto i = 0; i < output.tensor->get_element_num(); i++) {
      auto output_idx = output_dim_calc->index(i);
      auto input_idx = translate(output_idx);
      auto input_offset = input_dim_calc->offset(input_idx);
      output.data[i] = input.data[input_offset];
    }
    return 0;
  }

 private:
  std::vector<int32_t> translate(const std::vector<int32_t>& output) {
    auto ret = std::vector<int32_t>(output.size());
    for (auto dim = 0u; dim < output.size(); ++dim) {
      ret[dim] = translate1(begin_[dim], end_[dim], strides_[dim], output[dim]);
    }
    return ret;
  }

  int32_t translate1(int32_t begin, int32_t end, int32_t stride, int index) {
    return index * stride + begin;
  }

 private:
  std::vector<int32_t> begin_;
  std::vector<int32_t> end_;
  std::vector<int32_t> strides_;
};

}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
