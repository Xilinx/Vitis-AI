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

namespace {
static int size_from(const std::vector<int32_t>& input_idx, size_t from) {
    int ret = 1;
    for (; from < input_idx.size(); ++from) {
        ret = ret * input_idx[from];
    }
    return ret;
}

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    dim_ = (size_t)op->get_attr<int>("dim");

    if (op->has_attr("p")) {
      p_ = op->get_attr<int>("p");
      CHECK_EQ(p_, 2);
    }
  }
  int calculate(vart::simple_tensor_buffer_t<float> output,
                std::vector<vart::simple_tensor_buffer_t<float>> inputs) {
    auto input_shape = inputs[0].tensor->get_shape();
    auto output_shape = output.tensor->get_shape();
    auto dims = input_shape.size();
    CHECK_LT(dim_, dims);
    for (auto d = 0u; d < dims; ++d)
      CHECK_EQ(input_shape[d], output_shape[d]);
 
    auto num_of_dim_ = input_shape[dim_];
    auto low_step = size_from(input_shape, dim_+1);
    auto high_step = size_from(input_shape, dim_);
    auto loop_size = inputs[0].tensor->get_element_num() / num_of_dim_;

    for (auto n = 0; n < loop_size; ++n) {
      auto base_offset = (n/low_step * high_step) + n%low_step;
      float sum = 0.0f;
      for (auto d = 0; d < num_of_dim_; ++d) {
        auto element = inputs[0].data[base_offset + d * low_step];
        sum += element * element;
      }
      auto sum_sqrt = sqrt(sum);
      CHECK_GT(sum_sqrt, 0);
      for (auto d = 0; d < num_of_dim_; ++d) {
        auto data_offset = base_offset + d * low_step;
        output.data[data_offset] = inputs[0].data[data_offset] / sum_sqrt;
      }
    }

    return 0;
  }

 private:
  size_t dim_;
  int p_ = 2;
};
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
