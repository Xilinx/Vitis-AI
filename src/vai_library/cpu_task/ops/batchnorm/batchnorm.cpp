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
struct StrideInfo {
  int num_of_strides;
  int stride;
};
StrideInfo get_stride(int axis, const vector<int>& shape) {
  auto sz = (int)shape.size();
  CHECK_LT(axis, sz);
  auto ret = StrideInfo{1, 1};
  for (int x = 0; x < sz; ++x) {
    if (x < axis) {
      ret.num_of_strides = ret.num_of_strides * shape[x];
    } else {
      ret.stride = ret.stride * shape[x];
    }
  }
  return ret;
}

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    axis_ = op->get_attr<decltype(axis_)>("axis");
    auto input = op->get_input_tensors("input");
    CHECK_EQ(input.size(), 1u);
    auto input_shape = input[0]->get_shape();
    axis_ = axis_ < 0 ? axis_ + input_shape.size() : axis_;
    epsilon_ = op->get_attr<decltype(epsilon_)>("epsilon");
    CHECK_LT(axis_, (decltype(axis_))input_shape.size());
    CHECK_GE(axis_, 0);
    stride_ = get_stride(axis_, input_shape);
  }
  int calculate(vart::simple_tensor_buffer_t<float> output_v,
                vart::simple_tensor_buffer_t<float> input_v,
                vart::simple_tensor_buffer_t<float> gamma_v,
                vart::simple_tensor_buffer_t<float> beta_v,
                vart::simple_tensor_buffer_t<float> moving_mean_v,
                vart::simple_tensor_buffer_t<float> moving_var_v) {
    // "implements batchnorm along the last dimension of input feature "
    //          "maps.\n\n"
    //          "    output = (input - moving_mean) /\n"
    //          "             sqrt(moving_var + epsilon) * gamma + beta")
    auto output_shape = output_v.tensor->get_shape();
    auto input_shape = input_v.tensor->get_shape();
    CHECK_EQ(output_shape.size(), input_shape.size());
    auto output = output_v.data;
    auto input = input_v.data;
    for (auto n = 0; n < stride_.num_of_strides; ++n) {
      auto gamma = gamma_v.data;
      auto beta = beta_v.data;
      auto moving_mean = moving_mean_v.data;
      auto moving_var = moving_var_v.data;
      for (auto s = 0; s < stride_.stride; ++s) {
        const auto inv_var = (1.0f / std::sqrt((*moving_var) + epsilon_));
        const auto alpha = inv_var * (*gamma);
        const auto beta1 = (*beta) - (*moving_mean) * inv_var * (*gamma);
        *output = (*input) * alpha + beta1;
        // the following code has small error in comparison with the above code.
        // *output = ((*input) - (*moving_mean)) *  //
        //                 (1.0f / std::sqrt((*moving_var) + epsilon_)) *
        //                 (*gamma) +
        //     (*beta);
        output++;
        input++;
        gamma++;
        beta++;
        moving_mean++;
        moving_var++;
      }
    }
    return 0;
  }

 public:
  int axis_;
  float epsilon_;
  StrideInfo stride_;
};  // namespace
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
