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
#include <limits>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
#include "vitis/ai/dim_calc.hpp"

using namespace std;

namespace {
static inline int8_t float2fix(float data) {
  int data_max = std::numeric_limits<int8_t>::max();
  int data_min = std::numeric_limits<int8_t>::min();
  auto rlt = 0.0f;
  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else if (data < 0 && (data - floor(data)) == 0.5) {
    rlt = std::ceil(data);
  } else {
    rlt = std::round(data);
  }
  return rlt;
}

static inline uint8_t float2fix_u(float data) {
  int data_max = std::numeric_limits<uint8_t>::max();
  int data_min = 0;
  auto rlt = 0.0f;
  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else {
    rlt = std::round(data);
  }
  return rlt;
}

struct Float2FixOpImp : public vart::experimental::OpImpBase {
  Float2FixOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    CHECK(op->has_attr("fix_point"))
        << "get op fix_point error! has no fix_point attr, op name is "
        << op->get_name();
    CHECK(op->has_attr("if_signed"))
        << "get op if_signed error! has no if_signed attr, op name is "
        << op->get_name();

    auto fix_point = op->template get_attr<int>("fix_point");
    scale_ = std::exp2f(1.0f * (float)fix_point);
    if_signed_ = op->template get_attr<bool>("if_signed");
  }

  int calculate(vart::simple_tensor_buffer_t<void> result,
                vart::simple_tensor_buffer_t<float> input) {
    auto input_shape = input.tensor->get_shape();
    auto output_shape = result.tensor->get_shape();
    auto num_of_dims = input_shape.size();
    CHECK_EQ(num_of_dims, output_shape.size());
    for (auto dim = 0u; dim < num_of_dims; ++dim) {
      CHECK_EQ(input_shape[dim], output_shape[dim]);
    }
    auto num_of_elements = input.tensor->get_element_num();
    auto input_ptr = input.data;
    if (if_signed_) {
      auto output_ptr = (int8_t*)result.data;
      for (auto i = 0; i < num_of_elements; ++i) {
        output_ptr[i] = float2fix(input_ptr[i] * scale_);
      }
    } else {
      auto output_ptr = (uint8_t*)result.data;
      for (auto i = 0; i < num_of_elements; ++i) {
        output_ptr[i] = float2fix_u(input_ptr[i] * scale_);
      }
    }
    return 0;
  }
  float scale_;
  bool if_signed_;
};
}  // namespace

DEF_XIR_OP_IMP(Float2FixOpImp)
