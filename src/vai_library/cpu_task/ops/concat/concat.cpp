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

static inline char* go_offset(void* p, int offset) {
  return &((char*)p)[offset];
}

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    auto axis = op->get_attr<int>("axis");
    auto ins = op->get_input_tensors();
    axis = axis < 0 ? axis + ins[0]->get_shape().size() : axis;
    strides_.resize(ins.size());
    for (auto i = 0u; i < strides_.size(); ++i) {
      strides_[i] = get_stride(axis, ins[i]->get_shape());
    }
    for (auto i = 1u; i < strides_.size(); ++i) {
      CHECK_EQ(strides_[i].num_of_strides, strides_[0].num_of_strides);
    }
    num_of_strides_ = strides_[0].num_of_strides;
  }
  int calculate(vart::simple_tensor_buffer_t<void> output,
                std::vector<vart::simple_tensor_buffer_t<void>> input) {
    CHECK_EQ(input.size(), strides_.size());
    auto sz = input.size();
    auto p = output.data;
    for (auto s = 0; s < num_of_strides_; ++s) {
      for (auto i = 0u; i < sz; ++i) {
        auto elt_size = input[i].tensor->get_data_type().bit_width / 8;
        auto offset = s * strides_[i].stride * elt_size;
        auto size = strides_[i].stride * elt_size;
        memcpy(p, go_offset(input[i].data, offset), size);
        p = go_offset(p, strides_[i].stride * elt_size);
      }
    }
    CHECK(p == go_offset(output.data, output.mem_size));
    return 0;
  }

 public:
  vector<StrideInfo> strides_;
  int num_of_strides_;
};
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)
