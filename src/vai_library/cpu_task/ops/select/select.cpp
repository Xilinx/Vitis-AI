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

using namespace std;

namespace {

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    dim_ = op->get_attr<int>("dim");
    index_ = op->get_attr<int>("index");
  }
  int calculate(vart::simple_tensor_buffer_t<float> result,
                std::vector<vart::simple_tensor_buffer_t<float>> inputs) {
    CHECK_EQ(inputs.size(), 1);
    auto input_shape = inputs[0].tensor->get_shape();
    auto output_shape = result.tensor->get_shape();
    CHECK_LE(dim_, input_shape.size());
    CHECK_LE(index_, input_shape[dim_]);

    auto step = 1;
    auto one_copy_len = 1;
    auto begin_index = 0;
    for (auto i = dim_ + 1; i < (int)input_shape.size(); i++) {
      one_copy_len *= input_shape[i];
    }
    step = one_copy_len * input_shape[dim_];
    begin_index = one_copy_len * index_;

    auto num_of_elements = inputs[0].tensor->get_element_num();
    auto j = 0;
    for (auto i = begin_index; i < num_of_elements; i = i + step) {
      memcpy((void*)(result.data + one_copy_len * j),
             (void*)(inputs[0].data + i), one_copy_len * sizeof(float));
      j++;
    }
    return 0;
  }

 private:
  int dim_;
  int index_;
};
}  // namespace
DEF_XIR_OP_IMP(MyOpImp)
