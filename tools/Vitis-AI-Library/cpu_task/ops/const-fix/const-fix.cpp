/*
 * Copyright 2019 Xilinx Inc.
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
  MyOpImp(xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {
    data_ = op->get_attr<vector<char>>("data");
  }
  int calculate(vart::experimental::simple_tensor_buffer_t<int8_t> result) {
    CHECK_EQ(data_.size(), result.mem_size) << "data size mismatch";
    auto num_of_elements = result.tensor->get_element_num();
    memcpy(result.data, &data_[0], num_of_elements * sizeof(int8_t));
    return 0;
  }

 public:
  vector<char> data_;
};  // namespace
}  // namespace

extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t op) {
  return vart::experimental::make_vart_opt_imp<MyOpImp>();
}
