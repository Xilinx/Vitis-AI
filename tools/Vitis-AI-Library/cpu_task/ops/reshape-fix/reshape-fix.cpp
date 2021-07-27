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
#include <fstream>
#include <iostream>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
#include "vitis/ai/env_config.hpp"
using namespace std;

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs} {}

  int calculate(vart::experimental::simple_tensor_buffer_t<void> result,
                vart::experimental::simple_tensor_buffer_t<void> input) {
    CHECK_EQ(result.mem_size, input.mem_size);
    memcpy(result.data, input.data, input.mem_size);
    return 0;
  }

 private:
};

extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t op) {
  return vart::experimental::make_vart_opt_imp<MyOpImp>();
}
