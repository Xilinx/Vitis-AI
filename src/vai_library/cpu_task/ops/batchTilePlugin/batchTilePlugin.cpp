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
#include <math.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"

using namespace std;

namespace {

struct MyOpImp : public vart::experimental::OpImpBase {
  MyOpImp(const xir::Op* op, xir::Attrs* attrs)
      : vart::experimental::OpImpBase{op, attrs}{
  }

  int calculate(vart::simple_tensor_buffer_t<void> output, std::vector<vart::simple_tensor_buffer_t<void>> input) {
    CHECK_EQ( input.size(), 2);
    input_shape_0 = input[0].tensor->get_shape(); 
    input_shape_1 = input[1].tensor->get_shape(); 
    CHECK_EQ( input_shape_0.size(), 4);
    CHECK_EQ( input_shape_1.size(), 4);
    CHECK_EQ( input_shape_1[0], 1);

    float* in1 = (float*)input[1].data;
    float* outlayer = (float*)output.data;
    for(int i=0; i<input_shape_0[0]; i++) {
      memcpy(outlayer, in1, input[1].mem_size);
      outlayer+=input[1].mem_size;
    }
    return 0;
  }

private:
  std::vector<std::int32_t> input_shape_0, input_shape_1;
};
}  // namespace

DEF_XIR_OP_IMP(MyOpImp)

