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
#include <vart/op_imp.h>
class MyAddOp {
 public:
  MyAddOp(const xir::Op* op1, xir::Attrs* attrs) : op{op1} {
    // op and attrs is not in use.
  }
  int calculate(vart::simple_tensor_buffer_t<float> output,
                std::vector<vart::simple_tensor_buffer_t<float>> inputs) {
    for (auto i = 0u; i < output.mem_size / sizeof(float); ++i) {
      output.data[i] = 0.0f;
      for (auto input : inputs) {
        output.data[i] = output.data[i] + input.data[i];
      }
    }
    return 0;
  }

 public:
  const xir::Op* const op;
};

DEF_XIR_OP_IMP(MyAddOp)
