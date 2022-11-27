/*
 * Copyright 2019 xilinx Inc.
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

#include "vart/op_imp.h"
using namespace std;

namespace {
class YOUR_OP_TYPE_OpImp : public vart::OpImp {
 public:
  explicit YOUR_OP_TYPE_OpImp(const xir::Op* op, xir::Attrs * attrs);
  virtual ~YOUR_OP_TYPE_OpImp();
  YOUR_OP_TYPE_OpImp(const YOUR_OP_TYPE_OpImp& other) = delete;
  YOUR_OP_TYPE_OpImp& operator=(const YOUR_OP_TYPE_OpImp& rhs) = delete;

 public:
  virtual int calculate(const std::vector<vart::OpImpArg>& inputs,
                        vart::TensorBuffer* output) override;
};

YOUR_OP_TYPE_OpImp::YOUR_OP_TYPE_OpImp(const xir::Op* op, xir::Attrs * attrs) : vart::OpImp(op){};
YOUR_OP_TYPE_OpImp::~YOUR_OP_TYPE_OpImp() {}
int YOUR_OP_TYPE_OpImp::calculate(const std::vector<vart::OpImpArg>& inputs,
                                  vart::TensorBuffer* output) {
  return 0;
}

}  // namespace
extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t op) {
  return vart::make_vart_opt_imp<YOUR_OP_TYPE_OpImp>();
}
