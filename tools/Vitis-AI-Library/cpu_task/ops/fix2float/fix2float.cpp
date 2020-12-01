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
class Fix2FloatOpImp : public vart::OpImp {
 public:
  explicit Fix2FloatOpImp(const xir::Op* op);
  virtual ~Fix2FloatOpImp();
  Fix2FloatOpImp(const Fix2FloatOpImp& other) = delete;
  Fix2FloatOpImp& operator=(const Fix2FloatOpImp& rhs) = delete;

 public:
  virtual int calculate(const std::vector<vart::OpImpArg>& inputs,
                        vart::TensorBuffer* output) override;
};

Fix2FloatOpImp::Fix2FloatOpImp(const xir::Op* op) : vart::OpImp(op){};
Fix2FloatOpImp::~Fix2FloatOpImp() {}
int Fix2FloatOpImp::calculate(const std::vector<vart::OpImpArg>& inputs,
                              vart::TensorBuffer* output) {
  LOG(INFO) << "hello " << inputs.size() << "output "
            << output->get_tensor()->get_name();

  return 0;
}

}  // namespace
extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t op) {
  return vart::make_vart_opt_imp<Fix2FloatOpImp>();
}
