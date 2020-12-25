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

#include "vart/op_imp.h"

#include <dlfcn.h>
#include <glog/logging.h>
namespace vart {
class OpImpStub : public OpImp {
 public:
  explicit OpImpStub(const vart_op_imp_t& imp, const xir::Op* op);
  virtual ~OpImpStub();
  OpImpStub(const OpImpStub& other) = delete;
  OpImpStub& operator=(const OpImpStub& rhs) = delete;

 public:
  virtual int calculate(const std::vector<OpImpArg>& inputs,
                        vart::TensorBuffer* output) override;

 private:
  vart_op_imp_t imp_;
  void* self_;
};

OpImpStub::OpImpStub(const vart_op_imp_t& imp, const xir::Op* op)
    : OpImp(op), imp_{imp}, self_{imp.init((void*)(op))} {}

OpImpStub::~OpImpStub() { imp_.cleanup(self_); }

int OpImpStub::calculate(const std::vector<OpImpArg>& inputs1,
                         vart::TensorBuffer* output1) {
  std::vector<vart_op_imp_input_t> inputs(inputs1.size());
  for (auto i = 0u; i < inputs1.size(); ++i) {
    inputs[i].arg_name = inputs1[i].arg_name.c_str();
    inputs[i].num_of_args = inputs1[i].args.size();
    inputs[i].args = (vart_tensor_buffer_t*)&inputs1[i].args[0];
  }
  auto output = (vart_tensor_buffer_t)output1;
  return imp_.calculate(self_, &inputs[0], inputs.size(), output);
}

}  // namespace vart

static std::string find_dl_lib_for_op(const xir::Op* op) {
  auto ret = std::string("") + "libvart_op_imp_" + op->get_type() + ".so";
  return ret;
}

static vart_op_imp_t get_op_imp(const std::string& lib, const xir_op_t op) {
  vart_op_imp_t ret;
  typedef vart_op_imp_t (*INIT_FUN)(const xir_op_t op);
  auto handle = dlopen(lib.c_str(), RTLD_LAZY);
  CHECK(handle != NULL) << "cannot open library!"
                        << " lib=" << lib << ";error=" << dlerror() << ";"
                        << "op=" << ((const xir::Op*)(op))->to_string();
  auto init_fun = (INIT_FUN)dlsym(handle, "vart_init_op_imp");
  CHECK(init_fun != NULL) << "cannot load symbol 'vart_init_op_imp'!"
                          << "! lib=" << lib << ";error=" << dlerror();
  ret = init_fun(op);
  return ret;
}

std::unique_ptr<vart::OpImp> create_op_imp(const xir::Op* op) {
  auto dl_file_name = find_dl_lib_for_op(op);
  auto op_imp = get_op_imp(dl_file_name, (const xir_op_t)op);
  auto ret = new vart::OpImpStub(op_imp, op);
  return std::unique_ptr<vart::OpImp>(ret);
}
