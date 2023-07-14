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

#include "vart/op_imp.h"

#include <dlfcn.h>
#include <glog/logging.h>
#include <UniLog/UniLog.hpp>

#include <regex>

#include "vart/runner_helper.hpp"
#include "xir/op/op_def.hpp"
namespace vart {
class OpImpStub : public OpImp {
 public:
  explicit OpImpStub(const vart_op_imp_t& imp, const xir::Op* op,
                     xir::Attrs* attrs);
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

OpImpStub::OpImpStub(const vart_op_imp_t& imp, const xir::Op* op,
                     xir::Attrs* attrs)
    : OpImp(op), imp_{imp}, self_{imp.init((void*)(op), (void*)attrs)} {}

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

std::string to_string(const std::vector<vart::OpImpArg>& inputs) {
  std::ostringstream str;
  str << "{";
  for (auto& input : inputs) {
    str << "\n\t" << to_string(input);
  }
  str << "\n{";
  return str.str();
}

std::string to_string(const vart::OpImpArg& input) {
  std::ostringstream str;
  str << "{args: " << input.arg_name << "=";
  for (auto& arg : input.args) {
    str << " " << arg->to_string();
  }
  str << "}";
  return str.str();
}

}  // namespace vart

static std::string find_dl_lib_for_op(const xir::Op* op) {
  auto op_type = std::string(op->get_type());
  for (auto c : {':', '/', '\\'}) {
    std::replace(op_type.begin(), op_type.end(), c, '_');
  }
  auto ret = std::string("") + "libvart_op_imp_" + op_type + ".so";
  return ret;
}

static vart_op_imp_t get_op_imp(const std::string& lib, const xir_op_t op) {
  vart_op_imp_t ret;

  typedef vart_op_imp_t (*INIT_FUN)(const xir_op_t op);
  // python_cpu_op required RTLD_GLOBAL instead of RTLD_LOCAL,
  // otherwise, importing any python c extension will result in a
  // error, like undefined reference Py_Import. then it is end user's
  // resposibility to avoid symbol conflict as much as possible.
  // Solve the problem that the call to the symbolic link library does
  // not meet the expectations, and include the realization of op with
  // namespace{}

  auto handle = dlopen(lib.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  // CHECK(handle != NULL) << "cannot open library!"
  //                     << " lib=" << lib << ";error=" << dlerror() << ";"
  //                      << "op=" << ((const xir::Op*)(op))->to_string();
  UNI_LOG_CHECK(handle != NULL, VAILIB_CPU_RUNNER_OPEN_LIB_ERROR)
      << " lib=" << lib << ";error=" << dlerror() << ";"
      << "op=" << ((const xir::Op*)(op))->to_string();
  auto init_fun = (INIT_FUN)dlsym(handle, "vart_init_op_imp");
  // CHECK(init_fun != NULL) << "cannot load symbol 'vart_init_op_imp'!"
  //                        << "! lib=" << lib << ";error=" << dlerror();
  UNI_LOG_CHECK(init_fun != NULL, VAILIB_CPU_RUNNER_LOAD_LIB_SYM_ERROR)
      << "symbol = 'vart_init_op_imp'!"
      << "! lib=" << lib << ";error=" << dlerror();
  ret = init_fun(op);
  return ret;
}

std::unique_ptr<vart::OpImp> create_op_imp(const xir::Op* op,
                                           xir::Attrs* attrs) {
  auto dl_file_name = find_dl_lib_for_op(op);
  auto op_imp = get_op_imp(dl_file_name, (const xir_op_t)op);
  auto ret = new vart::OpImpStub(op_imp, op, attrs);
  return std::unique_ptr<vart::OpImp>(ret);
}
