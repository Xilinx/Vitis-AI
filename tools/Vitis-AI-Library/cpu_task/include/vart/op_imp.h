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
#pragma once
#ifdef __cplusplus
#include <vart/vart.h>
#include <xir/xir.h>
extern "C" {
// we define the interface in C
#endif
typedef struct {
  const char* arg_name;
  size_t num_of_args;
  vart_tensor_buffer_t* args;
} vart_op_imp_input_t;

typedef struct vart_op_imp_t {
  // constructor, an Op implementation will return an opague data structure for
  // its own use.
  void* (*init)(const xir_op_t op);

  // decontructor.
  void (*cleanup)(void* self);

  // do the real calculation.
  // the inputs should be in the same order of in the xir::OpDef, returned by
  // op->get_op_def();
  int (*calculate)(void* self, vart_op_imp_input_t inputs[],
                   size_t num_of_inputs, vart_tensor_buffer_t output);
} vart_op_imp_t;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
#include <vart/runner.hpp>
#include <vector>
#include <xir/op/op.hpp>
namespace vart {
struct OpImpArg {
  std::string arg_name;
  std::vector<vart::TensorBuffer*> args;
};
// the interface
class OpImp {
 public:
  explicit OpImp(const xir::Op* op){};
  virtual ~OpImp() = default;
  OpImp(const OpImp& other) = delete;
  OpImp& operator=(const OpImp& rhs) = delete;

 public:
  virtual int calculate(const std::vector<OpImpArg>& inputs,
                        vart::TensorBuffer* output) = 0;
};
template <typename T>
vart_op_imp_t make_vart_opt_imp() {
  vart_op_imp_t ret;
  ret.init = [](const xir_op_t op) -> void* {
    return reinterpret_cast<void*>(new T(reinterpret_cast<xir::Op*>(op)));
  };
  ret.cleanup = [](void* self) -> void { delete reinterpret_cast<T*>(self); };
  ret.calculate = [](void* self1, vart_op_imp_input_t inputs1[],
                     size_t num_of_inputs, vart_tensor_buffer_t output1) {
    auto self = reinterpret_cast<T*>(self1);
    auto inputs = std::vector<OpImpArg>(num_of_inputs);
    for (auto i = 0u; i < num_of_inputs; ++i) {
      inputs[i].arg_name = inputs1[i].arg_name;
      inputs[i].args.resize(inputs1[i].num_of_args);
      for (auto j = 0u; j < inputs[i].args.size(); ++j) {
        inputs[i].args[j] =
            reinterpret_cast<vart::TensorBuffer*>(inputs1[i].args[j]);
      }
    }
    auto output = reinterpret_cast<vart::TensorBuffer*>(output1);
    return self->calculate(inputs, output);
  };
  return ret;
}

}  // namespace vart
#endif
// Local Variables:
// mode: c++
// coding: utf-8-unix
// End:
