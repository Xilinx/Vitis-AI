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
#include <functional>

#include "vart/runner_helper.hpp"
#include "vart/simple_tensor_buffer.hpp"
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
  void* (*init)(const xir_op_t op, xir_attrs_t attrs);

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
#include <xir/op/op_def.hpp>
namespace vart {
struct OpImpArg {
  std::string arg_name;
  std::vector<vart::TensorBuffer*> args;
};
std::string to_string(const std::vector<OpImpArg>& inputs);
std::string to_string(const OpImpArg& input);
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
  ret.init = [](const xir_op_t op, xir_attrs_t attrs) -> void* {
    return reinterpret_cast<void*>(new T(reinterpret_cast<xir::Op*>(op),
                                         reinterpret_cast<xir::Attrs*>(attrs)));
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

namespace experimental {

struct OpImpBase {
  explicit OpImpBase(xir::Op* op1, xir::Attrs* attrs1)
      : op{op1}, attrs{attrs1} {};
  xir::Op* op;
  xir::Attrs* attrs;
};

template <typename T>
struct arg_converter_t {
  static T convert(vart::OpImpArg& arg, const xir::OpDef* opdef,
                   size_t arg_index);
};

template <typename R, typename T, typename... Args, size_t... Index>
int calculate_proxy0(vart::TensorBuffer* output, T* self,
                     int (T::*f)(simple_tensor_buffer_t<R>, Args...),
                     std::vector<vart::OpImpArg>& inputs,
                     std::integer_sequence<size_t, Index...> int_seq) {
  auto simple_output = simple_tensor_buffer_t<R>::create(output);
  auto op_def = self->op->get_opdef();
  CHECK_EQ(op_def->input_args().size(), sizeof...(Index))
      << "num of input arguments mismatch."
      << "op: " << to_string(op_def);
  return std::invoke(
      f, self, simple_output,
      (arg_converter_t<Args>::convert(inputs[Index], op_def, Index))...);
}

template <typename R, typename T, typename... Args>
int calculate_proxy(vart::TensorBuffer* output, T* self,
                    int (T::*f)(simple_tensor_buffer_t<R>, Args...),
                    std::vector<vart::OpImpArg>& inputs) {
  return calculate_proxy0(output, self, f, inputs,
                          std::make_index_sequence<sizeof...(Args)>());
}

template <typename T>
T arg_converter_t<T>::convert(vart::OpImpArg& arg, const xir::OpDef* opdef,
                              size_t arg_index) {
  LOG(FATAL) << "cannot convert argument: T=" << typeid(T).name();
  return T();
}

template <typename T>
struct arg_converter_t<simple_tensor_buffer_t<T>> {
  static simple_tensor_buffer_t<T> convert(vart::OpImpArg& arg,
                                           const xir::OpDef* opdef,
                                           size_t arg_index) {
    auto input_arg_defs = opdef->input_args();
    CHECK_LT(arg_index, input_arg_defs.size()) << "wrong number of argument";
    auto& arg_def = input_arg_defs[arg_index];
    CHECK(arg_def.occur_type == xir::OpArgDef::REQUIRED)
        << "it must be required single argument.";
    // CHECK_EQ(arg.arg_name, arg_def.name) << "name mismatch";
    CHECK_EQ(arg.args.size(), 1u)
        << "it must be single argument. name=" << arg.arg_name;
    return simple_tensor_buffer_t<T>::create(arg.args[0]);
  }
};

template <typename T>
struct arg_converter_t<std::unique_ptr<simple_tensor_buffer_t<T>>> {
  static std::unique_ptr<simple_tensor_buffer_t<T>> convert(
      vart::OpImpArg& arg, const xir::OpDef* opdef, size_t arg_index) {
    auto input_arg_defs = opdef->input_args();
    CHECK_LT(arg_index, input_arg_defs.size()) << "wrong number of argument";
    auto& arg_def = input_arg_defs[arg_index];
    CHECK(arg_def.occur_type == xir::OpArgDef::OPTIONAL)
        << "it must be required single argument.";
    CHECK_EQ(arg.arg_name, arg_def.name)
        << "name mismatch! opdef:" << to_string(opdef);
    CHECK_LE(arg.args.size(), 1u)
        << "it must be single argument. name=" << arg.arg_name
        << ". opdef=" << to_string(opdef);
    if (arg.args.size() == 1) {
      return std::make_unique<simple_tensor_buffer_t<T>>(
          simple_tensor_buffer_t<T>::create(arg.args[0]));
    }
    return nullptr;
  }
};

template <typename T>
struct arg_converter_t<std::vector<simple_tensor_buffer_t<T>>> {
  static std::vector<simple_tensor_buffer_t<T>> convert(vart::OpImpArg& arg,
                                                        const xir::OpDef* opdef,
                                                        size_t arg_index) {
    auto input_arg_defs = opdef->input_args();
    CHECK_LT(arg_index, input_arg_defs.size()) << "wrong number of argument";
    auto& arg_def = input_arg_defs[arg_index];
    CHECK(arg_def.occur_type == xir::OpArgDef::REQUIRED_AND_REPEATED)
        << "it must be required_and_repeated argument.";
    CHECK_EQ(arg.arg_name, arg_def.name)
        << "name mismatch! opdef:" << to_string(opdef);
    auto ret = std::vector<simple_tensor_buffer_t<T>>();
    ret.reserve(arg.args.size());
    for (auto tb : arg.args) {
      ret.emplace_back(simple_tensor_buffer_t<T>::create(tb));
    };
    return ret;
  }
};

template <typename T>
vart_op_imp_t make_vart_opt_imp() {
  vart_op_imp_t ret;
  ret.init = [](const xir_op_t op, xir_attrs_t attrs) -> void* {
    return reinterpret_cast<void*>(new T(reinterpret_cast<xir::Op*>(op),
                                         reinterpret_cast<xir::Attrs*>(attrs)));
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
    return calculate_proxy(output, self, &T::calculate, inputs);
  };
  return ret;
}

}  // namespace experimental
}  // namespace vart
#endif

// Local Variables:
// mode: c++
// coding: utf-8-unix
// End:
