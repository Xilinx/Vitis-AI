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

#include "./util.hpp"
#include "xir/op/op_imp.hpp"
namespace xir {
class c_api {
 public:
  static xir_string_t op_get_name(xir_op_t op) {
    auto self = static_cast<xir::OpImp*>(op);
    return conv_to_xir_string(self->name_);
  }
  static xir_string_t op_get_type(xir_op_t op) {
    auto self = static_cast<xir::OpImp*>(op);
    return conv_to_xir_string(self->type_);
  }
  static xir_attrs_t op_get_attrs(xir_op_t op) {
    auto self = static_cast<xir::OpImp*>(op);
    return static_cast<xir_op_t>(self->attrs_.get());
  }
};
}  // namespace xir

extern "C" xir_string_t xir_op_get_name(xir_op_t op) {
  return xir::c_api::op_get_name(op);
}

extern "C" xir_string_t xir_op_get_type(xir_op_t op) {
  return xir::c_api::op_get_type(op);
}

extern "C" int xir_op_get_input_num(xir_op_t op) {
  return static_cast<xir::Op*>(op)->get_input_num();
}

extern "C" int xir_op_get_input_num_by_name(xir_op_t op, char* arg_name) {
  return static_cast<xir::Op*>(op)->get_input_num(std::string(arg_name));
}

extern "C" void xir_op_get_input_ops(xir_op_t op, xir_string_t arg_name,
                                     void* data, xir_get_op_callback_t cb) {
  auto self = static_cast<xir::Op*>(op);
  for (auto o : self->get_input_ops(conv_to_std_string(arg_name))) {
    cb(data, o);
  }
}
extern "C" xir_op_t xir_op_get_input_op(xir_op_t op, xir_string_t arg_name,
                                        int idx) {
  return static_cast<xir_op_t>(static_cast<xir::Op*>(op)->get_input_op(
      conv_to_std_string(arg_name), idx));
}

extern "C" void xir_op_replace_input_op(xir_op_t op, xir_op_t old_op,
                                        xir_op_t new_op) {
  return static_cast<xir::Op*>(op)->replace_input_op(
      static_cast<xir::Op*>(old_op), static_cast<xir::Op*>(new_op));
}

extern "C" int xir_op_get_fanout_num(xir_op_t op) {
  return static_cast<xir::Op*>(op)->get_fanout_num();
}

extern "C" void xir_op_get_fanout_ops(xir_op_t op, void* data,
                                      xir_get_op_callback_t cb) {
  auto self = static_cast<xir::Op*>(op);
  for (auto o : self->get_fanout_ops()) {
    cb(data, o);
  }
}

extern "C" xir_tensor_t xir_op_get_input_tensor(xir_op_t op,
                                                xir_string_t arg_name,
                                                int idx) {
  return static_cast<xir_tensor_t>(static_cast<xir::Op*>(op)->get_input_tensor(
      conv_to_std_string(arg_name), idx));
}

extern "C" xir_tensor_t xir_op_get_output_tensor(xir_op_t op) {
  return static_cast<xir_tensor_t>(
      static_cast<xir::Op*>(op)->get_output_tensor());
}

extern "C" void xir_op_replace_output_tensor(xir_op_t op,
                                             xir_tensor_t tensor_new) {
  return static_cast<xir::Op*>(op)->replace_output_tensor(
      std::unique_ptr<xir::Tensor>(static_cast<xir::Tensor*>(tensor_new)));
}

extern "C" xir_graph_t xir_op_get_graph(xir_op_t op) {
  return static_cast<xir_graph_t>(static_cast<xir::Op*>(op)->get_graph());
}

extern "C" int xir_op_has_attrs(xir_op_t op) {
  return static_cast<xir::Op*>(op)->has_attrs();
}

extern "C" xir_attrs_t xir_op_get_attrs(xir_op_t op) {
  return xir::c_api::op_get_attrs(op);
}

extern "C" xir_op_t xir_op_set_attrs(xir_op_t op, xir_attrs_t attrs) {
  return static_cast<xir_op_t>(static_cast<xir::Op*>(op)->set_attrs(
      std::unique_ptr<xir::Attrs>(static_cast<xir::Attrs*>(attrs))));
}

extern "C" int xir_op_has_attr(xir_op_t op, const char* key) {
  return static_cast<xir::Op*>(op)->has_attr(std::string(key));
}

extern "C" void xir_op_shape_infer(xir_op_t op) {
  return static_cast<xir::Op*>(op)->shape_infer();
}

extern "C" xir_op_def_t xir_op_get_opdef(xir_op_t op) {
  return static_cast<xir_op_def_t>(
      const_cast<xir::OpDef*>(static_cast<xir::Op*>(op)->get_opdef()));
}
// extern "C" void xir_op_print_info(xir_op_t op) {
//  return static_cast<xir::Op*>(op)->print_info();
// }
