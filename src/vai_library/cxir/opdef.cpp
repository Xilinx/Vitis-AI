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

#include "util.hpp"
#include "xir/attrs/attr_def.hpp"
#include "xir/cxir.h"
#include "xir/op/op.hpp"
#include "xir/op/op_def.hpp"
#include "xir/util/data_type.hpp"
using namespace std;
// clang-format off
#include "./attrs2_primitive_values.hpp"
#include "./attrs2_any.hpp"
#include "./attrs2_vec.hpp"
#include "./attrs2_map.hpp"
// clang-format on
#include "./attrs2_any.tcc"

namespace xir {
class c_api {};
}  // namespace xir

extern "C" xir_op_def_t xir_op_def_create(xir_string_t name) {
  auto ret = make_unique<xir::OpDef>(conv_to_std_string(name));
  return static_cast<xir_op_def_t>(ret.release());
}

extern "C" void xir_op_def_destroy(xir_op_def_t self) {
  auto def = static_cast<xir::OpDef*>(self);
  delete def;
}

extern "C" xir_string_t xir_op_def_get_name(xir_op_def_t self) {
  auto def = static_cast<xir::OpDef*>(self);
  return conv_to_xir_string(def->name());
}

extern "C" xir_op_arg_def_t xir_op_arg_def_create(
    xir_string_t name, xir_op_arg_def_occurence_type_t occurence_type,
    enum xir_tensor_data_type_t data_type, const int32_t bit_width,
    xir_string_t annotation) {
  xir::DataType::Type dtype = (xir::DataType::Type)data_type;
  xir::OpArgDef::OccurenceType otype =
      (xir::OpArgDef::OccurenceType)occurence_type;
  auto ret = new xir::OpArgDef{conv_to_std_string(name), otype,
                               // xir::DataType(dtype, bit_width),
                               dtype, conv_to_std_string(annotation)};
  return static_cast<xir_op_arg_def_t>(ret);
}

extern "C" void xir_op_arg_def_destroy(xir_op_arg_def_t self) {
  auto def = static_cast<xir::OpArgDef*>(self);
  delete def;
}

extern "C" xir_string_t xir_op_arg_def_get_name(xir_op_arg_def_t self) {
  auto def = static_cast<xir::OpArgDef*>(self);
  return conv_to_xir_string(def->name);
}

extern "C" void xir_op_def_add_input_arg(xir_op_def_t self,
                                         xir_op_arg_def_t arg) {
  auto def = static_cast<xir::OpDef*>(self);
  auto argdef = static_cast<xir::OpArgDef*>(arg);
  def->add_input_arg(*argdef);
}

extern "C" xir_attr_def_t xir_attr_def_create(
    xir_string_t name1,  //
    xir_type_index_t data_type1,
    xir_attr_def_occurence_type_t otype1,  //
    uint32_t list_length,                  //
    xir_string_t annotation1, xir_attr_value_t default_value1) {
  // xir::AttrDef::OccurenceType otype = (xir::AttrDef::OccurenceType)otype1;
  string name = conv_to_std_string(name1);
  string annotation = conv_to_std_string(annotation1);
  auto& data_type = *static_cast<type_index*>(data_type1);
  auto occur_type = xir::AttrDef::OccurenceType(otype1);
  auto default_value = convert<xir_attr_value_t, any>::conv(default_value1);
  auto ret =
      new xir::AttrDef{/// Name of the op attribute
                       name,
                       /// Data type
                       data_type,
                       /// Occurence type
                       occur_type,
                       /// List size for validation, 0 for variable length
                       list_length,
                       /// Some comments
                       annotation,
                       /// Default value of the attribute
                       default_value};
  return static_cast<xir_attr_def_t>(ret);
}

extern "C" void xir_attr_def_destroy(xir_attr_def_t self) {
  auto def = static_cast<xir::AttrDef*>(self);
  delete def;
}

extern "C" xir_string_t xir_attr_def_get_name(xir_attr_def_t self) {
  auto def = static_cast<xir::AttrDef*>(self);
  return conv_to_xir_string(def->name);
}

extern "C" void xir_op_def_add_attr(xir_op_def_t self, xir_attr_def_t arg) {
  auto def = static_cast<xir::OpDef*>(self);
  auto attr_def = static_cast<xir::AttrDef*>(arg);
  def->add_attr(*attr_def);
}

extern "C" xir_type_index_t XIR_TYPE_INDEX_BOOL() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_BOOL);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_INT8() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_INT8);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_UINT8() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_UINT8);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_INT16() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_INT16);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_UINT16() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_UINT16);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_INT32() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_INT32);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_UINT32() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_UINT32);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_INT64() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_INT64);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_UINT64() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_UINT64);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_FLOAT() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_FLOAT);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_DOUBLE() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_DOUBLE);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_STRING() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_STRING);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_BYTES() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_BYTES);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_BOOL_VEC() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_BOOL_VEC);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_INT8_VEC() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_INT8_VEC);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_UINT8_VEC() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_UINT8_VEC);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_INT16_VEC() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_INT16_VEC);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_UINT16_VEC() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_UINT16_VEC);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_INT32_VEC() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_INT32_VEC);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_UINT32_VEC() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_UINT32_VEC);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_INT64_VEC() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_INT64_VEC);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_UINT64_VEC() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_UINT64_VEC);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_FLOAT_VEC() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_FLOAT_VEC);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_DOUBLE_VEC() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_DOUBLE_VEC);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_STRING_VEC() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_STRING_VEC);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_BYTES_VEC() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_BYTES_VEC);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_MAP_STR_2_INT32() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_MAP_STR_2_INT32);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_MAP_STR_2_VEC_CHAR() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_MAP_STR_2_VEC_CHAR);
}
extern "C" xir_type_index_t XIR_TYPE_INDEX_MAP_STR_2_STR() {
  return static_cast<xir_type_index_t>(&xir::TYPE_INDEX_MAP_STR_2_STR);
}

extern "C" void xir_op_def_set_annotation(xir_op_def_t self,
                                          xir_string_t annotation) {
  auto def = static_cast<xir::OpDef*>(self);
  def->set_annotation(conv_to_std_string(annotation));
}

extern "C" void xir_op_def_set_shape_infer(xir_op_def_t self, op_callback_t fun,
                                           void* data) {
  auto def = static_cast<xir::OpDef*>(self);
  def->set_shape_infer(
      [fun, data](xir::Op* op) { fun(data, static_cast<xir_op_t>(op)); });
}

extern "C" void xir_op_def_add_constraint(xir_op_def_t self, op_callback_t fun,
                                          void* data) {
  auto def = static_cast<xir::OpDef*>(self);
  def->set_shape_infer(
      [fun, data](xir::Op* op) { fun(data, static_cast<xir_op_t>(op)); });
}
