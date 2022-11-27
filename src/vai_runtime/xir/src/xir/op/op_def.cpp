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

#include "xir/op/op_def.hpp"

#include "UniLog/UniLog.hpp"
#include "xir/op/op.hpp"

namespace xir {

OpDef::OpDef(const std::string& name)
    : name_(name) {}

OpDef::OpDef(const std::string& name,
             const std::vector<OpArgDef>& input_args,
             const std::vector<AttrDef>& attrs,
             const std::function<void(Op* op)>& shape_infer,
             const std::string& annotation)
    : name_(name),
      input_args_(input_args),
      attrs_(attrs),
      shape_infer_(shape_infer),
      annotation_(annotation) {}

OpDef& OpDef::inherit_from(
  const std::function<void(xir::OpDef&)>& op_def) {
  op_def(*this);
  return *this;
}

OpDef& OpDef::add_attr(const AttrDef& attr_def) {
  UNI_LOG_CHECK(
    std::none_of(this->attrs_.begin(),
                 this->attrs_.end(),
                 [&](auto attr) {
                   return attr.name == attr_def.name;
                 }),
    XIR_INVALID_ARG_OCCUR)
    << "Attr " << attr_def.name << " has been set twice;";
  attrs_.push_back(attr_def);
  return *this;
}

OpDef& OpDef::add_input_arg(const OpArgDef& arg_def) {
  UNI_LOG_CHECK(
    std::none_of(this->input_args_.begin(),
                 this->input_args_.end(),
                 [&](auto arg) {
                   return arg.name == arg_def.name;
                 }),
    XIR_INVALID_ARG_OCCUR)
    << "Arg " << arg_def.name << " has been set twice;";
  input_args_.push_back(arg_def);
  return *this;
}

OpDef& OpDef::set_annotation(
  const std::string& annotation) {
  annotation_ = annotation;
  return *this;
}

OpDef& OpDef::set_shape_infer(
  const std::function<void(Op* op)>& shape_infer) {
  shape_infer_ = shape_infer;
  return *this;
}

OpDef& OpDef::add_constraint(
  const std::function<void(Op* op)>& constraint) {
  constraints_.push_back(constraint);
  return *this;
}

const std::string& OpDef::name() const {
  return name_;
}

const std::vector<OpArgDef>& OpDef::input_args() const {
  return input_args_;
}

const std::vector<AttrDef>& OpDef::attrs() const {
  return attrs_;
}

const std::function<void(Op* op)>&
OpDef::shape_infer() const {
  return shape_infer_;
}

const std::vector<std::function<void(Op* op)>>&
OpDef::constraints() const {
  return constraints_;
}

const std::string& OpDef::annotation() const {
  return annotation_;
}
}  // namespace xir