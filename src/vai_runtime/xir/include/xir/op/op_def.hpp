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

#include <algorithm>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "xir/attrs/attr_def.hpp"
#include "xir/util/any.hpp"
#include "xir/util/data_type.hpp"

#define XIR_REGISTER_OPS(...)                                                  \
  extern "C" void register_ops(xir::OpDefFactory* self) {                      \
    auto ops = std::vector<xir::OpDef>{__VA_ARGS__};                           \
    std::for_each(ops.begin(), ops.end(),                                      \
                  [self](const xir::OpDef& def) { self->register_h(def); });   \
  }

namespace xir {

/*
 *@struct OpArgDef
 *@brief Op argument definition
 *This struct defines an input argument of an op.
 */
struct XIR_DLLESPEC OpArgDef {
  /**
   * @brief Element Occurence Specifier
   */
  enum OccurenceType {
    /// Once and only once
    REQUIRED,
    /// Never or once
    OPTIONAL,
    /// No limitation
    REPEATED,
    /// At least once
    REQUIRED_AND_REPEATED,
    NUM
  };
  /// MSVC NOTE: member variable cannot be const, because
  /// vector<AttrDef> in OpDef requires it is copiable.  // when
  /// exported with XIR_DLLESPEC, a special default copy constructor
  /// is created, which is deleted.
  //

  /// Name of the op argument
  std::string name;
  /// Occurence type
  OccurenceType occur_type;
  /// DataType
  xir::DataType::Type data_type;
  /// Some comments
  std::string annotation;
};

/*
 *@class OpDef
 *@brief Op definition
 *This class defines an op.
 */
class XIR_DLLESPEC OpDef {
 public:
  /// Create a definition of an op by name
  OpDef(const std::string& name);
  /// Create a definition of an op by name, inputs,
  /// attributes, shape_infer function and annotation.
  OpDef(const std::string& name, const std::vector<OpArgDef>& input_args,
        const std::vector<AttrDef>& attrs,
        const std::function<void(Op* op)>& shape_infer,
        const std::string& annotation);
  OpDef() = delete;
  ~OpDef() = default;

  /// Update the current op definition with a function
  OpDef& inherit_from(const std::function<void(xir::OpDef&)>&);
  /// Add an argument of one of the inputs of this op definition
  OpDef& add_input_arg(const OpArgDef&);
  /// Add an attribute definition which may be required of this operator
  OpDef& add_attr(const AttrDef&);
  /// Set the description of this operator
  OpDef& set_annotation(const std::string&);
  /// Set the shape infer function of this operator
  OpDef& set_shape_infer(const std::function<void(Op* op)>&);
  /// Add a constraint which must be met if you want to create
  /// a new operator according to this op definition
  OpDef& add_constraint(const std::function<void(Op* op)>&);

  /// Get the type of this operator
  const std::string& name() const;
  /// Get input argument definitions
  const std::vector<OpArgDef>& input_args() const;
  /// Get attribue definitions
  const std::vector<AttrDef>& attrs() const;
  /// Get the shape infer function of this operator
  const std::function<void(Op* op)>& shape_infer() const;
  /// Get the constraints functions of this operator
  const std::vector<std::function<void(Op* op)>>& constraints() const;
  /// Get annotation
  const std::string& annotation() const;

 private:
  // MSVC NOTE: when export with XIR_DLLESPEC, we must not have
  // variable constructor syntax here, it causes `operator=()` deleted.
  std::string name_;
  std::vector<OpArgDef> input_args_;
  std::vector<AttrDef> attrs_;
  std::function<void(Op* op)> shape_infer_;
  std::vector<std::function<void(Op* op)>> constraints_;
  std::string annotation_;
};

class XIR_DLLESPEC OpDefFactory {
 public:
  virtual void register_h(const OpDef& def) = 0;

 public:
  virtual ~OpDefFactory() = default;
};

}  // namespace xir
