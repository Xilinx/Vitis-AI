//===- Matchers.h - Various common matchers ---------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file provides a simple and efficient mechanism for performing general
// tree-based pattern matching over MLIR. This mechanism is inspired by LLVM's
// include/llvm/IR/PatternMatch.h.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MATCHERS_H
#define MLIR_MATCHERS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include <type_traits>

namespace mlir {

namespace detail {

/// The matcher that matches a certain kind of Attribute and binds the value
/// inside the Attribute.
template <
    typename AttrClass,
    // Require AttrClass to be a derived class from Atribute and get its
    // value type
    typename ValueType =
        typename std::enable_if<std::is_base_of<Attribute, AttrClass>::value,
                                AttrClass>::type::ValueType,
    // Require the ValueType is not void
    typename = typename std::enable_if<!std::is_void<ValueType>::value>::type>
struct attr_value_binder {
  ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  attr_value_binder(ValueType *bv) : bind_value(bv) {}

  bool match(const Attribute &attr) {
    if (auto intAttr = attr.dyn_cast<AttrClass>()) {
      *bind_value = intAttr.getValue();
      return true;
    }
    return false;
  }
};

/// The matcher that matches a constant foldable operation that has no side
/// effect, no operands and produces a single result.
template <typename AttrT> struct constant_op_binder {
  AttrT *bind_value;

  /// Creates a matcher instance that binds the constant attribute value to
  /// bind_value if match succeeds.
  constant_op_binder(AttrT *bind_value) : bind_value(bind_value) {}

  bool match(Operation *op) {
    if (op->getNumOperands() > 0 || op->getNumResults() != 1)
      return false;
    if (!op->hasNoSideEffect())
      return false;

    SmallVector<OpFoldResult, 1> foldedOp;
    if (succeeded(op->fold(/*operands=*/llvm::None, foldedOp))) {
      if (auto attr = foldedOp.front().dyn_cast<Attribute>()) {
        if ((*bind_value = attr.dyn_cast<AttrT>()))
          return true;
      }
    }
    return false;
  }
};

/// The matcher that matches a constant scalar / vector splat / tensor splat
/// integer operation and binds the constant integer value.
struct constant_int_op_binder {
  IntegerAttr::ValueType *bind_value;

  /// Creates a matcher instance that binds the value to bv if match succeeds.
  constant_int_op_binder(IntegerAttr::ValueType *bv) : bind_value(bv) {}

  bool match(Operation *op) {
    Attribute attr;
    if (!constant_op_binder<Attribute>(&attr).match(op))
      return false;
    auto type = op->getResult(0)->getType();

    if (type.isa<IntegerType>()) {
      return attr_value_binder<IntegerAttr>(bind_value).match(attr);
    }
    if (type.isa<VectorType>() || type.isa<RankedTensorType>()) {
      if (auto splatAttr = attr.dyn_cast<SplatElementsAttr>()) {
        return attr_value_binder<IntegerAttr>(bind_value)
            .match(splatAttr.getSplatValue());
      }
    }
    return false;
  }
};

// The matcher that matches a given target constant scalar / vector splat /
// tensor splat integer value.
template <int64_t TargetValue> struct constant_int_value_matcher {
  bool match(Operation *op) {
    APInt value;

    return constant_int_op_binder(&value).match(op) && TargetValue == value;
  }
};

/// The matcher that matches a certain kind of op.
template <typename OpClass> struct op_matcher {
  bool match(Operation *op) { return isa<OpClass>(op); }
};

} // end namespace detail

/// Entry point for matching a pattern over a Value.
template <typename Pattern>
inline bool matchPattern(Value *value, const Pattern &pattern) {
  // TODO: handle other cases
  if (auto *op = value->getDefiningOp())
    return const_cast<Pattern &>(pattern).match(op);
  return false;
}

/// Entry point for matching a pattern over an Operation.
template <typename Pattern>
inline bool matchPattern(Operation *op, const Pattern &pattern) {
  return const_cast<Pattern &>(pattern).match(op);
}

/// Matches a constant holding a scalar/vector/tensor integer (splat) and
/// writes the integer value to bind_value.
inline detail::constant_int_op_binder
m_ConstantInt(IntegerAttr::ValueType *bind_value) {
  return detail::constant_int_op_binder(bind_value);
}

/// Matches a value from a constant foldable operation and writes the value to
/// bind_value.
template <typename AttrT>
inline detail::constant_op_binder<AttrT> m_Constant(AttrT *bind_value) {
  return detail::constant_op_binder<AttrT>(bind_value);
}

/// Matches a constant scalar / vector splat / tensor splat integer one.
inline detail::constant_int_value_matcher<1> m_One() {
  return detail::constant_int_value_matcher<1>();
}

/// Matches the given OpClass.
template <typename OpClass> inline detail::op_matcher<OpClass> m_Op() {
  return detail::op_matcher<OpClass>();
}

/// Matches a constant scalar / vector splat / tensor splat integer zero.
inline detail::constant_int_value_matcher<0> m_Zero() {
  return detail::constant_int_value_matcher<0>();
}

} // end namespace mlir

#endif // MLIR_MATCHERS_H
