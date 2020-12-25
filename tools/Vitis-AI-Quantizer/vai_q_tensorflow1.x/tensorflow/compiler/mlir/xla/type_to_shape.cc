/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/mlir/xla/type_to_shape.h"

#include <string>

#include "mlir/IR/AffineMap.h"  // TF:local_config_mlir
#include "mlir/IR/Diagnostics.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Support/DebugStringHelper.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

using mlir::IntegerType;
using mlir::MemRefType;
using mlir::RankedTensorType;
using mlir::VectorType;
using tensorflow::int64;
using xla::PrimitiveType;
using xla::ShapeUtil;

namespace xla {

PrimitiveType TypeToPrimitiveType(mlir::Type type) {
  switch (type.getKind()) {
    case mlir::StandardTypes::BF16:
      return PrimitiveType::BF16;
    case mlir::StandardTypes::F16:
      return PrimitiveType::F16;
    case mlir::StandardTypes::F32:
      return PrimitiveType::F32;
    case mlir::StandardTypes::F64:
      return PrimitiveType::F64;
    case mlir::StandardTypes::Integer: {
      const auto integer = type.cast<IntegerType>();
      switch (integer.getWidth()) {
        case 1:
          return PrimitiveType::PRED;
        case 8:
          return PrimitiveType::S8;
        case 16:
          return PrimitiveType::S16;
        case 32:
          return PrimitiveType::S32;
        case 64:
          return PrimitiveType::S64;
        default:
          return PrimitiveType::PRIMITIVE_TYPE_INVALID;
      }
    }
    default:
      return PrimitiveType::PRIMITIVE_TYPE_INVALID;
  }
}

Shape TypeToShape(mlir::Type type) {
  PrimitiveType ptype = TypeToPrimitiveType(type);
  if (ptype != PrimitiveType::PRIMITIVE_TYPE_INVALID)
    return ShapeUtil::MakeShape(ptype, {});

  switch (type.getKind()) {
    case mlir::StandardTypes::BF16:
    case mlir::StandardTypes::F32:
    case mlir::StandardTypes::F64:
    case mlir::StandardTypes::Integer: {
      auto* context = type.getContext();
      mlir::emitError(mlir::UnknownLoc::get(context))
          << "lowering should have been handled by primitive type lowering for "
          << debugString(type);
      break;
    }
    case mlir::StandardTypes::Vector: {
      const auto v = type.cast<VectorType>();
      llvm::SmallVector<int64, 4> span(v.getShape().begin(),
                                       v.getShape().end());
      mlir::Type element_type = v.getElementType();
      PrimitiveType primitive_type = TypeToPrimitiveType(element_type);
      if (primitive_type != PrimitiveType::PRIMITIVE_TYPE_INVALID)
        return ShapeUtil::MakeShape(primitive_type, span);
      break;
    }
    case mlir::StandardTypes::MemRef: {
      const auto m = type.cast<MemRefType>();
      llvm::SmallVector<int64, 6> span(m.getShape().begin(),
                                       m.getShape().end());
      mlir::Type element_type = m.getElementType();
      // Treat a memref of a vector as if it was a memref of primitive type with
      // the vector dimensions at the end.
      if (auto v = element_type.dyn_cast<VectorType>()) {
        element_type = v.getElementType();
        span.insert(span.end(), v.getShape().begin(), v.getShape().end());
      }
      PrimitiveType primitive_type = TypeToPrimitiveType(element_type);
      if (primitive_type == PrimitiveType::PRIMITIVE_TYPE_INVALID) break;
      // For the primitive type case, the shape of the memref is similar to the
      // vector type case (i.e., it is, modulo the layout, the same dimensions
      // and primitive type).
      // Currently we only return shapes for identity affine maps.
      // TODO(andydavis) Map affine map layout function to XLA layout.
      if (m.getAffineMaps().empty() ||
          (m.getAffineMaps().size() == 1 && m.getAffineMaps()[0].isIdentity()))
        return ShapeUtil::MakeShape(primitive_type, span);
      break;
    }
    case mlir::StandardTypes::RankedTensor: {
      // TODO(jpienaar): This is only handling the base case with primitive
      // element type.
      const auto t = type.cast<RankedTensorType>();
      llvm::SmallVector<int64, 4> span(t.getShape().begin(),
                                       t.getShape().end());
      // Only fully static shapes are supported.
      // TODO(b/115638799): Update once xla::Shape can support dynamic shapes.
      if (std::find(t.getShape().begin(), t.getShape().end(), -1) !=
          t.getShape().end())
        break;
      mlir::Type element_type = t.getElementType();
      PrimitiveType primitive_type = TypeToPrimitiveType(element_type);
      // Only primitive element type supported.
      if (primitive_type != PrimitiveType::PRIMITIVE_TYPE_INVALID)
        return ShapeUtil::MakeShape(primitive_type, span);
      break;
    }
    default:
      break;
  }
  // Return empty XLA shape to signify error. No MLIR Type maps to a empty
  // Shape.
  return {};
}

}  // namespace xla
