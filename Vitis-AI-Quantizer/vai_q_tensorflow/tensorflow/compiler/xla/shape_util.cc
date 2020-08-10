/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/shape_util.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/overflow_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/iterator_range.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/regexp.h"

namespace xla {

using absl::StrAppend;
using absl::StrCat;

string ShapeIndex::ToString() const { return ShapeIndexView(*this).ToString(); }

string ShapeIndexView::ToString() const {
  return StrCat("{", absl::StrJoin(indices_, ","), "}");
}

bool ShapeIndexView::operator==(const ShapeIndexView& other) const {
  return indices_ == other.indices_;
}

bool ShapeIndexView::operator!=(const ShapeIndexView& other) const {
  return !(*this == other);
}

std::ostream& operator<<(std::ostream& out, const ShapeIndex& shape_index) {
  out << shape_index.ToString();
  return out;
}

std::ostream& operator<<(std::ostream& out, const ShapeIndexView& shape_index) {
  out << shape_index.ToString();
  return out;
}

bool ShapeIndexView::StartsWith(ShapeIndexView prefix) const {
  return size() >= prefix.size() &&
         indices_.subspan(0, prefix.size()) == prefix.indices_;
}

/* static */ bool ShapeUtil::IsArrayPrimitiveType(
    PrimitiveType primitive_type) {
  return primitive_util::IsArrayType(primitive_type);
}

namespace {
// Constructs and returns the new shape with the given minor_to_major order in
// its Layout.
StatusOr<Shape> MakeShapeWithLayoutInternal(
    PrimitiveType element_type, absl::Span<const int64> dimensions,
    absl::Span<const int64> minor_to_major, absl::Span<const Tile> tiles,
    int64 element_size_in_bits, int64 memory_space) {
  if (dimensions.size() != minor_to_major.size()) {
    return InvalidArgument("Dimensions size is %ld, but layout size is %ld.",
                           dimensions.size(), minor_to_major.size());
  }
  if (element_type == OPAQUE_TYPE || element_type == TUPLE) {
    return InvalidArgument("Unsupported element type: %s",
                           PrimitiveType_Name(element_type));
  }
  TF_ASSIGN_OR_RETURN(Shape shape,
                      ShapeUtil::MakeValidatedShape(element_type, dimensions));
  if (element_size_in_bits ==
      ShapeUtil::ByteSizeOfPrimitiveType(element_type) * 8) {
    // Only set element_size_in_bits if it's different from the default value.
    element_size_in_bits = 0;
  }
  *shape.mutable_layout() = LayoutUtil::MakeLayout(
      minor_to_major, tiles, element_size_in_bits, memory_space);
  if (!shape.has_layout()) {
    return InvalidArgument("Shape has no layout.");
  }
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(shape));
  return shape;
}
}  // namespace

/* static */ bool ShapeUtil::Equal(const Shape& lhs, const Shape& rhs) {
  bool equal = Shape::Equal()(lhs, rhs);

  if (!equal && VLOG_IS_ON(3)) {
    VLOG(3) << "ShapeUtil::Equal differ: lhs = " << lhs.ShortDebugString()
            << ", rhs = " << rhs.ShortDebugString();
  }

  return equal;
}

/* static */ bool ShapeUtil::EqualIgnoringFpPrecision(const Shape& lhs,
                                                      const Shape& rhs) {
  bool equal = Shape::Equal().IgnoreFpPrecision()(lhs, rhs);
  if (!equal && VLOG_IS_ON(3)) {
    VLOG(3) << "ShapeUtil::EqualIgnoringFpPrecision differ: lhs = "
            << lhs.ShortDebugString() << ", rhs = " << rhs.ShortDebugString();
  }

  return equal;
}

/* static */ int64 ShapeUtil::TrueRank(const Shape& shape) {
  int64 accum = 0;
  for (int64 dimension : shape.dimensions()) {
    // We do not count zero dimensions.
    if (dimension != 1) {
      accum += 1;
    }
  }
  return accum;
}

/* static */ ProgramShape ShapeUtil::MakeProgramShape(
    std::initializer_list<Shape> parameters, Shape result) {
  ProgramShape program_shape;
  for (const Shape& shape : parameters) {
    *program_shape.add_parameters() = shape;
  }
  *program_shape.mutable_result() = std::move(result);
  return program_shape;
}

/* static */ Shape ShapeUtil::MakeShape(PrimitiveType element_type,
                                        absl::Span<const int64> dimensions) {
  return MakeValidatedShape(element_type, dimensions).ValueOrDie();
}

/* static */ Shape ShapeUtil::MakeShape(
    PrimitiveType element_type, absl::Span<const int64> dimensions,
    const std::vector<bool>& dynamic_dimensions) {
  return MakeValidatedShape(element_type, dimensions, dynamic_dimensions)
      .ValueOrDie();
}

/* static */ StatusOr<Shape> ShapeUtil::MakeValidatedShape(
    PrimitiveType element_type, absl::Span<const int64> dimensions) {
  CHECK(IsArrayPrimitiveType(element_type)) << element_type;
  Shape result;
  TF_RETURN_IF_ERROR(PopulateShape(element_type, dimensions, &result));
  return result;
}

/* static */ StatusOr<Shape> ShapeUtil::MakeValidatedShape(
    PrimitiveType element_type, absl::Span<const int64> dimensions,
    const std::vector<bool>& dynamic_dimensions) {
  TF_ASSIGN_OR_RETURN(Shape shape,
                      MakeValidatedShape(element_type, dimensions));
  for (int i = 0; i < dynamic_dimensions.size(); ++i) {
    shape.set_dynamic_dimension(i, dynamic_dimensions[i]);
  }
  return shape;
}

/* static */ Shape ShapeUtil::MakeShapeWithLayout(
    PrimitiveType element_type, absl::Span<const int64> dimensions,
    absl::Span<const int64> minor_to_major, absl::Span<const Tile> tiles,
    int64 element_size_in_bits, int64 memory_space) {
  return MakeShapeWithLayoutInternal(element_type, dimensions, minor_to_major,
                                     tiles, element_size_in_bits, memory_space)
      .ValueOrDie();
}

/* static */ Shape ShapeUtil::MakeShapeWithDescendingLayout(
    PrimitiveType element_type, absl::Span<const int64> dimensions) {
  std::vector<int64> layout(dimensions.size());
  std::iota(layout.rbegin(), layout.rend(), static_cast<int64>(0));
  return MakeShapeWithLayout(element_type, dimensions, layout);
}

/* static */ Shape ShapeUtil::MakeShapeWithSparseLayout(
    PrimitiveType element_type, absl::Span<const int64> dimensions,
    int64 max_sparse_elements) {
  CHECK(IsArrayPrimitiveType(element_type));
  Shape shape = ShapeUtil::MakeShape(element_type, dimensions);
  *shape.mutable_layout() = LayoutUtil::MakeSparseLayout(max_sparse_elements);
  TF_DCHECK_OK(ShapeUtil::ValidateShape(shape));
  return shape;
}

/* static */ Shape
ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
    const Shape& shape) {
  std::vector<int64> dims(shape.dimensions_size());
  for (int i = 0; i < shape.dimensions_size(); ++i) {
    dims[i] = shape.dimensions(LayoutUtil::Major(shape.layout(), i));
  }
  Shape new_shape = MakeShapeWithDescendingLayout(shape.element_type(), dims);
  // Since the physical layout is kept the same, the tiles and element size are
  // the same also.
  *new_shape.mutable_layout()->mutable_tiles() = shape.layout().tiles();
  new_shape.mutable_layout()->set_element_size_in_bits(
      shape.layout().element_size_in_bits());
  return new_shape;
}

/* static */ Status ShapeUtil::PopulateShape(PrimitiveType element_type,
                                             absl::Span<const int64> dimensions,
                                             Shape* shape) {
  shape->Clear();
  shape->set_element_type(element_type);
  for (int64 dimension : dimensions) {
    shape->add_dimensions(dimension);
  }
  LayoutUtil::SetToDefaultLayout(shape);
  return ValidateShape(*shape);
}

/* static */ Shape ShapeUtil::MakeTupleShape(absl::Span<const Shape> shapes) {
  Shape result;
  result.set_element_type(TUPLE);
  result.mutable_tuple_shapes()->reserve(shapes.size());
  for (const auto& shape : shapes) {
    AppendShapeToTuple(shape, &result);
  }
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(result));
  return result;
}

/* static */ Shape ShapeUtil::MakeOpaqueShape() {
  Shape result;
  result.set_element_type(OPAQUE_TYPE);
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(result));
  return result;
}

/* static */ Shape ShapeUtil::MakeTokenShape() {
  Shape result;
  result.set_element_type(TOKEN);
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(result));
  return result;
}

/* static */ void ShapeUtil::AppendShapeToTuple(const Shape& shape,
                                                Shape* tuple_shape) {
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(shape));
  *tuple_shape->add_tuple_shapes() = shape;
}

/* static */ void ShapeUtil::AppendMajorDimension(int bound, Shape* shape) {
  CHECK(LayoutUtil::IsDenseArray(*shape));
  shape->mutable_layout()->add_minor_to_major(shape->rank());
  shape->add_dimensions(bound);
  TF_DCHECK_OK(ValidateShape(*shape));
}

/* static */ bool ShapeUtil::ElementIsIntegral(const Shape& shape) {
  return primitive_util::IsIntegralType(shape.element_type());
}

/* static */ bool ShapeUtil::ElementIsIntegralWithBits(const Shape& shape,
                                                       int32 bits) {
  return ElementIsIntegral(shape) && ElementHasBitWidth(shape, bits);
}

/* static */ bool ShapeUtil::ElementHasBitWidth(const Shape& shape, int bits) {
  if (!shape.IsArray()) {
    return false;
  }
  return primitive_util::BitWidth(shape.element_type()) == bits;
}

/* static */ bool ShapeUtil::ElementIsSigned(const Shape& shape) {
  switch (shape.element_type()) {
    case S8:
    case S16:
    case S32:
    case S64:
    case F16:
    case BF16:
    case F32:
    case F64:
      return true;

    case PRED:
    case U8:
    case U16:
    case U32:
    case U64:
    case C64:
    case C128:
    case TUPLE:
    case OPAQUE_TYPE:
    case TOKEN:
      return false;

    default:
      LOG(FATAL) << "Unhandled element type " << shape.element_type();
  }
}

/* static */ bool ShapeUtil::ElementIsComplex(const Shape& shape) {
  return primitive_util::IsComplexType(shape.element_type());
}

/* static */ bool ShapeUtil::ElementIsFloating(const Shape& shape) {
  return primitive_util::IsFloatingPointType(shape.element_type());
}

/* static */ bool ShapeUtil::IsNestedTuple(const Shape& shape) {
  return shape.IsTuple() &&
         absl::c_any_of(shape.tuple_shapes(),
                        [](const Shape& s) { return s.IsTuple(); });
}

/* static */ bool ShapeUtil::IsEmptyTuple(const Shape& shape) {
  return shape.IsTuple() && TupleElementCount(shape) == 0;
}

/* static */ int64 ShapeUtil::TupleElementCount(const Shape& shape) {
  CHECK(shape.IsTuple()) << HumanString(shape);
  return shape.tuple_shapes_size();
}

/* static */ const Shape& ShapeUtil::GetTupleElementShape(const Shape& shape,
                                                          int64 index) {
  CHECK(shape.IsTuple());
  CHECK_GT(TupleElementCount(shape), index);
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(shape.tuple_shapes(index)));
  return shape.tuple_shapes(index);
}

/* static */ int64 ShapeUtil::SubshapeCount(const Shape& shape) {
  int64 n = 0;
  ForEachSubshape(shape, [&](const Shape& literal_subshape,
                             const ShapeIndex& index) { ++n; });
  return n;
}

/* static */ Shape ShapeUtil::SliceTuple(const Shape& tuple, int64 start,
                                         int64 limit) {
  TF_DCHECK_OK(ValidateShapeWithOptionalLayout(tuple));
  CHECK(tuple.IsTuple());
  CHECK_LE(start, TupleElementCount(tuple));
  CHECK_LE(limit, TupleElementCount(tuple));

  std::vector<Shape> new_elements(tuple.tuple_shapes().begin() + start,
                                  tuple.tuple_shapes().begin() + limit);
  return MakeTupleShape(new_elements);
}

// Returns the shape of a real or imaginary component.
/* static */ Shape ShapeUtil::ComplexComponentShape(
    const Shape& complex_shape) {
  CHECK(ElementIsComplex(complex_shape)) << HumanString(complex_shape);
  return ChangeElementType(complex_shape, primitive_util::ComplexComponentType(
                                              complex_shape.element_type()));
}

/* static */ int64 ShapeUtil::ElementsIn(const Shape& shape) {
  DCHECK(shape.IsArray()) << ShapeUtil::HumanString(shape);
  DCHECK_EQ(shape.dimensions_size(), shape.rank());
  if (shape.dimensions().size() == 1) {
    return shape.dimensions()[0];
  }
  return std::accumulate<decltype(shape.dimensions().begin()), int64>(
      shape.dimensions().begin(), shape.dimensions().end(), 1LL,
      std::multiplies<int64>());
}

/* static */ int64 ShapeUtil::ElementsInRecursive(const Shape& shape) {
  CHECK(shape.IsArray() || shape.IsTuple());
  if (shape.IsArray()) {
    return ElementsIn(shape);
  }
  int64 count = 0;
  for (const Shape& element_shape : shape.tuple_shapes()) {
    count += ElementsInRecursive(element_shape);
  }
  return count;
}

/* static */ bool ShapeUtil::HasPrimitiveType(const Shape& shape,
                                              PrimitiveType primitive_type) {
  if (shape.element_type() == primitive_type) {
    return true;
  }
  for (const Shape& element_shape : shape.tuple_shapes()) {
    if (HasPrimitiveType(element_shape, primitive_type)) {
      return true;
    }
  }
  return false;
}

/* static */ bool ShapeUtil::IsZeroElementArray(const Shape& shape) {
  return shape.IsArray() && ElementsIn(shape) == 0;
}

/* static */ bool ShapeUtil::IsScalarWithElementType(
    const Shape& shape, PrimitiveType element_type) {
  return IsScalar(shape) && shape.element_type() == element_type;
}

/* static */ string ShapeUtil::HumanString(const Shape& shape) {
  if (shape.IsTuple()) {
    string text = "(";
    const char* prefix = "";
    for (const Shape& elem_shape : shape.tuple_shapes()) {
      StrAppend(&text, prefix, HumanString(elem_shape));
      prefix = ", ";
    }
    text += ")";
    return text;
  }
  std::vector<string> dim_elements;
  for (int i = 0; i < shape.dimensions_size(); ++i) {
    if (shape.is_dynamic_dimension(i)) {
      dim_elements.push_back(StrCat("<=", shape.dimensions(i)));
    } else {
      dim_elements.push_back(StrCat(shape.dimensions(i)));
    }
  }
  return StrCat(
      primitive_util::LowercasePrimitiveTypeName(shape.element_type()), "[",
      absl::StrJoin(dim_elements, ","), "]");
}

/* static */ string ShapeUtil::HumanStringWithLayout(const Shape& shape) {
  if (shape.IsTuple()) {
    string text = "(";
    const char* prefix = "";
    for (const Shape& elem_shape : shape.tuple_shapes()) {
      StrAppend(&text, prefix, HumanStringWithLayout(elem_shape));
      prefix = ", ";
    }
    text += ")";
    return text;
  }
  string result = StrCat(
      primitive_util::LowercasePrimitiveTypeName(shape.element_type()), "[");
  for (int i = 0; i < shape.dimensions().size(); i++) {
    StrAppend(&result, (i > 0) ? "," : "",
              shape.is_dynamic_dimension(i) ? "<=" : "", shape.dimensions(i));
  }
  result += "]";
  if (IsScalar(shape)) {
    string layout_str = LayoutUtil::HumanString(shape.layout());
    // Don't print "{}" as layout for scalars.
    if (layout_str != "{}") {
      StrAppend(&result, layout_str);
    }
  } else if (shape.IsArray() && LayoutUtil::HasLayout(shape)) {
    StrAppend(&result, LayoutUtil::HumanString(shape.layout()));
  }
  return result;
}

/* static */ string ShapeUtil::HumanString(const ProgramShape& program_shape) {
  std::vector<string> parameters;
  for (auto& shape : program_shape.parameters()) {
    const int i = parameters.size();
    parameters.push_back(StrCat(i < program_shape.parameter_names_size()
                                    ? program_shape.parameter_names(i)
                                    : "(unknown)",
                                ": ", HumanString(shape)));
  }
  return StrCat("(", absl::StrJoin(parameters, ", "), ") -> ",
                HumanString(program_shape.result()));
}

/* static */ bool ShapeUtil::SameDimensions(const Shape& lhs,
                                            const Shape& rhs) {
  CHECK(lhs.IsArray());
  CHECK(rhs.IsArray());
  return absl::c_equal(lhs.dimensions(), rhs.dimensions());
}

/* static */ bool ShapeUtil::Compatible(const Shape& lhs, const Shape& rhs) {
  return Shape::Equal().IgnoreLayout()(lhs, rhs);
}

/* static */ bool ShapeUtil::CompatibleIgnoringElementType(const Shape& lhs,
                                                           const Shape& rhs) {
  return Shape::Equal().IgnoreElementType().IgnoreLayout()(lhs, rhs);
}

/* static */ bool ShapeUtil::CompatibleIgnoringFpPrecision(const Shape& lhs,
                                                           const Shape& rhs) {
  return Shape::Equal().IgnoreFpPrecision().IgnoreLayout()(lhs, rhs);
}

/* static */ int64 ShapeUtil::GetDimension(const Shape& shape,
                                           int64 dimension_number) {
  return shape.dimensions(GetDimensionNumber(shape, dimension_number));
}

/* static */ int64 ShapeUtil::GetDimensionNumber(const Shape& shape,
                                                 int64 dimension_number) {
  if (dimension_number < 0) {
    dimension_number += shape.rank();
  }
  CHECK_GE(dimension_number, 0);
  return dimension_number;
}

/* static */ int64 ShapeUtil::ByteSizeOfPrimitiveType(
    PrimitiveType primitive_type) {
  switch (primitive_type) {
    case PRED:
      return sizeof(int8);
    case S8:
      return sizeof(int8);
    case S16:
      return sizeof(int16);
    case S32:
      return sizeof(int32);
    case S64:
      return sizeof(int64);
    case U8:
      return sizeof(uint8);
    case U16:
      return sizeof(uint16);
    case U32:
      return sizeof(uint32);
    case U64:
      return sizeof(uint64);
    case BF16:
      return sizeof(float) / 2;
    case F16:
      return sizeof(float) / 2;
    case F32:
      return sizeof(float);
    case F64:
      return sizeof(double);
    case C64:
      return sizeof(complex64);
    case C128:
      return sizeof(complex128);
    case TOKEN:
      // Tokens require no space.
      return 0;
    case TUPLE:
    case OPAQUE_TYPE:
      LOG(FATAL) << PrimitiveType_Name(primitive_type)
                 << " primitive type has no definitive size";
    default:
      LOG(FATAL) << "Unhandled primitive type " << primitive_type;
  }
}

/* static */ int64 ShapeUtil::ByteSizeOf(const Shape& shape,
                                         int64 pointer_size) {
  TF_DCHECK_OK(ValidateShape(shape));
  if (shape.element_type() == TUPLE) {
    return ByteSizeOfTupleIndexTable(shape, pointer_size);
  } else if (shape.IsArray()) {
    int64 byte_size = ByteSizeOfElements(shape);
    if (LayoutUtil::IsSparseArray(shape)) {
      byte_size += ByteSizeOfSparseIndices(shape);
    }
    return byte_size;
  } else if (shape.element_type() == TOKEN) {
    return 0;
  } else if (shape.element_type() == OPAQUE_TYPE) {
    CHECK_GT(pointer_size, 0);
    return pointer_size;
  }
  LOG(FATAL) << PrimitiveType_Name(shape.element_type())
             << " primitive type has no definitive size";
}

/* static */ int64 ShapeUtil::ByteSizeOfTupleIndexTable(const Shape& shape,
                                                        int64 pointer_size) {
  TF_DCHECK_OK(ValidateShape(shape));
  CHECK_EQ(TUPLE, shape.element_type());
  CHECK_GT(pointer_size, 0);
  return pointer_size * shape.tuple_shapes_size();
}

/* static */ int64 ShapeUtil::ByteSizeOfElements(const Shape& shape) {
  TF_DCHECK_OK(ValidateShape(shape));
  CHECK(shape.IsArray());
  int64 allocated_element_count;

  if (LayoutUtil::IsSparseArray(shape)) {
    allocated_element_count = LayoutUtil::MaxSparseElements(shape.layout());
  } else {
    CHECK(LayoutUtil::IsDenseArray(shape)) << shape.ShortDebugString();
    allocated_element_count = ElementsIn(shape);
  }
  return allocated_element_count *
         ByteSizeOfPrimitiveType(shape.element_type());
}

/* static */ int64 ShapeUtil::ByteSizeOfSparseIndices(const Shape& shape) {
  TF_DCHECK_OK(ValidateShape(shape));
  CHECK(LayoutUtil::IsSparseArray(shape));
  return LayoutUtil::MaxSparseElements(shape.layout()) * shape.rank() *
         sizeof(int64);
}

/* static */ Status ShapeUtil::ValidateShapeWithOptionalLayoutInternal(
    const Shape& shape) {
  if (shape.element_type() == PRIMITIVE_TYPE_INVALID ||
      !PrimitiveType_IsValid(shape.element_type())) {
    return InvalidArgument("shape has invalid element type: %s",
                           shape.ShortDebugString());
  }
  if (shape.element_type() == TUPLE) {
    if (shape.dimensions_size() != 0) {
      return InvalidArgument("tuples must not have dimensions specified");
    }
    for (auto& element_shape : shape.tuple_shapes()) {
      TF_RETURN_IF_ERROR(
          ValidateShapeWithOptionalLayoutInternal(element_shape));
    }
    return Status::OK();
  }

  // Non-tuple shape.
  if (shape.tuple_shapes_size() > 0) {
    return InvalidArgument("non-tuple shape has tuple_shapes field");
  }

  // Tokens and opaques can should not have layout or dimensions.
  if (shape.element_type() == TOKEN || shape.element_type() == OPAQUE_TYPE) {
    if (shape.dimensions_size() != 0) {
      return InvalidArgument(
          "shape has %s element type, but has dimensions field: %s",
          primitive_util::LowercasePrimitiveTypeName(shape.element_type()),
          shape.ShortDebugString());
    }
    if (shape.has_layout()) {
      return InvalidArgument(
          "shape has %s element type, but has layout field: %s",
          primitive_util::LowercasePrimitiveTypeName(shape.element_type()),
          shape.ShortDebugString());
    }
    return Status::OK();
  }

  if (LayoutUtil::IsSparseArray(shape) && shape.rank() == 0) {
    return InvalidArgument("sparse arrays must have rank > 0");
  }
  for (int64 i = 0; i < shape.rank(); ++i) {
    int64 dimension = shape.dimensions(i);
    if (dimension < 0) {
      return InvalidArgument(
          "shape's dimensions must not be < 0; dimension at index %d was %d", i,
          dimension);
    }
  }

  TF_RETURN_IF_ERROR(ValidateShapeSize(shape));
  return Status::OK();
}

/* static */ Status ShapeUtil::ValidateShapeSize(const Shape& shape) {
  VLOG(3) << "Validating shape size: " << ShapeUtil::HumanString(shape);

  if (!shape.IsArray()) {
    return Status::OK();
  }

  // We can only reason about some aspects of array's shape if it has a valid
  // layout, these aspects will be ignored otherwise.
  bool shape_has_valid_layout = LayoutUtil::HasLayout(shape) &&
                                LayoutUtil::ValidateLayoutInShape(shape).ok();

  int64 shape_size = [&]() {
    if (shape_has_valid_layout && LayoutUtil::IsSparseArray(shape)) {
      int64 max_sparse_elements = LayoutUtil::MaxSparseElements(shape.layout());
      if (max_sparse_elements < 0) {
        return max_sparse_elements;
      }
      int64 sparse_elements_size = MultiplyWithoutOverflow(
          max_sparse_elements, ByteSizeOfPrimitiveType(shape.element_type()));
      if (sparse_elements_size < 0) {
        return sparse_elements_size;
      }
      int64 sparse_indices_size =
          MultiplyWithoutOverflow(max_sparse_elements, shape.rank());
      if (sparse_indices_size < 0) {
        return sparse_indices_size;
      }
      sparse_indices_size =
          MultiplyWithoutOverflow(sparse_indices_size, sizeof(int64));
      if (sparse_indices_size < 0) {
        return sparse_indices_size;
      }
      // At this point, both sparse_indices_size and sparse_elements_size are
      // non-negative, so we can easily check if adding them wraps.
      if (static_cast<uint64>(sparse_elements_size) +
              static_cast<uint64>(sparse_indices_size) >
          INT64_MAX) {
        return static_cast<int64>(-1);
      }
    }

    // This is intentionally unconditional: even if the shape is sparse, we want
    // to verify the densified version has a reasonable size.
    int64 dense_shape_size = 1;
    if (shape.dimensions().empty()) {
      return dense_shape_size;
    }

    absl::Span<const int64> shape_max_dimensions =
        AsInt64Slice(shape.dimensions());
    for (int64 dim : shape_max_dimensions) {
      dense_shape_size = MultiplyWithoutOverflow(dense_shape_size, dim);
      if (dense_shape_size < 0) {
        return dense_shape_size;
      }
    }
    dense_shape_size = MultiplyWithoutOverflow(
        dense_shape_size, ByteSizeOfPrimitiveType(shape.element_type()));
    return dense_shape_size;
  }();

  if (shape_size < 0) {
    return InvalidArgument("Shape %s size may overflow int64.",
                           ShapeUtil::HumanString(shape));
  }

  VLOG(3) << "Shape size is valid: " << shape_size;
  return Status::OK();
}

/* static */ Status ShapeUtil::ValidateShapeWithOptionalLayout(
    const Shape& shape) {
  TF_RETURN_IF_ERROR(ValidateShapeWithOptionalLayoutInternal(shape));

  return LayoutUtil::ValidateLayoutInShape(shape,
                                           /*allow_missing_layouts=*/true);
}

/* static */ Status ShapeUtil::ValidateShape(const Shape& shape) {
  TF_RETURN_IF_ERROR(ValidateShapeWithOptionalLayoutInternal(shape));

  return LayoutUtil::ValidateLayoutInShape(shape);
}

/* static */ Shape ShapeUtil::ChangeElementType(const Shape& original,
                                                PrimitiveType type) {
  Shape new_shape = original;
  new_shape.set_element_type(type);
  return new_shape;
}

/* static */ bool ShapeUtil::IndexIsValid(const Shape& shape,
                                          ShapeIndexView index) {
  const Shape* subshape = &shape;
  for (auto i : index) {
    if (!subshape->IsTuple() || i >= subshape->tuple_shapes_size() || i < 0) {
      return false;
    }
    subshape = &subshape->tuple_shapes(i);
  }
  return true;
}

/* static */ const Shape& ShapeUtil::GetSubshape(const Shape& shape,
                                                 ShapeIndexView index) {
  const Shape* return_shape = &shape;
  for (auto i : index) {
    CHECK(return_shape->IsTuple())
        << "Invalid index " << index << " for shape " << shape;
    return_shape = &return_shape->tuple_shapes(i);
  }
  return *return_shape;
}

/* static */ StatusOr<const Shape*> ShapeUtil::TryGetSubshape(
    const Shape& shape, ShapeIndexView index) {
  const Shape* return_shape = &shape;
  for (auto i : index) {
    if (!return_shape->IsTuple() || i < 0 ||
        i >= return_shape->tuple_shapes_size()) {
      return InvalidArgument(
          "Shape index %s not a valid subshape index for tuple with shape %s",
          index.ToString(), shape.DebugString());
    }
    return_shape = &return_shape->tuple_shapes(i);
  }
  return return_shape;
}

/* static */ Shape* ShapeUtil::GetMutableSubshape(Shape* shape,
                                                  ShapeIndexView index) {
  Shape* return_shape = shape;
  for (auto i : index) {
    CHECK(return_shape->IsTuple());
    return_shape = return_shape->mutable_tuple_shapes(i);
  }
  return return_shape;
}

/* static */
bool ShapeUtil::IsLeafIndex(const Shape& shape, const ShapeIndex& index) {
  return !GetSubshape(shape, index).IsTuple();
}

/* static */ int64 ShapeUtil::GetLeafCount(const Shape& shape) {
  if (!shape.IsTuple()) {
    return 1;
  }
  int64 count = 0;
  for (const Shape& subshape : shape.tuple_shapes()) {
    count += GetLeafCount(subshape);
  }
  return count;
}

/* static */ std::vector<ShapeUtil::IndexedShape> ShapeUtil::GetLeafShapes(
    const Shape& shape) {
  std::vector<IndexedShape> leaves;
  ForEachSubshape(shape, [&](const Shape& sub_shape, const ShapeIndex& index) {
    if (IsLeafIndex(shape, index)) {
      leaves.emplace_back(index, sub_shape);
    }
  });
  return leaves;
}

/* static */ bool ShapeUtil::HasDegenerateDimensions(const Shape& shape) {
  CHECK(shape.IsArray());
  return absl::c_linear_search(shape.dimensions(), 1);
}

/* static */ Shape ShapeUtil::DropDegenerateDimensions(const Shape& shape) {
  return FilterDimensions(
      [&](int64 dim) -> bool { return shape.dimensions()[dim] != 1; }, shape);
}

namespace {

// Helper for ForEachSubshape which visits the subshapes of the given shape in
// DFS pre-order starting with the index.
Status ForEachSubshapeHelper(const Shape& shape,
                             const ShapeUtil::StatusVisitorFunction& func,
                             ShapeIndex* index) {
  TF_RETURN_IF_ERROR(func(shape, *index));
  if (shape.IsTuple()) {
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
      index->push_back(i);
      TF_RETURN_IF_ERROR(ForEachSubshapeHelper(
          ShapeUtil::GetTupleElementShape(shape, i), func, index));
      index->pop_back();
    }
  }
  return Status::OK();
}

// Helper for ForEachMutableSubshape which visits the subshapes of the given
// shape in DFS pre-order starting with the index.
Status ForEachMutableSubshapeHelper(
    Shape* shape, const ShapeUtil::MutatingStatusVisitorFunction& func,
    ShapeIndex* index) {
  TF_RETURN_IF_ERROR(func(shape, *index));
  if (shape->IsTuple()) {
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(*shape); ++i) {
      index->push_back(i);
      TF_RETURN_IF_ERROR(ForEachMutableSubshapeHelper(
          shape->mutable_tuple_shapes(i), func, index));
      index->pop_back();
    }
  }
  return Status::OK();
}

}  // namespace

/* static */ void ShapeUtil::ForEachSubshape(const Shape& shape,
                                             const VisitorFunction& func) {
  ShapeIndex index;
  ForEachSubshapeHelper(
      shape,
      [&func](const Shape& subshape, const ShapeIndex& index) {
        func(subshape, index);
        return Status::OK();
      },
      &index)
      .IgnoreError();
}

/* static */ void ShapeUtil::ForEachMutableSubshape(
    Shape* shape, const MutatingVisitorFunction& func) {
  ShapeIndex index;
  ForEachMutableSubshapeHelper(
      shape,
      [&func](Shape* subshape, const ShapeIndex& index) {
        func(subshape, index);
        return Status::OK();
      },
      &index)
      .IgnoreError();
}

/* static */ Status ShapeUtil::ForEachSubshapeWithStatus(
    const Shape& shape, const StatusVisitorFunction& func) {
  ShapeIndex index;
  return ForEachSubshapeHelper(shape, func, &index);
}

/* static */ Status ShapeUtil::ForEachMutableSubshapeWithStatus(
    Shape* shape, const MutatingStatusVisitorFunction& func) {
  ShapeIndex index;
  return ForEachMutableSubshapeHelper(shape, func, &index);
}

/* static */ Shape ShapeUtil::PermuteDimensions(
    absl::Span<const int64> permutation, const Shape& shape) {
  Shape new_shape = shape;
  new_shape.clear_dimensions();
  for (auto dim : Permute(permutation, shape.dimensions())) {
    new_shape.add_dimensions(dim);
  }
  for (int64 i = 0; i < shape.rank(); i++) {
    new_shape.set_dynamic_dimension(permutation[i],
                                    shape.is_dynamic_dimension(i));
  }

  // If `shape` has a layout, by contract we choose a new layout such that the
  // transpose defined by this permutation is a bitcast.
  //
  // Some formalism helps to understand the correct way to do this.  We're going
  // to do algebra in the group of permutations of the dimensions of `shape`.
  //
  // Since the order of `shape`'s dimensions is not permuted relative to itself,
  // `shape`'s list of dimensions is isomorphic to the identity I.
  //
  // Let `shape`'s layout be L.  A layout is a permutation which maps a
  // minor-to-major physical layout to the order of a shape's logical dims.
  // Therefore inverse of a layout maps from logical to physical dims, and so
  // the physical layout of I is simply L'.I = L', where L' is the inverse of L.
  //
  // Let the argument `permutation` be P.  This is a permutation over `shape`'s
  // dimensions, so our return value will be a shape with dims P.I = P.  Our
  // goal is to construct a layout permutation L* that we can apply to P such
  // that the physical dimension ordering of the returned shape is the same
  // as that of the original shape, namely L'.
  //
  // Our returned shape has dims P and layout L*, so its in-memory layout is
  // L*'.P.  Setting this equal to L' and solving for L*, we get:
  //
  //   L*'.P = L'    =>
  //   L*'   = L'P'  =>
  //   L*    = P.L
  //
  if (shape.has_layout()) {
    CHECK(LayoutUtil::IsDenseArray(shape));
    Layout* new_layout = new_shape.mutable_layout();
    new_layout->set_format(DENSE);
    new_layout->clear_minor_to_major();
    for (auto index : ComposePermutations(
             permutation, AsInt64Slice(shape.layout().minor_to_major()))) {
      new_layout->add_minor_to_major(index);
    }
    // The permutation accepted by TransposeIsBitcast is the inverse of the
    // permutation here.
    CHECK(TransposeIsBitcast(shape, new_shape, InversePermutation(permutation)))
        << "shape=" << HumanStringWithLayout(shape)
        << ", new_shape=" << HumanStringWithLayout(new_shape)
        << ", permutation={" << absl::StrJoin(permutation, ",") << "}";
  }
  return new_shape;
}

/* static */ std::tuple<bool, std::vector<int64>, std::vector<int64>>
ShapeUtil::InsertedOrDeleted1SizedDimensions(const Shape& shape_pre,
                                             const Shape& shape_post) {
  CHECK(shape_pre.IsArray());
  CHECK(shape_post.IsArray());

  auto nil = std::make_tuple(false, std::vector<int64>(), std::vector<int64>());

  std::vector<int64> deleted_indices;
  std::vector<int64> inserted_indices;
  // Returns false if any input/output index between prior_unmodified_dim_pair
  // and unmodified_dim_pair have size >1. Otherwise, returns true and appends
  // the degerenate input/output dimensions in the gap to
  // deleted_indices/inserted_indices respectively.
  auto check_modified_dims =
      [&shape_pre, &shape_post, &deleted_indices, &inserted_indices](
          std::pair<int64, int64> prior_unmodified_dim_pair,
          std::pair<int64, int64> unmodified_dim_pair) {
        for (int64 modified_input_dim = prior_unmodified_dim_pair.first + 1;
             modified_input_dim < unmodified_dim_pair.first;
             ++modified_input_dim) {
          if (shape_pre.dimensions(modified_input_dim) > 1) {
            return false;
          }
          deleted_indices.push_back(modified_input_dim);
        }
        for (int64 modified_output_dim = prior_unmodified_dim_pair.second + 1;
             modified_output_dim < unmodified_dim_pair.second;
             ++modified_output_dim) {
          if (shape_post.dimensions(modified_output_dim) > 1) {
            return false;
          }
          inserted_indices.push_back(modified_output_dim);
        }
        return true;
      };

  std::vector<std::pair<int64, int64>> unmodified_dims =
      DimensionsUnmodifiedByReshape(shape_pre, shape_post);
  // Returns nil if the reshape modifies any non-degenerate input/output
  // dimension. DimensionsUnmodifiedByReshape gives us all unmodified
  // dimensions, so we only need to check whether dimensions in the gaps (thus
  // modified) have size >1.
  for (size_t i = 0; i <= unmodified_dims.size(); ++i) {
    // Check (modified) dimensions between unmodified_dims[i-1] and
    // unmodified_dims[i].
    auto prior_unmodified_dim_pair =
        i > 0 ? unmodified_dims[i - 1] : std::make_pair(-1LL, -1LL);
    auto unmodified_dim_pair =
        i < unmodified_dims.size()
            ? unmodified_dims[i]
            : std::make_pair(shape_pre.rank(), shape_post.rank());
    if (!check_modified_dims(prior_unmodified_dim_pair, unmodified_dim_pair)) {
      return nil;
    }
  }

  return std::make_tuple(true, deleted_indices, inserted_indices);
}

/* static */ std::vector<std::pair<int64, int64>>
ShapeUtil::DimensionsUnmodifiedByReshape(const Shape& input_shape,
                                         const Shape& output_shape) {
  CHECK(input_shape.IsArray());
  CHECK(output_shape.IsArray());

  // Unmodified dimensions are merely common factors of rank 1.
  auto common_factors = CommonFactors(AsInt64Slice(input_shape.dimensions()),
                                      AsInt64Slice(output_shape.dimensions()));
  for (size_t i = 0; i < common_factors.size() - 1;) {
    if (1 != common_factors[i + 1].first - common_factors[i].first ||
        1 != common_factors[i + 1].second - common_factors[i].second) {
      common_factors.erase(common_factors.begin() + i);
    } else {
      ++i;
    }
  }
  // `CommonFactors(a, b).back() == (a.rank, b.rank)` so we must pop it.
  common_factors.pop_back();
  return common_factors;
}

/* static */ absl::optional<std::vector<int64>>
ShapeUtil::ReshapeLeavesDimensionsUnmodified(
    const Shape& from_shape, const Shape& to_shape,
    absl::Span<const int64> input_dim_indices) {
  CHECK(std::is_sorted(input_dim_indices.begin(), input_dim_indices.end()));

  std::vector<int64> output_dim_indices;
  std::vector<std::pair<int64, int64>> unmodified_dims =
      ShapeUtil::DimensionsUnmodifiedByReshape(from_shape, to_shape);
  size_t i = 0;  // index to unmodified_dims
  for (int64 input_dim_index : input_dim_indices) {
    // Search unmodified_dims for input_dim_index. We can search from the last
    // matching position because input_dim_indices is guaranteed to be sorted.
    while (i < unmodified_dims.size() &&
           unmodified_dims[i].first < input_dim_index) {
      ++i;
    }
    if (i >= unmodified_dims.size() ||
        unmodified_dims[i].first != input_dim_index) {
      return absl::nullopt;
    }
    output_dim_indices.push_back(unmodified_dims[i].second);
  }
  return output_dim_indices;
}

/* static */ bool ShapeUtil::TransposeIsBitcast(
    const Shape& input_shape, const Shape& output_shape,
    absl::Span<const int64> dimension_mapping) {
  CHECK(LayoutUtil::HasLayout(input_shape) &&
        LayoutUtil::HasLayout(output_shape));

  if (!SameElementType(input_shape, output_shape)) {
    return false;
  }

  // Check the reshape permutes the positions of each dimension in the
  // minor-to-major order. positions[i]=k means dimension `i` is k-th minor.
  //   input_positions = apply(dimension_mapping, output_positions)
  //
  // Because the positions of each dimension are the inverse permutation of the
  // minor-to-major order, the above check is equivalent to
  //   inverse(input_dimensions) =
  //       apply(dimension_mapping, inverse(output_dimensions))
  //   # `I` indicates identity permutation.
  //   apply(input_dimensions, I) =
  //       apply(dimension_mapping, apply(output_dimensions, I))
  //   apply(input_dimensions, I) =
  //       apply((dimension_mapping * output_dimensions), I)
  //   input_dimensions = dimension_mapping * output_dimensions
  return absl::c_equal(
      ComposePermutations(dimension_mapping,
                          AsInt64Slice(output_shape.layout().minor_to_major())),
      input_shape.layout().minor_to_major());
}

/* static */ bool ShapeUtil::ReshapeIsBitcast(const Shape& input_shape,
                                              const Shape& output_shape) {
  CHECK(input_shape.IsArray());
  CHECK(output_shape.IsArray());
  CHECK(LayoutUtil::HasLayout(input_shape));
  CHECK(LayoutUtil::HasLayout(output_shape));

  if (!SameElementType(input_shape, output_shape)) {
    return false;
  }

  CHECK_EQ(ElementsIn(input_shape), ElementsIn(output_shape));
  if (ElementsIn(input_shape) == 0) {
    return true;
  }

  // TL;DR: The rest of the method checks that the reshape does not change the
  // physical location of any unit input or output index. Unit indices have
  // exactly one dimension that equals 1 and other dimensions 0. This condition
  // is necessary for the reshape to be a bitcast, because a bitcast-equivalent
  // reshape shouldn't change the physical location of any element. It is also a
  // sufficient condition as is proved below (note: many details are omitted for
  // space).
  //
  // Definitions:
  //
  // * Denote the input shape by IS and output shape by OS. IS[i] or OS[i] means
  // the size of i-th least significant dimension of IS or OS (this is opposite
  // to how we define the index of Shape::dimensions()).
  //
  // * Given an input or output index I, denote by p(I) I's physical linear
  // index (or physical index for short) and l(I) I's logical linear index (or
  // logical index for short).
  //
  // * Given a logical index k, denote by II(k) the input index whose linear
  // index is k, and OI(k) the corresponding output index.
  //
  // * Denote by IT[i] the increment of physical index if i-th dimension of the
  // input index is increased by 1. Similarly, OT[i] means the increment if i-th
  // dimension of the output index is increased by 1. Note that IT[i] or OT[i]
  // is a function of IS or OS and the layout, and not dependent on the specific
  // input or output index.
  //
  // To prove the reshape from IS to OS is a bitcast, it is sufficient to prove
  // that, for any linear index k, p(II(k))=p(OI(k)). We prove this by
  // induction. We know p(II(0))=p(OI(0)) is trivially true, so what's left is
  // to prove, with every increment on k, the above formula still holds.
  //
  // First, suppose reshaping from IS to OS is non-factorizable (we discuss
  // refactorizable reshapes later). A reshape from IS to OS is factorizable, if
  // there exists (i,j) such that
  //
  //   0<=i<=|IS|
  //   0<=j<=|OS|
  //   |IS|-i+|OS|-j > 0 (i.e., i,j mustn't both point to the end)
  //   product(IS[i], IS[i+1], ..., IS[|IS|-1])
  //     = product(OS[j], OS[j+1], ..., OS[|OS|-1])
  //
  // p(II(k))=p(OI(k)) is trivially true for k=0 because p(II(0)) and p(OI(0))
  // are both 0. It's also trivially true for k=1, because II(1) and OI(1) are
  // unit indices which are already tested. This also means IT[0]=OT[0]
  // because p(II(1))=IT[0] and p(OI(1))=OT[0].
  //
  // Furthermore, p(II(k))=p(OI(k)) for k<min(IS[0],OS[0]), because each
  // increment of k adds IT[0] to the input physical and OT[0] (same as IT[0])
  // to the output physical.
  //
  // When k=min(IS[0],OS[0]), the first wrap happens. Without losing generality,
  // suppose IS[0]<OS[0] and thus k=IS[0]. Similar proof applies to IS[0]>OS[0].
  // Note that IS[0]!=OS[0] because the reshape is non-factorizable. From
  // logical index k-1 to logical index k, dimension 1 of the input index
  // is increased by 1 and dimension 0 is reset to 0 thus decreased by
  // IS[0]-1. Therefore, the physical input index is increased by
  //
  //   p(II(k)) - p(II(k-1)) = IT[1] - (IS[0]-1) * IT[0]
  //
  // Because IS[0]<OS[0], the only change to the output index is that its
  // dimension 0 is increased by one. Therefore,
  //
  //   p(OI(k)) - p(OI(k-1)) = OT[0] = IT[0]
  //
  // Because II(k) is an unit index -- (0,..,0,1,0), we already tested that
  // p(II(k))=p(OI(k)). Therefore,
  //   IT[1] - (IS[0]-1) * IT[0] = IT[0]
  //   IT[1] = IS[0] * IT[0]
  // In other words, input dimension 1 is immediately more major than input
  // dimension 0. We can now conceptually collapse these two dimensions because
  // an increment in the logical index affecting only these two dimensions maps
  // to IT[0] in the physical index.
  //
  // By induction (omitted here), we can prove IT[i]=IS[i-1]*IT[i-1] and
  // OT[i]=OS[i-1]*OT[i-1]. Therefore, both IS and OS are row-major and bitwise
  // identical.
  //
  // A factorizable reshape can be factorized into a list of non-factorizable
  // sub-reshapes, each of which can be handled similarly to the proof above.
  // For example,
  //
  //   [7x9x2x15] -> [63x6x5]
  //
  // can be factorized into
  //
  //   [7x9] -> [63] and [2x15] -> [6x5].
  //
  // Suppose input index I=(x3,x2,x1,x0) and output index O=(y2,y1,y0) have the
  // same logical linear index. According to the factorization, we know
  // l(x3,x2,0,0)=l(y2,0,0) and l(0,0,x1,x0)=l(0,y1,y0). Using the proof for
  // non-factorizable reshapes, we can prove p(0,0,x1,x0)=p(0,y1,y0). Using a
  // similar proof, with the increment of the logical index set to
  // IS[1]*IS[0]=OS[1]*OS[0]=30 instead of 1, we can prove
  // p(x3,x2,0,0)=p(y2,0,0) too. Therefore,
  //
  //   p(x3,x2,x1,x0) = p(x3,x2,0,0) + p(0,0,x1,x0)
  //                  = p(y2,0,0) + p(0,0,y1,y0)
  //                  = p(y2,y1,y0)
  //
  // check_input_unit_indices checks one way of the condition: each input unit
  // index is mapped to an output index with the same physical location. This
  // lambda will be called again with input_shape and output_shape reversed to
  // check the other way.
  auto check_input_unit_indices = [](const Shape& input_shape,
                                     const Shape& output_shape) {
    // input_shape_dim0_major/output_shape_dim0_major has the same "dimensions"
    // as input_shape/output_shape and the dimension-0-major layout. These two
    // shapes are used for conversion between logical linear indices and
    // multi-dimensional indices.
    Shape input_shape_dim0_major = MakeShapeWithDescendingLayout(
        input_shape.element_type(), AsInt64Slice(input_shape.dimensions()));
    Shape output_shape_dim0_major = MakeShapeWithDescendingLayout(
        output_shape.element_type(), AsInt64Slice(output_shape.dimensions()));

    for (int64 input_dim = 0; input_dim < input_shape.rank(); ++input_dim) {
      if (input_shape.dimensions(input_dim) <= 1) {
        continue;
      }

      std::vector<int64> input_unit_index(input_shape.rank(), 0);
      input_unit_index[input_dim] = 1;
      int64 logical_linear_index =
          IndexUtil::MultidimensionalIndexToLinearIndex(input_shape_dim0_major,
                                                        input_unit_index);
      // output_index has the same logical linear index as input_unit_index.
      std::vector<int64> output_index =
          IndexUtil::LinearIndexToMultidimensionalIndex(output_shape_dim0_major,
                                                        logical_linear_index);
      // Check input_unit_index and output_index have the same physical linear
      // index.
      if (IndexUtil::MultidimensionalIndexToLinearIndex(input_shape,
                                                        input_unit_index) !=
          IndexUtil::MultidimensionalIndexToLinearIndex(output_shape,
                                                        output_index)) {
        return false;
      }
    }
    return true;
  };
  return check_input_unit_indices(input_shape, output_shape) &&
         check_input_unit_indices(output_shape, input_shape);
}

/* static */ absl::optional<Shape> ShapeUtil::AlignLayouts(
    const Shape& input_shape, const Shape& output_shape) {
  CHECK(input_shape.IsArray());
  CHECK(output_shape.IsArray());
  // Removing trivial dimensions from the shape simplifies the alignment
  // algorithm since ones can go in any position.
  if (HasDegenerateDimensions(input_shape) ||
      HasDegenerateDimensions(output_shape)) {
    auto simple_output_shape =
        AlignLayouts(DropDegenerateDimensions(input_shape),
                     DropDegenerateDimensions(output_shape));
    if (!simple_output_shape) {
      return absl::nullopt;
    }

    auto layout = simple_output_shape->layout().minor_to_major();
    // For each one sized dimension in the output, increment the dimension
    // numbers in layout that are more minor than the one.
    absl::InlinedVector<int64, 8> dim_map;
    dim_map.reserve(simple_output_shape->rank());
    for (int64 i = 0; i < output_shape.rank(); ++i) {
      if (output_shape.dimensions(i) != 1) {
        dim_map.push_back(i);
      }
    }
    for (int64& d : layout) {
      d = dim_map[d];
    }

    // Add the ones in descending order to the layout. Descending layouts tend
    // to reduce the number of copies inserted in layout assignment.
    for (int64 i = output_shape.rank() - 1; i >= 0; --i) {
      if (output_shape.dimensions(i) == 1) {
        layout.push_back(i);
      }
    }
    Shape output_shape_with_layout = output_shape;
    *output_shape_with_layout.mutable_layout() = Layout{layout};
    return output_shape_with_layout;
  }

  int64 input_rank = input_shape.rank();
  int64 output_rank = output_shape.rank();

  // First, calculate an alignment of the dimensions. A consecutive sequence of
  // input dimensions and output dimensions belong to the same alignment part if
  // the products of their dimension bounds are the same. In the easiest case,
  // an alignment part consists of one input dimension and one output dimension
  // which both have the same dimension bound. An alignment part specifies which
  // dimensions need to be kept together in a physical layout if we want a
  // reshape to be a bitcast. The order of the alignment parts is defined by the
  // physical layout of the input shape, so when we construct the layout for the
  // output shape we just process the alignment parts in this order, and then
  // layout the dimensions belonging to each part in descending (major to minor)
  // order.

  // Stores the input and output dimension numbers where each alignment part
  // starts.
  std::vector<std::pair<int64, int64>> alignment;
  alignment.push_back({0, 0});

  // Stores a mapping from the input dimension to the alignment part it belongs
  // to.
  std::vector<int64> dimension_to_alignment_index(input_rank);
  int64 input_dimension_product = 1, output_dimension_product = 1;
  for (int64 i = 0, j = 0; i < input_rank || j < output_rank;) {
    // Check if we have reached the end of an alignment part.
    if (input_dimension_product == output_dimension_product &&
        input_dimension_product > 1) {
      alignment.push_back({i, j});
      input_dimension_product = output_dimension_product = 1;
    }
    if (input_dimension_product < output_dimension_product ||
        j == output_rank) {
      if (i == input_rank) {
        return absl::nullopt;
      }
      dimension_to_alignment_index[i] = alignment.size() - 1;
      input_dimension_product *= input_shape.dimensions(i);
      ++i;
    } else {
      output_dimension_product *= output_shape.dimensions(j);
      ++j;
    }
  }
  if (input_dimension_product != output_dimension_product) {
    return absl::nullopt;
  }

  // We also need to store an end element so that we know where the last
  // alignment part ends.
  alignment.push_back({input_rank, output_rank});
  // Now check if the physical layout can potentially be aligned to the output
  // shape by changing the physical layout of the output shape. We need to check
  // that all dimension numbers that belong to the same alignment part appear
  // consecutively, and are in descending order. However we can ignore any
  // trivial dimension bounds of 1, because they can be placed anywhere.
  auto input_dimension_numbers = input_shape.layout().minor_to_major();
  std::vector<int64> output_layout;
  output_layout.reserve(output_rank);
  for (int64 i = 0; i < input_rank;) {
    int64 current_dimension_number = input_dimension_numbers[i];

    // Trivial dimensions are stripped.
    CHECK_NE(input_shape.dimensions(current_dimension_number), 1);
    const int64 current_alignment_index =
        dimension_to_alignment_index[current_dimension_number];
    // Because of the special end element that we added, we can be sure that
    // 'current_alignment_index' is < alignment.size() - 1.
    CHECK_LT(current_alignment_index, alignment.size() - 1);

    // Check that the following 'num_non_trivial_dimensions_in_alignment_part'
    // dimension numbers (ignoring dimension numbers with dimension bound 1) are
    // in descending order and belong to the current alignment part.
    for (int64 j = 0; j < alignment[current_alignment_index + 1].first -
                              alignment[current_alignment_index].first;
         ++i, ++j) {
      if (i == input_rank) {
        return absl::nullopt;
      }
      // If the current dimension number belongs to a different alignment part,
      // or the dimension numbers are not in descending order, we can return
      // early.
      if (dimension_to_alignment_index[input_dimension_numbers[i]] !=
              current_alignment_index ||
          input_dimension_numbers[i] > current_dimension_number) {
        return absl::nullopt;
      }
      current_dimension_number = input_dimension_numbers[i];
    }
    // The output dimension numbers that belong to the current alignment part
    // need to appear in the same descending order as in the input.
    for (int64 j = alignment[current_alignment_index + 1].second - 1;
         j >= alignment[current_alignment_index].second; --j) {
      output_layout.push_back(j);
    }
  }
  CHECK_EQ(output_layout.size(), output_rank);
  Shape output_shape_with_layout = MakeShapeWithLayout(
      output_shape.element_type(), AsInt64Slice(output_shape.dimensions()),
      output_layout);
  CHECK(ReshapeIsBitcast(input_shape, output_shape_with_layout))
      << "reshape is not a bitcast for input_shape: "
      << ShapeUtil::HumanStringWithLayout(input_shape)
      << " and output_shape_with_layout: "
      << ShapeUtil::HumanStringWithLayout(output_shape_with_layout);
  return output_shape_with_layout;
}

/* static */ Shape ShapeUtil::DeleteDimension(int64 dim_to_delete,
                                              Shape shape) {
  CHECK(shape.IsArray());
  shape.DeleteDimension(dim_to_delete);
  return shape;
}

/* static */ Shape ShapeUtil::FilterDimensions(
    const std::function<bool(int64)>& p, Shape shape) {
  CHECK(shape.IsArray());
  std::vector<int64> dims_to_delete;
  for (int64 i = shape.dimensions().size() - 1; i >= 0; --i) {
    if (!p(i)) {
      dims_to_delete.push_back(i);
    }
  }
  for (int64 dim : dims_to_delete) {
    shape = DeleteDimension(dim, shape);
  }
  return shape;
}

/*static*/ size_t ShapeUtil::Hash(const Shape& shape) {
  using tensorflow::hash;
  using tensorflow::Hash64Combine;

  size_t hash_value = hash<PrimitiveType>()(shape.element_type());

  if (shape.tuple_shapes().empty()) {
    for (int i = 0; i < shape.dimensions_size(); ++i) {
      hash_value =
          Hash64Combine(hash_value, hash<int64>()(shape.dimensions(i)));
      hash_value = Hash64Combine(hash_value,
                                 hash<bool>()(shape.is_dynamic_dimension(i)));
    }

    hash_value = Hash64Combine(hash_value, LayoutUtil::Hash(shape.layout()));
  } else {
    hash_value = 0;
    for (const Shape& subshape : shape.tuple_shapes()) {
      hash_value = Hash64Combine(hash_value, ShapeUtil::Hash(subshape));
    }
  }

  return hash_value;
}

}  // namespace xla
