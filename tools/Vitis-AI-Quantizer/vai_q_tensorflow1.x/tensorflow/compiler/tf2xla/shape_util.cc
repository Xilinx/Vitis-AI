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

#include "tensorflow/compiler/tf2xla/shape_util.h"

#include <numeric>

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {

Status PopulateInfeedLayoutVector(const xla::Shape& shape,
                                  std::vector<int>* layouts) {
  if (shape.IsTuple()) {
    int64 tuple_elements = xla::ShapeUtil::TupleElementCount(shape);
    for (int64 i = 0; i < tuple_elements; ++i) {
      const xla::Shape& subshape =
          xla::ShapeUtil::GetTupleElementShape(shape, i);
      TF_RETURN_IF_ERROR(PopulateInfeedLayoutVector(subshape, layouts));
    }
  } else if (xla::LayoutUtil::HasLayout(shape)) {
    for (auto dim : xla::LayoutUtil::MinorToMajor(shape)) {
      layouts->push_back(dim);
    }
  } else {
    layouts->insert(layouts->end(), shape.rank(), -1);
  }
  return Status::OK();
}

// Populate the output layout unless the minor_to_major array contains all -1
// value, in which case the layout is considered missing and the API returns
// false.
xla::StatusOr<bool> MakeLayout(absl::Span<const int64> minor_to_major,
                               xla::Layout* layout) {
  if (std::all_of(minor_to_major.begin(), minor_to_major.end(),
                  [](int64 dim) { return dim == -1; })) {
    return false;
  }
  std::vector<bool> dim_present(minor_to_major.size(), false);
  for (auto dim : minor_to_major) {
    if (dim < 0 || dim >= minor_to_major.size()) {
      return errors::InvalidArgument("Layout dimension out of range: dim=", dim,
                                     " rank=", minor_to_major.size());
    }
    if (dim_present[dim]) {
      return errors::InvalidArgument("Repeated layout dimension: dim=", dim);
    }
    dim_present[dim] = true;
  }
  *layout = xla::LayoutUtil::MakeLayout(minor_to_major);
  return true;
}

Status AssignLayout(
    absl::Span<const int64> minor_to_major,
    const std::function<xla::Layout(const xla::Shape&)>& layout_func,
    xla::Shape* shape) {
  xla::Layout layout;
  TF_ASSIGN_OR_RETURN(bool has_layout, MakeLayout(minor_to_major, &layout));
  if (!has_layout && layout_func) {
    layout = layout_func(*shape);
  }
  *shape->mutable_layout() = layout;
  return Status::OK();
}

}  // namespace

// Convert an XLA Shape into the equivalent TensorFlow shape.
Status XLAShapeToTensorShape(const xla::Shape& shape,
                             TensorShape* tensor_shape) {
  if (shape.IsTuple()) {
    return errors::InvalidArgument("XLA shape ",
                                   xla::ShapeUtil::HumanString(shape),
                                   " cannot be converted to a TensorShape");
  }
  *tensor_shape = TensorShape();
  for (int i = 0; i < shape.rank(); ++i) {
    tensor_shape->AddDim(shape.dimensions(i));
  }
  return Status::OK();
}

// Convert a TensorShape into the equivalent XLA Shape proto.
Status TensorShapeToXLAShape(DataType dtype, const TensorShape& tensor_shape,
                             xla::Shape* shape) {
  xla::PrimitiveType type;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(dtype, &type));
  *shape = TensorShapeToXLAShape(type, tensor_shape);
  return Status::OK();
}

xla::Shape TensorShapeToXLAShape(xla::PrimitiveType type,
                                 const TensorShape& tensor_shape) {
  int rank = tensor_shape.dims();
  std::vector<int64> dimensions(rank);
  std::vector<int64> layout(rank);
  for (int d = 0; d < rank; ++d) {
    dimensions[d] = tensor_shape.dim_size(d);
  }
  // XLA uses minor-to-major; Tensorflow uses major-to-minor.
  std::iota(layout.rbegin(), layout.rend(), 0);

  return xla::ShapeUtil::MakeShapeWithLayout(type, dimensions, layout);
}

xla::StatusOr<std::vector<int>> GetShapeLayoutVector(const xla::Shape& shape) {
  std::vector<int> layouts;
  TF_RETURN_IF_ERROR(PopulateInfeedLayoutVector(shape, &layouts));
  return layouts;
}

Status GetShapeWithLayout(
    const xla::Shape& input_shape, absl::Span<const int64> minor_to_major,
    const std::function<xla::Layout(const xla::Shape&)>& layout_func,
    xla::Shape* output_shape) {
  if (input_shape.IsTuple()) {
    int64 tuple_elements = xla::ShapeUtil::TupleElementCount(input_shape);
    std::vector<xla::Shape> shapes;
    shapes.reserve(tuple_elements);
    size_t position = 0;
    for (int64 i = 0; i < tuple_elements; ++i) {
      const xla::Shape& shape =
          xla::ShapeUtil::GetTupleElementShape(input_shape, i);
      if (shape.IsTuple()) {
        return errors::InvalidArgument(
            "Nested tuples not supported: ",
            xla::ShapeUtil::HumanString(input_shape));
      }
      int64 rank = shape.rank();
      if (position + rank > minor_to_major.size()) {
        return errors::InvalidArgument(
            "Not enough layout attribute elements: position=", position,
            " rank=", rank, " elements=", minor_to_major.size());
      }
      shapes.push_back(shape);
      TF_RETURN_IF_ERROR(AssignLayout(
          absl::Span<const int64>(minor_to_major).subspan(position, rank),
          layout_func, &shapes.back()));
      position += rank;

      VLOG(4) << "Shape[" << i
              << "] = " << xla::ShapeUtil::HumanStringWithLayout(shapes.back());
    }
    if (position != minor_to_major.size()) {
      return errors::InvalidArgument(
          "Too many elements passed in the layout attribute: position=",
          position, " size=", minor_to_major.size());
    }
    *output_shape = xla::ShapeUtil::MakeTupleShape(shapes);
  } else {
    int64 rank = input_shape.rank();
    if (rank != minor_to_major.size()) {
      return errors::InvalidArgument(
          "Wrong number of layout attribute elements: rank=", rank,
          " elements=", minor_to_major.size());
    }
    *output_shape = input_shape;
    TF_RETURN_IF_ERROR(AssignLayout(minor_to_major, layout_func, output_shape));

    VLOG(4) << "Shape[] = "
            << xla::ShapeUtil::HumanStringWithLayout(*output_shape);
  }
  return Status::OK();
}

}  // namespace tensorflow
