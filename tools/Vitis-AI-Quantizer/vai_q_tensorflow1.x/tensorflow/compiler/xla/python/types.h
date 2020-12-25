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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TYPES_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TYPES_H_

#include <memory>
#include <vector>

#include "absl/types/optional.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {

// Helper that converts a failing StatusOr to an exception.
// For use only inside pybind11 code.
template <typename T>
T ValueOrThrow(StatusOr<T> v) {
  if (!v.ok()) {
    throw std::runtime_error(v.status().ToString());
  }
  return v.ConsumeValueOrDie();
}

// Converts a NumPy dtype to a PrimitiveType.
StatusOr<PrimitiveType> DtypeToPrimitiveType(const pybind11::dtype& np_type);

// Converts a PrimitiveType to a Numpy dtype.
StatusOr<pybind11::dtype> PrimitiveTypeToDtype(PrimitiveType type);

// Converts a literal to (possibly-nested tuples of) NumPy arrays.
// The literal's leaf arrays are not copied; instead the NumPy arrays share
// buffers with the literals. Takes ownership of `literal` and keeps the
// necessary pieces alive using Python reference counting.
// Requires the GIL.
StatusOr<pybind11::object> LiteralToPython(std::shared_ptr<Literal> literal);

// Converts a Python object into an XLA shape and a vector of leaf buffers.
// The leaf buffers correspond to a depth-first, left-to-right traversal of
// the Python value.
// Requires the GIL.
struct PythonBufferTree {
  // Holds a reference to the arrays pointed to by `leaves`, since we may
  // need to make a copy if the array is not in a C-style layout.
  absl::InlinedVector<pybind11::object, 1> arrays;
  absl::InlinedVector<BorrowingLiteral, 1> leaves;
  Shape shape;
};
StatusOr<PythonBufferTree> GetPythonBufferTree(
    const pybind11::object& argument);

// Converts a sequence of int64s to a Python tuple of ints.
// Pybind11 by default converts a std::vector<int64> to a Python list; for
// shapes we frequently want a tuple instead.
pybind11::tuple IntSpanToTuple(absl::Span<int64 const> xs);

// Converts a Python sequence of integers to a std::vector<int64>
std::vector<int64> IntSequenceToVector(const pybind11::object& sequence);

}  // namespace xla

// This namespace is a documented pybind11 extension point.
// Caution: Unusually for Google code, this code uses C++ exceptions because
// they are the only mechanism for reporting cast failures to pybind11. However,
// the exceptions are local to the binding code.
namespace pybind11 {
namespace detail {

// When absl::optional is an alias for std::optional, the type_caster
// specializations are provided by pybind11.
#ifndef ABSL_HAVE_STD_OPTIONAL
// absl::optional
template <typename T>
struct type_caster<absl::optional<T>> : optional_caster<absl::optional<T>> {};

template <>
struct type_caster<absl::nullopt_t> : public void_caster<absl::nullopt_t> {};
#endif

// absl::Span
template <typename T>
struct type_caster<absl::Span<const T>> {
  using value_conv = make_caster<T>;

  PYBIND11_TYPE_CASTER(absl::Span<const T>,
                       _("Span[") + value_conv::name + _("]"));

  // absl::Span doesn't hold ownership. We therefore need a temporary array.
  // Pybind appears to keep type_casters alive until the callee has run.
  std::vector<T> storage_;

  bool load(handle src, bool convert) {
    if (!isinstance<sequence>(src)) {
      return false;
    }
    auto seq = reinterpret_borrow<sequence>(src);
    storage_.clear();
    storage_.reserve(seq.size());
    for (auto it : seq) {
      value_conv conv;
      if (!conv.load(it, convert)) {
        return false;
      }
      storage_.push_back(cast_op<T&&>(std::move(conv)));
    }
    value = absl::Span<const T>(storage_);
    return true;
  }
};

// Status, StatusOr. Failing statuses become Python exceptions; Status::OK()
// becomes None.
template <>
struct type_caster<xla::Status> {
 public:
  PYBIND11_TYPE_CASTER(xla::Status, _("Status"));

  static handle cast(xla::Status src, return_value_policy /* policy */,
                     handle /* parent */) {
    if (!src.ok()) {
      throw std::runtime_error(src.ToString());
    }
    return none().inc_ref();
  }
};

template <typename T>
struct type_caster<xla::StatusOr<T>> {
 public:
  using value_conv = make_caster<T>;

  PYBIND11_TYPE_CASTER(xla::StatusOr<T>,
                       _("StatusOr[") + value_conv::name + _("]"));

  static handle cast(xla::StatusOr<T> src, return_value_policy policy,
                     handle parent) {
    if (!src.ok()) {
      throw std::runtime_error(src.status().ToString());
    }
    return value_conv::cast(std::forward<xla::StatusOr<T>>(src).ValueOrDie(),
                            policy, parent);
  }
};

// Literals.
// Literal data can be passed to XLA as a NumPy array; its value can be
// cast to an xla::BorrowingLiteral or xla::LiteralSlice in a zero-copy way.
// We don't have any literal -> numpy conversions here, since all the methods
// that want to return arrays build Python objects directly.

template <>
struct type_caster<xla::BorrowingLiteral> {
 public:
  PYBIND11_TYPE_CASTER(xla::BorrowingLiteral, _("xla::BorrowingLiteral"));

  // Pybind appears to keep type_casters alive until the callee has run.
  pybind11::array array;

  bool load(handle handle, bool) {
    array = pybind11::array::ensure(
        handle, pybind11::array::c_style |
                    pybind11::detail::npy_api::NPY_ARRAY_ALIGNED_);
    if (!array) return false;
    pybind11::buffer_info buffer_info = array.request();

    absl::InlinedVector<xla::int64, 4> dims(array.ndim());
    for (int i = 0; i < array.ndim(); ++i) {
      dims[i] = array.shape(i);
    }
    auto type = xla::DtypeToPrimitiveType(array.dtype());
    if (!type.ok()) {
      throw std::runtime_error(type.status().ToString());
    }
    xla::Shape shape = xla::ShapeUtil::MakeShape(type.ValueOrDie(), dims);
    if (buffer_info.size * buffer_info.itemsize !=
        xla::ShapeUtil::ByteSizeOf(shape)) {
      throw std::runtime_error(absl::StrCat(
          "Size mismatch for buffer: ", buffer_info.size * buffer_info.itemsize,
          " vs. ", xla::ShapeUtil::ByteSizeOf(shape)));
    }
    value =
        xla::BorrowingLiteral(static_cast<const char*>(buffer_info.ptr), shape);
    return true;
  }
};

template <>
struct type_caster<xla::LiteralSlice> {
 public:
  PYBIND11_TYPE_CASTER(xla::LiteralSlice, _("xla::LiteralSlice"));

  // Pybind appears to keep type_casters alive until the callee has run.
  type_caster<xla::BorrowingLiteral> literal_caster;

  bool load(handle handle, bool convert) {
    if (!literal_caster.load(handle, convert)) {
      return false;
    }
    value = static_cast<const xla::BorrowingLiteral&>(literal_caster);
    return true;
  }
};

// XLA protocol buffers
// We don't actually care that these are the protocol buffers, we merely want
// objects that duck type as protocol buffers. The client code currently avoids
// depending on Python protocol buffers to avoid conflicting definitions from
// different modules that both include XLA.

template <>
struct type_caster<xla::ConvolutionDimensionNumbers> {
 public:
  PYBIND11_TYPE_CASTER(xla::ConvolutionDimensionNumbers,
                       _("xla::ConvolutionDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    value.set_input_batch_dimension(
        getattr(handle, "input_batch_dimension").cast<xla::int64>());
    value.set_input_feature_dimension(
        getattr(handle, "input_feature_dimension").cast<xla::int64>());
    value.set_output_batch_dimension(
        getattr(handle, "output_batch_dimension").cast<xla::int64>());
    value.set_output_feature_dimension(
        getattr(handle, "output_feature_dimension").cast<xla::int64>());
    value.set_kernel_input_feature_dimension(
        getattr(handle, "kernel_input_feature_dimension").cast<xla::int64>());
    value.set_kernel_output_feature_dimension(
        getattr(handle, "kernel_output_feature_dimension").cast<xla::int64>());
    std::vector<xla::int64> dims;
    dims = getattr(handle, "input_spatial_dimensions")
               .cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_input_spatial_dimensions()));
    dims = getattr(handle, "kernel_spatial_dimensions")
               .cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_kernel_spatial_dimensions()));
    dims = getattr(handle, "output_spatial_dimensions")
               .cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_output_spatial_dimensions()));
    return true;
  }
};

template <>
struct type_caster<xla::DotDimensionNumbers> {
 public:
  PYBIND11_TYPE_CASTER(xla::DotDimensionNumbers, _("xla::DotDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    std::vector<xla::int64> dims;
    dims = getattr(handle, "lhs_contracting_dimensions")
               .cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_lhs_contracting_dimensions()));
    dims = getattr(handle, "rhs_contracting_dimensions")
               .cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_rhs_contracting_dimensions()));
    dims =
        getattr(handle, "lhs_batch_dimensions").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_lhs_batch_dimensions()));
    dims =
        getattr(handle, "rhs_batch_dimensions").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_rhs_batch_dimensions()));
    return true;
  }
};

template <>
struct type_caster<xla::GatherDimensionNumbers> {
 public:
  PYBIND11_TYPE_CASTER(xla::GatherDimensionNumbers,
                       _("xla::GatherDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    std::vector<xla::int64> dims;
    dims = getattr(handle, "offset_dims").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_offset_dims()));
    dims =
        getattr(handle, "collapsed_slice_dims").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_collapsed_slice_dims()));
    dims = getattr(handle, "start_index_map").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_start_index_map()));
    value.set_index_vector_dim(
        getattr(handle, "index_vector_dim").cast<xla::int64>());
    return true;
  }
};

template <>
struct type_caster<xla::ScatterDimensionNumbers> {
 public:
  PYBIND11_TYPE_CASTER(xla::ScatterDimensionNumbers,
                       _("xla::ScatterDimensionNumbers"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    std::vector<xla::int64> dims;
    dims =
        getattr(handle, "update_window_dims").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_update_window_dims()));
    dims =
        getattr(handle, "inserted_window_dims").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_inserted_window_dims()));
    dims = getattr(handle, "scatter_dims_to_operand_dims")
               .cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_scatter_dims_to_operand_dims()));
    value.set_index_vector_dim(
        getattr(handle, "index_vector_dim").cast<xla::int64>());
    return true;
  }
};

template <>
struct type_caster<xla::ReplicaGroup> {
 public:
  PYBIND11_TYPE_CASTER(xla::ReplicaGroup, _("xla::ReplicaGroup"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    std::vector<xla::int64> dims;
    dims = getattr(handle, "replica_ids").cast<std::vector<xla::int64>>();
    std::copy(dims.begin(), dims.end(),
              tensorflow::protobuf::RepeatedFieldBackInserter(
                  value.mutable_replica_ids()));
    return true;
  }
};

template <>
struct type_caster<xla::PaddingConfig> {
 public:
  PYBIND11_TYPE_CASTER(xla::PaddingConfig, _("xla::PaddingConfig"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    sequence dimensions =
        reinterpret_borrow<sequence>(getattr(handle, "dimensions"));

    for (auto dimension : dimensions) {
      xla::PaddingConfig::PaddingConfigDimension* config_dim =
          value.add_dimensions();
      config_dim->set_edge_padding_low(
          getattr(dimension, "edge_padding_low").cast<xla::int64>());
      config_dim->set_edge_padding_high(
          getattr(dimension, "edge_padding_high").cast<xla::int64>());
      config_dim->set_interior_padding(
          getattr(dimension, "interior_padding").cast<xla::int64>());
    }
    return true;
  }
};

template <>
struct type_caster<xla::OpMetadata> {
 public:
  PYBIND11_TYPE_CASTER(xla::OpMetadata, _("xla::OpMetadata"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    pybind11::handle op_type = getattr(handle, "op_type");
    if (!op_type.is_none()) {
      value.set_op_type(op_type.cast<std::string>());
    }
    pybind11::handle op_name = getattr(handle, "op_name");
    if (!op_name.is_none()) {
      value.set_op_name(op_name.cast<std::string>());
    }
    pybind11::handle source_file = getattr(handle, "source_file");
    if (!source_file.is_none()) {
      value.set_source_file(source_file.cast<std::string>());
    }
    pybind11::handle source_line = getattr(handle, "source_line");
    if (!source_line.is_none()) {
      value.set_source_line(source_line.cast<xla::int32>());
    }
    return true;
  }
};

template <>
struct type_caster<xla::PrecisionConfig> {
 public:
  PYBIND11_TYPE_CASTER(xla::PrecisionConfig, _("xla::PrecisionConfig"));

  // PyObject -> C++ conversion.
  bool load(handle handle, bool) {
    if (handle.is_none()) {
      return true;
    }

    sequence operand_precisions =
        reinterpret_borrow<sequence>(getattr(handle, "operand_precision"));

    for (auto operand_precision : operand_precisions) {
      value.add_operand_precision(
          operand_precision.cast<xla::PrecisionConfig::Precision>());
    }
    return true;
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TYPES_H_
