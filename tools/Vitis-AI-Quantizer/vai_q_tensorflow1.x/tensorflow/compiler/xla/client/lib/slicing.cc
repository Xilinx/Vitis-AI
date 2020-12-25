/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/slicing.h"

#include <limits>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

XlaOp SliceInMinorDims(XlaOp x, absl::Span<const int64> start,
                       absl::Span<const int64> end) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RET_CHECK(start.size() == end.size());
    int64 n_minor_dims = start.size();

    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));

    const int64 n_dims = shape.rank();
    TF_RET_CHECK(n_minor_dims <= n_dims);
    auto major_dims = AsInt64Slice(shape.dimensions())
                          .subspan(
                              /*pos=*/0,
                              /*len=*/n_dims - n_minor_dims);

    // Prepends 0s in the major dim
    std::vector<int64> padded_start(n_dims, 0);
    std::copy(start.begin(), start.end(),
              padded_start.begin() + major_dims.size());

    // Prepends the shape of the major dims.
    std::vector<int64> padded_end(n_dims);
    std::copy(major_dims.begin(), major_dims.end(), padded_end.begin());
    std::copy(end.begin(), end.end(), padded_end.begin() + major_dims.size());

    std::vector<int64> strides(n_dims, 1);
    return Slice(x, padded_start, padded_end, strides);
  });
}

XlaOp UpdateSlice(XlaOp x, XlaOp update, absl::Span<const int64> start) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = shape.rank();
    TF_RET_CHECK(start.size() == n_dims);

    // TODO(phawkins): make int64 work on all backends, remove the int32 cast.
    std::vector<int32> start_as_int32(start.begin(), start.end());
    std::vector<XlaOp> start_ops(start.size());
    for (int i = 0; i < start.size(); ++i) {
      start_ops[i] = ConstantR0(builder, start_as_int32[i]);
    }
    return DynamicUpdateSlice(x, update, start_ops);
  });
}

XlaOp UpdateSliceInMinorDims(XlaOp x, XlaOp update,
                             absl::Span<const int64> start) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = shape.rank();
    const int64 n_minor_dims = start.size();
    TF_RET_CHECK(n_minor_dims <= n_dims);
    std::vector<int64> padded_start(n_dims, 0);
    std::copy(start.begin(), start.end(),
              padded_start.begin() + (n_dims - n_minor_dims));
    return UpdateSlice(x, update, padded_start);
  });
}

namespace {

std::vector<int64> ConcatVectors(absl::Span<const int64> xs,
                                 absl::Span<const int64> ys) {
  std::vector<int64> output(xs.size() + ys.size());
  std::copy(xs.begin(), xs.end(), output.begin());
  std::copy(ys.begin(), ys.end(), output.begin() + xs.size());
  return output;
}

StatusOr<std::vector<XlaOp>> PrependZerosInMajorDims(
    XlaOp x, absl::Span<const XlaOp> starts) {
  XlaBuilder* builder = x.builder();
  TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
  const int64 n_dims = shape.rank();
  auto zero = ConstantR0<int32>(builder, 0);
  std::vector<XlaOp> padded_starts(n_dims, zero);
  for (int i = 0; i < starts.size(); ++i) {
    padded_starts[n_dims - starts.size() + i] = starts[i];
  }
  return padded_starts;
}

}  // namespace

XlaOp DynamicSliceInMinorDims(XlaOp x, absl::Span<const XlaOp> starts,
                              absl::Span<const int64> sizes) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = shape.rank();
    int64 n_minor_dims = starts.size();
    TF_RET_CHECK(n_minor_dims == sizes.size());
    TF_RET_CHECK(n_minor_dims <= n_dims);
    auto major_dims = AsInt64Slice(shape.dimensions())
                          .subspan(
                              /*pos=*/0,
                              /*len=*/n_dims - sizes.size());
    TF_ASSIGN_OR_RETURN(auto padded_starts, PrependZerosInMajorDims(x, starts));
    auto padded_sizes = ConcatVectors(major_dims, sizes);
    return DynamicSlice(x, padded_starts, padded_sizes);
  });
}

XlaOp DynamicUpdateSliceInMinorDims(XlaOp x, XlaOp update,
                                    absl::Span<const XlaOp> starts) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto padded_starts, PrependZerosInMajorDims(x, starts));
    return DynamicUpdateSlice(x, update, padded_starts);
  });
}

XlaOp TorchGather(XlaOp input, XlaOp index, int64 dim, bool sparse) {
  XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape index_shape, builder->GetShape(index));
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    if (ShapeUtil::ElementHasBitWidth(index_shape, 64) &&
        input_shape.dimensions(dim) < std::numeric_limits<uint32>::max()) {
      index = ConvertElementType(index, U32);
      index_shape.set_element_type(U32);
    }
    if (index_shape.rank() == 1) {
      return TorchIndexSelect(input, index, 0);
    }
    if (!sparse) {
      std::vector<int64> index_broacast_dims;
      std::vector<int64> input_broacast_dims;
      std::vector<int64> sizes;
      for (int64 i = 0; i < index_shape.rank(); ++i) {
        if (i < dim) {
          input_broacast_dims.push_back(i);
          index_broacast_dims.push_back(i);
        } else if (i == dim) {
          sizes.push_back(input_shape.dimensions(i));
          input_broacast_dims.push_back(i);
          index_broacast_dims.push_back(i + 1);
        } else {
          input_broacast_dims.push_back(i + 1);
          index_broacast_dims.push_back(i + 1);
        }
        sizes.push_back(index_shape.dimensions(i));
      }
      auto mask = Eq(
          BroadcastInDim(index, sizes, index_broacast_dims),
          Iota(builder, ShapeUtil::MakeShape(index_shape.element_type(), sizes),
               dim));
      auto masked_input = Select(
          mask, BroadcastInDim(input, sizes, input_broacast_dims),
          Zeros(builder,
                ShapeUtil::MakeShape(input_shape.element_type(), sizes)));
      return Reduce(masked_input, Zero(builder, input_shape.element_type()),
                    CreateScalarIdentityWithZeroComputation(
                        input_shape.element_type(), builder),
                    {dim});
    }

    ShapeUtil::AppendMajorDimension(1, &index_shape);
    std::vector<XlaOp> to_concat;

    to_concat.reserve(input_shape.rank());
    for (int64 i = 0; i < input_shape.rank(); ++i) {
      if (i == dim) {
        to_concat.push_back(Reshape(index, index_shape.dimensions()));
      } else {
        to_concat.push_back(Iota(builder, index_shape, i));
      }
    }
    XlaOp gather_indices = ConcatInDim(builder, to_concat, input_shape.rank());
    std::vector<int64> slice_sizes(input_shape.rank(), 1);
    GatherDimensionNumbers gather_dnums;
    gather_dnums.set_index_vector_dim(input_shape.rank());
    for (int64 i = 0; i < input_shape.rank(); ++i) {
      gather_dnums.add_collapsed_slice_dims(i);
      gather_dnums.add_start_index_map(i);
    }
    return Gather(input, gather_indices, gather_dnums, slice_sizes);
  });
}

XlaOp TorchIndexSelect(XlaOp input, XlaOp index, int64 dim, int64 batch_dims) {
  XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    TF_ASSIGN_OR_RETURN(Shape index_shape, builder->GetShape(index));
    if (dim < batch_dims) {
      return InvalidArgument(
          "Gather dim must be greater than or equal to the number of batch "
          "dims");
    }
    if (ShapeUtil::ElementHasBitWidth(index_shape, 64) &&
        input_shape.dimensions(dim) < std::numeric_limits<uint32>::max()) {
      index = ConvertElementType(index, U32);
      index_shape.set_element_type(U32);
    }
    std::vector<int64> slice_sizes = input_shape.dimensions();
    GatherDimensionNumbers gather_dnums;
    gather_dnums.set_index_vector_dim(index_shape.rank());
    if (batch_dims > 0) {
      ShapeUtil::AppendMajorDimension(1, &index_shape);
      std::vector<XlaOp> to_concat;
      to_concat.reserve(batch_dims + 1);
      for (int64 batch_dim = 0; batch_dim < batch_dims; ++batch_dim) {
        to_concat.push_back(Iota(builder, index_shape, batch_dim));
      }
      to_concat.push_back(Reshape(index, index_shape.dimensions()));
      index = ConcatInDim(builder, to_concat, gather_dnums.index_vector_dim());
    }
    for (int64 i = 0; i < input_shape.rank(); ++i) {
      if (i < batch_dims || i == dim) {
        if (slice_sizes[i] != 0) {
          slice_sizes[i] = 1;
          gather_dnums.add_collapsed_slice_dims(i);
        }
        gather_dnums.add_start_index_map(i);
      } else {
        if (i < dim) {
          gather_dnums.add_offset_dims(i);
        } else {
          gather_dnums.add_offset_dims(i + gather_dnums.index_vector_dim() -
                                       (1 + batch_dims));
        }
      }
    }
    return Gather(input, index, gather_dnums, slice_sizes);
  });
}

}  // namespace xla
