/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/dnn.h"

#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

namespace stream_executor {
namespace dnn {

uint64 AlgorithmDesc::hash() const {
  auto p = std::make_pair(algo_id(), tensor_ops_enabled());
  return absl::Hash<decltype(p)>()(p);
}

bool DnnSupport::GetConvolveAlgorithms(
    bool with_winograd_nonfused, int cc_major, int cc_minor,
    std::vector<AlgorithmDesc>* out_algorithms) {
  return false;
}

bool DnnSupport::GetRnnAlgorithms(std::vector<AlgorithmDesc>* out_algorithms) {
  return false;
}

bool DnnSupport::GetConvolveBackwardDataAlgorithms(
    bool with_winograd_nonfused, int cc_major, int cc_minor,
    std::vector<AlgorithmDesc>* out_algorithms) {
  return false;
}

bool DnnSupport::GetConvolveBackwardFilterAlgorithms(
    bool with_winograd_nonfused, int cc_major, int cc_minor,
    std::vector<AlgorithmDesc>* out_algorithms) {
  return false;
}

string QuantizedActivationModeString(QuantizedActivationMode mode) {
  switch (mode) {
    case dnn::QuantizedActivationMode::k8Bit:
      return "uint8";
    case dnn::QuantizedActivationMode::k16Bit:
      return "uint16";
    case dnn::QuantizedActivationMode::k32Bit:
      return "int32";
    default:
      LOG(FATAL) << "Unknown quantized_activation_mode "
                 << static_cast<int32>(mode);
  }
  return "unknown quantized_activation_mode";
}

string ActivationModeString(ActivationMode mode) {
  switch (mode) {
    case ActivationMode::kSigmoid:
      return "sigmoid";
    case ActivationMode::kRelu:
      return "relu";
    case ActivationMode::kRelu6:
      return "relu6";
    case ActivationMode::kReluX:
      return "reluX";
    case ActivationMode::kTanh:
      return "tanh";
    case ActivationMode::kBandPass:
      return "bandpass";
    default:
      LOG(FATAL) << "Unknown activation_mode " << static_cast<int32>(mode);
  }
  return "unknown activation_mode";
}

string ElementwiseOperationString(ElementwiseOperation op) {
  switch (op) {
    case ElementwiseOperation::kAdd:
      return "add";
    case ElementwiseOperation::kMultiply:
      return "multiply";
    default:
      LOG(FATAL) << "Unknown elementwise op " << static_cast<int32>(op);
  }
  return "unknown element wise op";
}

string DataLayoutString(DataLayout layout) {
  switch (layout) {
    case DataLayout::kYXDepthBatch:
      return "YXDepthBatch";
    case DataLayout::kYXBatchDepth:
      return "YXBatchDepth";
    case DataLayout::kBatchYXDepth:
      return "BatchYXDepth";
    case DataLayout::kBatchDepthYX:
      return "BatchDepthYX";
    case DataLayout::kBatchDepthYX4:
      return "BatchDepthYX4";
    default:
      LOG(FATAL) << "Unknown data layout " << static_cast<int32>(layout);
  }
  return "unknown data layout";
}

string FilterLayoutString(FilterLayout layout) {
  switch (layout) {
    case FilterLayout::kOutputInputYX:
      return "OutputInputYX";
    case FilterLayout::kOutputYXInput:
      return "OutputYXInput";
    case FilterLayout::kOutputInputYX4:
      return "OutputInputYX4";
    case FilterLayout::kInputYXOutput:
      return "InputYXOutput";
    case FilterLayout::kYXInputOutput:
      return "YXInputOutput";
    default:
      LOG(FATAL) << "Unknown filter layout " << static_cast<int32>(layout);
  }
  return "unknown filter layout";
}

string PadAlignmentString(PadAlignment alignment) {
  switch (alignment) {
    case PadAlignment::kDefault:
      return "default";
    case PadAlignment::kCudnnPadding:
      return "cuDNN padding";
    case PadAlignment::kTensorFlowPadding:
      return "TensorFlow padding";
  }
  return "unknown pad alignment";
}

std::ostream& operator<<(std::ostream& str, dnn::PadAlignment alignment) {
  return str << PadAlignmentString(alignment);
}

string ShortPoolingModeString(PoolingMode mode) {
  switch (mode) {
    case PoolingMode::kMaximum:
      return "Max";
    case PoolingMode::kAverage:
      return "Avg";
    default:
      LOG(FATAL) << "Unknown filter layout " << static_cast<int32>(mode);
  }
  return "unknown filter layout";
}

std::tuple<int, int, int> GetDimIndices(const DataLayout& layout,
                                        const int data_dims) {
  int depth_idx, batch_idx, spatial_idx;
  switch (layout) {
    case DataLayout::kYXBatchDepth:
      depth_idx = data_dims - 1;
      batch_idx = data_dims - 2;
      spatial_idx = 0;
      break;

    case DataLayout::kYXDepthBatch:
      depth_idx = data_dims - 2;
      batch_idx = data_dims - 1;
      spatial_idx = 0;
      break;

    case DataLayout::kBatchYXDepth:
      depth_idx = data_dims - 1;
      batch_idx = 0;
      spatial_idx = 1;
      break;

    case DataLayout::kBatchDepthYX:
    case DataLayout::kBatchDepthYX4:
      depth_idx = 1;
      batch_idx = 0;
      spatial_idx = 2;
      break;

    default:
      LOG(FATAL) << "Unknown layout " << layout;
  }

  return std::make_tuple(depth_idx, batch_idx, spatial_idx);
}

std::vector<int64> ReorderDims(const std::vector<int64>& input,
                               const DataLayout& from, const DataLayout& to) {
  if (from == to) return input;

  int d_idx_from, b_idx_from, spatial_idx_from;
  int d_idx_to, b_idx_to, spatial_idx_to;

  std::tie(d_idx_from, b_idx_from, spatial_idx_from) =
      GetDimIndices(from, input.size());
  std::tie(d_idx_to, b_idx_to, spatial_idx_to) =
      GetDimIndices(to, input.size());

  std::vector<int64> reordered(input.size());
  reordered[b_idx_to] = input[b_idx_from];
  reordered[d_idx_to] = input[d_idx_from];

  for (size_t i = 0; i < input.size() - 2;
       i++, spatial_idx_from++, spatial_idx_to++) {
    reordered[spatial_idx_to] = input[spatial_idx_from];
  }

  return reordered;
}

// -- AlgorithmConfig

string AlgorithmConfig::ToString() const {
  AlgorithmDesc::Index algo_id = -1;
  if (algorithm().has_value()) {
    algo_id = algorithm()->algo_id();
  }
  AlgorithmDesc::Index algo_id_no_scratch = -1;
  if (algorithm_no_scratch().has_value()) {
    algo_id_no_scratch = algorithm_no_scratch()->algo_id();
  }
  return absl::StrCat(algo_id, ", ", algo_id_no_scratch);
}

// -- BatchDescriptor

BatchDescriptor::BatchDescriptor(int ndims)
    : value_max_(0.0),
      value_min_(0.0),
      quantized_activation_mode_(QuantizedActivationMode::k8Bit) {
  tensor_.mutable_dimensions()->Resize(ndims + 2, 0);
  set_layout(DataLayout::kYXDepthBatch);
}

BatchDescriptor::BatchDescriptor() : BatchDescriptor(/*ndims=*/2) {}

std::vector<int64> BatchDescriptor::full_dims(const DataLayout& layout) const {
  std::vector<int64> bdyx_dims(ndims() + 2);
  bdyx_dims[0] = count();
  bdyx_dims[1] = feature_map_count();
  std::copy(spatial_size().begin(), spatial_size().end(),
            bdyx_dims.begin() + 2);
  return ReorderDims(bdyx_dims, DataLayout::kBatchDepthYX, layout);
}

std::vector<int64> BatchDescriptor::full_strides(
    const DataLayout& layout) const {
  if (this->layout() == DataLayout::kBatchDepthYX4) {
    LOG(FATAL)
        << "Cannot compute full strides for batch descriptor " << ToString()
        << ", because its layout is kBatchDepthYX4. In fact, "
           "cudnnSetTensorNdDescriptor doesn't work for kBatchDepthYX4 at all. "
           "Use cudnnSetTensor4DDescriptor to set cudnnTensorDescriptor_t "
           "instead.";
  }
  std::vector<int64> phys_dims = full_dims(this->layout());
  std::vector<int64> phys_strides(phys_dims.size());
  phys_strides[ndims() + 1] = 1;
  for (int i = ndims(); i >= 0; i--) {
    phys_strides[i] = phys_strides[i + 1] * phys_dims[i + 1];
  }
  return ReorderDims(phys_strides, this->layout(), layout);
}

void BatchDescriptor::CloneFrom(const BatchDescriptor& other) {
  tensor_ = other.tensor_;
  value_max_ = other.value_max_;
  value_min_ = other.value_min_;
  quantized_activation_mode_ = other.quantized_activation_mode_;
}

string BatchDescriptor::ToString() const {
  string spatial;
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&spatial, "%d ", spatial_size()[i]);
  }
  return absl::StrFormat(
      "{count: %d feature_map_count: %d spatial: %s "
      "value_min: %f value_max: %f layout: %s}",
      count(), feature_map_count(), spatial, value_min_, value_max_,
      DataLayoutString(layout()));
}

string BatchDescriptor::ToShortString() const {
  // All the constituent strings are less than 15 characters, so the
  // small string optimization ensures that there will be at most one
  // heap memory allocation.
  string depth = absl::StrCat("d", feature_map_count());
  string batch = absl::StrCat("b", count());

  string spatial = "s";
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&spatial, "%d ", spatial_size()[i]);
  }

  string suffix;
  if (value_min() != value_max()) {
    absl::StrAppend(&suffix, "[", value_min(), ";", value_max(), "]");
  }
  if (quantized_activation_mode() == QuantizedActivationMode::k16Bit) {
    suffix += "_16bit";
  }

  switch (layout()) {
    case DataLayout::kYXDepthBatch:
      return absl::StrCat(spatial, depth, batch, suffix);
    case DataLayout::kYXBatchDepth:
      return absl::StrCat(spatial, batch, depth, suffix);
    case DataLayout::kBatchYXDepth:
      return absl::StrCat(batch, spatial, depth, suffix);
    case DataLayout::kBatchDepthYX:
      return absl::StrCat(batch, depth, spatial, suffix);
    case DataLayout::kBatchDepthYX4:
      return absl::StrCat(batch, depth, spatial, suffix, "(VECT_C)");
    default:
      LOG(FATAL) << "Unknown layout " << static_cast<int32>(layout());
      return "";  // Avoid return warning (unreachable)
  }
}

int64 BatchDescriptor::NodesPerFeatureMap() const {
  int64 ret = 1;
  for (int i = 0; i < ndims(); i++) {
    ret *= spatial_size()[i];
  }
  return ret;
}

int64 BatchDescriptor::NodesAcrossFeatureMaps() const {
  return NodesPerFeatureMap() * feature_map_count();
}

int64 BatchDescriptor::ElementCount() const {
  return count() * feature_map_count() * NodesPerFeatureMap();
}

int64 BatchDescriptor::FullyConnectedWeightCount(
    const BatchDescriptor& input, const BatchDescriptor& output) {
  return input.NodesAcrossFeatureMaps() * output.NodesAcrossFeatureMaps();
}

int64 BatchDescriptor::FullyConnectedBiasCount(const BatchDescriptor& output) {
  return output.NodesAcrossFeatureMaps();
}

BatchDescriptor BatchDescriptor::DepthConcatenateOutputDescriptor(
    port::ArraySlice<dnn::BatchDescriptor> inputs) {
  if (inputs.empty()) {
    return BatchDescriptor();
  }
  int feature_map_count = 0;
  for (const auto& dimensions : inputs) {
    feature_map_count += dimensions.feature_map_count();
  }
  BatchDescriptor output = inputs[0];
  output.set_feature_map_count(feature_map_count);
  return output;
}

TensorDescriptorProto BatchDescriptor::ToProto(DataType data_type) const {
  CHECK_EQ(0.0, value_max_);
  CHECK_EQ(0.0, value_min_);
  CHECK(quantized_activation_mode_ == QuantizedActivationMode::k8Bit);

  TensorDescriptorProto ret = tensor_;
  ret.set_data_type(data_type);
  return ret;
}

// -- FilterDescriptor

FilterDescriptor::FilterDescriptor(int ndims) {
  tensor_.mutable_dimensions()->Resize(ndims + 2, 0);
  set_layout(FilterLayout::kOutputInputYX);
}

FilterDescriptor::FilterDescriptor() : FilterDescriptor(/*ndims=*/2) {}

FilterDescriptor::~FilterDescriptor() {}

void FilterDescriptor::CloneFrom(const FilterDescriptor& other) {
  tensor_ = other.tensor_;
}

string FilterDescriptor::ToString() const {
  string desc = absl::StrFormat(
      "{output_feature_map_count: %d input_feature_map_count: %d "
      "layout: %s shape: ",
      output_feature_map_count(), input_feature_map_count(),
      FilterLayoutString(layout()));
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&desc, "%d ", input_filter_dims()[i]);
  }
  absl::StrAppend(&desc, "}");

  return desc;
}

string FilterDescriptor::ToShortString() const {
  // All the constituent strings are less than 15 characters, so the
  // small string optimization ensures that there will be at most one
  // heap memory allocation.
  string od = absl::StrCat("od", output_feature_map_count());
  string id = absl::StrCat("id", input_feature_map_count());

  string spatial = "s";
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&spatial, "%d ", input_filter_dims()[i]);
  }

  switch (layout()) {
    case FilterLayout::kOutputInputYX:
      return absl::StrCat(od, id, spatial);
    case FilterLayout::kOutputYXInput:
      return absl::StrCat(od, spatial, id);
    case FilterLayout::kOutputInputYX4:
      return absl::StrCat(od, id, spatial, "(VECT_C)");
    case FilterLayout::kInputYXOutput:
      return absl::StrCat(id, spatial, od);
    case FilterLayout::kYXInputOutput:
      return absl::StrCat(spatial, id, od);
    default:
      LOG(FATAL) << "Unknown layout " << static_cast<int32>(layout());
      return "";  // Avoid return warning (unreachable)
  }
}

int64 FilterDescriptor::ComputeWeightCount() const {
  int64 ret = output_feature_map_count() * input_feature_map_count();
  for (int i = 0; i < ndims(); i++) {
    ret *= input_filter_dims()[i];
  }
  return ret;
}

TensorDescriptorProto FilterDescriptor::ToProto(DataType data_type) const {
  TensorDescriptorProto ret = tensor_;
  ret.set_data_type(data_type);
  return ret;
}

// -- ConvolutionDescriptor

ConvolutionDescriptor::ConvolutionDescriptor(int ndims) {
  proto_.mutable_paddings()->Resize(ndims, 0);
  proto_.mutable_strides()->Resize(ndims, 1);
  proto_.mutable_dilations()->Resize(ndims, 1);
  proto_.set_group_count(1);
  proto_.set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);
}

ConvolutionDescriptor::ConvolutionDescriptor()
    : ConvolutionDescriptor(/*ndims=*/2) {}

ConvolutionDescriptor::~ConvolutionDescriptor() {}

string ConvolutionDescriptor::ToString() const {
  string padding;
  string strides;
  string dilations;
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&padding, "%d ", this->padding()[i]);
    absl::StrAppendFormat(&strides, "%d ", this->strides()[i]);
    absl::StrAppendFormat(&dilations, "%d ", this->dilations()[i]);
  }

  return absl::StrFormat(
      "{zero_padding: %s pad_alignment: %s filter_strides: %s dilation_rates: "
      "%s}",
      padding, PadAlignmentString(pad_alignment()), strides, dilations);
}

string ConvolutionDescriptor::ToShortString() const {
  string desc;
  for (int i = 0; i < ndims(); i++) {
    if (i > 0) absl::StrAppend(&desc, "_");
    absl::StrAppendFormat(&desc, "p%d:%d", i, padding()[i]);
  }
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&desc, "_s%d:%d", i, strides()[i]);
  }
  for (int i = 0; i < ndims(); i++) {
    absl::StrAppendFormat(&desc, "_d%d:%d", i, dilations()[i]);
  }
  return desc;
}

// -- PoolingDescriptor

PoolingDescriptor::PoolingDescriptor(int ndims)
    : mode_(dnn::PoolingMode::kMaximum),
      ndims_(ndims),
      propagate_nans_(false),
      window_(ndims, 0),
      padding_(ndims, 0),
      strides_(ndims, 1) {}

PoolingDescriptor::PoolingDescriptor() : PoolingDescriptor(/*ndims=*/2) {}

void PoolingDescriptor::CloneFrom(const PoolingDescriptor& other) {
  mode_ = other.mode_;
  ndims_ = other.ndims_;
  window_ = other.window_;
  padding_ = other.padding_;
  strides_ = other.strides_;
  propagate_nans_ = other.propagate_nans_;
}

string PoolingDescriptor::ToString() const {
  const char* mode_string =
      mode_ == dnn::PoolingMode::kMaximum ? "kMaximum" : "kAverage";

  string window, strides, padding;
  for (int i = 0; i < ndims_; i++) {
    absl::StrAppendFormat(&window, "%d ", window_[i]);
    absl::StrAppendFormat(&strides, "%d ", strides_[i]);
    absl::StrAppendFormat(&padding, "%d", padding_[i]);
  }

  const char* propagate_string = propagate_nans_ ? "Yes" : "No";

  return absl::StrFormat(
      "{mode: %s window: %s strides: %s padding: %s propagate NaNs: %s}",
      mode_string, window, strides, padding, propagate_string);
}

string PoolingDescriptor::ToShortString() const {
  string window, strides, padding;
  for (int i = 0; i < ndims_; i++) {
    absl::StrAppendFormat(&window, "_w%d:%d", i, window_[i]);
    absl::StrAppendFormat(&strides, "_s%d:%d", i, strides_[i]);
    absl::StrAppendFormat(&padding, "_p%d:%d", i, padding_[i]);
  }
  return absl::StrCat(mode_ == dnn::PoolingMode::kMaximum ? "max" : "avg",
                      window, strides, padding,
                      propagate_nans_ ? "propagate_nans" : "ignore_nans");
}

// -- NormalizeDescriptor

NormalizeDescriptor::NormalizeDescriptor()
    : bias_(0.0),
      range_(0),
      alpha_(0.0),
      beta_(0.0),
      wrap_around_(false),
      segment_size_(0) {}

void NormalizeDescriptor::CloneFrom(const NormalizeDescriptor& other) {
  bias_ = other.bias_;
  range_ = other.range_;
  alpha_ = other.alpha_;
  beta_ = other.beta_;
  wrap_around_ = other.wrap_around_;
  segment_size_ = other.segment_size_;
}

string NormalizeDescriptor::ToString() const {
  return absl::StrFormat(
      "{bias: %f range: %d alpha: %f beta: %f wrap_around: %d "
      "segment_size: %d}",
      bias_, range_, alpha_, beta_, wrap_around_, segment_size_);
}

string NormalizeDescriptor::ToShortString() const {
  return absl::StrCat("bias:", bias_, "_range:", range_, "_alpha:", alpha_,
                      "_beta:", beta_, "_wrap:", wrap_around_,
                      "_size:", segment_size_);
}

bool DnnSupport::IsStatusOk(const port::Status& status, bool report_error) {
  if (status.ok()) {
    return true;
  }
  if (report_error) {
    LOG(ERROR) << status.error_message();
  }
  return false;
}

}  // namespace dnn
}  // namespace stream_executor
