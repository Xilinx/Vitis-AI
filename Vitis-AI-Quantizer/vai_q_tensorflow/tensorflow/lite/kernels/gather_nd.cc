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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace gather_nd {
constexpr int kParams = 0;
constexpr int kIndices = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* params = GetInput(context, node, kParams);
  const TfLiteTensor* indices = GetInput(context, node, kIndices);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (params->type) {
    case kTfLiteFloat32:
    case kTfLiteUInt8:
    case kTfLiteInt8:
    case kTfLiteInt64:
    case kTfLiteInt32:
      break;
    default:
      context->ReportError(
          context, "Params of type '%s' are not supported by gather_nd.",
          TfLiteTypeGetName(params->type));
      return kTfLiteError;
  }
  switch (indices->type) {
    case kTfLiteInt64:
    case kTfLiteInt32:
      break;
    default:
      context->ReportError(
          context, "Indices of type '%s' are not supported by gather_nd.",
          TfLiteTypeGetName(indices->type));
      return kTfLiteError;
  }

  const int params_rank = NumDimensions(params);
  const int indices_rank = NumDimensions(indices);
  const int indices_nd = SizeOfDimension(indices, indices_rank - 1);
  if (params_rank < 1) {
    context->ReportError(context, "Params must be at least a vector.");
    return kTfLiteError;
  }
  if (indices_rank < 1) {
    context->ReportError(context, "Indices must be at least a vector.");
    return kTfLiteError;
  }
  if (indices_nd > params_rank) {
    context->ReportError(
        context, "Index innermost dimension length must be <= params rank.");
    return kTfLiteError;
  }

  // Assign to output the input type.
  output->type = params->type;

  // The result shape is
  // indices.shape[:-1] + params.shape[indices.shape[-1]:]
  const int output_rank = indices_rank + params_rank - indices_nd - 1;
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(output_rank);
  int output_index = 0;
  for (int i = 0; i < indices_rank - 1; ++i) {
    output_shape->data[output_index++] = indices->dims->data[i];
  }
  for (int i = indices_nd; i < params_rank; ++i) {
    output_shape->data[output_index++] = params->dims->data[i];
  }
  return context->ResizeTensor(context, output, output_shape);
}

template <typename ParamsT, typename IndicesT>
TfLiteStatus GatherNd(const TfLiteTensor* params, const TfLiteTensor* indices,
                      TfLiteTensor* output) {
  reference_ops::GatherNd(
      GetTensorShape(params), GetTensorData<ParamsT>(params),
      GetTensorShape(indices), GetTensorData<IndicesT>(indices),
      GetTensorShape(output), GetTensorData<ParamsT>(output));
  return kTfLiteOk;
}

template <typename IndicesT>
TfLiteStatus EvalGatherNd(TfLiteContext* context, const TfLiteTensor* params,
                          const TfLiteTensor* indices, TfLiteTensor* output) {
  switch (params->type) {
    case kTfLiteFloat32:
      return GatherNd<float, IndicesT>(params, indices, output);
    case kTfLiteUInt8:
      return GatherNd<uint8_t, IndicesT>(params, indices, output);
    case kTfLiteInt8:
      return GatherNd<int8_t, IndicesT>(params, indices, output);
    case kTfLiteInt32:
      return GatherNd<int32_t, IndicesT>(params, indices, output);
    case kTfLiteInt64:
      return GatherNd<int64_t, IndicesT>(params, indices, output);
    default:
      context->ReportError(context,
                           "Params type '%s' are not supported by gather_nd.",
                           TfLiteTypeGetName(params->type));
      return kTfLiteError;
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* params = GetInput(context, node, kParams);
  const TfLiteTensor* indices = GetInput(context, node, kIndices);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (indices->type) {
    case kTfLiteInt32:
      return EvalGatherNd<int32_t>(context, params, indices, output);
    case kTfLiteInt64:
      return EvalGatherNd<int64_t>(context, params, indices, output);
    default:
      context->ReportError(
          context, "Indices of type '%s' are not supported by gather_nd.",
          TfLiteTypeGetName(indices->type));
      return kTfLiteError;
  }
}
}  // namespace gather_nd

TfLiteRegistration* Register_GATHER_ND() {
  static TfLiteRegistration r = {/*init*/ nullptr, /*free*/ nullptr,
                                 gather_nd::Prepare, gather_nd::Eval};
  return &r;
}
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
