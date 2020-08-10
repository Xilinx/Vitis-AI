/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Implements convolution operations with other kernels baked into the
// processing, to optimize latency and memory usage:
//  - Conv2D + BiasAdd + <Activation>
//  - Conv2D + FusedBatchNorm + <Activation>
//
// Activation: Relu, Relu6, Elu, etc...
//
// Kernels for convolutions fused with image transformations (resize and mirror
// padding) defined in `conv_ops_fused_image_transform.cc`.
//
// For the CPU device we implement fusion with an Eigen tensor contraction
// output kernel. For the GPU device we rely on CuDNN primitives.
//
// NOTE: GPU only supports fusion of Conv2D + BiasAdd + <optional Relu>.

#ifndef TENSORFLOW_CORE_KERNELS_CONV_OPS_FUSED_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_CONV_OPS_FUSED_IMPL_H_

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#include "tensorflow/stream_executor/cuda/ptxas_utils.h"
#include "tensorflow/stream_executor/cuda/redzone_allocator.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

class AutotuneResult;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchFusedConv2DOp {
  void operator()(OpKernelContext* context, bool use_cudnn,
                  bool cudnn_use_autotune, const Tensor& input,
                  const Tensor& filter, FusedComputationType fusion,
                  const FusedComputationArgs& fusion_args,
                  const Conv2DParameters& params,
                  const Conv2DDimensions& dimensions, Tensor* output);
};

// This is CPU-only implementation that uses Eigen contraction output kernels.
//
// Dispatch 2D convolution to the appropriate primitive operation:
//   (1) MatMul for the case of 1x1 convolution.
//   (2) MatMul for the case when filter size equals to the input size.
//   (3) General spatial 2D convolution for all other cases.
template <typename T>
class LaunchFusedConv2DWithOutputKernel {
 public:
  LaunchFusedConv2DWithOutputKernel(int row_stride, int col_stride,      //
                                    int row_dilation, int col_dilation,  //
                                    Padding padding,
                                    const std::vector<int64>& explicit_paddings)
      : row_stride_(row_stride),
        col_stride_(col_stride),
        row_dilation_(row_dilation),
        col_dilation_(col_dilation),
        padding_(padding),
        explicit_paddings_(explicit_paddings) {}

  template <typename OutputKernel>
  void operator()(const OutputKernel& output_kernel, OpKernelContext* ctx,
                  const Tensor& input, const Tensor& filter, Tensor* output) {
    if (filter.dim_size(0) == 1 && filter.dim_size(1) == 1 &&
        row_stride_ == 1 && col_stride_ == 1 && padding_ != EXPLICIT) {
      int conv_width = 1;  // Width for the convolution step.
      for (int i = 0; i < 3; ++i) {
        conv_width *= output->dim_size(i);
      }

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<CPUDevice, T, OutputKernel>()(
          ctx->eigen_device<CPUDevice>(),
          output->shaped<T, 2>({conv_width, filter.dim_size(3)}),
          input.shaped<T, 2>({conv_width, filter.dim_size(2)}),
          filter.shaped<T, 2>({filter.dim_size(2), filter.dim_size(3)}),
          dim_pair, output_kernel);

    } else if (filter.dim_size(0) == input.dim_size(1) &&
               filter.dim_size(1) == input.dim_size(2) && row_dilation_ == 1 &&
               col_dilation_ == 1 && padding_ == VALID) {
      // If the input data and filter have the same height/width,
      // reduce the 2D convolution to matrix multiplication.
      const auto k =  // Length of reduction dimension.
          filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<CPUDevice, T, OutputKernel>()(
          ctx->eigen_device<CPUDevice>(),
          output->shaped<T, 2>({input.dim_size(0), filter.dim_size(3)}),
          input.shaped<T, 2>({input.dim_size(0), k}),
          filter.shaped<T, 2>({k, filter.dim_size(3)}), dim_pair,
          output_kernel);

    } else {
      if (padding_ == EXPLICIT) {
        functor::SpatialConvolution<CPUDevice, T, OutputKernel>()(
            ctx->eigen_device<CPUDevice>(), output->tensor<T, 4>(),
            input.tensor<T, 4>(), filter.tensor<T, 4>(), row_stride_,
            col_stride_, row_dilation_, col_dilation_,
            static_cast<int>(explicit_paddings_[2]),
            static_cast<int>(explicit_paddings_[3]),
            static_cast<int>(explicit_paddings_[4]),
            static_cast<int>(explicit_paddings_[5]), output_kernel);
      } else {
        functor::SpatialConvolution<CPUDevice, T, OutputKernel>()(
            ctx->eigen_device<CPUDevice>(), output->tensor<T, 4>(),
            input.tensor<T, 4>(), filter.tensor<T, 4>(), row_stride_,
            col_stride_, row_dilation_, col_dilation_,
            BrainPadding2EigenPadding(padding_), output_kernel);
      }
    }
  }

 private:
  int row_stride_;
  int col_stride_;
  int row_dilation_;
  int col_dilation_;
  const Padding padding_;
  const std::vector<int64>& explicit_paddings_;
};

template <typename T>
struct LaunchFusedConv2DOp<CPUDevice, T> {
  void operator()(OpKernelContext* context, bool use_cudnn,
                  bool cudnn_use_autotune, const Tensor& input,
                  const Tensor& filter, const FusedComputationType fusion,
                  const FusedComputationArgs& fusion_args,
                  const Conv2DParameters& params,
                  const Conv2DDimensions& dimensions, Tensor* output) {
    OP_REQUIRES(context, dimensions.in_depth == filter.dim_size(2),
                errors::Unimplemented("Fused conv implementation does not "
                                      "support grouped convolutions for now."));
    OP_REQUIRES(context, params.data_format == FORMAT_NHWC,
                errors::Unimplemented("Fused conv implementation only supports "
                                      "NHWC tensor format for now."));

    BiasAddArgs<T> bias_add_args;
    if (BiasAddArgs<T>::IsSupported(fusion)) {
      OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add_args));
    }

    FusedBatchNormArgs<T> fused_batch_norm_args;
    if (FusedBatchNormArgs<T>::IsSupported(fusion)) {
      OP_REQUIRES_OK(context,
                     InitFusedBatchNormArgs(context, fusion_args.epsilon,
                                            &fused_batch_norm_args));
    }

    LaunchFusedConv2DWithOutputKernel<T> conv2d(
        dimensions.stride_rows, dimensions.stride_cols,
        dimensions.dilation_rows, dimensions.dilation_cols, params.padding,
        params.explicit_paddings);

    switch (fusion) {
      case FusedComputationType::kUndefined:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type is undefined"));
        break;
      case FusedComputationType::kBiasAdd:
        conv2d(WithBiasAdd<T>(bias_add_args), context, input, filter, output);
        break;
      case FusedComputationType::kBiasAddWithRelu:
        conv2d(WithBiasAddAndRelu<T>(bias_add_args), context, input, filter,
               output);
        break;
      case FusedComputationType::kBiasAddWithRelu6:
        conv2d(WithBiasAddAndRelu6<T>(bias_add_args), context, input, filter,
               output);
        break;
      case FusedComputationType::kBiasAddWithElu:
        conv2d(WithBiasAddAndElu<T>(bias_add_args), context, input, filter,
               output);
        break;
      case FusedComputationType::kFusedBatchNorm:
        conv2d(
            WithFusedBatchNorm<T>(fusion_args.epsilon, fused_batch_norm_args),
            context, input, filter, output);
        break;
      case FusedComputationType::kFusedBatchNormWithRelu:
        conv2d(WithFusedBatchNormAndRelu<T>(fusion_args.epsilon,
                                            fused_batch_norm_args),
               context, input, filter, output);
        break;
      case FusedComputationType::kFusedBatchNormWithRelu6:
        conv2d(WithFusedBatchNormAndRelu6<T>(fusion_args.epsilon,
                                             fused_batch_norm_args),
               context, input, filter, output);
        break;
      case FusedComputationType::kFusedBatchNormWithElu:
        conv2d(WithFusedBatchNormAndElu<T>(fusion_args.epsilon,
                                           fused_batch_norm_args),
               context, input, filter, output);
        break;
    }
  }
};

#if GOOGLE_CUDA

// Encapsulate the default shape information that is used by the convolution
// operation, and add an activation mode for the fusion.
class FusedConvParameters : public ConvParameters {
 public:
  FusedConvParameters(const ConvParameters& base,
                      const se::dnn::ActivationMode activation_mode)
      : ConvParameters(base), activation_mode_(activation_mode) {}

  string ToString() const {
    return absl::StrCat(ConvParameters::ToString(), ", ", activation_mode_);
  }

 private:
  friend bool operator==(const FusedConvParameters& lhs,
                         const FusedConvParameters& rhs);

  using ParameterDataType =
      std::tuple<ConvParameters::ParameterDataType, se::dnn::ActivationMode>;

  ParameterDataType get_data_as_tuple() const {
    return std::make_tuple(ConvParameters::get_data_as_tuple(),
                           activation_mode_);
  }

  se::dnn::ActivationMode activation_mode_;
};

inline bool operator==(const FusedConvParameters& lhs,
                       const FusedConvParameters& rhs) {
  return lhs.get_data_as_tuple() == rhs.get_data_as_tuple();
}

inline bool operator!=(const FusedConvParameters& lhs,
                       const FusedConvParameters& rhs) {
  return !(lhs == rhs);
}

// A dummy type to group forward convolution autotune results together.
struct FusedConvAutoTuneGroup {
  static string name() { return "FusedConv"; }
};

using AutoTuneFusedConv =
    AutoTuneSingleton<FusedConvAutoTuneGroup, FusedConvParameters,
                      se::dnn::AlgorithmConfig>;

inline int64 ConvolveScratchSize() {
  static int64 convolve_scratch_size = GetDnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );
  return convolve_scratch_size;
}

// Finds the best convolutiun algorithm for the given ConvLaunch (cuda
// convolution on the stream) and parameters, by running all possible
// algorithms and measuring execution time.
// TODO(ezhulenev): Move it to conv_ops_gpu.h and share with conv_ops.cc.
template <typename T, typename ConvLaunch, typename LogFunc>
Status FindBestConvolveAlgorithm(const FusedConvParameters& params,
                                 const ConvLaunch launch,
                                 OpKernelContext* context, se::Stream* stream,
                                 se::DeviceMemory<T> output_ptr,
                                 const LogFunc& log,
                                 se::dnn::AlgorithmConfig* algorithm_config) {
  // Check if we already have an algorithm selected for the given parameters.
  if (AutoTuneFusedConv::GetInstance()->Find(params, algorithm_config)) {
    return Status::OK();
  }

  // Find all candidate algorithms.
  std::vector<se::dnn::AlgorithmDesc> algorithms;
  if (!stream->parent()->GetConvolveAlgorithms(
          params.ShouldIncludeWinogradNonfusedAlgo<T>(stream->parent()),
          &algorithms)) {
    return errors::Unknown(
        "Failed to get convolution algorithm. This is probably "
        "because cuDNN failed to initialize, so try looking to "
        "see if a warning log message was printed above.");
  }

  se::TfAllocatorAdapter tf_allocator_adapter(
      context->device()->GetAllocator({}), stream);
  se::cuda::RedzoneAllocator rz_allocator(stream, &tf_allocator_adapter,
                                          se::cuda::PtxCompilationOptions());
  se::DeviceMemory<T> output_ptr_rz(
      WrapRedzoneBestEffort(&rz_allocator, output_ptr));

  std::vector<tensorflow::AutotuneResult> results;
  for (auto profile_algorithm : algorithms) {
    DnnScratchAllocator scratch_allocator(ConvolveScratchSize(), context);
    se::cuda::RedzoneAllocator rz_scratch_allocator(
        stream, &tf_allocator_adapter, se::cuda::PtxCompilationOptions(),
        /*memory_limit=*/ConvolveScratchSize());
    se::ScratchAllocator* allocator_used =
        !RedzoneCheckDisabled()
            ? static_cast<se::ScratchAllocator*>(&rz_scratch_allocator)
            : static_cast<se::ScratchAllocator*>(&scratch_allocator);
    se::dnn::ProfileResult profile_result;

    bool cudnn_launch_status =
        launch(se::dnn::AlgorithmConfig(profile_algorithm), allocator_used,
               output_ptr_rz, &profile_result);

    if (cudnn_launch_status && profile_result.is_valid()) {
      results.emplace_back();
      auto& result = results.back();
      result.mutable_conv()->set_algorithm(profile_algorithm.algo_id());
      result.mutable_conv()->set_tensor_ops_enabled(
          profile_algorithm.tensor_ops_enabled());
      result.set_scratch_bytes(
          !RedzoneCheckDisabled()
              ? rz_scratch_allocator.TotalAllocatedBytesExcludingRedzones()
              : scratch_allocator.TotalByteSize());
      *result.mutable_run_time() = proto_utils::ToDurationProto(
          absl::Milliseconds(profile_result.elapsed_time_in_ms()));
      CheckRedzones(rz_scratch_allocator, &result);
      CheckRedzones(rz_allocator, &result);
    }
  }
  // Only log on an AutoTuneFusedConv cache miss.
  log(results);
  TF_RETURN_IF_ERROR(BestCudnnConvAlgorithm(results, algorithm_config));
  AutoTuneFusedConv::GetInstance()->Insert(params, *algorithm_config);
  return Status::OK();
}

template <typename T>
struct LaunchFusedConv2DOp<GPUDevice, T> {
  void operator()(OpKernelContext* context, bool use_cudnn,
                  bool cudnn_use_autotune, const Tensor& input_param,
                  const Tensor& filter, FusedComputationType fusion,
                  const FusedComputationArgs& fusion_args,
                  const Conv2DParameters& params,
                  const Conv2DDimensions& dimensions, Tensor* output) {
    OP_REQUIRES(
        context,
        params.data_format == FORMAT_NHWC || params.data_format == FORMAT_NCHW,
        errors::Unimplemented("Fused conv implementation only supports "
                              "NHWC and HCHW tensor formats for now."));

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
    OP_REQUIRES(
        context, use_cudnn,
        errors::Unimplemented("FusedConv2D for GPU is not currently supported "
                              "without cudnn"));

    OP_REQUIRES(
        context, fusion == FusedComputationType::kBiasAddWithRelu,
        errors::Unimplemented("FusedConv2D implementation only supports "
                              "fusing with `BiasAdd + Relu` for now."));

    Tensor input = input_param;

    const int64 in_batch = GetTensorDim(input, params.data_format, 'N');
    int64 in_rows = GetTensorDim(input, params.data_format, 'H');
    int64 in_cols = GetTensorDim(input, params.data_format, 'W');
    const int64 in_depths = GetTensorDim(input, params.data_format, 'C');

    const int64 patch_rows = filter.dim_size(0);
    const int64 patch_cols = filter.dim_size(1);
    const int64 patch_depths = filter.dim_size(2);

    const int64 out_batch = GetTensorDim(*output, params.data_format, 'N');
    const int64 out_rows = GetTensorDim(*output, params.data_format, 'H');
    const int64 out_cols = GetTensorDim(*output, params.data_format, 'W');
    const int64 out_depths = GetTensorDim(*output, params.data_format, 'C');

    // Bias of the following dimensions: [ output_depth ]
    const Tensor& bias = context->input(2);
    OP_REQUIRES(context, bias.dims() == 1,
                errors::InvalidArgument("bias must be 1-dimensional",
                                        bias.shape().DebugString()));
    OP_REQUIRES(context, bias.dim_size(0) == out_depths,
                errors::InvalidArgument("bias depth must be equal to out depth",
                                        bias.shape().DebugString()));

    const int64 common_padding_rows =
        std::min(dimensions.pad_rows_before, dimensions.pad_rows_after);
    const int64 common_padding_cols =
        std::min(dimensions.pad_cols_before, dimensions.pad_cols_after);
    if (dimensions.pad_rows_before != dimensions.pad_rows_after ||
        dimensions.pad_cols_before != dimensions.pad_cols_after) {
      // cuDNN only supports padding the same amount on the left and right
      // sides, and on the top and bottom sides. So we manually create a new
      // padded input tensor such that we can pass it to cuDNN.

      // TODO(reedwm): In some cases, we can avoid an allocation even if the two
      // padding sides are different. For example, if the input is 2x2, the
      // filter is 1x1, the stride is 2, and the padding is (1, 0, 1, 0), the
      // result is equivalent to as if the padding is (1, 1, 1, 1). Changing the
      // padding in such a way would allow us to avoid the allocation.
      Tensor transformed_input;
      const int64 padding_rows_diff =
          std::abs(dimensions.pad_rows_after - dimensions.pad_rows_before);
      const int64 padding_cols_diff =
          std::abs(dimensions.pad_cols_after - dimensions.pad_cols_before);
      const int64 new_in_rows = in_rows + padding_rows_diff;
      const int64 new_in_cols = in_cols + padding_cols_diff;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(
                         DataTypeToEnum<T>::value,
                         ShapeFromFormat(params.data_format, in_batch,
                                         new_in_rows, new_in_cols, in_depths),
                         &transformed_input));
      const int64 input_pad_top =
          dimensions.pad_rows_before - common_padding_rows;
      const int64 input_pad_bottom =
          dimensions.pad_rows_after - common_padding_rows;
      const int64 input_pad_left =
          dimensions.pad_cols_before - common_padding_cols;
      const int64 input_pad_right =
          dimensions.pad_cols_after - common_padding_cols;
      bool in_bounds =
          FastBoundsCheck(input_pad_top, std::numeric_limits<int>::max()) &&
          FastBoundsCheck(input_pad_bottom, std::numeric_limits<int>::max()) &&
          FastBoundsCheck(input_pad_left, std::numeric_limits<int>::max()) &&
          FastBoundsCheck(input_pad_right, std::numeric_limits<int>::max());
      if (!in_bounds) {
        context->SetStatus(errors::InvalidArgument("Padding is too large."));
        return;
      }
      functor::PadInput<GPUDevice, T, int, 4>()(
          context->eigen_device<GPUDevice>(),
          To32Bit(input_param.tensor<T, 4>()),
          {{static_cast<int>(input_pad_top), static_cast<int>(input_pad_left)}},
          {{static_cast<int>(input_pad_bottom),
            static_cast<int>(input_pad_right)}},
          To32Bit(transformed_input.tensor<T, 4>()), params.data_format);
      input = transformed_input;
      in_rows = new_in_rows;
      in_cols = new_in_cols;
    }

    if (params.data_format == FORMAT_NHWC) {
      // Convert the input tensor from NHWC to NCHW.
      TensorShape nchw_shape =
          ShapeFromFormat(FORMAT_NCHW, in_batch, in_rows, in_cols, in_depths);
      if (in_depths > 1) {
        Tensor transformed_input;
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<T>::value,
                                              nchw_shape, &transformed_input));
        functor::NHWCToNCHW<GPUDevice, T, 4>()(
            context->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(input).tensor<T, 4>(),
            transformed_input.tensor<T, 4>());
        input = transformed_input;
      } else {
        // If depth <= 1, then just reshape.
        CHECK(input.CopyFrom(input, nchw_shape));  // Crash OK
      }
    }

    CHECK(common_padding_rows >= 0) << "Negative padding rows";  // Crash OK
    CHECK(common_padding_rows >= 0) << "Negative padding cols";  // Crash OK

    se::dnn::ActivationMode dnn_activation_mode;
    switch (fusion) {
      case FusedComputationType::kBiasAddWithRelu:
        dnn_activation_mode = se::dnn::ActivationMode::kRelu;
        break;
      default:
        LOG(FATAL) << "Unsupported fusion type";  // Crash OK
    }

    se::dnn::BatchDescriptor input_desc;
    input_desc.set_count(in_batch)
        .set_feature_map_count(in_depths)
        .set_height(in_rows)
        .set_width(in_cols)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);
    se::dnn::FilterDescriptor filter_desc;
    filter_desc.set_input_filter_height(patch_rows)
        .set_input_filter_width(patch_cols)
        .set_input_feature_map_count(patch_depths)
        .set_output_feature_map_count(filter.dim_size(3));
    se::dnn::BatchDescriptor bias_desc;
    bias_desc.set_count(1)
        .set_height(1)
        .set_width(1)
        .set_feature_map_count(out_depths)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);
    se::dnn::ConvolutionDescriptor conv_desc;
    conv_desc.set_vertical_dilation_rate(dimensions.dilation_rows)
        .set_horizontal_dilation_rate(dimensions.dilation_cols)
        .set_vertical_filter_stride(dimensions.stride_rows)
        .set_horizontal_filter_stride(dimensions.stride_cols)
        .set_zero_padding_height(common_padding_rows)
        .set_zero_padding_width(common_padding_cols)
        .set_group_count(in_depths / patch_depths);
    se::dnn::BatchDescriptor output_desc;
    output_desc.set_count(out_batch)
        .set_height(out_rows)
        .set_width(out_cols)
        .set_feature_map_count(out_depths)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);

    Tensor transformed_filter;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DataTypeToEnum<T>::value,
                       TensorShape({filter.dim_size(3), filter.dim_size(2),
                                    filter.dim_size(0), filter.dim_size(1)}),
                       &transformed_filter));
    functor::TransformFilter<GPUDevice, T, int, 4>()(
        context->eigen_device<GPUDevice>(), FORMAT_OIHW,
        To32Bit(filter.tensor<T, 4>()),
        To32Bit(transformed_filter.tensor<T, 4>()));

    Tensor transformed_output;
    if (params.data_format == FORMAT_NHWC) {
      // Only allocate temporary memory when a layout transformation is needed.
      OP_REQUIRES_OK(context,
                     context->allocate_temp(
                         DataTypeToEnum<T>::value,
                         ShapeFromFormat(FORMAT_NCHW, out_batch, out_rows,
                                         out_cols, out_depths),
                         &transformed_output));
    } else {
      transformed_output = *output;
    }

    const auto tensor_on_device = [](const Tensor& t) -> se::DeviceMemory<T> {
      return AsDeviceMemory(t.template flat<T>().data(),
                            t.template flat<T>().size());
    };

    se::DeviceMemory<T> input_ptr = tensor_on_device(input);
    se::DeviceMemory<T> filter_ptr = tensor_on_device(transformed_filter);
    se::DeviceMemory<T> bias_ptr = tensor_on_device(bias);
    se::DeviceMemory<T> output_ptr = tensor_on_device(transformed_output);

    // We do not use side inputs, so we can safely pass nullptr.
    se::DeviceMemory<T> side_input_ptr =
        AsDeviceMemory(static_cast<T*>(nullptr), 0);

    int device_id = stream->parent()->device_ordinal();
    DataType dtype = input.dtype();
    FusedConvParameters conv_parameters = {
        {in_batch,                      // batch
         in_depths,                     // in_depths
         {{in_rows,                     // in_rows
           in_cols}},                   // in_cols
         FORMAT_NCHW,                   // compute_data_format
         out_depths,                    // out_depths
         {{patch_rows,                  // filter_rows
           patch_cols,                  // filter_cols
           patch_depths}},              // filter_depths
         {{dimensions.dilation_rows,    // dilation_rows
           dimensions.dilation_cols}},  // dilation_cols
         {{dimensions.stride_rows,      // stride_rows
           dimensions.stride_cols}},    // stride_cols
         {{common_padding_rows,         // padding_rows
           common_padding_cols}},       // padding_cols
         dtype,                         // tensor datatype
         device_id,                     // device_id
         conv_desc.group_count()},
        dnn_activation_mode  // activation_mode
    };

    // Launch fused convolution with given parameters and scratch allocator.
    // Record profile result into `profile_result` if it's not nullptr.
    const auto launch = [&](se::dnn::AlgorithmConfig algorithm_config,
                            se::ScratchAllocator* scratch_allocator,
                            se::DeviceMemory<T> output_ptr_to_use,
                            se::dnn::ProfileResult* profile_result) -> bool {
      return stream
          ->ThenFusedConvolveWithAlgorithm(
              input_desc, input_ptr,                     // input
              /*conv_input_scale=*/1.0,                  // input_scale
              filter_desc, filter_ptr,                   // filter
              conv_desc,                                 // conv
              side_input_ptr, /*side_input_scale=*/0.0,  // side_input
              bias_desc, bias_ptr,                       // bias
              dnn_activation_mode,                       // activation
              output_desc, &output_ptr_to_use,           // output
              scratch_allocator, algorithm_config, profile_result)
          .ok();
    };

    se::dnn::AlgorithmConfig algorithm_config;
    if (cudnn_use_autotune) {
      auto status = FindBestConvolveAlgorithm<T>(
          conv_parameters, launch, context, stream, output_ptr,
          [&](absl::Span<const tensorflow::AutotuneResult> results) {
            LogFusedConvForwardAutotuneResults(
                se::dnn::ToDataType<T>::value, input_ptr, filter_ptr,
                output_ptr, bias_ptr, side_input_ptr, input_desc, filter_desc,
                output_desc, conv_desc, 1.0, 0.0, dnn_activation_mode,
                stream->parent(), results);
          },
          &algorithm_config);
      OP_REQUIRES_OK(context, status);
    }

    DnnScratchAllocator scratch_allocator(ConvolveScratchSize(), context);
    bool cudnn_launch_status = launch(algorithm_config, &scratch_allocator,
                                      output_ptr, /*profile_result=*/nullptr);
    OP_REQUIRES(
        context, cudnn_launch_status,
        errors::Internal(absl::Substitute(
            "cuDNN launch failure: input shape($0) filter shape($1)",
            input.shape().DebugString(), filter.shape().DebugString())));

    // Convert the output tensor back from NCHW to NHWC.
    if (params.data_format == FORMAT_NHWC) {
      functor::NCHWToNHWC<GPUDevice, T, 4>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(transformed_output).tensor<T, 4>(),
          output->tensor<T, 4>());
    }
  }
};

#endif  // GOOGLE_CUDA

template <typename Device, typename T>
class FusedConv2DOp : public OpKernel {
 public:
  explicit FusedConv2DOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));

    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();

    using FCT = FusedComputationType;

    std::vector<FusedComputationPattern> patterns;
    if (std::is_same<Device, CPUDevice>::value) {
      patterns = {
          {FCT::kBiasAdd, {"BiasAdd"}},
          {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
          {FCT::kBiasAddWithRelu6, {"BiasAdd", "Relu6"}},
          {FCT::kBiasAddWithElu, {"BiasAdd", "Elu"}},
          {FCT::kFusedBatchNorm, {"FusedBatchNorm"}},
          {FCT::kFusedBatchNormWithRelu, {"FusedBatchNorm", "Relu"}},
          {FCT::kFusedBatchNormWithRelu6, {"FusedBatchNorm", "Relu6"}},
          {FCT::kFusedBatchNormWithElu, {"FusedBatchNorm", "Elu"}},
      };
    }

    // NOTE(ezhulenev): CuDNN `cudnnConvolutionBiasActivationForward` supports
    // identity activation function, it in theory should allow to fuse
    // convolution with BiasAdd, but in practice it doesn't work, cuDNN ignores
    // this parameter and always does Relu activation.
    if (std::is_same<Device, GPUDevice>::value) {
      patterns = {{FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}}};
    }

    OP_REQUIRES_OK(context, InitializeFusedComputation(
                                context, "Conv2D", patterns,
                                &fused_computation_, &fused_computation_args_));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    Conv2DDimensions dimensions;
    OP_REQUIRES_OK(context,
                   ComputeConv2DDimension(params_, input, filter, &dimensions));

    TensorShape out_shape = ShapeFromFormat(
        params_.data_format, dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "FusedConv2D: in_depth = " << dimensions.in_depth
            << ", patch_depth = " << dimensions.patch_depth
            << ", input_cols = " << dimensions.input_cols
            << ", filter_cols = " << dimensions.filter_cols
            << ", input_rows = " << dimensions.input_rows
            << ", filter_rows = " << dimensions.filter_rows
            << ", stride_rows = " << dimensions.stride_rows
            << ", stride_cols = " << dimensions.stride_cols
            << ", dilation_rows = " << dimensions.dilation_rows
            << ", dilation_cols = " << dimensions.dilation_cols
            << ", out_depth = " << dimensions.out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    LaunchFusedConv2DOp<Device, T>()(context, use_cudnn_, cudnn_use_autotune_,
                                     input, filter, fused_computation_,
                                     fused_computation_args_, params_,
                                     dimensions, output);
  }

 private:
  Conv2DParameters params_;
  bool use_cudnn_;
  bool cudnn_use_autotune_;

  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;

  TF_DISALLOW_COPY_AND_ASSIGN(FusedConv2DOp);
};

// Registration of the CPU implementations.
#define REGISTER_FUSED_CPU_CONV2D(T)                                  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_FusedConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FusedConv2DOp<CPUDevice, T>);

#if GOOGLE_CUDA

#define DECLARE_FUNCTOR_GPU_SPEC(T)                                      \
  template <>                                                            \
  void TransformFilter<GPUDevice, T, int, 4>::operator()(                \
      const GPUDevice& d, FilterTensorFormat dst_filter_format,          \
      typename TTypes<T, 4, int>::ConstTensor in,                        \
      typename TTypes<T, 4, int>::Tensor out);                           \
  extern template struct TransformFilter<GPUDevice, T, int, 4>;          \
  template <>                                                            \
  void PadInput<GPUDevice, T, int, 4>::operator()(                       \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,    \
      const std::array<int, 2>& padding_left,                            \
      const std::array<int, 2>& padding_right,                           \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format); \
  extern template struct PadInput<GPUDevice, T, int, 4>

// Registration of the GPU implementations.
#define REGISTER_FUSED_GPU_CONV2D(T)                                  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_FusedConv2D").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      FusedConv2DOp<GPUDevice, T>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_FUSED_IMPL_H_
