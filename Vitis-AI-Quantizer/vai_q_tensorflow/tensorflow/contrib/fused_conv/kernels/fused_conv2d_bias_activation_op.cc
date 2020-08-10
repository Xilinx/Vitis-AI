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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#define EIGEN_USE_THREADS

#include "tensorflow/contrib/fused_conv/kernels/fused_conv2d_bias_activation_op.h"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/use_cudnn.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif  // defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)

#if GOOGLE_CUDA
#include "google/protobuf/duration.pb.h"
#include "absl/time/time.h"
#include "third_party/gpus/cudnn/cudnn.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/logger.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/core/protobuf/conv_autotuning.pb.h"
#include "tensorflow/core/util/activation_mode.h"
#include "tensorflow/stream_executor/dnn.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

namespace {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct RawType {
  using type = T;
};

template <>
struct RawType<qint8> {
  using type = int8;
};

// Template struct to convert int8x4 to int32.
// (for NCHW_VECT_C with element type int8, we can consider it to be
// an NCHW layout with element type int32 for operations like padding).
template <typename T>
struct Int8x4ToInt32 {
  // By default, do not change T.
  using type = T;
};

template <>
struct Int8x4ToInt32<int8> {
  using type = int32;
};
}  // namespace

// WARNING: Packing specializations defined in eigen_spatial_convolutions.h do
// not support packing expressions of QInt8 type. However, default Eigen
// gebp_kernel for QInt8 is too slow to be considered useful for anything.
#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)

template <typename BiasType, typename ScaleType>
class LaunchFusedConv2DBiasActivationOp<CPUDevice, qint8, BiasType, ScaleType> {
  using T = qint8;         // conv_input and filter type
  using ComputeT = float;  // convert inputs to fp32 for tensor contraction
  using TempT = float;     // temporary accumulator type for tensor contraction

 public:
  void launch(OpKernelContext* ctx, bool cudnn_use_autotune,
              const Tensor& conv_input, ScaleType conv_input_scale,
              const Tensor& filter, int32 row_stride, int32 col_stride,
              const Eigen::PaddingType& padding, const Tensor& side_input,
              ScaleType side_input_scale, const Tensor& bias,
              ActivationMode activation_mode, TensorFormat data_format,
              FilterTensorFormat filter_format, Tensor* output) {
    static_assert(std::is_same<BiasType, ScaleType>::value,
                  "Scale and Bias must be of the same type.");

    // Output tensor has type T (QInt8), but we can only evaluate Int8 Tensor
    // contraction using 32-bit accumulation (QInt32).
    Tensor temp_output(DataTypeToEnum<TempT>::value, output->shape());

    constexpr int32 row_dilation = 1;
    constexpr int32 col_dilation = 1;

    auto& device = ctx->eigen_device<CPUDevice>();

    // CPU convolution works with input in NHWC and filter in HWIO data formats.
    // NOTE: This code is mostly shared with 'Conv2D' and 'FusedConv2D'.

    BiasActivationOutputKernel output_kernel(conv_input_scale, side_input,
                                             side_input_scale, bias,
                                             activation_mode, output);

    if (filter.dim_size(0) == 1 && filter.dim_size(1) == 1 && row_stride == 1 &&
        col_stride == 1) {
      int conv_width =  // Width for the convolution step.
          output->dim_size(0) * output->dim_size(1) * output->dim_size(2);

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);

      auto out = temp_output.shaped<TempT, 2>({conv_width, filter.dim_size(3)});
      auto in0 = conv_input.shaped<T, 2>({conv_width, filter.dim_size(2)});
      auto in1 = filter.shaped<T, 2>({filter.dim_size(2), filter.dim_size(3)});

      out.device(device) = in0.cast<ComputeT>().contract(
          in1.cast<ComputeT>(), dim_pair, output_kernel);

    } else if (filter.dim_size(0) == conv_input.dim_size(1) &&
               filter.dim_size(1) == conv_input.dim_size(2) &&
               row_dilation == 1 && col_dilation == 1 &&
               padding == Eigen::PaddingType::PADDING_VALID) {
      // If the input data and filter have the same height/width,
      // reduce the 2D convolution to matrix multiplication.
      const auto k =  // Length of reduction dimension.
          filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);

      auto out = temp_output.shaped<TempT, 2>(
          {conv_input.dim_size(0), filter.dim_size(3)});
      auto in0 = conv_input.shaped<T, 2>({conv_input.dim_size(0), k});
      auto in1 = filter.shaped<T, 2>({k, filter.dim_size(3)});

      out.device(device) = in0.cast<ComputeT>().contract(
          in1.cast<ComputeT>(), dim_pair, output_kernel);

    } else {
      auto out = temp_output.tensor<TempT, 4>();
      auto in0 = conv_input.tensor<T, 4>();
      auto in1 = filter.tensor<T, 4>();

      // Need to swap row/col when calling Eigen.
      out.device(device) = Eigen::SpatialConvolution(
          in0.cast<ComputeT>(), in1.cast<ComputeT>(), col_stride, row_stride,
          padding, col_dilation, row_dilation, output_kernel);
    }
  }

 private:
  // Contraction output mapper for temporary QInt32 tensor.
  using ContractionOutputMapper =
      Eigen::internal::blas_data_mapper<TempT, Eigen::Index, Eigen::ColMajor>;

  // This output kernel computes an expressions corresponding to cuDNN
  // implementation of INT8 cudnnConvolutionBiasActivationForward:
  // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#scaling-parameters__fig-conv-bias-activation-forward
  struct BiasActivationOutputKernel {
    static constexpr ScaleType kMaxRange = static_cast<ScaleType>(127.f);
    static constexpr ScaleType kMinRange = static_cast<ScaleType>(-128.f);

    explicit BiasActivationOutputKernel(ScaleType conv_input_scale,
                                        const Tensor& side_input,
                                        ScaleType side_input_scale,
                                        const Tensor& bias,
                                        ActivationMode activation_mode,
                                        Tensor* output)
        : activation_mode(activation_mode),
          conv_input_scale(conv_input_scale),
          bias_data(bias.flat<BiasType>().data()),
          side_input_data(side_input.flat<T>().data()),
          side_input_scale(side_input_scale),
          output_data(const_cast<T*>(output->flat<T>().data())) {}

    EIGEN_ALWAYS_INLINE void operator()(
        const ContractionOutputMapper& conv_output_mapper,
        const Eigen::TensorContractionParams& params, Eigen::Index i,
        Eigen::Index j, Eigen::Index num_rows, Eigen::Index num_cols) const {
      DCHECK(params.swapped_arguments);

      const auto stride = conv_output_mapper.stride();

      const BiasType* bias_base = bias_data + i;
      const T* side_input_base = side_input_data + i + j * stride;
      T* output_base = output_data + i + j * stride;

      for (int col = 0; col < num_cols; ++col) {
        // A column of an output tensor after QInt8xQInt8 -> QInt32 contraction.
        // This is a temporary tensor, that we will scale, add bias with
        // side_input, and quantize before writing to final output tensor.
        typename TTypes<TempT>::UnalignedTensor conv_output(
            &conv_output_mapper(0, col), num_rows);

        // A column of output quantized tensor corresponding to conv output row.
        typename TTypes<T>::UnalignedTensor output(output_base + col * stride,
                                                   num_rows);

        // Pointers to the input data accounting for the column offset.
        TempT* conv_output_ptr = conv_output.data();
        const T* side_input_ptr = side_input_base + col * stride;
        const BiasType* bias_ptr = bias_base;

        static_assert(
            std::is_same<TempT, ScaleType>::value,
            "Temporary contraction result type must match with scale type.");

        // (1) Scale and add bias.
        // NOTE(ezhulenev): We do not use Eigen expressions for this loop,
        // because it seems that packet FMA produces slightly different results,
        // and we are targeting close equality with Nvidia implementation.
        // We could use std::fmaf, but it can be ~50x slower, on machines
        // without fma instruction.
        for (int idx = 0; idx < num_rows; ++idx) {
          conv_output_ptr[idx] = static_cast<double>(conv_output_ptr[idx]) *
                                     static_cast<double>(conv_input_scale) +
                                 static_cast<double>(bias_ptr[idx]);
          if (side_input_scale != 0.0f) {
            conv_output_ptr[idx] = static_cast<double>(side_input_ptr[idx]) *
                                       static_cast<double>(side_input_scale) +
                                   static_cast<double>(conv_output_ptr[idx]);
          }
        }

        // (2) Round-up, clip and apply activation function.
        ScaleType lower_bound =
            (activation_mode == ActivationMode::NONE ? kMinRange : 0);
        output =
            conv_output
                // scalar_round_op_google uses HALF_TO_EVEN.
                .unaryExpr(Eigen::internal::scalar_round_op_google<float>())
                .clip(lower_bound, kMaxRange)
                .template cast<T>();
      }
    }

   private:
    ActivationMode activation_mode;
    ScaleType conv_input_scale;
    const BiasType* bias_data;
    const T* side_input_data;
    ScaleType side_input_scale;
    T* output_data;
  };
};
#endif  // defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)

// T is the element type of the conv_input, filter and side_input tensors.
// BiasType is the element type of the bias tensor, which can be different.
// ScaleType is the type used for conv_input_scale, side_input_scale.
template <typename Device, typename T, typename BiasType, typename ScaleType>
class FusedConv2DBiasActivationOp : public OpKernel {
 public:
  enum InputIndexes {
    kConvInput = 0,
    kFilter,
    kBias,
    kSideInput,
    kConvInputScale,
    kSideInputScale,
    kNumInputs
  };

  explicit FusedConv2DBiasActivationOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format_str, filter_format_str;
    CHECK_EQ(kNumInputs, context->num_inputs());
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(context, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context,
                   context->GetAttr("filter_format", &filter_format_str));
    OP_REQUIRES(context,
                FilterFormatFromString(filter_format_str, &filter_format_),
                errors::InvalidArgument("Invalid filter format"));

    std::vector<int32> strides;
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
    OP_REQUIRES(context, strides.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));

    stride_rows_ = GetTensorDim(strides, data_format_, 'H');
    stride_cols_ = GetTensorDim(strides, data_format_, 'W');
    OP_REQUIRES(
        context,
        (GetTensorDim(strides, data_format_, 'N') == 1 &&
         GetTensorDim(strides, data_format_, 'C') == 1),
        errors::Unimplemented("Convolutional strides are not supported in "
                              "the batch and depth dimensions."));

    std::vector<int32> dilations;
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations));
    OP_REQUIRES(context, dilations == std::vector<int32>({1, 1, 1, 1}),
                errors::InvalidArgument("Dilations must be all equal to 1."));

    constexpr bool is_cpu = std::is_same<Device, CPUDevice>::value;
    constexpr bool is_gpu = std::is_same<Device, GPUDevice>::value;
    OP_REQUIRES(context, is_cpu || is_gpu,
                errors::InvalidArgument("Unknown Device type."));

    constexpr bool is_qint8 = std::is_same<T, qint8>::value;

    if (is_qint8 && is_gpu) {
      // Assuming qint8 <--> NCHW_VECT_C, OIHW_VECT_I (int8x4) here.

      // Note: Only NCHW_VECT_C format is supported for int8 on GPU.
      // This is because it is expected to be the fastest, and our previous
      // tests found cudnn 6 does not fully support the other formats for int8
      // mode.
      OP_REQUIRES(
          context, data_format_ == FORMAT_NCHW_VECT_C,
          errors::InvalidArgument(
              "qint8 should be used with data_format NCHW_VECT_C on GPU."));
      OP_REQUIRES(
          context, filter_format_ == FORMAT_OIHW_VECT_I,
          errors::InvalidArgument(
              "qint8 should be used with filter_format OIHW_VECT_I on GPU."));

    } else if (is_qint8 && is_cpu) {
      // On CPU we implement convolution with Eigen Tensor contraction, it
      // requries NHWC and HWIO formats for input and kernel.

      OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                  errors::InvalidArgument(
                      "qint8 should be used with data_format NHWC on CPU."));
      OP_REQUIRES(context, filter_format_ == FORMAT_HWIO,
                  errors::InvalidArgument(
                      "qint8 should be used with filter_format HWIO on CPU."));
    }

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_type_));
    eigen_padding_type_ = BrainPadding2EigenPadding(padding_type_);
    string activation_mode_str;
    OP_REQUIRES_OK(context,
                   context->GetAttr("activation_mode", &activation_mode_str));
    OP_REQUIRES_OK(context, GetActivationModeFromString(activation_mode_str,
                                                        &activation_mode_));
    OP_REQUIRES(context,
                activation_mode_ == ActivationMode::RELU ||
                    activation_mode_ == ActivationMode::NONE,
                errors::InvalidArgument(
                    "Current implementation only supports RELU or NONE "
                    "as the activation function."));
    cudnn_use_autotune_ = CudnnUseAutotune();
  }

  Status CheckShape(const Tensor& tensor, const string& tensor_name) {
    const int num_dims = tensor.dims();
    for (int i = 0; i < num_dims; i++) {
      if (!FastBoundsCheck(tensor.dim_size(i),
                           std::numeric_limits<int32>::max())) {
        return errors::InvalidArgument(tensor_name, " dimension ", i,
                                       " too large");
      }
    }
    // If there is a 5th dimension it is the VECT_C or VECT_I dimension.
    if (num_dims == 5 && tensor.dim_size(4) != 4) {
      return errors::InvalidArgument("The last dimension of ", tensor_name,
                                     " must be of size 4 for qint8.");
    }
    return Status::OK();
  }

  void Compute(OpKernelContext* context) override {
    // The conv_input tensor is one of the following formats:
    // NHWC, NCHW, NCHW_VECT_C.
    const Tensor& conv_input = context->input(kConvInput);
    OP_REQUIRES_OK(context, CheckShape(conv_input, "conv_input"));

    // The filter tensor is one of the following formats:
    // HWIO, OIHW, OIHW_VECT_I.
    const Tensor& filter = context->input(kFilter);
    OP_REQUIRES_OK(context, CheckShape(filter, "filter"));

    // Input bias is a 1-D tensor, with size matching output depth.
    const Tensor& bias = context->input(kBias);
    OP_REQUIRES_OK(context, CheckShape(bias, "bias"));

    const Tensor& conv_input_scale_tensor = context->input(kConvInputScale);
    const Tensor& side_input_scale_tensor = context->input(kSideInputScale);

    auto conv_input_scale = *reinterpret_cast<const ScaleType*>(
        conv_input_scale_tensor.tensor_data().data());
    auto side_input_scale = *reinterpret_cast<const ScaleType*>(
        side_input_scale_tensor.tensor_data().data());

    // If side_input_scale != 0, then side_input is not ignored and
    // has the same type and dimensions as the output.
    const Tensor& side_input = context->input(kSideInput);
    if (side_input_scale != 0) {
      OP_REQUIRES_OK(context, CheckShape(side_input, "side_input"));
    }

    // TODO(pauldonnelly): Switch to a more efficient mechanism to access
    // dimension indexes and per-dimension attributes.
    const int32 filter_rows = GetFilterDim(filter, filter_format_, 'H');
    const int32 filter_cols = GetFilterDim(filter, filter_format_, 'W');
    const int32 output_depth = GetFilterDim(filter, filter_format_, 'O');

    const int32 batch_size = GetTensorDim(conv_input, data_format_, 'N');
    const int32 conv_input_rows = GetTensorDim(conv_input, data_format_, 'H');
    const int32 conv_input_cols = GetTensorDim(conv_input, data_format_, 'W');

    int64 output_rows = 0, output_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context, GetWindowedOutputSize(conv_input_rows, filter_rows,
                                                  stride_rows_, padding_type_,
                                                  &output_rows, &pad_rows));
    OP_REQUIRES_OK(context, GetWindowedOutputSize(conv_input_cols, filter_cols,
                                                  stride_cols_, padding_type_,
                                                  &output_cols, &pad_cols));
    // Initialize the output tensor shape according to data_format_
    TensorShape output_shape = ShapeFromFormat(
        data_format_, batch_size, output_rows, output_cols, output_depth);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    VLOG(2) << "FusedConv2DBiasActivation: conv_input_cols = "
            << conv_input_cols << ", conv_input_rows = " << conv_input_rows
            << ", filter_cols = " << filter_cols
            << ", filter_rows = " << filter_rows
            << ", stride_cols = " << stride_cols_
            << ", stride_rows = " << stride_rows_
            << ", output_depth = " << output_depth
            << ", output_cols = " << output_cols
            << ", output_rows = " << output_rows
            << ", output_shape.num_elements = " << output_shape.num_elements();

    // If there is nothing to compute, return.
    if (output_shape.num_elements() == 0) {
      return;
    }

    launcher_.launch(context, cudnn_use_autotune_, conv_input, conv_input_scale,
                     filter, stride_rows_, stride_cols_, eigen_padding_type_,
                     side_input, side_input_scale, bias, activation_mode_,
                     data_format_, filter_format_, output);
  }

 private:
  int32 stride_rows_, stride_cols_;
  Padding padding_type_;
  Eigen::PaddingType eigen_padding_type_;
  ActivationMode activation_mode_;
  TensorFormat data_format_;
  FilterTensorFormat filter_format_;
  LaunchFusedConv2DBiasActivationOp<Device, T, BiasType, ScaleType> launcher_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(FusedConv2DBiasActivationOp);
};

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
REGISTER_KERNEL_BUILDER(
    Name("FusedConv2DBiasActivation")
        .Device(DEVICE_CPU)
        .TypeConstraint<qint8>("T")
        .TypeConstraint<float>("Tbias"),
    FusedConv2DBiasActivationOp<CPUDevice, qint8, float, float>);
#endif  // defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)

#if GOOGLE_CUDA
namespace dnn = se::dnn;

// Several functions are copyed over from tensorflow/core/kernels/gpu_utils,
// since this file may be compiled down to a tf_custom_op_library .so file,
// which can't depend on basic dependencies like tensorflow/core:lib. Instead,
// the code has to depend on whatever is the same in libtensorflow_framework.so.
//
// In theory, we can lift the dependencies of gpu_utils by turning it into a
// template library that provides duck typing, but I think duplication is the
// lesser of two evils.
namespace internal {
namespace {

tensorflow::CudnnVersion GetCudnnVersion(se::StreamExecutor* stream_executor) {
  tensorflow::CudnnVersion cudnn_version;
  if (auto* dnn = stream_executor->AsDnn()) {
    se::port::StatusOr<se::dnn::VersionInfo> version_or = dnn->GetVersion();
    if (version_or.ok()) {
      const auto& version = version_or.ValueOrDie();
      cudnn_version.set_major(version.major_version());
      cudnn_version.set_minor(version.minor_version());
      cudnn_version.set_patch(version.patch());
    }
  }
  return cudnn_version;
}

// Converts an absl::Duration to a google::protobuf::Duration.
inline google::protobuf::Duration ToDurationProto(absl::Duration duration) {
  google::protobuf::Duration proto;
  proto.set_seconds(absl::IDivDuration(duration, absl::Seconds(1), &duration));
  proto.set_nanos(
      absl::IDivDuration(duration, absl::Nanoseconds(1), &duration));
  return proto;
}

// Converts a google::protobuf::Duration to an absl::Duration.
inline absl::Duration FromDurationProto(google::protobuf::Duration proto) {
  return absl::Seconds(proto.seconds()) + absl::Nanoseconds(proto.nanos());
}

tensorflow::ComputeCapability GetComputeCapability(
    se::StreamExecutor* stream_executor) {
  tensorflow::ComputeCapability cc;
  int cc_major, cc_minor;
  stream_executor->GetDeviceDescription().cuda_compute_capability(&cc_major,
                                                                  &cc_minor);
  cc.set_major(cc_major);
  cc.set_minor(cc_minor);
  return cc;
}

void LogFusedConvForwardAutotuneResults(
    se::dnn::DataType element_type, se::DeviceMemoryBase input_buffer,
    se::DeviceMemoryBase filter_buffer, se::DeviceMemoryBase output_buffer,
    se::DeviceMemoryBase bias_buffer, se::DeviceMemoryBase side_input_buffer,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc, double conv_scale,
    double side_value_scale, se::dnn::ActivationMode activation_mode,
    se::StreamExecutor* stream_exec, absl::Span<const AutotuneResult> results) {
  AutotuningLog log;
  {
    ConvolutionProto instr;
    instr.set_kind(se::dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION);
    *instr.mutable_input() = input_desc.ToProto(element_type);
    *instr.mutable_filter() = filter_desc.ToProto(element_type);
    *instr.mutable_output() = output_desc.ToProto(element_type);
    *instr.mutable_conv_desc() = conv_desc.ToProto();
    instr.set_conv_scale(conv_scale);
    instr.set_side_value_scale(side_value_scale);
    instr.set_activation(activation_mode);
    instr.set_input_address(reinterpret_cast<uint64>(input_buffer.opaque()));
    instr.set_filter_address(reinterpret_cast<uint64>(filter_buffer.opaque()));
    instr.set_output_address(reinterpret_cast<uint64>(output_buffer.opaque()));
    instr.set_bias_address(reinterpret_cast<uint64>(bias_buffer.opaque()));
    instr.set_side_input_address(
        reinterpret_cast<uint64>(side_input_buffer.opaque()));
    log.mutable_instr()->PackFrom(std::move(instr));
  }
  *log.mutable_cudnn_version() = GetCudnnVersion(stream_exec);
  *log.mutable_compute_capability() = GetComputeCapability(stream_exec);
  log.set_device_pci_bus_id(stream_exec->GetDeviceDescription().pci_bus_id());
  {
    string blas_version;
    if (auto* blas = stream_exec->AsBlas()) {
      if (blas->GetVersion(&blas_version).ok()) {
        log.set_blas_version(blas_version);
      }
    }
  }
  for (const auto& result : results) {
    *log.add_results() = result;
  }
  Logger::GetSingleton()->LogProto(log);
}

Status BestCudnnConvAlgorithm(absl::Span<const AutotuneResult> results,
                              se::dnn::AlgorithmConfig* algo) {
  const AutotuneResult* best_result = std::min_element(
      results.begin(), results.end(),
      [](const AutotuneResult& lhs, const AutotuneResult& rhs) {
        return internal::FromDurationProto(lhs.run_time()) <
               internal::FromDurationProto(rhs.run_time());
      });

  const AutotuneResult* best_result_no_scratch = std::min_element(
      results.begin(), results.end(),
      [](const AutotuneResult& lhs, const AutotuneResult& rhs) {
        return std::make_tuple(lhs.scratch_bytes(),
                               internal::FromDurationProto(lhs.run_time())) <
               std::make_tuple(rhs.scratch_bytes(),
                               internal::FromDurationProto(rhs.run_time()));
      });

  if (best_result == results.end()) {
    return errors::NotFound("No algorithm worked!");
  }
  algo->set_algorithm({best_result->conv().algorithm(),
                       best_result->conv().tensor_ops_enabled()});
  if (best_result_no_scratch != results.end() &&
      best_result_no_scratch->scratch_bytes() == 0) {
    algo->set_algorithm_no_scratch(
        {best_result_no_scratch->conv().algorithm(),
         best_result_no_scratch->conv().tensor_ops_enabled()});
  }
  return Status::OK();
}

}  // namespace
}  // namespace internal

// A dummy type to group forward convolution autotune results together.
struct ConvBiasActivationAutoTuneGroup {
  static string name() { return "ConvBiasActivation"; }
};
typedef AutoTuneSingleton<ConvBiasActivationAutoTuneGroup, FusedConvParameters,
                          dnn::AlgorithmConfig>
    AutoTuneConvBiasActivation;

// Allocates 'transformed_tensor' and transforms 'nhwc_tensor' into it
// using the specified 'batch_size', 'rows', 'cols', and 'depth' dimensions.
template <typename T, size_t NDIMS>
Status TransformNHWCToNCHW(OpKernelContext* ctx, const Tensor& nhwc_tensor,
                           int batch_size, int rows, int cols, int depth,
                           Tensor* transformed_tensor, const Tensor** result) {
  TensorShape nchw_shape =
      ShapeFromFormat(FORMAT_NCHW, batch_size, rows, cols, depth);
  if (depth > 1) {
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<T>::value, nchw_shape,
                                          transformed_tensor));
    functor::NHWCToNCHW<GPUDevice, T, NDIMS>()(
        ctx->eigen_device<GPUDevice>(), nhwc_tensor.tensor<T, NDIMS>(),
        transformed_tensor->tensor<T, NDIMS>());
  } else {
    // If depth <= 1, then just reshape.
    CHECK(transformed_tensor->CopyFrom(nhwc_tensor, nchw_shape));
  }
  *result = transformed_tensor;
  return Status::OK();
}

// Adjusts padding so cudnn supports it. Sets `adjusted_padding` to be the
// adjusted padding, and `extra_padding_before` and `extra_padding_after` to be
// the extra padding that FusedConv needs to apply before calling cudnn.
void AdjustPaddingForCudnn(int padding, bool is_int8x4, int filter_size,
                           int* adjusted_padding, int* extra_padding_before,
                           int* extra_padding_after) {
#if CUDNN_VERSION < 7000
  if (is_int8x4 && filter_size >= 6) {
    // TODO(b/70795525): Remove after NVIDIA fixes this bug with int8 fused
    // convolution. I don't know cuDNN7 still has the bug, so enable this
    // workaround for cuDNN6 or older.
    *adjusted_padding = 0;
    *extra_padding_before = padding / 2;
    *extra_padding_after = padding - *extra_padding_before;
    return;
  }
#endif
  *adjusted_padding = padding / 2 * 2;
  *extra_padding_before = 0;
  *extra_padding_after = padding % 2;
}

template <typename T, typename BiasType, typename ScaleType>
void LaunchFusedConv2DBiasActivationOp<GPUDevice, T, BiasType, ScaleType>::
    launch(OpKernelContext* ctx, bool cudnn_use_autotune,
           const Tensor& conv_input_param, ScaleType conv_input_scale,
           const Tensor& filter_param, int32 row_stride, int32 col_stride,
           const Eigen::PaddingType& padding, const Tensor& side_input_param,
           ScaleType side_input_scale, const Tensor& bias,
           ActivationMode activation_mode, TensorFormat data_format,
           FilterTensorFormat filter_format, Tensor* output_param) {
  auto* stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

  // TODO(yangzihao): refactor all the complicated/duplicated code in regular
  // conv ops to a shared conv utility.

  // Assuming qint8 <--> NCHW_VECT_C, OIHW_VECT_I (int8x4) here.
  constexpr bool is_int8x4 = std::is_same<T, qint8>::value;
  constexpr int rank = is_int8x4 ? 5 : 4;
  constexpr int vect = is_int8x4 ? 4 : 1;

  if (is_int8x4) {
    int cc_major, cc_minor;
    stream->parent()->GetDeviceDescription().cuda_compute_capability(&cc_major,
                                                                     &cc_minor);
    OP_REQUIRES(
        ctx, ((cc_major == 6 && cc_minor >= 1) || cc_major > 6),
        errors::Unimplemented(
            "FusedConv2DBiasActivation for int8 is only supported on GPUs with "
            "compute capability 6.1 or later."));
  }

  const int batch_size = GetTensorDim(conv_input_param, data_format, 'N');
  int conv_input_rows = GetTensorDim(conv_input_param, data_format, 'H');
  int conv_input_cols = GetTensorDim(conv_input_param, data_format, 'W');

  const int conv_input_depth =
      GetTensorDim(conv_input_param, data_format, 'C') * vect;
  const int output_rows = GetTensorDim(*output_param, data_format, 'H');
  const int output_cols = GetTensorDim(*output_param, data_format, 'W');
  const int output_depth = GetFilterDim(filter_param, filter_format, 'O');
  const int filter_rows = GetFilterDim(filter_param, filter_format, 'H');
  const int filter_cols = GetFilterDim(filter_param, filter_format, 'W');
  int padding_rows = 0;
  int padding_cols = 0;
  const Tensor* conv_input = &conv_input_param;

  Tensor maybe_padded_conv_input;
  if (padding == Eigen::PADDING_SAME) {
    // Total padding on rows and cols is
    // Pr = (R' - 1) * S + Kr - R
    // Pc = (C' - 1) * S + Kc - C
    // where (R', C') are output dimensions, (R, C) are input dimensions, S
    // is stride, (Kr, Kc) are filter dimensions.
    // We pad Pr/2 on the left and Pr - Pr/2 on the right, Pc/2 on the top
    // and Pc - Pc/2 on the bottom.  When Pr or Pc is odd, this means
    // we pad more on the right and bottom than on the top and left.
    padding_rows = std::max<int>(
        0, (output_rows - 1) * row_stride + filter_rows - conv_input_rows);
    padding_cols = std::max<int>(
        0, (output_cols - 1) * col_stride + filter_cols - conv_input_cols);
    int extra_top_padding = 0;
    int extra_bottom_padding = 0;
    int extra_left_padding = 0;
    int extra_right_padding = 0;
    AdjustPaddingForCudnn(padding_rows, is_int8x4, filter_rows, &padding_rows,
                          &extra_top_padding, &extra_bottom_padding);
    AdjustPaddingForCudnn(padding_cols, is_int8x4, filter_cols, &padding_cols,
                          &extra_left_padding, &extra_right_padding);
    if (extra_top_padding != 0 || extra_bottom_padding != 0 ||
        extra_left_padding != 0 || extra_right_padding != 0) {
      const int new_conv_input_rows =
          conv_input_rows + extra_top_padding + extra_bottom_padding;
      const int new_conv_input_cols =
          conv_input_cols + extra_left_padding + extra_right_padding;

      using VectT = typename Int8x4ToInt32<typename RawType<T>::type>::type;
      auto pad_data_format = is_int8x4 ? FORMAT_NCHW : data_format;

      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(
                   DataTypeToEnum<T>::value,
                   ShapeFromFormat(data_format, batch_size, new_conv_input_rows,
                                   new_conv_input_cols, conv_input_depth),
                   &maybe_padded_conv_input));

      auto conv_input_eigen_tensor =
          To32Bit(conv_input_param.reinterpret_last_dimension<VectT, 4>());
      auto padded_conv_input_eigen_tensor = To32Bit(
          maybe_padded_conv_input.reinterpret_last_dimension<VectT, 4>());

      functor::PadInput<GPUDevice, VectT, int, 4>()(
          ctx->eigen_device<GPUDevice>(), conv_input_eigen_tensor,
          {{extra_top_padding, extra_left_padding}},
          {{extra_bottom_padding, extra_right_padding}},
          padded_conv_input_eigen_tensor, pad_data_format);

      conv_input = &maybe_padded_conv_input;
      conv_input_rows = new_conv_input_rows;
      conv_input_cols = new_conv_input_cols;
    }
  }

  Tensor maybe_transformed_conv_input, maybe_transformed_side_input;
  Tensor maybe_transformed_output;
  const Tensor* side_input = &side_input_param;
  Tensor* output = output_param;

  // NOTE: Here and elsewhere, checking 'is_int8x4' may look unnecessary
  // and inefficient, but it is actually both a time and code size optimization,
  // since 'is_int8x4' is a constexpr determined by the template parameter.
  if (!is_int8x4 && data_format == FORMAT_NHWC) {
    OP_REQUIRES_OK(ctx, (TransformNHWCToNCHW<T, rank>(
                            ctx, *conv_input, batch_size, conv_input_rows,
                            conv_input_cols, conv_input_depth,
                            &maybe_transformed_conv_input, &conv_input)));
    if (side_input_scale != 0) {
      OP_REQUIRES_OK(
          ctx, (TransformNHWCToNCHW<T, rank>(
                   ctx, side_input_param, batch_size, output_rows, output_cols,
                   output_depth, &maybe_transformed_side_input, &side_input)));
    }
    if (output_depth > 1) {
      // Allocate a tensor for the NCHW output of the kernel and point output
      // to it. Afterwards, we will transform it to NHWC while copying back to
      // 'output_param'.
      TensorShape nchw_shape = ShapeFromFormat(
          FORMAT_NCHW, batch_size, output_rows, output_cols, output_depth);
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<T>::value, nchw_shape,
                                        &maybe_transformed_output));
      output = &maybe_transformed_output;
    }
  }

  constexpr auto data_layout = is_int8x4 ? dnn::DataLayout::kBatchDepthYX4
                                         : dnn::DataLayout::kBatchDepthYX;
  constexpr auto filter_layout = is_int8x4 ? dnn::FilterLayout::kOutputInputYX4
                                           : dnn::FilterLayout::kOutputInputYX;
  constexpr auto compute_data_format =
      is_int8x4 ? FORMAT_NCHW_VECT_C : FORMAT_NCHW;

  dnn::BatchDescriptor conv_input_desc;
  conv_input_desc.set_count(batch_size)
      .set_feature_map_count(conv_input_depth)
      .set_height(conv_input_rows)
      .set_width(conv_input_cols)
      .set_layout(data_layout);
  dnn::FilterDescriptor filter_desc;
  filter_desc.set_input_filter_height(filter_rows)
      .set_input_filter_width(filter_cols)
      .set_input_feature_map_count(conv_input_depth)
      .set_output_feature_map_count(output_depth)
      .set_layout(filter_layout);
  dnn::BatchDescriptor side_input_desc;
  side_input_desc.set_count(batch_size)
      .set_height(output_rows)
      .set_width(output_cols)
      .set_feature_map_count(output_depth)
      .set_layout(data_layout);
  dnn::BatchDescriptor bias_desc;
  bias_desc.set_count(1)
      .set_height(1)
      .set_width(1)
      .set_feature_map_count(output_depth)
      .set_layout(dnn::DataLayout::kBatchDepthYX);
  dnn::BatchDescriptor output_desc;
  output_desc.set_count(batch_size)
      .set_height(output_rows)
      .set_width(output_cols)
      .set_feature_map_count(output_depth)
      .set_layout(data_layout);
  dnn::ConvolutionDescriptor conv_desc;
  CHECK_EQ(0, padding_rows % 2);
  CHECK_EQ(0, padding_cols % 2);
  conv_desc.set_vertical_filter_stride(row_stride)
      .set_horizontal_filter_stride(col_stride)
      .set_zero_padding_height(padding_rows / 2)
      .set_zero_padding_width(padding_cols / 2);

  Tensor maybe_transformed_filter;
  const Tensor* filter = &filter_param;
  // For qint8, we have already checked filter is OIHW_VECT_I in the
  // constructor, but we need to test for is_int8x4 so the if block doesn't
  // generate code for qint8.
  if (!is_int8x4 && filter_format == FORMAT_HWIO) {
    // Shuffle filter tensor from HWIO to OIHW:
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DataTypeToEnum<T>::value,
                            ShapeFromFilterFormat(
                                FORMAT_OIHW, filter_param.shape(), FORMAT_HWIO),
                            &maybe_transformed_filter));
    functor::TransformFilter<GPUDevice, T, int, 4>()(
        ctx->eigen_device<GPUDevice>(), FORMAT_OIHW,
        To32Bit(filter_param.tensor<T, 4>()),
        To32Bit(maybe_transformed_filter.tensor<T, 4>()));
    filter = &maybe_transformed_filter;
  }

  auto conv_input_ptr =
      AsDeviceMemory(reinterpret_cast<const typename RawType<T>::type*>(
                         conv_input->template flat<T>().data()),
                     conv_input->template flat<T>().size());
  auto filter_ptr =
      AsDeviceMemory(reinterpret_cast<const typename RawType<T>::type*>(
                         filter->template flat<T>().data()),
                     filter->template flat<T>().size());
  auto side_input_ptr =
      AsDeviceMemory(reinterpret_cast<const typename RawType<T>::type*>(
                         side_input->template flat<T>().data()),
                     side_input->template flat<T>().size());
  auto output_ptr =
      AsDeviceMemory(reinterpret_cast<const typename RawType<T>::type*>(
                         output->template flat<T>().data()),
                     output->template flat<T>().size());
  auto bias_ptr = AsDeviceMemory(bias.template flat<BiasType>().data(),
                                 bias.template flat<BiasType>().size());

  static int64 ConvolveScratchSize = GetDnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );

  int device_id = stream->parent()->device_ordinal();
  FusedConvParameters fused_conv_parameters = {
      batch_size,
      conv_input_depth,
      {{conv_input_rows, conv_input_cols}},
      compute_data_format,
      output_depth,
      {{filter_rows, filter_cols}},
      // TODO(yangzihao): Add support for arbitrary dilations for fused conv.
      {{1, 1}},  // dilation_rows, dilation_cols
      {{row_stride, col_stride}},
      {{padding_rows, padding_cols}},
      conv_input->dtype(),
      device_id,
      (side_input_scale != 0),
      activation_mode,
  };

  dnn::ActivationMode dnn_activation_mode;
  switch (activation_mode) {
    case ActivationMode::NONE:
      dnn_activation_mode = dnn::ActivationMode::kNone;
      break;
    case ActivationMode::RELU:
      dnn_activation_mode = dnn::ActivationMode::kRelu;
      break;
    default:
      LOG(FATAL) << "Activation mode " << activation_mode << " not supported";
  }

  dnn::AlgorithmConfig algorithm_config;
  if (cudnn_use_autotune && !AutoTuneConvBiasActivation::GetInstance()->Find(
                                fused_conv_parameters, &algorithm_config)) {
    std::vector<dnn::AlgorithmDesc> algorithms;
    CHECK(stream->parent()->GetConvolveAlgorithms(
        fused_conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(
            stream->parent()),
        &algorithms));
    if (activation_mode == ActivationMode::NONE) {
      // Only CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM is supported for
      // identity activation, other algs seem to quietly do Relu.
      // See
      // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBiasActivationForward
      algorithms.erase(
          std::remove_if(
              algorithms.begin(), algorithms.end(),
              [](dnn::AlgorithmDesc alg) {
                return alg.algo_id() !=
                       CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
              }),
          algorithms.end());
    }
    std::vector<tensorflow::AutotuneResult> results;
    for (auto profile_algorithm : algorithms) {
      // TODO(zhengxq): profile each algorithm multiple times to better
      // accuracy.
      DnnScratchAllocator scratch_allocator(ConvolveScratchSize, ctx);
      dnn::ProfileResult profile_result;
      bool cudnn_launch_status =
          stream
              ->ThenFusedConvolveWithAlgorithm(
                  conv_input_desc, conv_input_ptr, conv_input_scale,
                  filter_desc, filter_ptr, conv_desc, side_input_ptr,
                  side_input_scale, bias_desc, bias_ptr, dnn_activation_mode,
                  output_desc, &output_ptr, &scratch_allocator,
                  dnn::AlgorithmConfig(profile_algorithm), &profile_result)
              .ok();
      if (cudnn_launch_status && profile_result.is_valid()) {
        results.emplace_back();
        auto& result = results.back();
        result.mutable_conv()->set_algorithm(profile_algorithm.algo_id());
        result.mutable_conv()->set_tensor_ops_enabled(
            profile_algorithm.tensor_ops_enabled());
        result.set_scratch_bytes(scratch_allocator.TotalByteSize());
        *result.mutable_run_time() = internal::ToDurationProto(
            absl::Milliseconds(profile_result.elapsed_time_in_ms()));
      }
    }
    internal::LogFusedConvForwardAutotuneResults(
        se::dnn::ToDataType<typename RawType<T>::type>::value, conv_input_ptr,
        filter_ptr, output_ptr, bias_ptr, side_input_ptr, conv_input_desc,
        filter_desc, output_desc, conv_desc, conv_input_scale, side_input_scale,
        dnn_activation_mode, stream->parent(), results);
    OP_REQUIRES_OK(
        ctx, internal::BestCudnnConvAlgorithm(results, &algorithm_config));
    AutoTuneConvBiasActivation::GetInstance()->Insert(fused_conv_parameters,
                                                      algorithm_config);
  }

  DnnScratchAllocator scratch_allocator(ConvolveScratchSize, ctx);
  bool cudnn_launch_status =
      stream
          ->ThenFusedConvolveWithAlgorithm(
              conv_input_desc, conv_input_ptr, conv_input_scale, filter_desc,
              filter_ptr, conv_desc, side_input_ptr, side_input_scale,
              bias_desc, bias_ptr, dnn_activation_mode, output_desc,
              &output_ptr, &scratch_allocator, algorithm_config,
              /*output_profile_result=*/nullptr)
          .ok();

  if (!cudnn_launch_status) {
    ctx->SetStatus(errors::Internal("cuDNN launch failure : conv_input shape(",
                                    conv_input->shape().DebugString(),
                                    ") filter shape(",
                                    filter->shape().DebugString(), ")"));
  }

  // Convert the output tensor back from NCHW to NHWC if necessary.
  if (!is_int8x4 && (data_format == FORMAT_NHWC) && (output_depth > 1)) {
    functor::NCHWToNHWC<GPUDevice, T, 4>()(
        ctx->eigen_device<GPUDevice>(),
        const_cast<const Tensor*>(output)->tensor<T, 4>(),
        output_param->tensor<T, 4>());
  }
}

// Forward declarations of the functor specializations for GPU used above.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                              \
  template <>                                                            \
  void PadInput<GPUDevice, T, int, 4>::operator()(                       \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,    \
      const std::array<int, 2>& padding_left,                            \
      const std::array<int, 2>& padding_right,                           \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format); \
  extern template struct PadInput<GPUDevice, T, int, 4>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(int32);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.

REGISTER_KERNEL_BUILDER(
    Name("FusedConv2DBiasActivation")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T")
        .TypeConstraint<float>("Tbias")
        .HostMemory("conv_input_scale")
        .HostMemory("side_input_scale"),
    FusedConv2DBiasActivationOp<GPUDevice, float, float, float>);

REGISTER_KERNEL_BUILDER(
    Name("FusedConv2DBiasActivation")
        .Device(DEVICE_GPU)
        .TypeConstraint<qint8>("T")
        .TypeConstraint<float>("Tbias")
        .HostMemory("conv_input_scale")
        .HostMemory("side_input_scale"),
    FusedConv2DBiasActivationOp<GPUDevice, qint8, float, float>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
