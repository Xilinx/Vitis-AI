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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_OPTIMIZED_OPS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_OPTIMIZED_OPS_H_

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

#include "tensorflow/lite/kernels/internal/compatibility.h"

#if defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "fixedpoint/fixedpoint.h"
#include "profiling/instrumentation.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/im2col_utils.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/round.h"
#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

// Unoptimized reference ops:
using reference_ops::ArgMax;
using reference_ops::ArgMinMax;
using reference_ops::Broadcast4DSlowGreater;
using reference_ops::Broadcast4DSlowGreaterEqual;
using reference_ops::Broadcast4DSlowGreaterEqualWithScaling;
using reference_ops::Broadcast4DSlowGreaterWithScaling;
using reference_ops::Broadcast4DSlowLess;
using reference_ops::Broadcast4DSlowLessEqual;
using reference_ops::Broadcast4DSlowLessEqualWithScaling;
using reference_ops::Broadcast4DSlowLessWithScaling;
using reference_ops::BroadcastAdd4DSlow;
using reference_ops::BroadcastMul4DSlow;
using reference_ops::BroadcastSub4DSlow;
using reference_ops::Concatenation;
using reference_ops::ConcatenationWithScaling;
using reference_ops::DepthConcatenation;
using reference_ops::Div;
using reference_ops::Elu;
using reference_ops::FakeQuant;
using reference_ops::Fill;
using reference_ops::Gather;
using reference_ops::Greater;
using reference_ops::GreaterEqual;
using reference_ops::GreaterEqualWithScaling;
using reference_ops::GreaterWithScaling;
using reference_ops::LeakyRelu;
using reference_ops::Less;
using reference_ops::LessEqual;
using reference_ops::LessEqualWithScaling;
using reference_ops::LessWithScaling;
using reference_ops::Mean;
using reference_ops::ProcessBroadcastShapes;
using reference_ops::RankOneSelect;
using reference_ops::Relu1;
using reference_ops::Relu6;
using reference_ops::ReluX;
using reference_ops::Round;
using reference_ops::Select;
using reference_ops::SpaceToBatchND;
using reference_ops::Split;
using reference_ops::StridedSlice;
using reference_ops::Sub16;
using reference_ops::Transpose;

// TODO(b/80247582) Remove this constant.
// This will be phased out as the shifts are revised with more thought. Use of a
// constant enables us to track progress on this work.
//
// Used to convert from old-style shifts (right) to new-style (left).
static constexpr int kReverseShift = -1;

// Make a local VectorMap typedef allowing to map a float array
// as a Eigen vector expression. The std::conditional here is to
// construct the suitable Eigen type for the constness of the
// data. Indeed, for const data, we need to produce
//    Eigen::Map<const Eigen::Matrix<float, ...>>
// and not the more straightforward
//    Eigen::Map<Eigen::Matrix<const float, ...>>
template <typename Scalar>
using VectorMap = typename std::conditional<
    std::is_const<Scalar>::value,
    Eigen::Map<const Eigen::Matrix<typename std::remove_const<Scalar>::type,
                                   Eigen::Dynamic, 1>>,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>>::type;

template <typename Scalar>
VectorMap<Scalar> MapAsVector(Scalar* data, const RuntimeShape& shape) {
  const int size = shape.FlatSize();
  return VectorMap<Scalar>(data, size, 1);
}

// Make a local VectorMap typedef allowing to map a float array
// as a Eigen matrix expression. The same explanation as for VectorMap
// above also applies here.
template <typename Scalar>
using MatrixMap = typename std::conditional<
    std::is_const<Scalar>::value,
    Eigen::Map<const Eigen::Matrix<typename std::remove_const<Scalar>::type,
                                   Eigen::Dynamic, Eigen::Dynamic>>,
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>>::type;

template <typename Scalar>
MatrixMap<Scalar> MapAsMatrixWithLastDimAsRows(Scalar* data,
                                               const RuntimeShape& shape) {
  const int dims_count = shape.DimensionsCount();
  const int rows = shape.Dims(dims_count - 1);
  const int cols = FlatSizeSkipDim(shape, dims_count - 1);
  return MatrixMap<Scalar>(data, rows, cols);
}

template <typename Scalar>
MatrixMap<Scalar> MapAsMatrixWithFirstDimAsCols(Scalar* data,
                                                const RuntimeShape& shape) {
  const int cols = shape.Dims(0);
  const int rows = FlatSizeSkipDim(shape, 0);
  return MatrixMap<Scalar>(data, rows, cols);
}

template <typename Scalar>
using ArrayMap = typename std::conditional<
    std::is_const<Scalar>::value,
    Eigen::Map<const Eigen::Array<typename std::remove_const<Scalar>::type,
                                  Eigen::Dynamic, Eigen::Dynamic>>,
    Eigen::Map<Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic>>>::type;

template <typename Scalar>
ArrayMap<Scalar> MapAsArrayWithLastDimAsRows(Scalar* data,
                                             const RuntimeShape& shape) {
  const int dims_count = shape.DimensionsCount();
  const int rows = shape.Dims(dims_count - 1);
  const int cols = FlatSizeSkipDim(shape, dims_count - 1);
  return ArrayMap<Scalar>(data, rows, cols);
}

// Copied from tensorflow/core/framework/tensor_types.h
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
struct TTypes {
  // Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Flat;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>>
      UnalignedConstMatrix;
};

// TODO(b/62193649): this function is only needed as long
// as we have the --variable_batch hack.
template <typename Scalar>
MatrixMap<Scalar> MapAsMatrixWithGivenNumberOfRows(Scalar* data,
                                                   const RuntimeShape& shape,
                                                   int rows) {
  const int flatsize = shape.FlatSize();
  TFLITE_DCHECK_EQ(flatsize % rows, 0);
  const int cols = flatsize / rows;
  return MatrixMap<Scalar>(data, rows, cols);
}

inline void AddBiasAndEvalActivationFunction(float output_activation_min,
                                             float output_activation_max,
                                             const RuntimeShape& bias_shape,
                                             const float* bias_data,
                                             const RuntimeShape& array_shape,
                                             float* array_data) {
  BiasAndClamp(output_activation_min, output_activation_max,
               bias_shape.FlatSize(), bias_data, array_shape.FlatSize(),
               array_data);
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& weights_shape,
    const float* weights_data, const RuntimeShape& bias_shape,
    const float* optional_bias_data, const RuntimeShape& output_shape,
    float* output_data, CpuBackendContext* cpu_backend_context) {
  gemmlowp::ScopedProfilingLabel label("FullyConnected");
  const int dims_count = weights_shape.DimensionsCount();
  const int input_rows = weights_shape.Dims(dims_count - 1);
  cpu_backend_gemm::MatrixParams<float> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = input_rows;
  rhs_params.cols = input_shape.FlatSize() / input_rows;
  TFLITE_DCHECK_EQ(input_shape.FlatSize(), rhs_params.rows * rhs_params.cols);
  cpu_backend_gemm::MatrixParams<float> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.cols = weights_shape.Dims(dims_count - 1);
  lhs_params.rows = FlatSizeSkipDim(weights_shape, dims_count - 1);
  cpu_backend_gemm::MatrixParams<float> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = output_shape.Dims(output_shape.DimensionsCount() - 1);
  dst_params.cols =
      FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);
  cpu_backend_gemm::GemmParams<float, float> gemm_params;
  gemm_params.bias = optional_bias_data;
  gemm_params.clamp_min = params.float_activation_min;
  gemm_params.clamp_max = params.float_activation_max;
  cpu_backend_gemm::Gemm(lhs_params, weights_data, rhs_params, input_data,
                         dst_params, output_data, gemm_params,
                         cpu_backend_context);
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data, CpuBackendContext* cpu_backend_context) {
  gemmlowp::ScopedProfilingLabel label("FullyConnected/8bit");
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int filter_rows = filter_shape.Dims(filter_dim_count - 2);
  const int filter_cols = filter_shape.Dims(filter_dim_count - 1);
  TFLITE_DCHECK_EQ(filter_shape.FlatSize(), filter_rows * filter_cols);
  const int output_rows = output_shape.Dims(output_dim_count - 1);
  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_rows);
  }

  cpu_backend_gemm::MatrixParams<uint8> lhs_params;
  lhs_params.rows = filter_rows;
  lhs_params.cols = filter_cols;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.zero_point = -filter_offset;
  cpu_backend_gemm::MatrixParams<uint8> rhs_params;
  rhs_params.rows = filter_cols;
  rhs_params.cols = batches;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.zero_point = -input_offset;
  cpu_backend_gemm::MatrixParams<uint8> dst_params;
  dst_params.rows = filter_rows;
  dst_params.cols = batches;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.zero_point = output_offset;
  cpu_backend_gemm::GemmParams<int32, uint8> gemm_params;
  gemm_params.bias = bias_data;
  gemm_params.clamp_min = output_activation_min;
  gemm_params.clamp_max = output_activation_max;
  gemm_params.multiplier_fixedpoint = output_multiplier;
  gemm_params.multiplier_exponent = output_shift;
  cpu_backend_gemm::Gemm(lhs_params, filter_data, rhs_params, input_data,
                         dst_params, output_data, gemm_params,
                         cpu_backend_context);
}

inline void FullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data_int32, const RuntimeShape& output_shape,
    int16* output_data, CpuBackendContext* cpu_backend_context) {
  gemmlowp::ScopedProfilingLabel label("FullyConnected/Uint8Int16");
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(output_offset, 0);
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);

  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = MatchingDim(filter_shape, filter_dim_count - 2,
                                       output_shape, output_dim_count - 1);
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

  cpu_backend_gemm::MatrixParams<uint8> lhs_params;
  lhs_params.rows = output_depth;
  lhs_params.cols = accum_depth;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.zero_point = -filter_offset;
  cpu_backend_gemm::MatrixParams<uint8> rhs_params;
  rhs_params.rows = accum_depth;
  rhs_params.cols = batches;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.zero_point = -input_offset;
  cpu_backend_gemm::MatrixParams<int16> dst_params;
  dst_params.rows = output_depth;
  dst_params.cols = batches;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.zero_point = 0;
  cpu_backend_gemm::GemmParams<int32, int16> gemm_params;
  gemm_params.bias = bias_data_int32;
  gemm_params.clamp_min = output_activation_min;
  gemm_params.clamp_max = output_activation_max;
  gemm_params.multiplier_fixedpoint = output_multiplier;
  gemm_params.multiplier_exponent = output_shift;
  cpu_backend_gemm::Gemm(lhs_params, filter_data, rhs_params, input_data,
                         dst_params, output_data, gemm_params,
                         cpu_backend_context);
}

// Internal function doing the actual arithmetic work for
// ShuffledFullyConnected.
// May be called either directly by it (single-threaded case) or may be used
// as the 'task' for worker threads to run (multi-threaded case, see
// ShuffledFullyConnectedWorkerTask below).
inline void ShuffledFullyConnectedWorkerImpl(
    const uint8* shuffled_input_workspace_data,
    const int8* shuffled_weights_data, int batches, int output_depth,
    int output_stride, int accum_depth, const int32* bias_data,
    int32 output_multiplier, int output_shift, int16* output_data) {
#if defined USE_NEON
  const int8* shuffled_weights_ptr = shuffled_weights_data;
  if (batches == 1) {
    const int right_shift = output_shift > 0 ? 0 : -output_shift;
    const int left_shift = output_shift > 0 ? output_shift : 0;
    for (int c = 0; c < output_depth; c += 4) {
      // Accumulation loop.
      int32x4_t row_accum0 = vdupq_n_s32(0);
      int32x4_t row_accum1 = vdupq_n_s32(0);
      int32x4_t row_accum2 = vdupq_n_s32(0);
      int32x4_t row_accum3 = vdupq_n_s32(0);
      for (int d = 0; d < accum_depth; d += 16) {
        int8x16_t weights0 = vld1q_s8(shuffled_weights_ptr + 0);
        int8x16_t weights1 = vld1q_s8(shuffled_weights_ptr + 16);
        int8x16_t weights2 = vld1q_s8(shuffled_weights_ptr + 32);
        int8x16_t weights3 = vld1q_s8(shuffled_weights_ptr + 48);
        shuffled_weights_ptr += 64;
        int8x16_t input =
            vreinterpretq_s8_u8(vld1q_u8(shuffled_input_workspace_data + d));
        int16x8_t local_accum0 =
            vmull_s8(vget_low_s8(weights0), vget_low_s8(input));
        int16x8_t local_accum1 =
            vmull_s8(vget_low_s8(weights1), vget_low_s8(input));
        int16x8_t local_accum2 =
            vmull_s8(vget_low_s8(weights2), vget_low_s8(input));
        int16x8_t local_accum3 =
            vmull_s8(vget_low_s8(weights3), vget_low_s8(input));
        local_accum0 =
            vmlal_s8(local_accum0, vget_high_s8(weights0), vget_high_s8(input));
        local_accum1 =
            vmlal_s8(local_accum1, vget_high_s8(weights1), vget_high_s8(input));
        local_accum2 =
            vmlal_s8(local_accum2, vget_high_s8(weights2), vget_high_s8(input));
        local_accum3 =
            vmlal_s8(local_accum3, vget_high_s8(weights3), vget_high_s8(input));
        row_accum0 = vpadalq_s16(row_accum0, local_accum0);
        row_accum1 = vpadalq_s16(row_accum1, local_accum1);
        row_accum2 = vpadalq_s16(row_accum2, local_accum2);
        row_accum3 = vpadalq_s16(row_accum3, local_accum3);
      }
      // Horizontally reduce accumulators
      int32x2_t pairwise_reduced_acc_0, pairwise_reduced_acc_1,
          pairwise_reduced_acc_2, pairwise_reduced_acc_3;
      pairwise_reduced_acc_0 =
          vpadd_s32(vget_low_s32(row_accum0), vget_high_s32(row_accum0));
      pairwise_reduced_acc_1 =
          vpadd_s32(vget_low_s32(row_accum1), vget_high_s32(row_accum1));
      pairwise_reduced_acc_2 =
          vpadd_s32(vget_low_s32(row_accum2), vget_high_s32(row_accum2));
      pairwise_reduced_acc_3 =
          vpadd_s32(vget_low_s32(row_accum3), vget_high_s32(row_accum3));
      const int32x2_t reduced_lo =
          vpadd_s32(pairwise_reduced_acc_0, pairwise_reduced_acc_1);
      const int32x2_t reduced_hi =
          vpadd_s32(pairwise_reduced_acc_2, pairwise_reduced_acc_3);
      int32x4_t reduced = vcombine_s32(reduced_lo, reduced_hi);
      // Add bias values.
      int32x4_t bias_vec = vld1q_s32(bias_data + c);
      reduced = vaddq_s32(reduced, bias_vec);
      reduced = vshlq_s32(reduced, vdupq_n_s32(left_shift));
      // Multiply by the fixed-point multiplier.
      reduced = vqrdmulhq_n_s32(reduced, output_multiplier);
      // Rounding-shift-right.
      using gemmlowp::RoundingDivideByPOT;
      reduced = RoundingDivideByPOT(reduced, right_shift);
      // Narrow values down to 16 bit signed.
      const int16x4_t res16 = vqmovn_s32(reduced);
      vst1_s16(output_data + c, res16);
    }
  } else if (batches == 4) {
    const int right_shift = output_shift > 0 ? 0 : -output_shift;
    const int left_shift = output_shift > 0 ? output_shift : 0;
    for (int c = 0; c < output_depth; c += 4) {
      const int8* shuffled_input_ptr =
          reinterpret_cast<const int8*>(shuffled_input_workspace_data);
      // Accumulation loop.
      int32x4_t row_accum00 = vdupq_n_s32(0);
      int32x4_t row_accum10 = vdupq_n_s32(0);
      int32x4_t row_accum20 = vdupq_n_s32(0);
      int32x4_t row_accum30 = vdupq_n_s32(0);
      int32x4_t row_accum01 = vdupq_n_s32(0);
      int32x4_t row_accum11 = vdupq_n_s32(0);
      int32x4_t row_accum21 = vdupq_n_s32(0);
      int32x4_t row_accum31 = vdupq_n_s32(0);
      int32x4_t row_accum02 = vdupq_n_s32(0);
      int32x4_t row_accum12 = vdupq_n_s32(0);
      int32x4_t row_accum22 = vdupq_n_s32(0);
      int32x4_t row_accum32 = vdupq_n_s32(0);
      int32x4_t row_accum03 = vdupq_n_s32(0);
      int32x4_t row_accum13 = vdupq_n_s32(0);
      int32x4_t row_accum23 = vdupq_n_s32(0);
      int32x4_t row_accum33 = vdupq_n_s32(0);
      for (int d = 0; d < accum_depth; d += 16) {
        int8x16_t weights0 = vld1q_s8(shuffled_weights_ptr + 0);
        int8x16_t weights1 = vld1q_s8(shuffled_weights_ptr + 16);
        int8x16_t weights2 = vld1q_s8(shuffled_weights_ptr + 32);
        int8x16_t weights3 = vld1q_s8(shuffled_weights_ptr + 48);
        shuffled_weights_ptr += 64;
        int8x16_t input0 = vld1q_s8(shuffled_input_ptr + 0);
        int8x16_t input1 = vld1q_s8(shuffled_input_ptr + 16);
        int8x16_t input2 = vld1q_s8(shuffled_input_ptr + 32);
        int8x16_t input3 = vld1q_s8(shuffled_input_ptr + 48);
        shuffled_input_ptr += 64;
        int16x8_t local_accum0, local_accum1, local_accum2, local_accum3;
#define TFLITE_SHUFFLED_FC_ACCUM(B)                                           \
  local_accum0 = vmull_s8(vget_low_s8(weights0), vget_low_s8(input##B));      \
  local_accum1 = vmull_s8(vget_low_s8(weights1), vget_low_s8(input##B));      \
  local_accum2 = vmull_s8(vget_low_s8(weights2), vget_low_s8(input##B));      \
  local_accum3 = vmull_s8(vget_low_s8(weights3), vget_low_s8(input##B));      \
  local_accum0 =                                                              \
      vmlal_s8(local_accum0, vget_high_s8(weights0), vget_high_s8(input##B)); \
  local_accum1 =                                                              \
      vmlal_s8(local_accum1, vget_high_s8(weights1), vget_high_s8(input##B)); \
  local_accum2 =                                                              \
      vmlal_s8(local_accum2, vget_high_s8(weights2), vget_high_s8(input##B)); \
  local_accum3 =                                                              \
      vmlal_s8(local_accum3, vget_high_s8(weights3), vget_high_s8(input##B)); \
  row_accum0##B = vpadalq_s16(row_accum0##B, local_accum0);                   \
  row_accum1##B = vpadalq_s16(row_accum1##B, local_accum1);                   \
  row_accum2##B = vpadalq_s16(row_accum2##B, local_accum2);                   \
  row_accum3##B = vpadalq_s16(row_accum3##B, local_accum3);

        TFLITE_SHUFFLED_FC_ACCUM(0)
        TFLITE_SHUFFLED_FC_ACCUM(1)
        TFLITE_SHUFFLED_FC_ACCUM(2)
        TFLITE_SHUFFLED_FC_ACCUM(3)

#undef TFLITE_SHUFFLED_FC_ACCUM
      }
      // Horizontally reduce accumulators

#define TFLITE_SHUFFLED_FC_STORE(B)                                           \
  {                                                                           \
    int32x2_t pairwise_reduced_acc_0, pairwise_reduced_acc_1,                 \
        pairwise_reduced_acc_2, pairwise_reduced_acc_3;                       \
    pairwise_reduced_acc_0 =                                                  \
        vpadd_s32(vget_low_s32(row_accum0##B), vget_high_s32(row_accum0##B)); \
    pairwise_reduced_acc_1 =                                                  \
        vpadd_s32(vget_low_s32(row_accum1##B), vget_high_s32(row_accum1##B)); \
    pairwise_reduced_acc_2 =                                                  \
        vpadd_s32(vget_low_s32(row_accum2##B), vget_high_s32(row_accum2##B)); \
    pairwise_reduced_acc_3 =                                                  \
        vpadd_s32(vget_low_s32(row_accum3##B), vget_high_s32(row_accum3##B)); \
    const int32x2_t reduced_lo =                                              \
        vpadd_s32(pairwise_reduced_acc_0, pairwise_reduced_acc_1);            \
    const int32x2_t reduced_hi =                                              \
        vpadd_s32(pairwise_reduced_acc_2, pairwise_reduced_acc_3);            \
    int32x4_t reduced = vcombine_s32(reduced_lo, reduced_hi);                 \
    int32x4_t bias_vec = vld1q_s32(bias_data + c);                            \
    reduced = vaddq_s32(reduced, bias_vec);                                   \
    reduced = vshlq_s32(reduced, vdupq_n_s32(left_shift));                    \
    reduced = vqrdmulhq_n_s32(reduced, output_multiplier);                    \
    using gemmlowp::RoundingDivideByPOT;                                      \
    reduced = RoundingDivideByPOT(reduced, right_shift);                      \
    const int16x4_t res16 = vqmovn_s32(reduced);                              \
    vst1_s16(output_data + c + B * output_stride, res16);                     \
  }

      TFLITE_SHUFFLED_FC_STORE(0);
      TFLITE_SHUFFLED_FC_STORE(1);
      TFLITE_SHUFFLED_FC_STORE(2);
      TFLITE_SHUFFLED_FC_STORE(3);

#undef TFLITE_SHUFFLED_FC_STORE
    }
  } else {
    TFLITE_DCHECK(false);
    return;
  }
#else
  if (batches == 1) {
    int16* output_ptr = output_data;
    // Shuffled weights have had their sign bit (0x80) pre-flipped (xor'd)
    // so that just reinterpreting them as int8 values is equivalent to
    // subtracting 128 from them, thus implementing for free the subtraction of
    // the zero_point value 128.
    const int8* shuffled_weights_ptr =
        reinterpret_cast<const int8*>(shuffled_weights_data);
    // Likewise, we preshuffled and pre-xored the input data above.
    const int8* shuffled_input_data =
        reinterpret_cast<const int8*>(shuffled_input_workspace_data);
    for (int c = 0; c < output_depth; c += 4) {
      // Internal accumulation.
      // Initialize accumulator with the bias-value.
      int32 accum[4] = {0};
      // Accumulation loop.
      for (int d = 0; d < accum_depth; d += 16) {
        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 16; j++) {
            int8 input_val = shuffled_input_data[d + j];
            int8 weights_val = *shuffled_weights_ptr++;
            accum[i] += weights_val * input_val;
          }
        }
      }
      for (int i = 0; i < 4; i++) {
        // Add bias value
        int acc = accum[i] + bias_data[c + i];
        // Down-scale the final int32 accumulator to the scale used by our
        // (16-bit, typically 3 integer bits) fixed-point format. The quantized
        // multiplier and shift here have been pre-computed offline
        // (e.g. by toco).
        acc =
            MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
        // Saturate, cast to int16, and store to output array.
        acc = std::max(acc, -32768);
        acc = std::min(acc, 32767);
        output_ptr[c + i] = acc;
      }
    }
  } else if (batches == 4) {
    int16* output_ptr = output_data;
    // Shuffled weights have had their sign bit (0x80) pre-flipped (xor'd)
    // so that just reinterpreting them as int8 values is equivalent to
    // subtracting 128 from them, thus implementing for free the subtraction of
    // the zero_point value 128.
    const int8* shuffled_weights_ptr =
        reinterpret_cast<const int8*>(shuffled_weights_data);
    // Likewise, we preshuffled and pre-xored the input data above.
    const int8* shuffled_input_data =
        reinterpret_cast<const int8*>(shuffled_input_workspace_data);
    for (int c = 0; c < output_depth; c += 4) {
      const int8* shuffled_input_ptr = shuffled_input_data;
      // Accumulation loop.
      // Internal accumulation.
      // Initialize accumulator with the bias-value.
      int32 accum[4][4];
      for (int i = 0; i < 4; i++) {
        for (int b = 0; b < 4; b++) {
          accum[i][b] = 0;
        }
      }
      for (int d = 0; d < accum_depth; d += 16) {
        for (int i = 0; i < 4; i++) {
          for (int b = 0; b < 4; b++) {
            for (int j = 0; j < 16; j++) {
              int8 input_val = shuffled_input_ptr[16 * b + j];
              int8 weights_val = shuffled_weights_ptr[16 * i + j];
              accum[i][b] += weights_val * input_val;
            }
          }
        }
        shuffled_input_ptr += 64;
        shuffled_weights_ptr += 64;
      }
      for (int i = 0; i < 4; i++) {
        for (int b = 0; b < 4; b++) {
          // Add bias value
          int acc = accum[i][b] + bias_data[c + i];
          // Down-scale the final int32 accumulator to the scale used by our
          // (16-bit, typically 3 integer bits) fixed-point format. The
          // quantized multiplier and shift here have been pre-computed offline
          // (e.g. by toco).
          acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                              output_shift);
          // Saturate, cast to int16, and store to output array.
          acc = std::max(acc, -32768);
          acc = std::min(acc, 32767);
          output_ptr[b * output_stride + c + i] = acc;
        }
      }
    }
  } else {
    TFLITE_DCHECK(false);
    return;
  }
#endif
}

// Wraps ShuffledFullyConnectedWorkerImpl into a Task class
// to allow using gemmlowp's threadpool.
struct ShuffledFullyConnectedWorkerTask : cpu_backend_threadpool::Task {
  ShuffledFullyConnectedWorkerTask(const uint8* input_data,
                                   const int8* shuffled_weights_data,
                                   int batches, int output_depth,
                                   int output_stride, int accum_depth,
                                   const int32* bias_data,
                                   int32 output_multiplier, int output_shift,
                                   int16* output_data)
      : input_data_(input_data),
        shuffled_weights_data_(shuffled_weights_data),
        batches_(batches),
        output_depth_(output_depth),
        output_stride_(output_stride),
        accum_depth_(accum_depth),
        bias_data_(bias_data),
        output_multiplier_(output_multiplier),
        output_shift_(output_shift),
        output_data_(output_data) {}

  void Run() override {
    ShuffledFullyConnectedWorkerImpl(
        input_data_, shuffled_weights_data_, batches_, output_depth_,
        output_stride_, accum_depth_, bias_data_, output_multiplier_,
        output_shift_, output_data_);
  }

  const uint8* input_data_;
  const int8* shuffled_weights_data_;
  int batches_;
  int output_depth_;
  int output_stride_;
  int accum_depth_;
  const int32* bias_data_;
  int32 output_multiplier_;
  int output_shift_;
  int16* output_data_;
};

inline void ShuffledFullyConnected(
    const FullyConnectedParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& weights_shape,
    const uint8* shuffled_weights_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    int16* output_data, uint8* shuffled_input_workspace_data,
    CpuBackendContext* cpu_backend_context) {
  gemmlowp::ScopedProfilingLabel label("ShuffledFullyConnected/8bit");
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_EQ(output_activation_min, -32768);
  TFLITE_DCHECK_EQ(output_activation_max, 32767);
  TFLITE_DCHECK_GE(input_shape.DimensionsCount(), 1);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_GE(output_shape.DimensionsCount(), 1);
  // TODO(benoitjacob): This really should be:
  //     const int batches = ArraySize(output_dims, 1);
  // but the current --variable_batch hack consists in overwriting the 3rd
  // dimension with the runtime batch size, as we don't keep track for each
  // array of which dimension is the batch dimension in it.
  const int output_dim_count = output_shape.DimensionsCount();
  const int weights_dim_count = weights_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dim_count - 2,
                                       output_shape, output_dim_count - 1);
  const int accum_depth = weights_shape.Dims(weights_dim_count - 1);
  TFLITE_DCHECK((accum_depth % 16) == 0);
  TFLITE_DCHECK((output_depth % 4) == 0);
  // Shuffled weights have had their sign bit (0x80) pre-flipped (xor'd)
  // so that just reinterpreting them as int8 values is equivalent to
  // subtracting 128 from them, thus implementing for free the subtraction of
  // the zero_point value 128.
  const int8* int8_shuffled_weights_data =
      reinterpret_cast<const int8*>(shuffled_weights_data);

  // Shuffling and xoring of input activations into the workspace buffer
  if (batches == 1) {
#ifdef USE_NEON
    const uint8x16_t signbit = vdupq_n_u8(0x80);
    for (int i = 0; i < accum_depth; i += 16) {
      uint8x16_t val = vld1q_u8(input_data + i);
      val = veorq_u8(val, signbit);
      vst1q_u8(shuffled_input_workspace_data + i, val);
    }
#else
    for (int i = 0; i < accum_depth; i++) {
      shuffled_input_workspace_data[i] = input_data[i] ^ 0x80;
    }
#endif
  } else if (batches == 4) {
    uint8* shuffled_input_workspace_ptr = shuffled_input_workspace_data;
    int c = 0;
#ifdef USE_NEON
    const uint8x16_t signbit = vdupq_n_u8(0x80);
    for (c = 0; c < accum_depth; c += 16) {
      const uint8* src_data_ptr = input_data + c;
      uint8x16_t val0 = vld1q_u8(src_data_ptr + 0 * accum_depth);
      uint8x16_t val1 = vld1q_u8(src_data_ptr + 1 * accum_depth);
      uint8x16_t val2 = vld1q_u8(src_data_ptr + 2 * accum_depth);
      uint8x16_t val3 = vld1q_u8(src_data_ptr + 3 * accum_depth);
      val0 = veorq_u8(val0, signbit);
      val1 = veorq_u8(val1, signbit);
      val2 = veorq_u8(val2, signbit);
      val3 = veorq_u8(val3, signbit);
      vst1q_u8(shuffled_input_workspace_ptr + 0, val0);
      vst1q_u8(shuffled_input_workspace_ptr + 16, val1);
      vst1q_u8(shuffled_input_workspace_ptr + 32, val2);
      vst1q_u8(shuffled_input_workspace_ptr + 48, val3);
      shuffled_input_workspace_ptr += 64;
    }
#else
    for (c = 0; c < accum_depth; c += 16) {
      for (int b = 0; b < 4; b++) {
        const uint8* src_data_ptr = input_data + b * accum_depth + c;
        for (int j = 0; j < 16; j++) {
          uint8 src_val = *src_data_ptr++;
          // Flip the sign bit, so that the kernel will only need to
          // reinterpret these uint8 values as int8, getting for free the
          // subtraction of the zero_point value 128.
          uint8 dst_val = src_val ^ 0x80;
          *shuffled_input_workspace_ptr++ = dst_val;
        }
      }
    }
#endif
  } else {
    TFLITE_DCHECK(false);
    return;
  }

  static constexpr int kKernelRows = 4;
  const int thread_count =
      LegacyHowManyThreads<kKernelRows>(cpu_backend_context->max_num_threads(),
                                        output_depth, batches, accum_depth);
  if (thread_count == 1) {
    // Single-thread case: do the computation on the current thread, don't
    // use a threadpool
    ShuffledFullyConnectedWorkerImpl(
        shuffled_input_workspace_data, int8_shuffled_weights_data, batches,
        output_depth, output_depth, accum_depth, bias_data, output_multiplier,
        output_shift, output_data);
    return;
  }

  // Multi-threaded case: use the gemmlowp context's threadpool.
  TFLITE_DCHECK_GT(thread_count, 1);
  std::vector<ShuffledFullyConnectedWorkerTask> tasks;
  // TODO(b/131746020) don't create new heap allocations every time.
  // At least we make it a single heap allocation by using reserve().
  tasks.reserve(thread_count);
  const int kRowsPerWorker =
      RoundUp<kKernelRows>(CeilQuotient(output_depth, thread_count));
  int row_start = 0;
  for (int i = 0; i < thread_count; i++) {
    int row_end = std::min(output_depth, row_start + kRowsPerWorker);
    tasks.emplace_back(shuffled_input_workspace_data,
                       int8_shuffled_weights_data + row_start * accum_depth,
                       batches, row_end - row_start, output_depth, accum_depth,
                       bias_data + row_start, output_multiplier, output_shift,
                       output_data + row_start);
    row_start = row_end;
  }
  TFLITE_DCHECK_EQ(row_start, output_depth);
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                  cpu_backend_context);
}

#ifdef USE_NEON

inline float32x4_t DivideSumForMeanImpl(
    const float32x4_t sum, const float32x4_t num_elements_reverse,
    const bool ordinary_mean, const float32x4_t scale_dup,
    const float32x4_t zero_point_with_bias_dup) {
  const float32x4_t val = vmulq_f32(sum, num_elements_reverse);
  if (!ordinary_mean) {
#ifdef ARM_FEATURE_FMA
    return vfmaq_f32(zero_point_with_bias_dup, scale_dup, val);
#else
    return vmlaq_f32(zero_point_with_bias_dup, scale_dup, val);
#endif  // ARM_FEATURE_FMA
  }
  return val;
}

inline int32x4_t RoundToNearest(const float32x4_t input) {
#if !defined(__aarch64__) && !defined(__SSE4_1__)
  static const float32x4_t zero_val_dup = vdupq_n_f32(0.0f);
  static const float32x4_t point5_val_dup = vdupq_n_f32(0.5f);
  static const float32x4_t minus_point5_val_dup = vdupq_n_f32(-0.5f);

  const uint32x4_t mask = vcltq_f32(input, zero_val_dup);
  const float32x4_t round =
      vbslq_f32(mask, minus_point5_val_dup, point5_val_dup);
  return vcvtq_s32_f32(vaddq_f32(input, round));
#else
  return vcvtnq_s32_f32(input);
#endif  // !defined(__aarch64__)
}

inline uint32x4_t RoundToNearestUnsigned(const float32x4_t input) {
#if defined(__aarch64__) && !defined(__SSE4_1__)
  // Note that vcvtnq_u32_f32 is not available on the arm_neon_sse.h.
  return vcvtnq_u32_f32(input);
#else
  static const float32x4_t point5_val_dup = vdupq_n_f32(0.5f);

  return vcvtq_u32_f32(vaddq_f32(input, point5_val_dup));
#endif  // defined(__aarch64__) && !defined(__SSE4_1__)
}

#endif  // USE_NEON

inline void MeanImpl(const tflite::MeanParams& op_params,
                     const RuntimeShape& input_shape, const uint8_t* input_data,
                     int32 input_zero_point, float input_scale,
                     const RuntimeShape& output_shape, uint8_t* output_data,
                     int32 output_zero_point, float output_scale,
                     int start_depth, int end_depth) {
  gemmlowp::ScopedProfilingLabel label("Mean4D/Uint8/MeanImpl");

  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  const int output_batch = output_shape.Dims(0);
  const int output_height = output_shape.Dims(2);
  const int output_width = output_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const float num_elements_in_axis = input_width * input_height;

  TFLITE_DCHECK_EQ(op_params.axis_count, 2);
  TFLITE_DCHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
                (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_DCHECK_EQ(output_height, 1);
  TFLITE_DCHECK_EQ(output_width, 1);

  const bool ordinary_mean =
      (input_zero_point == output_zero_point && input_scale == output_scale);
  float scale = 0.0f, bias = 0.0f;
  if (!ordinary_mean) {
    scale = input_scale / output_scale;
    bias = -input_zero_point * scale + 0.5;
  }

#ifdef USE_NEON
  const float32x4_t num_elements_dup = vdupq_n_f32(num_elements_in_axis);
  // This is only an approximation as NEON does not offer division instruction.
  const float32x4_t scale_dup = vdupq_n_f32(scale);
  const float32x4_t num_elements_reverse = vrecpeq_f32(num_elements_dup);
  float32x4_t zero_point_with_bias_dup = vdupq_n_f32(output_zero_point + bias);
#endif  // USE_NEON

  for (int out_b = 0; out_b < output_batch; ++out_b) {
    int out_d = start_depth;
#ifdef USE_NEON

    for (; out_d < end_depth - 8; out_d += 8) {
      float32x4_t temp_sum_1 = vdupq_n_f32(0);
      float32x4_t temp_sum_2 = vdupq_n_f32(0);
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          const uint8_t* input_data_ptr =
              input_data + Offset(input_shape, out_b, in_h, in_w, out_d);
          uint8x8_t input_data_val = vld1_u8(input_data_ptr);
          int16x8_t input_data_val_shift =
              vreinterpretq_s16_u16(vmovl_u8(input_data_val));
          float32x4_t input_float_1 =
              vcvtq_f32_s32(vmovl_s16(vget_high_s16(input_data_val_shift)));
          float32x4_t input_float_2 =
              vcvtq_f32_s32(vmovl_s16(vget_low_s16(input_data_val_shift)));
          temp_sum_1 = vaddq_f32(temp_sum_1, input_float_1);
          temp_sum_2 = vaddq_f32(temp_sum_2, input_float_2);
        }
      }

      const float32x4_t mean_1 =
          DivideSumForMeanImpl(temp_sum_1, num_elements_reverse, ordinary_mean,
                               scale_dup, zero_point_with_bias_dup);
      const float32x4_t mean_2 =
          DivideSumForMeanImpl(temp_sum_2, num_elements_reverse, ordinary_mean,
                               scale_dup, zero_point_with_bias_dup);

      uint32x4_t casted_mean_1 = RoundToNearestUnsigned(mean_1);
      uint16x4_t narrow_range_mean_1 = vmovn_u32(casted_mean_1);
      uint32x4_t casted_mean_2 = RoundToNearestUnsigned(mean_2);
      uint16x4_t narrow_range_mean_2 = vmovn_u32(casted_mean_2);
      uint16x8_t combined_mean =
          vcombine_u16(narrow_range_mean_2, narrow_range_mean_1);
      uint8x8_t narrowed_combined_mean = vmovn_u16(combined_mean);
      uint8_t* output_data_ptr =
          output_data + Offset(output_shape, out_b, 0, 0, out_d);
      vst1_u8(output_data_ptr, narrowed_combined_mean);
    }
#endif  // USE_NEON

    for (; out_d < end_depth; ++out_d) {
      float temp_value = 0;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          temp_value +=
              input_data[Offset(input_shape, out_b, in_h, in_w, out_d)];
        }
      }

      temp_value = temp_value / num_elements_in_axis;
      if (ordinary_mean) {
        output_data[Offset(output_shape, out_b, 0, 0, out_d)] =
            static_cast<uint8_t>(round(temp_value));
      } else {
        output_data[Offset(output_shape, out_b, 0, 0, out_d)] =
            static_cast<uint8_t>(round(temp_value * scale + bias)) +
            output_zero_point;
      }
    }
  }
}

struct MeanWorkerTask : cpu_backend_threadpool::Task {
  MeanWorkerTask(const tflite::MeanParams& op_params,
                 const RuntimeShape& input_shape, const uint8_t* input_data,
                 int32 input_zero_point, float input_scale,
                 const RuntimeShape& output_shape, uint8_t* output_data,
                 int32 output_zero_point, float output_scale, int start_height,
                 int end_height)
      : op_params_(op_params),
        input_shape_(input_shape),
        input_data_(input_data),
        input_zero_point_(input_zero_point),
        input_scale_(input_scale),
        output_shape_(output_shape),
        output_data_(output_data),
        output_zero_point_(output_zero_point),
        output_scale_(output_scale),
        start_height_(start_height),
        end_height_(end_height) {}

  void Run() override {
    MeanImpl(op_params_, input_shape_, input_data_, input_zero_point_,
             input_scale_, output_shape_, output_data_, output_zero_point_,
             output_scale_, start_height_, end_height_);
  }

 private:
  const tflite::MeanParams& op_params_;
  const RuntimeShape& input_shape_;
  const uint8_t* input_data_;
  int32 input_zero_point_;
  float input_scale_;
  const RuntimeShape& output_shape_;
  uint8_t* output_data_;
  int32 output_zero_point_;
  float output_scale_;
  int start_height_;
  int end_height_;
};

inline void Mean(const tflite::MeanParams& op_params,
                 const RuntimeShape& unextended_input_shape,
                 const uint8_t* input_data, int32 input_zero_point,
                 float input_scale, const RuntimeShape& unextended_output_shape,
                 uint8_t* output_data, int32 output_zero_point,
                 float output_scale, CpuBackendContext* cpu_backend_context) {
  gemmlowp::ScopedProfilingLabel label("Mean4D/Uint8");
  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  TFLITE_CHECK_EQ(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_CHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  TFLITE_DCHECK_EQ(op_params.axis_count, 2);
  TFLITE_DCHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
                (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_DCHECK_EQ(output_height, 1);
  TFLITE_DCHECK_EQ(output_width, 1);

  constexpr int kMinDepthPerThread = 8;
  int thread_count = output_depth / kMinDepthPerThread;
  thread_count = thread_count > 0 ? thread_count : 1;
  const int capped_thread_count =
      std::min(thread_count, cpu_backend_context->max_num_threads());

  if (capped_thread_count == 1) {
    MeanImpl(op_params, input_shape, input_data, input_zero_point, input_scale,
             output_shape, output_data, output_zero_point, output_scale, 0,
             output_depth);
  } else {
    // Instead parrallel for batch, we loop for the output_depth since batch
    // is typical 1.
    std::vector<MeanWorkerTask> tasks;
    // TODO(b/131746020) don't create new heap allocations every time.
    // At least we make it a single heap allocation by using reserve().
    tasks.reserve(capped_thread_count);
    int depth_start = 0;
    for (int i = 0; i < capped_thread_count; ++i) {
      // Try to distribute the tasks as even as possible.
      int depth_end = depth_start +
                      (output_depth - depth_start) / (capped_thread_count - i);
      tasks.emplace_back(op_params, input_shape, input_data, input_zero_point,
                         input_scale, output_shape, output_data,
                         output_zero_point, output_scale, depth_start,
                         depth_end);
      depth_start = depth_end;
    }
    cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                    cpu_backend_context);
  }
}

inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& filter_shape,
                 const float* filter_data, const RuntimeShape& bias_shape,
                 const float* bias_data, const RuntimeShape& output_shape,
                 float* output_data, const RuntimeShape& im2col_shape,
                 float* im2col_data, CpuBackendContext* cpu_backend_context) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  (void)im2col_data;
  (void)im2col_shape;
  gemmlowp::ScopedProfilingLabel label("Conv");

  // NB: the float 0.0f value is represented by all zero bytes.
  const uint8 float_zero_byte = 0x00;
  const float* gemm_input_data = nullptr;
  const RuntimeShape* gemm_input_shape = nullptr;
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;
  if (need_dilated_im2col) {
    DilatedIm2col(params, float_zero_byte, input_shape, input_data,
                  filter_shape, output_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    Im2col(params, filter_height, filter_width, float_zero_byte, input_shape,
           input_data, im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else {
    // TODO(aselle): We need to make sure to not send im2col if it is not
    // needed.
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_shape = &input_shape;
  }

  const int gemm_input_dims = gemm_input_shape->DimensionsCount();
  int m = FlatSizeSkipDim(*gemm_input_shape, gemm_input_dims - 1);
  int n = output_shape.Dims(3);
  int k = gemm_input_shape->Dims(gemm_input_dims - 1);

#if defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)
  // The following code computes matrix multiplication c = a * transponse(b)
  // with CBLAS, where:
  // * `a` is a matrix with dimensions (m, k).
  // * `b` is a matrix with dimensions (n, k), so transpose(b) is (k, n).
  // * `c` is a matrix with dimensions (m, n).
  // The naming of variables are aligned with CBLAS specification here.
  const float* a = gemm_input_data;
  const float* b = filter_data;
  float* c = output_data;
  // The stride of matrix a, b and c respectively.
  int stride_a = k;
  int stride_b = k;
  int stride_c = n;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a,
              stride_a, b, stride_b, 0.0f, c, stride_c);
  optimized_ops::AddBiasAndEvalActivationFunction(
      output_activation_min, output_activation_max, bias_shape, bias_data,
      output_shape, output_data);
#else
  // When an optimized CBLAS implementation is not available, fall back
  // to using cpu_backend_gemm.
  cpu_backend_gemm::MatrixParams<float> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = n;
  lhs_params.cols = k;
  cpu_backend_gemm::MatrixParams<float> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = k;
  rhs_params.cols = m;
  cpu_backend_gemm::MatrixParams<float> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = n;
  dst_params.cols = m;
  cpu_backend_gemm::GemmParams<float, float> gemm_params;
  gemm_params.bias = bias_data;
  gemm_params.clamp_min = output_activation_min;
  gemm_params.clamp_max = output_activation_max;
  cpu_backend_gemm::Gemm(lhs_params, filter_data, rhs_params, gemm_input_data,
                         dst_params, output_data, gemm_params,
                         cpu_backend_context);
#endif  //  defined(TF_LITE_USE_CBLAS) && defined(__APPLE__)
}

inline void HybridConv(const ConvParams& params, float* scaling_factors_ptr,
                       const RuntimeShape& input_shape,
                       const int8_t* input_data,
                       const RuntimeShape& filter_shape,
                       const int8_t* filter_data,
                       const RuntimeShape& bias_shape, const float* bias_data,
                       const RuntimeShape& output_shape, float* output_data,
                       const RuntimeShape& im2col_shape, int8_t* im2col_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const int batch_size = input_shape.Dims(0);
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);

  const int8_t* gemm_input_data = nullptr;
  int num_input;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;

  if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    // symmetric quantization assumes zero point of 0.
    const int input_zero_point = 0;

    Im2col(params, filter_height, filter_width, input_zero_point, input_shape,
           input_data, im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    num_input = im2col_shape.FlatSize();
  } else {
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    num_input = input_shape.FlatSize();
  }

  // Flatten 4D matrices into 2D matrices for matrix multiplication.

  // Flatten so that each filter has its own row.
  const int filter_rows = filter_shape.Dims(0);
  const int filter_cols = FlatSizeSkipDim(filter_shape, 0);

  // In MatrixBatchVectorMultiplyAccumulate, each output value is the
  // dot product of one row of the first matrix with one row of the second
  // matrix. Therefore, the number of cols in each matrix are equivalent.
  //
  // After Im2Col, each input patch becomes a row.
  const int gemm_input_cols = filter_cols;
  const int gemm_input_rows = num_input / gemm_input_cols;

  const int output_cols = output_shape.Dims(3);
  const int output_rows = FlatSizeSkipDim(output_shape, 3);
  TFLITE_DCHECK_EQ(output_cols, filter_rows);
  TFLITE_DCHECK_EQ(output_rows, gemm_input_rows);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_cols);

  // MatrixBatchVectorMultiplyAccumulate assumes that each row of the second
  // input matrix has its own scale factor. This code duplicates the scale
  // factors for each row in the same batch.
  const int rows_per_batch = gemm_input_rows / batch_size;
  for (int i = gemm_input_rows - 1; i >= 0; --i) {
    scaling_factors_ptr[i] = scaling_factors_ptr[i / rows_per_batch];
  }

  std::fill_n(output_data, output_rows * output_cols, 0.0f);

  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      filter_data, filter_rows, filter_cols, gemm_input_data,
      scaling_factors_ptr, /*n_batch=*/gemm_input_rows, output_data,
      /*result_stride=*/1);

  AddBiasAndEvalActivationFunction(output_activation_min, output_activation_max,
                                   bias_shape, bias_data, output_shape,
                                   output_data);
}

inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const uint8* input_data, const RuntimeShape& filter_shape,
                 const uint8* filter_data, const RuntimeShape& bias_shape,
                 const int32* bias_data, const RuntimeShape& output_shape,
                 uint8* output_data, const RuntimeShape& im2col_shape,
                 uint8* im2col_data, CpuBackendContext* cpu_backend_context) {
  gemmlowp::ScopedProfilingLabel label("Conv/8bit");

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  const uint8* gemm_input_data = nullptr;
  const RuntimeShape* gemm_input_shape = nullptr;
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const bool need_dilated_im2col =
      dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;
  if (need_dilated_im2col) {
    TFLITE_DCHECK(im2col_data);
    const int input_zero_point = -input_offset;
    TFLITE_DCHECK_GE(input_zero_point, 0);
    TFLITE_DCHECK_LE(input_zero_point, 255);
    DilatedIm2col(params, input_zero_point, input_shape, input_data,
                  filter_shape, output_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    const int input_zero_point = -input_offset;
    TFLITE_DCHECK_GE(input_zero_point, 0);
    TFLITE_DCHECK_LE(input_zero_point, 255);
    Im2col(params, filter_height, filter_width, input_zero_point, input_shape,
           input_data, im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else {
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_shape = &input_shape;
  }

  const int gemm_input_rows = gemm_input_shape->Dims(3);
  // Using FlatSizeSkipDim causes segfault in some contexts (see b/79927784).
  // The root cause has not yet been identified though. Same applies below for
  // the other calls commented out. This is a partial rollback of cl/196819423.
  // const int gemm_input_cols = FlatSizeSkipDim(*gemm_input_shape, 3);
  const int gemm_input_cols = gemm_input_shape->Dims(0) *
                              gemm_input_shape->Dims(1) *
                              gemm_input_shape->Dims(2);
  const int filter_rows = filter_shape.Dims(0);
  // See b/79927784.
  // const int filter_cols = FlatSizeSkipDim(filter_shape, 0);
  const int filter_cols =
      filter_shape.Dims(1) * filter_shape.Dims(2) * filter_shape.Dims(3);
  const int output_rows = output_shape.Dims(3);
  // See b/79927784.
  // const int output_cols = FlatSizeSkipDim(output_shape, 3);
  const int output_cols =
      output_shape.Dims(0) * output_shape.Dims(1) * output_shape.Dims(2);
  TFLITE_DCHECK_EQ(output_rows, filter_rows);
  TFLITE_DCHECK_EQ(output_cols, gemm_input_cols);
  TFLITE_DCHECK_EQ(filter_cols, gemm_input_rows);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_rows);

  cpu_backend_gemm::MatrixParams<uint8> lhs_params;
  lhs_params.rows = filter_rows;
  lhs_params.cols = filter_cols;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.zero_point = -filter_offset;
  cpu_backend_gemm::MatrixParams<uint8> rhs_params;
  rhs_params.rows = gemm_input_rows;
  rhs_params.cols = gemm_input_cols;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.zero_point = -input_offset;
  cpu_backend_gemm::MatrixParams<uint8> dst_params;
  dst_params.rows = output_rows;
  dst_params.cols = output_cols;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.zero_point = output_offset;
  cpu_backend_gemm::GemmParams<int32, uint8> gemm_params;
  gemm_params.bias = bias_data;
  gemm_params.clamp_min = output_activation_min;
  gemm_params.clamp_max = output_activation_max;
  gemm_params.multiplier_fixedpoint = output_multiplier;
  gemm_params.multiplier_exponent = output_shift;
  cpu_backend_gemm::Gemm(lhs_params, filter_data, rhs_params, gemm_input_data,
                         dst_params, output_data, gemm_params,
                         cpu_backend_context);
}

template <typename T>
inline void DepthToSpace(const tflite::DepthToSpaceParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const T* input_data,
                         const RuntimeShape& unextended_output_shape,
                         T* output_data) {
  gemmlowp::ScopedProfilingLabel label("DepthToSpace");

  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int input_depth = input_shape.Dims(3);
  const int input_width = input_shape.Dims(2);
  const int input_height = input_shape.Dims(1);

  const int output_depth = output_shape.Dims(3);
  const int batch_size = output_shape.Dims(0);

  // Number of continuous values that we can copy in one interation.
  const int stride = op_params.block_size * output_depth;

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int in_h = 0; in_h < input_height; ++in_h) {
      const T* input_ptr = input_data + Offset(input_shape, batch, in_h, 0, 0);
      for (int offset_h = 0; offset_h < op_params.block_size; ++offset_h) {
        const T* src = input_ptr;
        for (int in_w = 0; in_w < input_width; ++in_w) {
          memcpy(output_data, src, stride * sizeof(T));
          output_data += stride;
          src += input_depth;
        }
        input_ptr += stride;
      }
    }
  }
}

template <typename T>
inline void SpaceToDepth(const tflite::SpaceToDepthParams& op_params,
                         const RuntimeShape& unextended_input_shape,
                         const T* input_data,
                         const RuntimeShape& unextended_output_shape,
                         T* output_data) {
  gemmlowp::ScopedProfilingLabel label("SpaceToDepth");

  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int output_depth = output_shape.Dims(3);
  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);

  const int input_depth = input_shape.Dims(3);
  const int batch_size = input_shape.Dims(0);

  // Number of continuous values that we can copy in one interation.
  const int stride = op_params.block_size * input_depth;

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int out_h = 0; out_h < output_height; ++out_h) {
      T* output_ptr = output_data + Offset(output_shape, batch, out_h, 0, 0);
      for (int offset_h = 0; offset_h < op_params.block_size; ++offset_h) {
        T* dst = output_ptr;
        for (int out_w = 0; out_w < output_width; ++out_w) {
          memcpy(dst, input_data, stride * sizeof(T));
          input_data += stride;
          dst += output_depth;
        }
        output_ptr += stride;
      }
    }
  }
}

inline void Relu(const RuntimeShape& input_shape, const float* input_data,
                 const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("Relu (not fused)");

  const auto input = MapAsVector(input_data, input_shape);
  auto output = MapAsVector(output_data, output_shape);
  output = input.cwiseMax(0.0f);
}

inline void L2Normalization(const tflite::L2NormalizationParams& op_params,
                            const RuntimeShape& input_shape,
                            const float* input_data,
                            const RuntimeShape& output_shape,
                            float* output_data) {
  gemmlowp::ScopedProfilingLabel label("L2Normalization");
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  for (int i = 0; i < outer_size; ++i) {
    float squared_l2_norm = 0;
    for (int c = 0; c < depth; ++c) {
      const float val = input_data[c];
      squared_l2_norm += val * val;
    }
    const float l2_norm = std::sqrt(squared_l2_norm);
    for (int c = 0; c < depth; ++c) {
      *output_data = *input_data / l2_norm;
      ++output_data;
      ++input_data;
    }
  }
}

inline void L2Normalization(const tflite::L2NormalizationParams& op_params,
                            const RuntimeShape& input_shape,
                            const uint8* input_data,
                            const RuntimeShape& output_shape,
                            uint8* output_data) {
  gemmlowp::ScopedProfilingLabel label("L2Normalization/8bit");
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int32 input_zero_point = op_params.input_zero_point;
  for (int i = 0; i < outer_size; ++i) {
    int32 square_l2_norm = 0;
    for (int c = 0; c < depth; c++) {
      // Note that input_data advances by depth in the second pass below.
      int32 diff = input_data[c] - input_zero_point;
      square_l2_norm += diff * diff;
    }
    // TODO(b/29395854): add clamping to TOCO and TF Lite kernel
    // for all zero tensors in the input_data
    int32 inv_l2norm_multiplier;
    int inv_l2norm_shift;
    GetInvSqrtQuantizedMultiplierExp(square_l2_norm, kReverseShift,
                                     &inv_l2norm_multiplier, &inv_l2norm_shift);

    for (int c = 0; c < depth; c++) {
      int32 diff = *input_data - input_zero_point;
      int32 rescaled_diff = MultiplyByQuantizedMultiplierSmallerThanOneExp(
          128 * diff, inv_l2norm_multiplier, inv_l2norm_shift);
      int32 unclamped_output_val = 128 + rescaled_diff;
      int32 output_val = std::min(255, std::max(0, unclamped_output_val));
      *output_data = static_cast<uint8>(output_val);
      ++input_data;
      ++output_data;
    }
  }
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const float* input1_data,
                const RuntimeShape& input2_shape, const float* input2_data,
                const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("Add");

  int i = 0;
  const int size = MatchingFlatSize(input1_shape, input2_shape, output_shape);
#ifdef USE_NEON
  const auto activation_min = vdupq_n_f32(params.float_activation_min);
  const auto activation_max = vdupq_n_f32(params.float_activation_max);
  for (; i <= size - 16; i += 16) {
    auto a10 = vld1q_f32(input1_data + i);
    auto a11 = vld1q_f32(input1_data + i + 4);
    auto a12 = vld1q_f32(input1_data + i + 8);
    auto a13 = vld1q_f32(input1_data + i + 12);
    auto a20 = vld1q_f32(input2_data + i);
    auto a21 = vld1q_f32(input2_data + i + 4);
    auto a22 = vld1q_f32(input2_data + i + 8);
    auto a23 = vld1q_f32(input2_data + i + 12);
    auto x0 = vaddq_f32(a10, a20);
    auto x1 = vaddq_f32(a11, a21);
    auto x2 = vaddq_f32(a12, a22);
    auto x3 = vaddq_f32(a13, a23);
    x0 = vmaxq_f32(activation_min, x0);
    x1 = vmaxq_f32(activation_min, x1);
    x2 = vmaxq_f32(activation_min, x2);
    x3 = vmaxq_f32(activation_min, x3);
    x0 = vminq_f32(activation_max, x0);
    x1 = vminq_f32(activation_max, x1);
    x2 = vminq_f32(activation_max, x2);
    x3 = vminq_f32(activation_max, x3);
    vst1q_f32(output_data + i, x0);
    vst1q_f32(output_data + i + 4, x1);
    vst1q_f32(output_data + i + 8, x2);
    vst1q_f32(output_data + i + 12, x3);
  }
  for (; i <= size - 4; i += 4) {
    auto a1 = vld1q_f32(input1_data + i);
    auto a2 = vld1q_f32(input2_data + i);
    auto x = vaddq_f32(a1, a2);
    x = vmaxq_f32(activation_min, x);
    x = vminq_f32(activation_max, x);
    vst1q_f32(output_data + i, x);
  }
#endif  // NEON

  for (; i < size; i++) {
    auto x = input1_data[i] + input2_data[i];
    output_data[i] = ActivationFunctionWithMinMax(
        x, params.float_activation_min, params.float_activation_max);
  }
}

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.
inline void AddElementwise(int size, const ArithmeticParams& params,
                           const uint8* input1_data, const uint8* input2_data,
                           uint8* output_data) {
  gemmlowp::ScopedProfilingLabel label("AddElementwise/8bit");
  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
#ifdef USE_NEON
  const uint8x8_t output_activation_min_vector =
      vdup_n_u8(params.quantized_activation_min);
  const uint8x8_t output_activation_max_vector =
      vdup_n_u8(params.quantized_activation_max);
  for (; i <= size - 8; i += 8) {
    const uint8x8_t input1_val_original = vld1_u8(input1_data + i);
    const uint8x8_t input2_val_original = vld1_u8(input2_data + i);
    const int16x8_t input1_val_s16 =
        vreinterpretq_s16_u16(vmovl_u8(input1_val_original));
    const int16x8_t input2_val_s16 =
        vreinterpretq_s16_u16(vmovl_u8(input2_val_original));
    const int16x8_t input1_val =
        vaddq_s16(input1_val_s16, vdupq_n_s16(params.input1_offset));
    const int16x8_t input2_val =
        vaddq_s16(input2_val_s16, vdupq_n_s16(params.input2_offset));
    const int16x4_t input1_val_high = vget_high_s16(input1_val);
    const int16x4_t input1_val_low = vget_low_s16(input1_val);
    const int16x4_t input2_val_high = vget_high_s16(input2_val);
    const int16x4_t input2_val_low = vget_low_s16(input2_val);
    int32x4_t x11 = vmovl_s16(input1_val_low);
    int32x4_t x12 = vmovl_s16(input1_val_high);
    int32x4_t x21 = vmovl_s16(input2_val_low);
    int32x4_t x22 = vmovl_s16(input2_val_high);
    const int32x4_t left_shift_dup = vdupq_n_s32(params.left_shift);
    x11 = vshlq_s32(x11, left_shift_dup);
    x12 = vshlq_s32(x12, left_shift_dup);
    x21 = vshlq_s32(x21, left_shift_dup);
    x22 = vshlq_s32(x22, left_shift_dup);
    x11 = vqrdmulhq_n_s32(x11, params.input1_multiplier);
    x12 = vqrdmulhq_n_s32(x12, params.input1_multiplier);
    x21 = vqrdmulhq_n_s32(x21, params.input2_multiplier);
    x22 = vqrdmulhq_n_s32(x22, params.input2_multiplier);
    const int32x4_t input1_shift_dup = vdupq_n_s32(params.input1_shift);
    const int32x4_t input2_shift_dup = vdupq_n_s32(params.input2_shift);
    x11 = vshlq_s32(x11, input1_shift_dup);
    x12 = vshlq_s32(x12, input1_shift_dup);
    x21 = vshlq_s32(x21, input2_shift_dup);
    x22 = vshlq_s32(x22, input2_shift_dup);
    int32x4_t s1 = vaddq_s32(x11, x21);
    int32x4_t s2 = vaddq_s32(x12, x22);
    s1 = vqrdmulhq_n_s32(s1, params.output_multiplier);
    s2 = vqrdmulhq_n_s32(s2, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    s1 = RoundingDivideByPOT(s1, -params.output_shift);
    s2 = RoundingDivideByPOT(s2, -params.output_shift);
    const int16x4_t s1_narrowed = vmovn_s32(s1);
    const int16x4_t s2_narrowed = vmovn_s32(s2);
    const int16x8_t s = vaddq_s16(vcombine_s16(s1_narrowed, s2_narrowed),
                                  vdupq_n_s16(params.output_offset));
    const uint8x8_t clamped =
        vmax_u8(output_activation_min_vector,
                vmin_u8(output_activation_max_vector, vqmovun_s16(s)));
    vst1_u8(output_data + i, clamped);
  }
#endif  // NEON

  for (; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32 shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32 scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);
    const int32 scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32 raw_sum = scaled_input1_val + scaled_input2_val;
    const int32 raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sum, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<uint8>(clamped_output);
  }
}

// Scalar-broadcast add that can be used for inner loop of more general
// broadcast add, so that, for example, scalar-broadcast with batch will still
// be fast.
inline void AddScalarBroadcast(int size, const ArithmeticParams& params,
                               uint8 input1_data, const uint8* input2_data,
                               uint8* output_data) {
  using gemmlowp::RoundingDivideByPOT;

  gemmlowp::ScopedProfilingLabel label("AddScalarBroadcast/8bit");
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);

  int i = 0;

#ifdef USE_NEON
  const int32x4_t left_shift_dup = vdupq_n_s32(params.left_shift);
  const uint8x8_t output_activation_min_vector =
      vdup_n_u8(params.quantized_activation_min);
  const uint8x8_t output_activation_max_vector =
      vdup_n_u8(params.quantized_activation_max);

  // Process broadcast scalar.
  const uint8x8_t input1_val_original = vdup_n_u8(input1_data);
  const int16x8_t input1_val_s16 =
      vreinterpretq_s16_u16(vmovl_u8(input1_val_original));
  const int16x8_t input1_val =
      vaddq_s16(input1_val_s16, vdupq_n_s16(params.input1_offset));
  const int16x4_t input1_val_high = vget_high_s16(input1_val);
  const int16x4_t input1_val_low = vget_low_s16(input1_val);
  int32x4_t x11 = vmovl_s16(input1_val_low);
  int32x4_t x12 = vmovl_s16(input1_val_high);
  x11 = vshlq_s32(x11, left_shift_dup);
  x12 = vshlq_s32(x12, left_shift_dup);
  x11 = vqrdmulhq_n_s32(x11, params.input1_multiplier);
  x12 = vqrdmulhq_n_s32(x12, params.input1_multiplier);
  const int32x4_t input1_shift_dup = vdupq_n_s32(params.input1_shift);
  x11 = vshlq_s32(x11, input1_shift_dup);
  x12 = vshlq_s32(x12, input1_shift_dup);

  for (; i <= size - 8; i += 8) {
    const uint8x8_t input2_val_original = vld1_u8(input2_data + i);
    const int16x8_t input2_val_s16 =
        vreinterpretq_s16_u16(vmovl_u8(input2_val_original));
    const int16x8_t input2_val =
        vaddq_s16(input2_val_s16, vdupq_n_s16(params.input2_offset));
    const int16x4_t input2_val_high = vget_high_s16(input2_val);
    const int16x4_t input2_val_low = vget_low_s16(input2_val);
    int32x4_t x21 = vmovl_s16(input2_val_low);
    int32x4_t x22 = vmovl_s16(input2_val_high);
    x21 = vshlq_s32(x21, left_shift_dup);
    x22 = vshlq_s32(x22, left_shift_dup);
    x21 = vqrdmulhq_n_s32(x21, params.input2_multiplier);
    x22 = vqrdmulhq_n_s32(x22, params.input2_multiplier);
    const int32x4_t input2_shift_dup = vdupq_n_s32(params.input2_shift);
    x21 = vshlq_s32(x21, input2_shift_dup);
    x22 = vshlq_s32(x22, input2_shift_dup);
    int32x4_t s1 = vaddq_s32(x11, x21);
    int32x4_t s2 = vaddq_s32(x12, x22);
    s1 = vqrdmulhq_n_s32(s1, params.output_multiplier);
    s2 = vqrdmulhq_n_s32(s2, params.output_multiplier);
    s1 = RoundingDivideByPOT(s1, -params.output_shift);
    s2 = RoundingDivideByPOT(s2, -params.output_shift);
    const int16x4_t s1_narrowed = vmovn_s32(s1);
    const int16x4_t s2_narrowed = vmovn_s32(s2);
    const int16x8_t s = vaddq_s16(vcombine_s16(s1_narrowed, s2_narrowed),
                                  vdupq_n_s16(params.output_offset));
    const uint8x8_t clamped =
        vmax_u8(output_activation_min_vector,
                vmin_u8(output_activation_max_vector, vqmovun_s16(s)));
    vst1_u8(output_data + i, clamped);
  }
#endif  // NEON

  if (i < size) {
    // Process broadcast scalar.
    const int32 input1_val = params.input1_offset + input1_data;
    const int32 shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32 scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);

    for (; i < size; ++i) {
      const int32 input2_val = params.input2_offset + input2_data[i];
      const int32 shifted_input2_val = input2_val * (1 << params.left_shift);
      const int32 scaled_input2_val =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              shifted_input2_val, params.input2_multiplier,
              params.input2_shift);
      const int32 raw_sum = scaled_input1_val + scaled_input2_val;
      const int32 raw_output =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              raw_sum, params.output_multiplier, params.output_shift) +
          params.output_offset;
      const int32 clamped_output =
          std::min(params.quantized_activation_max,
                   std::max(params.quantized_activation_min, raw_output));
      output_data[i] = static_cast<uint8>(clamped_output);
    }
  }
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const uint8* input1_data,
                const RuntimeShape& input2_shape, const uint8* input2_data,
                const RuntimeShape& output_shape, uint8* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  gemmlowp::ScopedProfilingLabel label("Add/8bit");
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);

  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  AddElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16* input1_data,
                const RuntimeShape& input2_shape, const int16* input2_data,
                const RuntimeShape& output_shape, int16* output_data) {
  gemmlowp::ScopedProfilingLabel label("Add/Int16");
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);

  const int input1_shift = params.input1_shift;
  const int flat_size =
      MatchingFlatSize(output_shape, input1_shape, input2_shape);
  const int16 output_activation_min = params.quantized_activation_min;
  const int16 output_activation_max = params.quantized_activation_max;

  TFLITE_DCHECK(input1_shift == 0 || params.input2_shift == 0);
  TFLITE_DCHECK_LE(input1_shift, 0);
  TFLITE_DCHECK_LE(params.input2_shift, 0);
  const int16* not_shift_input = input1_shift == 0 ? input1_data : input2_data;
  const int16* shift_input = input1_shift == 0 ? input2_data : input1_data;
  const int input_right_shift =
      input1_shift == 0 ? -params.input2_shift : -input1_shift;

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 input_ready_scaled = F0::FromRaw(not_shift_input[i]);
    F0 scaled_input = F0::FromRaw(
        gemmlowp::RoundingDivideByPOT(shift_input[i], input_right_shift));
    F0 result = gemmlowp::SaturatingAdd(scaled_input, input_ready_scaled);
    const int16 raw_output = result.raw();
    const int16 clamped_output = std::min(
        output_activation_max, std::max(output_activation_min, raw_output));
    output_data[i] = clamped_output;
  }
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int32* input1_data,
                const RuntimeShape& input2_shape, const int32* input2_data,
                const RuntimeShape& output_shape, int32* output_data) {
  gemmlowp::ScopedProfilingLabel label("Add/int32");

  auto input1_map = MapAsVector(input1_data, input1_shape);
  auto input2_map = MapAsVector(input2_data, input2_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  if (input1_shape == input2_shape) {
    output_map.array() = input1_map.array() + input2_map.array();
  } else if (input2_shape.FlatSize() == 1) {
    auto scalar = input2_data[0];
    output_map.array() = input1_map.array() + scalar;
  } else if (input1_shape.FlatSize() == 1) {
    auto scalar = input1_data[0];
    output_map.array() = scalar + input2_map.array();
  } else {
    // Should not come here.
    TFLITE_DCHECK(false);
  }
  output_map = output_map.cwiseMax(params.quantized_activation_min);
  output_map = output_map.cwiseMin(params.quantized_activation_max);
}

inline void BroadcastAddFivefold(const ArithmeticParams& unswitched_params,
                                 const RuntimeShape& unswitched_input1_shape,
                                 const uint8* unswitched_input1_data,
                                 const RuntimeShape& unswitched_input2_shape,
                                 const uint8* unswitched_input2_data,
                                 const RuntimeShape& output_shape,
                                 uint8* output_data) {
  gemmlowp::ScopedProfilingLabel label("BroadcastAddFivefold/8bit");

  ArithmeticParams switched_params = unswitched_params;
  switched_params.input1_offset = unswitched_params.input2_offset;
  switched_params.input1_multiplier = unswitched_params.input2_multiplier;
  switched_params.input1_shift = unswitched_params.input2_shift;
  switched_params.input2_offset = unswitched_params.input1_offset;
  switched_params.input2_multiplier = unswitched_params.input1_multiplier;
  switched_params.input2_shift = unswitched_params.input1_shift;

  const bool use_unswitched =
      unswitched_params.broadcast_category ==
      tflite::BroadcastableOpCategory::kFirstInputBroadcastsFast;

  const ArithmeticParams& params =
      use_unswitched ? unswitched_params : switched_params;
  const uint8* input1_data =
      use_unswitched ? unswitched_input1_data : unswitched_input2_data;
  const uint8* input2_data =
      use_unswitched ? unswitched_input2_data : unswitched_input1_data;

  // Fivefold nested loops. The second input resets its position for each
  // iteration of the second loop. The first input resets its position at the
  // beginning of the fourth loop. The innermost loop is an elementwise add of
  // sections of the arrays.
  uint8* output_data_ptr = output_data;
  const uint8* input1_data_ptr = input1_data;
  const uint8* input2_data_reset = input2_data;
  // In the fivefold pattern, y0, y2 and y4 are not broadcast, and so shared
  // between input shapes. y3 for input 1 is always broadcast, and so the
  // dimension there is 1, whereas optionally y1 might be broadcast for input 2.
  // Put another way,
  // input1.shape.FlatSize = y0 * y1 * y2 * y4,
  // input2.shape.FlatSize = y0 * y2 * y3 * y4.
  int y0 = params.broadcast_shape[0];
  int y1 = params.broadcast_shape[1];
  int y2 = params.broadcast_shape[2];
  int y3 = params.broadcast_shape[3];
  int y4 = params.broadcast_shape[4];
  if (y4 > 1) {
    // General fivefold pattern, with y4 > 1 so there is a non-broadcast inner
    // dimension.
    for (int i0 = 0; i0 < y0; ++i0) {
      const uint8* input2_data_ptr = nullptr;
      for (int i1 = 0; i1 < y1; ++i1) {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2) {
          for (int i3 = 0; i3 < y3; ++i3) {
            AddElementwise(y4, params, input1_data_ptr, input2_data_ptr,
                           output_data_ptr);
            input2_data_ptr += y4;
            output_data_ptr += y4;
          }
          // We have broadcast y4 of input1 data y3 times, and now move on.
          input1_data_ptr += y4;
        }
      }
      // We have broadcast y2*y3*y4 of input2 data y1 times, and now move on.
      input2_data_reset = input2_data_ptr;
    }
  } else {
    // Special case of y4 == 1, in which the innermost loop is a single element
    // and can be combined with the next (y3) as an inner broadcast.
    //
    // Note that this handles the case of pure scalar broadcast when
    // y0 == y1 == y2 == 1. With low overhead it handles cases such as scalar
    // broadcast with batch (as y2 > 1).
    //
    // NOTE The process is the same as the above general case except simplified
    // for y4 == 1 and the loop over y3 is contained within the
    // AddScalarBroadcast function.
    for (int i0 = 0; i0 < y0; ++i0) {
      const uint8* input2_data_ptr = nullptr;
      for (int i1 = 0; i1 < y1; ++i1) {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2) {
          AddScalarBroadcast(y3, params, *input1_data_ptr, input2_data_ptr,
                             output_data_ptr);
          input2_data_ptr += y3;
          output_data_ptr += y3;
          input1_data_ptr += 1;
        }
      }
      input2_data_reset = input2_data_ptr;
    }
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const float* input1_data,
                const RuntimeShape& input2_shape, const float* input2_data,
                const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("Mul");
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;

  int i = 0;
  const int size = MatchingFlatSize(input1_shape, input2_shape, output_shape);
#ifdef USE_NEON
  const auto activation_min = vdupq_n_f32(output_activation_min);
  const auto activation_max = vdupq_n_f32(output_activation_max);
  for (; i <= size - 16; i += 16) {
    auto a10 = vld1q_f32(input1_data + i);
    auto a11 = vld1q_f32(input1_data + i + 4);
    auto a12 = vld1q_f32(input1_data + i + 8);
    auto a13 = vld1q_f32(input1_data + i + 12);
    auto a20 = vld1q_f32(input2_data + i);
    auto a21 = vld1q_f32(input2_data + i + 4);
    auto a22 = vld1q_f32(input2_data + i + 8);
    auto a23 = vld1q_f32(input2_data + i + 12);
    auto x0 = vmulq_f32(a10, a20);
    auto x1 = vmulq_f32(a11, a21);
    auto x2 = vmulq_f32(a12, a22);
    auto x3 = vmulq_f32(a13, a23);

    x0 = vmaxq_f32(activation_min, x0);
    x1 = vmaxq_f32(activation_min, x1);
    x2 = vmaxq_f32(activation_min, x2);
    x3 = vmaxq_f32(activation_min, x3);
    x0 = vminq_f32(activation_max, x0);
    x1 = vminq_f32(activation_max, x1);
    x2 = vminq_f32(activation_max, x2);
    x3 = vminq_f32(activation_max, x3);

    vst1q_f32(output_data + i, x0);
    vst1q_f32(output_data + i + 4, x1);
    vst1q_f32(output_data + i + 8, x2);
    vst1q_f32(output_data + i + 12, x3);
  }
  for (; i <= size - 4; i += 4) {
    auto a1 = vld1q_f32(input1_data + i);
    auto a2 = vld1q_f32(input2_data + i);
    auto x = vmulq_f32(a1, a2);

    x = vmaxq_f32(activation_min, x);
    x = vminq_f32(activation_max, x);

    vst1q_f32(output_data + i, x);
  }
#endif  // NEON

  for (; i < size; i++) {
    auto x = input1_data[i] * input2_data[i];
    output_data[i] = ActivationFunctionWithMinMax(x, output_activation_min,
                                                  output_activation_max);
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int32* input1_data,
                const RuntimeShape& input2_shape, const int32* input2_data,
                const RuntimeShape& output_shape, int32* output_data) {
  gemmlowp::ScopedProfilingLabel label("Mul/int32/activation");

  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] * input2_data[i], output_activation_min,
        output_activation_max);
  }
}

inline void MulNoActivation(const ArithmeticParams& params,
                            const RuntimeShape& input1_shape,
                            const int32* input1_data,
                            const RuntimeShape& input2_shape,
                            const int32* input2_data,
                            const RuntimeShape& output_shape,
                            int32* output_data) {
  gemmlowp::ScopedProfilingLabel label("Mul/int32");

  auto input1_map = MapAsVector(input1_data, input1_shape);
  auto input2_map = MapAsVector(input2_data, input2_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  if (input1_shape == input2_shape) {
    output_map.array() = input1_map.array() * input2_map.array();
  } else if (input2_shape.FlatSize() == 1) {
    auto scalar = input2_data[0];
    output_map.array() = input1_map.array() * scalar;
  } else if (input1_shape.FlatSize() == 1) {
    auto scalar = input1_data[0];
    output_map.array() = scalar * input2_map.array();
  } else {
    // Should not come here.
    TFLITE_DCHECK(false);
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16* input1_data,
                const RuntimeShape& input2_shape, const int16* input2_data,
                const RuntimeShape& output_shape, int16* output_data) {
  gemmlowp::ScopedProfilingLabel label("Mul/Int16/NoActivation");
  // This is a copy of the reference implementation. We do not currently have a
  // properly optimized version.

  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 unclamped_result =
        F0::FromRaw(input1_data[i]) * F0::FromRaw(input2_data[i]);
    output_data[i] = unclamped_result.raw();
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16* input1_data,
                const RuntimeShape& input2_shape, const int16* input2_data,
                const RuntimeShape& output_shape, uint8* output_data) {
  gemmlowp::ScopedProfilingLabel label("Mul/Int16Uint8");
  // This is a copy of the reference implementation. We do not currently have a
  // properly optimized version.
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  const int32 output_offset = params.output_offset;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 unclamped_result =
        F0::FromRaw(input1_data[i]) * F0::FromRaw(input2_data[i]);
    int16 rescaled_result =
        gemmlowp::RoundingDivideByPOT(unclamped_result.raw(), 8);
    int16 clamped_result =
        std::min<int16>(output_activation_max - output_offset, rescaled_result);
    clamped_result =
        std::max<int16>(output_activation_min - output_offset, clamped_result);
    output_data[i] = output_offset + clamped_result;
  }
}

// Element-wise mul that can often be used for inner loop of broadcast Mul as
// well as the non-broadcast Mul.
inline void MulElementwise(int size, const ArithmeticParams& params,
                           const uint8* input1_data, const uint8* input2_data,
                           uint8* output_data) {
  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);
#ifdef USE_NEON
  const auto input1_offset_vector = vdupq_n_s16(params.input1_offset);
  const auto input2_offset_vector = vdupq_n_s16(params.input2_offset);
  const auto output_offset_vector = vdupq_n_s16(params.output_offset);
  const auto output_activation_min_vector =
      vdup_n_u8(params.quantized_activation_min);
  const auto output_activation_max_vector =
      vdup_n_u8(params.quantized_activation_max);
  const int left_shift = std::max(0, params.output_shift);
  const int right_shift = std::max(0, -params.output_shift);
  const int32x4_t left_shift_vec = vdupq_n_s32(left_shift);
  for (; i <= size - 8; i += 8) {
    // We load / store 8 at a time, multiplying as two sets of 4 int32s.
    const auto input1_val_original = vld1_u8(input1_data + i);
    const auto input2_val_original = vld1_u8(input2_data + i);
    const auto input1_val_s16 =
        vreinterpretq_s16_u16(vmovl_u8(input1_val_original));
    const auto input2_val_s16 =
        vreinterpretq_s16_u16(vmovl_u8(input2_val_original));
    const auto input1_val = vaddq_s16(input1_val_s16, input1_offset_vector);
    const auto input2_val = vaddq_s16(input2_val_s16, input2_offset_vector);

    const auto input1_val_low = vget_low_s16(input1_val);
    const auto input1_val_high = vget_high_s16(input1_val);
    const auto input2_val_low = vget_low_s16(input2_val);
    const auto input2_val_high = vget_high_s16(input2_val);

    auto p1 = vmull_s16(input2_val_low, input1_val_low);
    auto p2 = vmull_s16(input2_val_high, input1_val_high);

    p1 = vshlq_s32(p1, left_shift_vec);
    p2 = vshlq_s32(p2, left_shift_vec);
    p1 = vqrdmulhq_n_s32(p1, params.output_multiplier);
    p2 = vqrdmulhq_n_s32(p2, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    p1 = RoundingDivideByPOT(p1, right_shift);
    p2 = RoundingDivideByPOT(p2, right_shift);

    const auto p1_narrowed = vqmovn_s32(p1);
    const auto p2_narrowed = vqmovn_s32(p2);
    const auto p =
        vaddq_s16(vcombine_s16(p1_narrowed, p2_narrowed), output_offset_vector);
    const auto clamped =
        vmax_u8(output_activation_min_vector,
                vmin_u8(output_activation_max_vector, vqmovun_s16(p)));
    vst1_u8(output_data + i, clamped);
  }
#endif  // NEON

  for (; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                      params.output_multiplier,
                                      params.output_shift);
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<uint8>(clamped_output);
  }
}

// Broadcast mul that can often be used for inner loop of broadcast Mul.
inline void MulSimpleBroadcast(int size, const ArithmeticParams& params,
                               const uint8 broadcast_value,
                               const uint8* input2_data, uint8* output_data) {
  const int16 input1_val = params.input1_offset + broadcast_value;

  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);
#ifdef USE_NEON
  const auto input2_offset_vector = vdupq_n_s16(params.input2_offset);
  const auto output_offset_vector = vdupq_n_s16(params.output_offset);
  const auto output_activation_min_vector =
      vdup_n_u8(params.quantized_activation_min);
  const auto output_activation_max_vector =
      vdup_n_u8(params.quantized_activation_max);
  const int left_shift = std::max(0, params.output_shift);
  const int right_shift = std::max(0, -params.output_shift);
  const int32x4_t left_shift_vec = vdupq_n_s32(left_shift);
  for (; i <= size - 8; i += 8) {
    // We load / store 8 at a time, multiplying as two sets of 4 int32s.
    const auto input2_val_original = vld1_u8(input2_data + i);
    const auto input2_val_s16 =
        vreinterpretq_s16_u16(vmovl_u8(input2_val_original));
    const auto input2_val = vaddq_s16(input2_val_s16, input2_offset_vector);

    const auto input2_val_low = vget_low_s16(input2_val);
    const auto input2_val_high = vget_high_s16(input2_val);

    auto p1 = vmull_n_s16(input2_val_low, input1_val);
    auto p2 = vmull_n_s16(input2_val_high, input1_val);

    p1 = vshlq_s32(p1, left_shift_vec);
    p2 = vshlq_s32(p2, left_shift_vec);
    p1 = vqrdmulhq_n_s32(p1, params.output_multiplier);
    p2 = vqrdmulhq_n_s32(p2, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    p1 = RoundingDivideByPOT(p1, right_shift);
    p2 = RoundingDivideByPOT(p2, right_shift);

    const auto p1_narrowed = vmovn_s32(p1);
    const auto p2_narrowed = vmovn_s32(p2);
    const auto p =
        vaddq_s16(vcombine_s16(p1_narrowed, p2_narrowed), output_offset_vector);
    const auto clamped =
        vmax_u8(output_activation_min_vector,
                vmin_u8(output_activation_max_vector, vqmovun_s16(p)));
    vst1_u8(output_data + i, clamped);
  }
#endif  // NEON

  for (; i < size; ++i) {
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplierSmallerThanOneExp(input1_val * input2_val,
                                                       params.output_multiplier,
                                                       params.output_shift);
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<uint8>(clamped_output);
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const uint8* input1_data,
                const RuntimeShape& input2_shape, const uint8* input2_data,
                const RuntimeShape& output_shape, uint8* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  gemmlowp::ScopedProfilingLabel label("Mul/8bit");
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);

  MulElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void BroadcastMulFivefold(const ArithmeticParams& unswitched_params,
                                 const RuntimeShape& unswitched_input1_shape,
                                 const uint8* unswitched_input1_data,
                                 const RuntimeShape& unswitched_input2_shape,
                                 const uint8* unswitched_input2_data,
                                 const RuntimeShape& output_shape,
                                 uint8* output_data) {
  gemmlowp::ScopedProfilingLabel label("BroadcastMulFivefold/8bit");

  ArithmeticParams switched_params = unswitched_params;
  switched_params.input1_offset = unswitched_params.input2_offset;
  switched_params.input2_offset = unswitched_params.input1_offset;

  const bool use_unswitched =
      unswitched_params.broadcast_category ==
      tflite::BroadcastableOpCategory::kFirstInputBroadcastsFast;

  const ArithmeticParams& params =
      use_unswitched ? unswitched_params : switched_params;
  const uint8* input1_data =
      use_unswitched ? unswitched_input1_data : unswitched_input2_data;
  const uint8* input2_data =
      use_unswitched ? unswitched_input2_data : unswitched_input1_data;

  // Fivefold nested loops. The second input resets its position for each
  // iteration of the second loop. The first input resets its position at the
  // beginning of the fourth loop. The innermost loop is an elementwise Mul of
  // sections of the arrays.
  uint8* output_data_ptr = output_data;
  const uint8* input1_data_ptr = input1_data;
  const uint8* input2_data_reset = input2_data;
  int y0 = params.broadcast_shape[0];
  int y1 = params.broadcast_shape[1];
  int y2 = params.broadcast_shape[2];
  int y3 = params.broadcast_shape[3];
  int y4 = params.broadcast_shape[4];
  if (y4 > 1) {
    for (int i0 = 0; i0 < y0; ++i0) {
      const uint8* input2_data_ptr = nullptr;
      for (int i1 = 0; i1 < y1; ++i1) {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2) {
          for (int i3 = 0; i3 < y3; ++i3) {
            MulElementwise(y4, params, input1_data_ptr, input2_data_ptr,
                           output_data_ptr);
            input2_data_ptr += y4;
            output_data_ptr += y4;
          }
          input1_data_ptr += y4;
        }
      }
      input2_data_reset = input2_data_ptr;
    }
  } else {
    for (int i0 = 0; i0 < y0; ++i0) {
      const uint8* input2_data_ptr = nullptr;
      for (int i1 = 0; i1 < y1; ++i1) {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2) {
          MulSimpleBroadcast(y3, params, *input1_data_ptr, input2_data_ptr,
                             output_data_ptr);
          input2_data_ptr += y3;
          output_data_ptr += y3;
          ++input1_data_ptr;
        }
      }
      input2_data_reset = input2_data_ptr;
    }
  }
}

// TODO(jiawen): We can implement BroadcastDiv on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
// TODO(benoitjacob): BroadcastDiv is intentionally duplicated from
// reference_ops.h. Once an optimized version is implemented and NdArrayDesc<T>
// is no longer referenced in this file, move NdArrayDesc<T> from types.h to
// reference_ops.h.
template <typename T>
void BroadcastDiv4DSlow(const ArithmeticParams& params,
                        const RuntimeShape& unextended_input1_shape,
                        const T* input1_data,
                        const RuntimeShape& unextended_input2_shape,
                        const T* input2_data,
                        const RuntimeShape& unextended_output_shape,
                        T* output_data) {
  gemmlowp::ScopedProfilingLabel label("BroadcastDiv4DSlow");
  T output_activation_min;
  T output_activation_max;
  GetActivationParams(params, &output_activation_min, &output_activation_max);

  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          output_data[Offset(output_shape, b, y, x, c)] =
              ActivationFunctionWithMinMax(
                  input1_data[SubscriptToIndex(desc1, b, y, x, c)] /
                      input2_data[SubscriptToIndex(desc2, b, y, x, c)],
                  output_activation_min, output_activation_max);
        }
      }
    }
  }
}

// TODO: BroadcastDiv is intentionally duplicated from reference_ops.h.
// For more details see the comment above the generic version of
// BroadcastDiv4DSlow.
inline void BroadcastDiv4DSlow(const ArithmeticParams& params,
                               const RuntimeShape& unextended_input1_shape,
                               const uint8* input1_data,
                               const RuntimeShape& unextended_input2_shape,
                               const uint8* input2_data,
                               const RuntimeShape& unextended_output_shape,
                               uint8* output_data) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);

  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  TFLITE_DCHECK_GT(params.output_offset, -256);
  TFLITE_DCHECK_LT(params.output_offset, 256);

  for (int b = 0; b < output_shape.Dims(0); ++b) {
    for (int y = 0; y < output_shape.Dims(1); ++y) {
      for (int x = 0; x < output_shape.Dims(2); ++x) {
        for (int c = 0; c < output_shape.Dims(3); ++c) {
          const int32 input1_val =
              params.input1_offset +
              input1_data[SubscriptToIndex(desc1, b, y, x, c)];
          const int32 input2_val =
              params.input2_offset +
              input2_data[SubscriptToIndex(desc2, b, y, x, c)];
          TFLITE_DCHECK_NE(input2_val, 0);
          int recip_shift;
          const int32 input2_inv =
              (input2_val > 0) ? GetReciprocal(input2_val, 31, &recip_shift)
                               : -GetReciprocal(-input2_val, 31, &recip_shift);
          const int headroom = CountLeadingSignBits(input1_val);
          const int32 unscaled_quotient =
              MultiplyByQuantizedMultiplierGreaterThanOne(input1_val,
                                                          input2_inv, headroom);
          const int total_shift = params.output_shift - recip_shift - headroom;
          const int32 unclamped_result =
              params.output_offset +
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  unscaled_quotient, params.output_multiplier, total_shift);
          const int32 clamped_output = std::min(
              params.quantized_activation_max,
              std::max(params.quantized_activation_min, unclamped_result));
          output_data[Offset(output_shape, b, y, x, c)] =
              static_cast<uint8>(clamped_output);
        }
      }
    }
  }
}

// TODO(aselle): This is not actually optimized yet.
inline void SubNonBroadcast(const ArithmeticParams& params,
                            const RuntimeShape& input1_shape,
                            const float* input1_data,
                            const RuntimeShape& input2_shape,
                            const float* input2_data,
                            const RuntimeShape& output_shape,
                            float* output_data) {
  gemmlowp::ScopedProfilingLabel label("SubNonBroadcast");
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], params.float_activation_min,
        params.float_activation_max);
  }
}

inline void SubWithActivation(const ArithmeticParams& params,
                              const RuntimeShape& input1_shape,
                              const int32* input1_data,
                              const RuntimeShape& input2_shape,
                              const int32* input2_data,
                              const RuntimeShape& output_shape,
                              int32* output_data) {
  gemmlowp::ScopedProfilingLabel label("SubWithActivation/int32");
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, input2_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], params.quantized_activation_min,
        params.quantized_activation_max);
  }
}

inline void SubWithActivation(const ArithmeticParams& params,
                              const RuntimeShape& input1_shape,
                              const float* input1_data,
                              const RuntimeShape& input2_shape,
                              const float* input2_data,
                              const RuntimeShape& output_shape,
                              float* output_data) {
  gemmlowp::ScopedProfilingLabel label("SubWithActivation/float");
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, input2_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], params.float_activation_min,
        params.float_activation_max);
  }
}

template <typename T>
void Sub(const ArithmeticParams& params, const RuntimeShape& input1_shape,
         const T* input1_data, const RuntimeShape& input2_shape,
         const T* input2_data, const RuntimeShape& output_shape,
         T* output_data) {
  gemmlowp::ScopedProfilingLabel label("Sub");

  auto input1_map = MapAsVector(input1_data, input1_shape);
  auto input2_map = MapAsVector(input2_data, input2_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  if (input1_shape == input2_shape) {
    output_map.array() = input1_map.array() - input2_map.array();
  } else if (input1_shape.FlatSize() == 1) {
    auto scalar = input1_data[0];
    output_map.array() = scalar - input2_map.array();
  } else if (input2_shape.FlatSize() == 1) {
    auto scalar = input2_data[0];
    output_map.array() = input1_map.array() - scalar;
  } else {
    BroadcastSub4DSlow(params, input1_shape, input1_data, input2_shape,
                       input2_data, output_shape, output_data);
  }
}

inline void LstmCell(
    const LstmCellParams& params, const RuntimeShape& unextended_input_shape,
    const float* input_data, const RuntimeShape& unextended_prev_activ_shape,
    const float* prev_activ_data, const RuntimeShape& weights_shape,
    const float* weights_data, const RuntimeShape& unextended_bias_shape,
    const float* bias_data, const RuntimeShape& unextended_prev_state_shape,
    const float* prev_state_data,
    const RuntimeShape& unextended_output_state_shape, float* output_state_data,
    const RuntimeShape& unextended_output_activ_shape, float* output_activ_data,
    const RuntimeShape& unextended_concat_temp_shape, float* concat_temp_data,
    const RuntimeShape& unextended_activ_temp_shape, float* activ_temp_data,
    CpuBackendContext* cpu_backend_context) {
  gemmlowp::ScopedProfilingLabel label("LstmCell");
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_prev_activ_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_bias_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_prev_state_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_state_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_activ_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_concat_temp_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_activ_temp_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape prev_activ_shape =
      RuntimeShape::ExtendedShape(4, unextended_prev_activ_shape);
  const RuntimeShape bias_shape =
      RuntimeShape::ExtendedShape(4, unextended_bias_shape);
  const RuntimeShape prev_state_shape =
      RuntimeShape::ExtendedShape(4, unextended_prev_state_shape);
  const RuntimeShape output_state_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_state_shape);
  const RuntimeShape output_activ_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_activ_shape);
  const RuntimeShape concat_temp_shape =
      RuntimeShape::ExtendedShape(4, unextended_concat_temp_shape);
  const RuntimeShape activ_temp_shape =
      RuntimeShape::ExtendedShape(4, unextended_activ_temp_shape);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);

  const int weights_dim_count = weights_shape.DimensionsCount();
  MatchingDim(  // batches
      input_shape, 0, prev_activ_shape, 0, prev_state_shape, 0,
      output_state_shape, 0, output_activ_shape, 0);
  MatchingDim(  // height
      input_shape, 1, prev_activ_shape, 1, prev_state_shape, 1,
      output_state_shape, 1, output_activ_shape, 1);
  MatchingDim(  // width
      input_shape, 2, prev_activ_shape, 2, prev_state_shape, 2,
      output_state_shape, 2, output_activ_shape, 2);
  const int input_depth = input_shape.Dims(3);
  const int prev_activ_depth = prev_activ_shape.Dims(3);
  const int total_input_depth = prev_activ_depth + input_depth;
  TFLITE_DCHECK_EQ(weights_shape.Dims(weights_dim_count - 1),
                   total_input_depth);
  TFLITE_DCHECK_EQ(FlatSizeSkipDim(bias_shape, 3), 1);
  const int intern_activ_depth =
      MatchingDim(weights_shape, weights_dim_count - 2, bias_shape, 3);
  TFLITE_DCHECK_EQ(weights_shape.FlatSize(),
                   intern_activ_depth * total_input_depth);
  TFLITE_DCHECK_EQ(intern_activ_depth % 4, 0);
  const int output_depth =
      MatchingDim(prev_state_shape, 3, prev_activ_shape, 3, output_state_shape,
                  3, output_activ_shape, 3);
  TFLITE_DCHECK_EQ(output_depth, intern_activ_depth / 4);

  // Concatenate prev_activ and input data together
  std::vector<float const*> concat_input_arrays_data;
  std::vector<RuntimeShape const*> concat_input_arrays_shapes;
  concat_input_arrays_data.push_back(input_data);
  concat_input_arrays_data.push_back(prev_activ_data);
  concat_input_arrays_shapes.push_back(&input_shape);
  concat_input_arrays_shapes.push_back(&prev_activ_shape);
  tflite::ConcatenationParams concat_params;
  concat_params.axis = 3;
  concat_params.inputs_count = concat_input_arrays_data.size();
  Concatenation(concat_params, &(concat_input_arrays_shapes[0]),
                &(concat_input_arrays_data[0]), concat_temp_shape,
                concat_temp_data);

  // Fully connected
  tflite::FullyConnectedParams fc_params;
  fc_params.float_activation_min = std::numeric_limits<float>::lowest();
  fc_params.float_activation_max = std::numeric_limits<float>::max();
  FullyConnected(fc_params, concat_temp_shape, concat_temp_data, weights_shape,
                 weights_data, bias_shape, bias_data, activ_temp_shape,
                 activ_temp_data, cpu_backend_context);

  // Map raw arrays to Eigen arrays so we can use Eigen's optimized array
  // operations.
  ArrayMap<float> activ_temp_map =
      MapAsArrayWithLastDimAsRows(activ_temp_data, activ_temp_shape);
  auto input_gate_sm = activ_temp_map.block(0 * output_depth, 0, output_depth,
                                            activ_temp_map.cols());
  auto new_input_sm = activ_temp_map.block(1 * output_depth, 0, output_depth,
                                           activ_temp_map.cols());
  auto forget_gate_sm = activ_temp_map.block(2 * output_depth, 0, output_depth,
                                             activ_temp_map.cols());
  auto output_gate_sm = activ_temp_map.block(3 * output_depth, 0, output_depth,
                                             activ_temp_map.cols());
  ArrayMap<const float> prev_state_map =
      MapAsArrayWithLastDimAsRows(prev_state_data, prev_state_shape);
  ArrayMap<float> output_state_map =
      MapAsArrayWithLastDimAsRows(output_state_data, output_state_shape);
  ArrayMap<float> output_activ_map =
      MapAsArrayWithLastDimAsRows(output_activ_data, output_activ_shape);

  // Combined memory state and final output calculation
  gemmlowp::ScopedProfilingLabel label2("MemoryStateAndFinalOutput");
  output_state_map =
      input_gate_sm.unaryExpr(Eigen::internal::scalar_logistic_op<float>()) *
          new_input_sm.tanh() +
      forget_gate_sm.unaryExpr(Eigen::internal::scalar_logistic_op<float>()) *
          prev_state_map;
  output_activ_map =
      output_gate_sm.unaryExpr(Eigen::internal::scalar_logistic_op<float>()) *
      output_state_map.tanh();
}

template <int StateIntegerBits>
inline void LstmCell(
    const LstmCellParams& params, const RuntimeShape& unextended_input_shape,
    const uint8* input_data_uint8,
    const RuntimeShape& unextended_prev_activ_shape,
    const uint8* prev_activ_data_uint8, const RuntimeShape& weights_shape,
    const uint8* weights_data_uint8, const RuntimeShape& unextended_bias_shape,
    const int32* bias_data_int32,
    const RuntimeShape& unextended_prev_state_shape,
    const int16* prev_state_data_int16,
    const RuntimeShape& unextended_output_state_shape,
    int16* output_state_data_int16,
    const RuntimeShape& unextended_output_activ_shape,
    uint8* output_activ_data_uint8,
    const RuntimeShape& unextended_concat_temp_shape,
    uint8* concat_temp_data_uint8,
    const RuntimeShape& unextended_activ_temp_shape,
    int16* activ_temp_data_int16, CpuBackendContext* cpu_backend_context) {
  gemmlowp::ScopedProfilingLabel label(
      "LstmCell/quantized (8bit external, 16bit internal)");
  int32 weights_zero_point = params.weights_zero_point;
  int32 accum_multiplier = params.accum_multiplier;
  int accum_shift = params.accum_shift;
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_prev_activ_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_bias_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_prev_state_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_state_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_activ_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_concat_temp_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_activ_temp_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape prev_activ_shape =
      RuntimeShape::ExtendedShape(4, unextended_prev_activ_shape);
  const RuntimeShape bias_shape =
      RuntimeShape::ExtendedShape(4, unextended_bias_shape);
  const RuntimeShape prev_state_shape =
      RuntimeShape::ExtendedShape(4, unextended_prev_state_shape);
  const RuntimeShape output_state_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_state_shape);
  const RuntimeShape output_activ_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_activ_shape);
  const RuntimeShape concat_temp_shape =
      RuntimeShape::ExtendedShape(4, unextended_concat_temp_shape);
  const RuntimeShape activ_temp_shape =
      RuntimeShape::ExtendedShape(4, unextended_activ_temp_shape);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);

  // Gather dimensions information, and perform consistency checks.
  const int weights_dim_count = weights_shape.DimensionsCount();
  const int outer_size = MatchingFlatSizeSkipDim(
      input_shape, 3, prev_activ_shape, prev_state_shape, output_state_shape,
      output_activ_shape);
  const int input_depth = input_shape.Dims(3);
  const int prev_activ_depth = prev_activ_shape.Dims(3);
  const int total_input_depth = prev_activ_depth + input_depth;
  TFLITE_DCHECK_EQ(weights_shape.Dims(weights_dim_count - 1),
                   total_input_depth);
  const int intern_activ_depth =
      MatchingDim(weights_shape, weights_dim_count - 2, bias_shape, 3);
  TFLITE_DCHECK_EQ(weights_shape.FlatSize(),
                   intern_activ_depth * total_input_depth);
  TFLITE_DCHECK_EQ(FlatSizeSkipDim(bias_shape, 3), 1);
  TFLITE_DCHECK_EQ(intern_activ_depth % 4, 0);
  const int output_depth =
      MatchingDim(prev_state_shape, 3, prev_activ_shape, 3, output_state_shape,
                  3, output_activ_shape, 3);
  TFLITE_DCHECK_EQ(output_depth, intern_activ_depth / 4);
  const int fc_batches = FlatSizeSkipDim(activ_temp_shape, 3);
  const int fc_output_depth =
      MatchingDim(weights_shape, weights_dim_count - 2, activ_temp_shape, 3);
  const int fc_accum_depth = total_input_depth;
  TFLITE_DCHECK_EQ(fc_output_depth, 4 * output_depth);

  // Depth-concatenate prev_activ and input data together.
  uint8 const* concat_input_arrays_data[2] = {input_data_uint8,
                                              prev_activ_data_uint8};
  const RuntimeShape* concat_input_arrays_shapes[2] = {&input_shape,
                                                       &prev_activ_shape};
  tflite::ConcatenationParams concat_params;
  concat_params.axis = 3;
  concat_params.inputs_count = 2;
  Concatenation(concat_params, concat_input_arrays_shapes,
                concat_input_arrays_data, concat_temp_shape,
                concat_temp_data_uint8);

  // Implementation of the fully connected node inside the LSTM cell.
  // The operands are 8-bit integers, the accumulators are internally 32bit
  // integers, and the output is 16-bit fixed-point with 3 integer bits so
  // the output range is [-2^3, 2^3] == [-8, 8]. The rationale for that
  // is explained in the function comment above.
  cpu_backend_gemm::MatrixParams<uint8> lhs_params;
  lhs_params.rows = fc_output_depth;
  lhs_params.cols = fc_accum_depth;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.zero_point = weights_zero_point;
  cpu_backend_gemm::MatrixParams<uint8> rhs_params;
  rhs_params.rows = fc_accum_depth;
  rhs_params.cols = fc_batches;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.zero_point = 128;
  cpu_backend_gemm::MatrixParams<int16> dst_params;
  dst_params.rows = fc_output_depth;
  dst_params.cols = fc_batches;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.zero_point = 0;
  cpu_backend_gemm::GemmParams<int32, int16> gemm_params;
  gemm_params.bias = bias_data_int32;
  gemm_params.multiplier_fixedpoint = accum_multiplier;
  gemm_params.multiplier_exponent = accum_shift;
  cpu_backend_gemm::Gemm(
      lhs_params, weights_data_uint8, rhs_params, concat_temp_data_uint8,
      dst_params, activ_temp_data_int16, gemm_params, cpu_backend_context);

  // Rest of the LSTM cell: tanh and logistic math functions, and some adds
  // and muls, all done in 16-bit fixed-point.
  const int16* input_gate_input_ptr = activ_temp_data_int16;
  const int16* input_modulation_gate_input_ptr =
      activ_temp_data_int16 + output_depth;
  const int16* forget_gate_input_ptr = activ_temp_data_int16 + 2 * output_depth;
  const int16* output_gate_input_ptr = activ_temp_data_int16 + 3 * output_depth;
  const int16* prev_state_ptr = prev_state_data_int16;
  int16* output_state_data_ptr = output_state_data_int16;
  uint8* output_activ_data_ptr = output_activ_data_uint8;

  for (int b = 0; b < outer_size; ++b) {
    int c = 0;
#ifdef GEMMLOWP_NEON
    for (; c <= output_depth - 8; c += 8) {
      // Define the fixed-point data types that we will use here. All use
      // int16 as the underlying integer type i.e. all are 16-bit fixed-point.
      // They only differ by the number of integral vs. fractional bits,
      // determining the range of values that they can represent.
      //
      // F0 uses 0 integer bits, range [-1, 1].
      // This is the return type of math functions such as tanh, logistic,
      // whose range is in [-1, 1].
      using F0 = gemmlowp::FixedPoint<int16x8_t, 0>;
      // F3 uses 3 integer bits, range [-8, 8].
      // This is the range of the previous fully-connected node's output,
      // which is our input here.
      using F3 = gemmlowp::FixedPoint<int16x8_t, 3>;
      // FS uses StateIntegerBits integer bits, range [-2^StateIntegerBits,
      // 2^StateIntegerBits]. It's used to represent the internal state, whose
      // number of integer bits is currently dictated by the model. See comment
      // on the StateIntegerBits template parameter above.
      using FS = gemmlowp::FixedPoint<int16x8_t, StateIntegerBits>;
      // Implementation of input gate, using fixed-point logistic function.
      F3 input_gate_input = F3::FromRaw(vld1q_s16(input_gate_input_ptr));
      input_gate_input_ptr += 8;
      F0 input_gate_output = gemmlowp::logistic(input_gate_input);
      // Implementation of input modulation gate, using fixed-point tanh
      // function.
      F3 input_modulation_gate_input =
          F3::FromRaw(vld1q_s16(input_modulation_gate_input_ptr));
      input_modulation_gate_input_ptr += 8;
      F0 input_modulation_gate_output =
          gemmlowp::tanh(input_modulation_gate_input);
      // Implementation of forget gate, using fixed-point logistic function.
      F3 forget_gate_input = F3::FromRaw(vld1q_s16(forget_gate_input_ptr));
      forget_gate_input_ptr += 8;
      F0 forget_gate_output = gemmlowp::logistic(forget_gate_input);
      // Implementation of output gate, using fixed-point logistic function.
      F3 output_gate_input = F3::FromRaw(vld1q_s16(output_gate_input_ptr));
      output_gate_input_ptr += 8;
      F0 output_gate_output = gemmlowp::logistic(output_gate_input);
      // Implementation of internal multiplication nodes, still in fixed-point.
      F0 input_times_input_modulation =
          input_gate_output * input_modulation_gate_output;
      FS prev_state = FS::FromRaw(vld1q_s16(prev_state_ptr));
      prev_state_ptr += 8;
      FS prev_state_times_forget_state = forget_gate_output * prev_state;
      // Implementation of internal addition node, saturating.
      FS new_state = gemmlowp::SaturatingAdd(
          gemmlowp::Rescale<StateIntegerBits>(input_times_input_modulation),
          prev_state_times_forget_state);
      // Implementation of last internal Tanh node, still in fixed-point.
      // Since a Tanh fixed-point implementation is specialized for a given
      // number or integer bits, and each specialization can have a substantial
      // code size, and we already used above a Tanh on an input with 3 integer
      // bits, and per the table in the above function comment there is no
      // significant accuracy to be lost by clamping to [-8, +8] for a
      // 3-integer-bits representation, let us just do that. This helps people
      // porting this to targets where code footprint must be minimized.
      F3 new_state_f3 = gemmlowp::Rescale<3>(new_state);
      F0 output_activ_int16 = output_gate_output * gemmlowp::tanh(new_state_f3);
      // Store the new internal state back to memory, as 16-bit integers.
      // Note: here we store the original value with StateIntegerBits, not
      // the rescaled 3-integer-bits value fed to tanh.
      vst1q_s16(output_state_data_ptr, new_state.raw());
      output_state_data_ptr += 8;
      // Down-scale the output activations to 8-bit integers, saturating,
      // and store back to memory.
      int16x8_t rescaled_output_activ =
          gemmlowp::RoundingDivideByPOT(output_activ_int16.raw(), 8);
      int8x8_t int8_output_activ = vqmovn_s16(rescaled_output_activ);
      uint8x8_t uint8_output_activ =
          vadd_u8(vdup_n_u8(128), vreinterpret_u8_s8(int8_output_activ));
      vst1_u8(output_activ_data_ptr, uint8_output_activ);
      output_activ_data_ptr += 8;
    }
#endif
    for (; c < output_depth; ++c) {
      // Define the fixed-point data types that we will use here. All use
      // int16 as the underlying integer type i.e. all are 16-bit fixed-point.
      // They only differ by the number of integral vs. fractional bits,
      // determining the range of values that they can represent.
      //
      // F0 uses 0 integer bits, range [-1, 1].
      // This is the return type of math functions such as tanh, logistic,
      // whose range is in [-1, 1].
      using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
      // F3 uses 3 integer bits, range [-8, 8].
      // This is the range of the previous fully-connected node's output,
      // which is our input here.
      using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;
      // FS uses StateIntegerBits integer bits, range [-2^StateIntegerBits,
      // 2^StateIntegerBits]. It's used to represent the internal state, whose
      // number of integer bits is currently dictated by the model. See comment
      // on the StateIntegerBits template parameter above.
      using FS = gemmlowp::FixedPoint<std::int16_t, StateIntegerBits>;
      // Implementation of input gate, using fixed-point logistic function.
      F3 input_gate_input = F3::FromRaw(*input_gate_input_ptr++);
      F0 input_gate_output = gemmlowp::logistic(input_gate_input);
      // Implementation of input modulation gate, using fixed-point tanh
      // function.
      F3 input_modulation_gate_input =
          F3::FromRaw(*input_modulation_gate_input_ptr++);
      F0 input_modulation_gate_output =
          gemmlowp::tanh(input_modulation_gate_input);
      // Implementation of forget gate, using fixed-point logistic function.
      F3 forget_gate_input = F3::FromRaw(*forget_gate_input_ptr++);
      F0 forget_gate_output = gemmlowp::logistic(forget_gate_input);
      // Implementation of output gate, using fixed-point logistic function.
      F3 output_gate_input = F3::FromRaw(*output_gate_input_ptr++);
      F0 output_gate_output = gemmlowp::logistic(output_gate_input);
      // Implementation of internal multiplication nodes, still in fixed-point.
      F0 input_times_input_modulation =
          input_gate_output * input_modulation_gate_output;
      FS prev_state = FS::FromRaw(*prev_state_ptr++);
      FS prev_state_times_forget_state = forget_gate_output * prev_state;
      // Implementation of internal addition node, saturating.
      FS new_state = gemmlowp::SaturatingAdd(
          gemmlowp::Rescale<StateIntegerBits>(input_times_input_modulation),
          prev_state_times_forget_state);
      // Implementation of last internal Tanh node, still in fixed-point.
      // Since a Tanh fixed-point implementation is specialized for a given
      // number or integer bits, and each specialization can have a substantial
      // code size, and we already used above a Tanh on an input with 3 integer
      // bits, and per the table in the above function comment there is no
      // significant accuracy to be lost by clamping to [-8, +8] for a
      // 3-integer-bits representation, let us just do that. This helps people
      // porting this to targets where code footprint must be minimized.
      F3 new_state_f3 = gemmlowp::Rescale<3>(new_state);
      F0 output_activ_int16 = output_gate_output * gemmlowp::tanh(new_state_f3);
      // Store the new internal state back to memory, as 16-bit integers.
      // Note: here we store the original value with StateIntegerBits, not
      // the rescaled 3-integer-bits value fed to tanh.
      *output_state_data_ptr++ = new_state.raw();
      // Down-scale the output activations to 8-bit integers, saturating,
      // and store back to memory.
      int16 rescaled_output_activ =
          gemmlowp::RoundingDivideByPOT(output_activ_int16.raw(), 8);
      int16 clamped_output_activ =
          std::max<int16>(-128, std::min<int16>(127, rescaled_output_activ));
      *output_activ_data_ptr++ = 128 + clamped_output_activ;
    }
    input_gate_input_ptr += 3 * output_depth;
    input_modulation_gate_input_ptr += 3 * output_depth;
    forget_gate_input_ptr += 3 * output_depth;
    output_gate_input_ptr += 3 * output_depth;
  }
}

inline int NodeOffset(int b, int h, int w, int height, int width) {
  return (b * height + h) * width + w;
}

inline void AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const float* input_data,
                        const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("AveragePool");
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  // TODO(benoitjacob) make this a proper reference impl without Eigen!
  const auto in_mat = MapAsMatrixWithLastDimAsRows(input_data, input_shape);
  auto out_mat = MapAsMatrixWithLastDimAsRows(output_data, output_shape);
  // TODO(benoitjacob) get rid of the dynamic memory allocation here!
  Eigen::VectorXf out_count(out_mat.cols());
  out_count.setZero();
  // Prefill the output to 0.
  out_mat.setZero();
  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < input_height; ++h) {
      for (int w = 0; w < input_width; ++w) {
        // (h_start, h_end) * (w_start, w_end) is the range that the input
        // vector projects to.
        int hpad = h + params.padding_values.height;
        int wpad = w + params.padding_values.width;
        int h_start = (hpad < params.filter_height)
                          ? 0
                          : (hpad - params.filter_height) / stride_height + 1;
        int h_end = std::min(hpad / stride_height + 1, output_height);
        int w_start = (wpad < params.filter_width)
                          ? 0
                          : (wpad - params.filter_width) / stride_width + 1;
        int w_end = std::min(wpad / stride_width + 1, output_width);
        // compute elementwise sum
        for (int ph = h_start; ph < h_end; ++ph) {
          for (int pw = w_start; pw < w_end; ++pw) {
            int out_offset = NodeOffset(b, ph, pw, output_height, output_width);
            out_mat.col(out_offset) +=
                in_mat.col(NodeOffset(b, h, w, input_height, input_width));
            out_count(out_offset)++;
          }
        }
      }
    }
  }
  // Divide the output by the actual number of elements being averaged over
  TFLITE_DCHECK_GT(out_count.minCoeff(), 0);
  out_mat.array().rowwise() /= out_count.transpose().array();

  const int flat_size = output_shape.FlatSize();
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(output_data[i],
                                                  params.float_activation_min,
                                                  params.float_activation_max);
  }
}

inline void AveragePool16(const PoolParams& params,
                          const RuntimeShape& input_shape,
                          const uint8* input_data,
                          const RuntimeShape& output_shape,
                          uint8* output_data) {
  gemmlowp::ScopedProfilingLabel label("AveragePool/8bit");

  // Here, and in other pooling ops, in order to maintain locality of reference,
  // to minimize some recalculations, and to load into NEON vector registers, we
  // use an inner loop down the depth. Since depths can be large and hence we
  // would need arbitrarily large temporary storage, we divide the work up into
  // depth tranches just within the batch loop.
  static constexpr int kPoolingAccTrancheSize = 256;

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  uint16 acc[kPoolingAccTrancheSize];
  for (int batch = 0; batch < batches; ++batch) {
    // We proceed through the depth in tranches (see comment above). The
    // depth_base is the depth at the beginning of the tranche. The
    // tranche_depth is the depth dimension of the tranche.
    for (int depth_base = 0; depth_base < depth;
         depth_base += kPoolingAccTrancheSize) {
      const int tranche_depth =
          std::min(depth - depth_base, kPoolingAccTrancheSize);
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          const int filter_count =
              (filter_x_end - filter_x_start) * (filter_y_end - filter_y_start);
          memset(acc, 0, tranche_depth * sizeof(acc[0]));
          const uint8* input_ptr =
              input_data + depth_base +
              depth * (in_x_origin +
                       input_width * (in_y_origin + input_height * batch));
          for (int fy = filter_y_start; fy < filter_y_end; fy++) {
            const uint8* input_row_ptr =
                input_ptr + depth * (fy * input_width + filter_x_start);
            for (int fx = filter_x_start; fx < filter_x_end; fx++) {
              const uint8* input_channel_ptr = input_row_ptr;
              int channel = 0;
#ifdef USE_NEON
              for (; channel <= tranche_depth - 16; channel += 16) {
                uint16x8_t acc_reg[2];
                for (int i = 0; i < 2; i++) {
                  acc_reg[i] = vld1q_u16(acc + channel + 8 * i);
                }
                uint8x16_t input_reg = vld1q_u8(input_channel_ptr);
                input_channel_ptr += 16;
                acc_reg[0] = vaddw_u8(acc_reg[0], vget_low_u8(input_reg));
                acc_reg[1] = vaddw_u8(acc_reg[1], vget_high_u8(input_reg));
                for (int i = 0; i < 2; i++) {
                  vst1q_u16(acc + channel + 8 * i, acc_reg[i]);
                }
              }
              for (; channel <= tranche_depth - 8; channel += 8) {
                uint16x8_t acc_reg = vld1q_u16(acc + channel);
                uint8x8_t input_reg = vld1_u8(input_channel_ptr);
                input_channel_ptr += 8;
                acc_reg = vaddw_u8(acc_reg, input_reg);
                vst1q_u16(acc + channel, acc_reg);
              }
#endif
              for (; channel < tranche_depth; ++channel) {
                acc[channel] += *input_channel_ptr++;
              }
              input_row_ptr += depth;
            }
          }
          uint8* output_ptr = output_data + Offset(output_shape, batch, out_y,
                                                   out_x, depth_base);
          int channel = 0;
#ifdef USE_NEON
#define AVGPOOL_DIVIDING_BY(FILTER_COUNT)                               \
  if (filter_count == FILTER_COUNT) {                                   \
    for (; channel <= tranche_depth - 8; channel += 8) {                \
      uint16 buf[8];                                                    \
      for (int i = 0; i < 8; i++) {                                     \
        buf[i] = (acc[channel + i] + FILTER_COUNT / 2) / FILTER_COUNT;  \
      }                                                                 \
      uint8x8_t buf8 = vqmovn_u16(vld1q_u16(buf));                      \
      buf8 = vmin_u8(buf8, vdup_n_u8(params.quantized_activation_max)); \
      buf8 = vmax_u8(buf8, vdup_n_u8(params.quantized_activation_min)); \
      vst1_u8(output_ptr + channel, buf8);                              \
    }                                                                   \
  }
          AVGPOOL_DIVIDING_BY(9)
          AVGPOOL_DIVIDING_BY(15)
#undef AVGPOOL_DIVIDING_BY
          for (; channel <= tranche_depth - 8; channel += 8) {
            uint16 buf[8];
            for (int i = 0; i < 8; i++) {
              buf[i] = (acc[channel + i] + filter_count / 2) / filter_count;
            }
            uint8x8_t buf8 = vqmovn_u16(vld1q_u16(buf));
            buf8 = vmin_u8(buf8, vdup_n_u8(params.quantized_activation_max));
            buf8 = vmax_u8(buf8, vdup_n_u8(params.quantized_activation_min));
            vst1_u8(output_ptr + channel, buf8);
          }
#endif
          for (; channel < tranche_depth; ++channel) {
            uint16 a = (acc[channel] + filter_count / 2) / filter_count;
            a = std::max<uint16>(a, params.quantized_activation_min);
            a = std::min<uint16>(a, params.quantized_activation_max);
            output_ptr[channel] = static_cast<uint8>(a);
          }
        }
      }
    }
  }
}

inline void AveragePool32(const PoolParams& params,
                          const RuntimeShape& input_shape,
                          const uint8* input_data,
                          const RuntimeShape& output_shape,
                          uint8* output_data) {
  gemmlowp::ScopedProfilingLabel label("AveragePool/8bit");

  // Here, and in other pooling ops, in order to maintain locality of reference,
  // to minimize some recalculations, and to load into NEON vector registers, we
  // use an inner loop down the depth. Since depths can be large and hence we
  // would need arbitrarily large temporary storage, we divide the work up into
  // depth tranches just within the batch loop.
  static constexpr int kPoolingAccTrancheSize = 256;

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  uint32 acc[kPoolingAccTrancheSize];
  for (int batch = 0; batch < batches; ++batch) {
    // We proceed through the depth in tranches (see comment above). The
    // depth_base is the depth at the beginning of the tranche. The
    // tranche_depth is the depth dimension of the tranche.
    for (int depth_base = 0; depth_base < depth;
         depth_base += kPoolingAccTrancheSize) {
      const int tranche_depth =
          std::min(depth - depth_base, kPoolingAccTrancheSize);
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          const int filter_count =
              (filter_x_end - filter_x_start) * (filter_y_end - filter_y_start);
          memset(acc, 0, tranche_depth * sizeof(acc[0]));
          const uint8* input_ptr =
              input_data + depth_base +
              depth * (in_x_origin +
                       input_width * (in_y_origin + input_height * batch));
          for (int fy = filter_y_start; fy < filter_y_end; fy++) {
            const uint8* input_row_ptr =
                input_ptr + depth * (fy * input_width + filter_x_start);
            for (int fx = filter_x_start; fx < filter_x_end; fx++) {
              const uint8* input_channel_ptr = input_row_ptr;
              int channel = 0;
#ifdef USE_NEON
              for (; channel <= tranche_depth - 16; channel += 16) {
                uint16x4_t acc_reg[4];
                uint8x16_t input_reg = vld1q_u8(input_channel_ptr);
                input_channel_ptr += 16;
                acc_reg[0] = vget_low_u16(vmovl_u8(vget_low_u8(input_reg)));
                acc_reg[1] = vget_high_u16(vmovl_u8(vget_low_u8(input_reg)));
                acc_reg[2] = vget_low_u16(vmovl_u8(vget_high_u8(input_reg)));
                acc_reg[3] = vget_high_u16(vmovl_u8(vget_high_u8(input_reg)));
                for (int i = 0; i < 4; i++) {
                  vst1q_u32(
                      acc + channel + 4 * i,
                      vaddw_u16(vld1q_u32(acc + channel + 4 * i), acc_reg[i]));
                }
              }
              for (; channel <= tranche_depth - 8; channel += 8) {
                uint16x4_t acc_reg[2];
                uint16x8_t input_reg = vmovl_u8(vld1_u8(input_channel_ptr));
                input_channel_ptr += 8;
                acc_reg[0] = vget_low_u16(input_reg);
                acc_reg[1] = vget_high_u16(input_reg);
                for (int i = 0; i < 2; i++) {
                  vst1q_u32(
                      acc + channel + 4 * i,
                      vaddw_u16(vld1q_u32(acc + channel + 4 * i), acc_reg[i]));
                }
              }
#endif
              for (; channel < tranche_depth; ++channel) {
                acc[channel] += *input_channel_ptr++;
              }
              input_row_ptr += depth;
            }
          }
          uint8* output_ptr = output_data + Offset(output_shape, batch, out_y,
                                                   out_x, depth_base);
          int channel = 0;
#ifdef USE_NEON
#define AVGPOOL_DIVIDING_BY(FILTER_COUNT)                               \
  if (filter_count == FILTER_COUNT) {                                   \
    for (; channel <= tranche_depth - 8; channel += 8) {                \
      uint16 buf[8];                                                    \
      for (int i = 0; i < 8; i++) {                                     \
        buf[i] = (acc[channel + i] + FILTER_COUNT / 2) / FILTER_COUNT;  \
      }                                                                 \
      uint8x8_t buf8 = vqmovn_u16(vld1q_u16(buf));                      \
      buf8 = vmin_u8(buf8, vdup_n_u8(params.quantized_activation_max)); \
      buf8 = vmax_u8(buf8, vdup_n_u8(params.quantized_activation_min)); \
      vst1_u8(output_ptr + channel, buf8);                              \
    }                                                                   \
  }
          AVGPOOL_DIVIDING_BY(9)
          AVGPOOL_DIVIDING_BY(15)
#undef AVGPOOL_DIVIDING_BY
          for (; channel <= tranche_depth - 8; channel += 8) {
            uint16 buf[8];
            for (int i = 0; i < 8; i++) {
              buf[i] = (acc[channel + i] + filter_count / 2) / filter_count;
            }
            uint8x8_t buf8 = vqmovn_u16(vld1q_u16(buf));
            buf8 = vmin_u8(buf8, vdup_n_u8(params.quantized_activation_max));
            buf8 = vmax_u8(buf8, vdup_n_u8(params.quantized_activation_min));
            vst1_u8(output_ptr + channel, buf8);
          }
#endif
          for (; channel < tranche_depth; ++channel) {
            uint16 a = (acc[channel] + filter_count / 2) / filter_count;
            a = std::max<uint16>(a, params.quantized_activation_min);
            a = std::min<uint16>(a, params.quantized_activation_max);
            output_ptr[channel] = static_cast<uint8>(a);
          }
        }
      }
    }
  }
}

inline void AveragePool(const PoolParams& params,
                        const RuntimeShape& input_shape,
                        const uint8* input_data,
                        const RuntimeShape& output_shape, uint8* output_data) {
  if (params.filter_height * params.filter_width > 16 * 16) {
    AveragePool32(params, input_shape, input_data, output_shape, output_data);
  } else {
    AveragePool16(params, input_shape, input_data, output_shape, output_data);
  }
}

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const float* input_data, const RuntimeShape& output_shape,
                    float* output_data) {
  gemmlowp::ScopedProfilingLabel label("MaxPool");
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  const auto in_mat = MapAsMatrixWithLastDimAsRows(input_data, input_shape);
  auto out_mat = MapAsMatrixWithLastDimAsRows(output_data, output_shape);
  // Prefill the output to minimum representable float value
  out_mat.setConstant(std::numeric_limits<float>::lowest());
  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < input_height; ++h) {
      for (int w = 0; w < input_width; ++w) {
        // (h_start, h_end) * (w_start, w_end) is the range that the input
        // vector projects to.
        int hpad = h + params.padding_values.height;
        int wpad = w + params.padding_values.width;
        int h_start = (hpad < params.filter_height)
                          ? 0
                          : (hpad - params.filter_height) / stride_height + 1;
        int h_end = std::min(hpad / stride_height + 1, output_height);
        int w_start = (wpad < params.filter_width)
                          ? 0
                          : (wpad - params.filter_width) / stride_width + 1;
        int w_end = std::min(wpad / stride_width + 1, output_width);
        // compute elementwise sum
        for (int ph = h_start; ph < h_end; ++ph) {
          for (int pw = w_start; pw < w_end; ++pw) {
            int out_offset = NodeOffset(b, ph, pw, output_height, output_width);
            out_mat.col(out_offset) =
                out_mat.col(out_offset)
                    .cwiseMax(in_mat.col(
                        NodeOffset(b, h, w, input_height, input_width)));
          }
        }
      }
    }
  }
  const int flat_size = output_shape.FlatSize();
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(output_data[i],
                                                  params.float_activation_min,
                                                  params.float_activation_max);
  }
}

inline void MaxPool(const PoolParams& params, const RuntimeShape& input_shape,
                    const uint8* input_data, const RuntimeShape& output_shape,
                    uint8* output_data) {
  gemmlowp::ScopedProfilingLabel label("MaxPool/8bit");

  // Here, and in other pooling ops, in order to maintain locality of reference,
  // to minimize some recalculations, and to load into NEON vector registers, we
  // use an inner loop down the depth. Since depths can be large and hence we
  // would need arbitrarily large temporary storage, we divide the work up into
  // depth tranches just within the batch loop.
  static constexpr int kPoolingAccTrancheSize = 256;

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  uint8 acc[kPoolingAccTrancheSize];
  for (int batch = 0; batch < batches; ++batch) {
    // We proceed through the depth in tranches (see comment above). The
    // depth_base is the depth at the beginning of the tranche. The
    // tranche_depth is the depth dimension of the tranche.
    for (int depth_base = 0; depth_base < depth;
         depth_base += kPoolingAccTrancheSize) {
      const int tranche_depth =
          std::min(depth - depth_base, kPoolingAccTrancheSize);
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin =
              (out_x * stride_width) - params.padding_values.width;
          const int in_y_origin =
              (out_y * stride_height) - params.padding_values.height;
          const int filter_x_start = std::max(0, -in_x_origin);
          const int filter_x_end =
              std::min(params.filter_width, input_width - in_x_origin);
          const int filter_y_start = std::max(0, -in_y_origin);
          const int filter_y_end =
              std::min(params.filter_height, input_height - in_y_origin);
          memset(acc, 0, tranche_depth * sizeof(acc[0]));
          const uint8* input_ptr =
              input_data + depth_base +
              depth * (in_x_origin +
                       input_width * (in_y_origin + input_height * batch));
          for (int fy = filter_y_start; fy < filter_y_end; fy++) {
            const uint8* input_row_ptr =
                input_ptr + depth * (fy * input_width + filter_x_start);
            for (int fx = filter_x_start; fx < filter_x_end; fx++) {
              const uint8* input_channel_ptr = input_row_ptr;
              int channel = 0;
#ifdef USE_NEON
              for (; channel <= tranche_depth - 16; channel += 16) {
                uint8x16_t acc_reg = vld1q_u8(acc + channel);
                uint8x16_t input_reg = vld1q_u8(input_channel_ptr);
                input_channel_ptr += 16;
                acc_reg = vmaxq_u8(acc_reg, input_reg);
                vst1q_u8(acc + channel, acc_reg);
              }

              for (; channel <= tranche_depth - 8; channel += 8) {
                uint8x8_t acc_reg = vld1_u8(acc + channel);
                uint8x8_t input_reg = vld1_u8(input_channel_ptr);
                input_channel_ptr += 8;
                acc_reg = vmax_u8(acc_reg, input_reg);
                vst1_u8(acc + channel, acc_reg);
              }
#endif
              for (; channel < tranche_depth; ++channel) {
                acc[channel] = std::max(acc[channel], *input_channel_ptr++);
              }
              input_row_ptr += depth;
            }
          }
          uint8* output_ptr = output_data + Offset(output_shape, batch, out_y,
                                                   out_x, depth_base);
          int channel = 0;
#ifdef USE_NEON
          for (; channel <= tranche_depth - 16; channel += 16) {
            uint8x16_t a = vld1q_u8(acc + channel);
            a = vminq_u8(a, vdupq_n_u8(params.quantized_activation_max));
            a = vmaxq_u8(a, vdupq_n_u8(params.quantized_activation_min));
            vst1q_u8(output_ptr + channel, a);
          }
          for (; channel <= tranche_depth - 8; channel += 8) {
            uint8x8_t a = vld1_u8(acc + channel);
            a = vmin_u8(a, vdup_n_u8(params.quantized_activation_max));
            a = vmax_u8(a, vdup_n_u8(params.quantized_activation_min));
            vst1_u8(output_ptr + channel, a);
          }
#endif
          for (; channel < tranche_depth; ++channel) {
            uint8 a = acc[channel];
            a = std::max<uint8>(a, params.quantized_activation_min);
            a = std::min<uint8>(a, params.quantized_activation_max);
            output_ptr[channel] = static_cast<uint8>(a);
          }
        }
      }
    }
  }
}

inline void L2Pool(const PoolParams& params, const RuntimeShape& input_shape,
                   const float* input_data, const RuntimeShape& output_shape,
                   float* output_data) {
  gemmlowp::ScopedProfilingLabel label("L2Pool");
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  // Actually carry out L2 Pool. Code is written in forward mode: we go through
  // the input values once, and write to all the pooled regions that it maps to.
  const auto in_mat = MapAsMatrixWithLastDimAsRows(input_data, input_shape);
  auto out_mat = MapAsMatrixWithLastDimAsRows(output_data, output_shape);
  Eigen::VectorXf in_square(in_mat.rows());
  Eigen::VectorXf out_count(out_mat.cols());
  out_count.setZero();
  // Prefill the output to 0.
  out_mat.setZero();
  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < input_height; ++h) {
      for (int w = 0; w < input_width; ++w) {
        // (h_start, h_end) * (w_start, w_end) is the range that the input
        // vector projects to.
        const int hpad = h + params.padding_values.height;
        const int wpad = w + params.padding_values.width;
        const int h_start =
            (hpad < params.filter_height)
                ? 0
                : (hpad - params.filter_height) / stride_height + 1;
        const int h_end = std::min(hpad / stride_height + 1, output_height);
        const int w_start =
            (wpad < params.filter_width)
                ? 0
                : (wpad - params.filter_width) / stride_width + 1;
        const int w_end = std::min(wpad / stride_width + 1, output_width);
        // pre-compute square
        const int in_offset = w + input_width * (h + input_height * b);
        in_square =
            in_mat.col(in_offset).array() * in_mat.col(in_offset).array();
        // compute elementwise sum of squares
        for (int ph = h_start; ph < h_end; ++ph) {
          for (int pw = w_start; pw < w_end; ++pw) {
            const int out_offset = pw + output_width * (ph + output_height * b);
            out_mat.col(out_offset) += in_square;
            out_count(out_offset)++;
          }
        }
      }
    }
  }

  out_count = out_count.array().inverse();
  out_mat =
      (out_mat.array().rowwise() * out_count.transpose().array()).cwiseSqrt();

  const int flat_size = output_shape.FlatSize();
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(output_data[i],
                                                  params.float_activation_min,
                                                  params.float_activation_max);
  }
}

inline void LocalResponseNormalization(
    const tflite::LocalResponseNormalizationParams& op_params,
    const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("LocalResponseNormalization");
  MatchingFlatSize(input_shape, output_shape);

  const auto data_in = MapAsMatrixWithLastDimAsRows(input_data, input_shape);
  auto data_out = MapAsMatrixWithLastDimAsRows(output_data, output_shape);

  // Carry out local response normalization, vector by vector.
  // Since the data are stored column major, making row-wise operation
  // probably not memory efficient anyway, we do an explicit for loop over
  // the columns.
  const int double_range = op_params.range * 2;
  Eigen::VectorXf padded_square(data_in.rows() + double_range);
  padded_square.setZero();
  for (int r = 0; r < data_in.cols(); ++r) {
    // Do local response normalization for data_in(:, r)
    // first, compute the square and store them in buffer for repeated use
    padded_square.block(op_params.range, 0, data_in.rows(), 1) =
        data_in.col(r).cwiseProduct(data_in.col(r)) * op_params.alpha;
    // Then, compute the scale and writes them to data_out
    float accumulated_scale = 0;
    for (int i = 0; i < double_range; ++i) {
      accumulated_scale += padded_square(i);
    }
    for (int i = 0; i < data_in.rows(); ++i) {
      accumulated_scale += padded_square(i + double_range);
      data_out(i, r) = op_params.bias + accumulated_scale;
      accumulated_scale -= padded_square(i);
    }
  }

  // In a few cases, the pow computation could benefit from speedups.
  if (op_params.beta == 1) {
    data_out.array() = data_in.array() * data_out.array().inverse();
  } else if (op_params.beta == 0.5) {
    data_out.array() = data_in.array() * data_out.array().sqrt().inverse();
  } else {
    data_out.array() = data_in.array() * data_out.array().pow(-op_params.beta);
  }
}

inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const float* input_data,
                    const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("Softmax");
  MatchingFlatSize(input_shape, output_shape);

  const auto in_mat = MapAsMatrixWithLastDimAsRows(input_data, input_shape);
  auto out_mat = MapAsMatrixWithLastDimAsRows(output_data, output_shape);
  // Compute the exponential first, removing the max coefficient for numerical
  // stability.
  out_mat =
      (in_mat.rowwise() - in_mat.colwise().maxCoeff()).array() * params.beta;
  // We are separating out the exp function so that exp can be vectorized.
  out_mat = out_mat.array().exp();
  // Normalize to get the activations.
  Eigen::Array<float, 1, Eigen::Dynamic> scale =
      out_mat.array().colwise().sum().inverse();
  out_mat.array().rowwise() *= scale;
}

inline int32_t QuantizeSoftmaxOutput(int8_t* output_data, float prob_rescaled,
                                     int32_t zero_point) {
  const int32_t prob_rnd = static_cast<int32_t>(std::round(prob_rescaled));
  return prob_rnd + zero_point;
}

inline int32_t QuantizeSoftmaxOutput(uint8_t* output_data, float prob_rescaled,
                                     int32_t zero_point) {
  return static_cast<int32_t>(prob_rescaled + 0.5);
}

inline void PopulateSoftmaxLookupTable(SoftmaxParams* data, float input_scale,
                                       float beta) {
  const float scale = -input_scale * beta;
  const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
  for (int32_t val = 0; val <= max_uint8; ++val) {
    data->table[max_uint8 - val] = expf(scale * val);
  }
}

template <typename T>
inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const T* input_data,
                    const RuntimeShape& output_shape, T* output_data) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int excluding_last_dim =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int last_dim =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  const int32_t clamp_max = std::numeric_limits<T>::max();
  const int32_t clamp_min = std::numeric_limits<T>::min();
  for (int i = 0; i < excluding_last_dim; ++i) {
    int32_t max_val = std::numeric_limits<T>::min();
    // Find max quantized value.
    for (int j = 0; j < last_dim; ++j) {
      max_val = std::max(max_val, static_cast<int32_t>(input_data[j]));
    }

    float sum_exp = 0.0f;
    const int32_t max_uint8 = std::numeric_limits<uint8_t>::max();
    const float* table_offset = &params.table[max_uint8 - max_val];
    // Calculate normalizer sum(exp(x)).
    for (int j = 0; j < last_dim; ++j) {
      sum_exp += table_offset[input_data[j]];
    }

    const float inv_sum_exp = 1.0f / (sum_exp * params.scale);
    // Normalize and quantize probabilities.
    for (int j = 0; j < last_dim; ++j) {
      const float prob_rescaled = table_offset[input_data[j]] * inv_sum_exp;
      const int32_t prob_quantized =
          QuantizeSoftmaxOutput(output_data, prob_rescaled, params.zero_point);
      output_data[j] = static_cast<T>(
          std::max(std::min(clamp_max, prob_quantized), clamp_min));
    }
    input_data += last_dim;
    output_data += last_dim;
  }
}

// TODO(myenik): This is the same as the reference implementation, not actually
// optimized yet.
inline void LogSoftmax(const SoftmaxParams& params,
                       const RuntimeShape& input_shape, const float* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("LogSoftmax");
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    const float* block_input_data = input_data + i * depth;
    float* block_output_data = output_data + i * depth;
    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // log(exp(x[i])/sum(exp(x[i]))) == log(exp(x[i]+C)/sum(exp(x[i]+C)))
    float max = std::numeric_limits<float>::lowest();
    for (int c = 0; c < depth; ++c) {
      max = std::max(max, block_input_data[c]);
    }

    // Compute sum.
    float sum = 0.f;
    for (int c = 0; c < depth; ++c) {
      sum += std::exp(block_input_data[c] - max);
    }

    // Compute result.
    const float log_sum = std::log(sum);
    for (int c = 0; c < depth; ++c) {
      block_output_data[c] = block_input_data[c] - max - log_sum;
    }
  }
}

// Currently just a copy of the reference code.
inline void LogSoftmax(const SoftmaxParams& params,
                       const RuntimeShape& input_shape, const uint8* input_data,
                       const RuntimeShape& output_shape, uint8* output_data) {
  gemmlowp::ScopedProfilingLabel label("LogSoftmax/Uint8");
  const int32 input_multiplier = params.input_multiplier;
  const int32 input_left_shift = params.input_left_shift;
  const int32 reverse_scaling_divisor = params.reverse_scaling_divisor;
  const int32 reverse_scaling_right_shift = params.reverse_scaling_right_shift;
  const int diff_min = params.diff_min;
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static constexpr int kScaledDiffIntegerBits = 5;
  static constexpr int kAccumulationIntegerBits = 12;
  static constexpr int kOutputIntegerBits = 4;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32, kScaledDiffIntegerBits>;
  using FixedPointAccum = gemmlowp::FixedPoint<int32, kAccumulationIntegerBits>;

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    const uint8* block_input_data = input_data + i * depth;
    uint8* block_output_data = output_data + i * depth;
    uint8 max_in_row = 0;
    for (int c = 0; c < depth; ++c) {
      max_in_row = std::max(max_in_row, block_input_data[c]);
    }

    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    for (int c = 0; c < depth; ++c) {
      int32 input_diff = static_cast<int32>(block_input_data[c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32 input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_multiplier, input_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);
        sum_of_exps = sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                        exp_on_negative_values(scaled_diff_f8));
      }
    }

    const int32 fixed_log_sum_of_exps =
        log_x_for_x_greater_than_or_equal_to_1<kScaledDiffIntegerBits>(
            sum_of_exps)
            .raw();

    // rescaled_diff_min is smallest representable in
    // Q(kScaledDiffIntegerBits).(31-kScaledDiffIntegerBits) plus the
    // log-sub-exps that will be subtracted in the loop.
    //
    // The thresholds diff_min, etc are negative.
    const int rescaled_diff_min =
        fixed_log_sum_of_exps + std::numeric_limits<int32>::lowest();
    const int adjusted_diff_min =
        std::max(diff_min - 1,  // Note use of > below instead of >= above.
                 MultiplyByQuantizedMultiplierSmallerThanOneExp(
                     rescaled_diff_min, reverse_scaling_divisor,
                     -reverse_scaling_right_shift));

    for (int c = 0; c < depth; ++c) {
      int32 input_diff = static_cast<int32>(block_input_data[c]) - max_in_row;
      if (input_diff > adjusted_diff_min) {
        const int32 input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_multiplier, input_left_shift);
        int32 unsat_output =
            gemmlowp::RoundingDivideByPOT(
                (input_diff_rescaled - fixed_log_sum_of_exps),
                31 - kScaledDiffIntegerBits - kOutputIntegerBits) +
            255;

        block_output_data[c] = static_cast<uint8>(
            std::max(std::min(unsat_output, static_cast<int32>(255)), 0));
      } else {
        // Set output to smallest value.
        block_output_data[c] = 0;
      }
    }
  }
}

inline void Logistic(const RuntimeShape& input_shape, const float* input_data,
                     const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("Logistic");
  auto input_map = MapAsVector(input_data, input_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  output_map.array() =
      input_map.array().unaryExpr(Eigen::internal::scalar_logistic_op<float>());
}

// Convenience version that allows, for example, generated-code calls to be
// uniform between data types.
inline void Logistic(const LogisticParams&, const RuntimeShape& input_shape,
                     const float* input_data, const RuntimeShape& output_shape,
                     float* output_data) {
  // Drop params: not needed.
  Logistic(input_shape, input_data, output_shape, output_data);
}

inline void Logistic(const LogisticParams& params,
                     const RuntimeShape& input_shape, const int16* input_data,
                     const RuntimeShape& output_shape, int16* output_data) {
  gemmlowp::ScopedProfilingLabel label("Logistic/Int16");
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
  }

  int c = 0;
  const int16* input_data_ptr = input_data;
  int16* output_data_ptr = output_data;
#ifdef GEMMLOWP_NEON
  {
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<int16x8_t, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<int16x8_t, 3>;

    for (; c <= flat_size - 16; c += 16) {
      F3 input0 = F3::FromRaw(vld1q_s16(input_data_ptr));
      F3 input1 = F3::FromRaw(vld1q_s16(input_data_ptr + 8));
      F0 output0 = gemmlowp::logistic(input0);
      F0 output1 = gemmlowp::logistic(input1);
      vst1q_s16(output_data_ptr, output0.raw());
      vst1q_s16(output_data_ptr + 8, output1.raw());

      input_data_ptr += 16;
      output_data_ptr += 16;
    }
    for (; c <= flat_size - 8; c += 8) {
      F3 input = F3::FromRaw(vld1q_s16(input_data_ptr));
      F0 output = gemmlowp::logistic(input);
      vst1q_s16(output_data_ptr, output.raw());

      input_data_ptr += 8;
      output_data_ptr += 8;
    }
  }
#endif
  {
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;

    for (; c < flat_size; ++c) {
      F3 input = F3::FromRaw(*input_data_ptr);
      F0 output = gemmlowp::logistic(input);
      *output_data_ptr = output.raw();

      ++input_data_ptr;
      ++output_data_ptr;
    }
  }
}

inline void Tanh(const RuntimeShape& input_shape, const float* input_data,
                 const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("Tanh");
  auto input_map = MapAsVector(input_data, input_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  output_map.array() = input_map.array().tanh();
}

// Convenience version that allows, for example, generated-code calls to be
// uniform between data types.
inline void Tanh(const TanhParams&, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& output_shape,
                 float* output_data) {
  // Drop params: not needed.
  Tanh(input_shape, input_data, output_shape, output_data);
}

inline void Tanh(const TanhParams& params, const RuntimeShape& input_shape,
                 const int16* input_data, const RuntimeShape& output_shape,
                 int16* output_data) {
  gemmlowp::ScopedProfilingLabel label("Tanh/Int16");
  const int input_left_shift = params.input_left_shift;
  // Support for shifts is limited until we have a parameterized version of
  // SaturatingRoundingMultiplyByPOT().
  TFLITE_DCHECK_GE(input_left_shift, 0);
  TFLITE_DCHECK_LE(input_left_shift, 1);

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
  const int16* input_data_ptr = input_data;
  int16* output_data_ptr = output_data;
#ifdef GEMMLOWP_NEON
  {
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<int16x8_t, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<int16x8_t, 3>;

    if (input_left_shift == 0) {
      for (; c <= flat_size - 16; c += 16) {
        F3 input0 = F3::FromRaw(vld1q_s16(input_data_ptr));
        F3 input1 = F3::FromRaw(vld1q_s16(input_data_ptr + 8));
        F0 output0 = gemmlowp::tanh(input0);
        F0 output1 = gemmlowp::tanh(input1);
        vst1q_s16(output_data_ptr, output0.raw());
        vst1q_s16(output_data_ptr + 8, output1.raw());

        input_data_ptr += 16;
        output_data_ptr += 16;
      }
      for (; c <= flat_size - 8; c += 8) {
        F3 input = F3::FromRaw(vld1q_s16(input_data_ptr));
        F0 output = gemmlowp::tanh(input);
        vst1q_s16(output_data_ptr, output.raw());

        input_data_ptr += 8;
        output_data_ptr += 8;
      }
    } else {
      for (; c <= flat_size - 16; c += 16) {
        F3 input0 = F3::FromRaw(gemmlowp::SaturatingRoundingMultiplyByPOT<1>(
            vld1q_s16(input_data_ptr)));
        F3 input1 = F3::FromRaw(gemmlowp::SaturatingRoundingMultiplyByPOT<1>(
            vld1q_s16(input_data_ptr + 8)));
        F0 output0 = gemmlowp::tanh(input0);
        F0 output1 = gemmlowp::tanh(input1);
        vst1q_s16(output_data_ptr, output0.raw());
        vst1q_s16(output_data_ptr + 8, output1.raw());

        input_data_ptr += 16;
        output_data_ptr += 16;
      }
      for (; c <= flat_size - 8; c += 8) {
        F3 input = F3::FromRaw(gemmlowp::SaturatingRoundingMultiplyByPOT<1>(
            vld1q_s16(input_data_ptr)));
        F0 output = gemmlowp::tanh(input);
        vst1q_s16(output_data_ptr, output.raw());

        input_data_ptr += 8;
        output_data_ptr += 8;
      }
    }
  }
#endif
  {
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;

    if (input_left_shift == 0) {
      for (; c < flat_size; ++c) {
        F3 input = F3::FromRaw(*input_data_ptr);
        F0 output = gemmlowp::tanh(input);
        *output_data_ptr = output.raw();

        ++input_data_ptr;
        ++output_data_ptr;
      }
    } else {
      for (; c < flat_size; ++c) {
        F3 input = F3::FromRaw(
            gemmlowp::SaturatingRoundingMultiplyByPOT<1>(*input_data_ptr));
        F0 output = gemmlowp::tanh(input);
        *output_data_ptr = output.raw();

        ++input_data_ptr;
        ++output_data_ptr;
      }
    }
  }
}

template <typename SrcT, typename DstT>
inline void Cast(const RuntimeShape& input_shape, const SrcT* input_data,
                 const RuntimeShape& output_shape, DstT* output_data) {
  gemmlowp::ScopedProfilingLabel label("Cast");
  auto input_map = MapAsVector(input_data, input_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  output_map.array() = input_map.array().template cast<DstT>();
}

inline void Floor(const RuntimeShape& input_shape, const float* input_data,
                  const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("Floor");
  auto input_map = MapAsVector(input_data, input_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  output_map.array() = Eigen::floor(input_map.array());
}

inline void Ceil(const RuntimeShape& input_shape, const float* input_data,
                 const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("Ceil");
  auto input_map = MapAsVector(input_data, input_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  output_map.array() = Eigen::ceil(input_map.array());
}

#ifdef USE_NEON
inline void ResizeBilinearKernel(const float* input_ptr, int32 depth,
                                 float scale, float* output_ptr) {
  int ic = 0;
  // Handle 32 input channels at a time.
  for (; ic <= depth - 32; ic += 32) {
    float32x4x2_t input[4];
    for (int i = 0; i < 4; i++) {
      input[i].val[0] = vld1q_f32(input_ptr + 8 * i);
      input[i].val[1] = vld1q_f32(input_ptr + 8 * i + 4);
    }
    float32x4x2_t acc[4];
    for (int i = 0; i < 4; i++) {
      acc[i].val[0] = vld1q_f32(output_ptr + 8 * i);
      acc[i].val[1] = vld1q_f32(output_ptr + 8 * i + 4);
    }
    for (int i = 0; i < 4; i++) {
      acc[i].val[0] = vmlaq_n_f32(acc[i].val[0], input[i].val[0], scale);
      acc[i].val[1] = vmlaq_n_f32(acc[i].val[1], input[i].val[1], scale);
    }
    for (int i = 0; i < 4; i++) {
      vst1q_f32(output_ptr, acc[i].val[0]);
      vst1q_f32(output_ptr + 4, acc[i].val[1]);
      output_ptr += 8;
    }
    input_ptr += 32;
  }
  // Handle 16 input channels at a time.
  for (; ic <= depth - 16; ic += 16) {
    float32x4x2_t input[2];
    for (int i = 0; i < 2; i++) {
      input[i].val[0] = vld1q_f32(input_ptr + 8 * i);
      input[i].val[1] = vld1q_f32(input_ptr + 8 * i + 4);
    }
    float32x4x2_t acc[2];
    for (int i = 0; i < 2; i++) {
      acc[i].val[0] = vld1q_f32(output_ptr + 8 * i);
      acc[i].val[1] = vld1q_f32(output_ptr + 8 * i + 4);
    }
    for (int i = 0; i < 2; i++) {
      acc[i].val[0] = vmlaq_n_f32(acc[i].val[0], input[i].val[0], scale);
      acc[i].val[1] = vmlaq_n_f32(acc[i].val[1], input[i].val[1], scale);
    }
    for (int i = 0; i < 2; i++) {
      vst1q_f32(output_ptr, acc[i].val[0]);
      vst1q_f32(output_ptr + 4, acc[i].val[1]);
      output_ptr += 8;
    }
    input_ptr += 16;
  }
  // Handle 8 input channels at a time.
  for (; ic <= depth - 8; ic += 8) {
    float32x4x2_t input;
    input.val[0] = vld1q_f32(input_ptr);
    input.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t acc;
    acc.val[0] = vld1q_f32(output_ptr);
    acc.val[1] = vld1q_f32(output_ptr + 4);
    acc.val[0] = vmlaq_n_f32(acc.val[0], input.val[0], scale);
    acc.val[1] = vmlaq_n_f32(acc.val[1], input.val[1], scale);

    vst1q_f32(output_ptr, acc.val[0]);
    vst1q_f32(output_ptr + 4, acc.val[1]);

    input_ptr += 8;
    output_ptr += 8;
  }
  // Handle 4 input channels at a time.
  for (; ic <= depth - 4; ic += 4) {
    float32x4_t input = vld1q_f32(input_ptr);
    float32x4_t acc = vld1q_f32(output_ptr);

    acc = vmlaq_n_f32(acc, input, scale);
    vst1q_f32(output_ptr, acc);

    input_ptr += 4;
    output_ptr += 4;
  }
  // Handle 1 input channel at a time.
  for (; ic < depth; ic++) {
    *output_ptr += *input_ptr * scale;
    output_ptr++;
    input_ptr++;
  }
}
#else
inline void ResizeBilinearKernel(const float* input_ptr, int32 depth,
                                 float scale, float* output_ptr) {
  for (int32 i = 0; i < depth; i++) {
    *output_ptr += *input_ptr * scale;
    output_ptr++;
    input_ptr++;
  }
}
#endif

inline void ResizeBilinearKernel2x2(int32 x0, int32 x1, int32 y0, int32 y1,
                                    int32 x, int32 y, int32 depth, int32 batch,
                                    const RuntimeShape& input_shape,
                                    const float* input_data,
                                    const RuntimeShape& output_shape,
                                    float* output_data) {
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int32 input_width = input_shape.Dims(2);
  const int32 output_width = output_shape.Dims(2);

  const int32 input_x_offset = (x1 - x0) * depth;
  const int32 input_y_offset = (y1 - y0) * depth * input_width;
  const int32 output_x_offset = depth;
  const int32 output_y_offset = depth * output_width;

#ifdef USE_NEON
  TFLITE_DCHECK(x1 >= x0);
  TFLITE_DCHECK(y1 >= y0);

  int ic = 0;
  // Handle 8 input channels at a time.
  for (; ic <= depth - 8; ic += 8) {
    const float* input_ptr = nullptr;

    float32x4x2_t x0y0;
    input_ptr = &input_data[Offset(input_shape, batch, y0, x0, ic)];
    x0y0.val[0] = vld1q_f32(input_ptr);
    x0y0.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t x1y0;
    input_ptr += input_x_offset;
    x1y0.val[0] = vld1q_f32(input_ptr);
    x1y0.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t x0y1;
    input_ptr += -input_x_offset + input_y_offset;
    x0y1.val[0] = vld1q_f32(input_ptr);
    x0y1.val[1] = vld1q_f32(input_ptr + 4);

    float32x4x2_t x1y1;
    input_ptr += input_x_offset;
    x1y1.val[0] = vld1q_f32(input_ptr);
    x1y1.val[1] = vld1q_f32(input_ptr + 4);

    // Top left corner.
    float* output_ptr = &output_data[Offset(output_shape, batch, y, x, ic)];
    vst1q_f32(output_ptr, x0y0.val[0]);
    vst1q_f32(output_ptr + 4, x0y0.val[1]);

    // Top right corner.
    output_ptr += output_x_offset;
    float32x4x2_t tr;
    tr.val[0] = vaddq_f32(x0y0.val[0], x1y0.val[0]);
    tr.val[1] = vaddq_f32(x0y0.val[1], x1y0.val[1]);
    tr.val[0] = vmulq_n_f32(tr.val[0], 0.5f);
    tr.val[1] = vmulq_n_f32(tr.val[1], 0.5f);

    vst1q_f32(output_ptr, tr.val[0]);
    vst1q_f32(output_ptr + 4, tr.val[1]);

    // Bottom left corner.
    output_ptr += -output_x_offset + output_y_offset;
    float32x4x2_t bl;
    bl.val[0] = vaddq_f32(x0y0.val[0], x0y1.val[0]);
    bl.val[1] = vaddq_f32(x0y0.val[1], x0y1.val[1]);
    bl.val[0] = vmulq_n_f32(bl.val[0], 0.5f);
    bl.val[1] = vmulq_n_f32(bl.val[1], 0.5f);
    vst1q_f32(output_ptr, bl.val[0]);
    vst1q_f32(output_ptr + 4, bl.val[1]);

    // Bottom right corner.
    output_ptr += output_x_offset;
    float32x4x2_t br;
    br.val[0] = vaddq_f32(x1y0.val[0], x1y1.val[0]);
    br.val[1] = vaddq_f32(x1y0.val[1], x1y1.val[1]);
    br.val[0] = vmlaq_n_f32(bl.val[0], br.val[0], 0.5f);
    br.val[1] = vmlaq_n_f32(bl.val[1], br.val[1], 0.5f);
    br.val[0] = vmulq_n_f32(br.val[0], 0.5f);
    br.val[1] = vmulq_n_f32(br.val[1], 0.5f);
    vst1q_f32(output_ptr, br.val[0]);
    vst1q_f32(output_ptr + 4, br.val[1]);
  }
  // Handle 4 input channels at a time.
  for (; ic <= depth - 4; ic += 4) {
    const float* input_ptr =
        &input_data[Offset(input_shape, batch, y0, x0, ic)];
    float32x4_t x0y0 = vld1q_f32(input_ptr);
    float32x4_t x1y0 = vld1q_f32(input_ptr + input_x_offset);
    float32x4_t x0y1 = vld1q_f32(input_ptr + input_y_offset);
    float32x4_t x1y1 = vld1q_f32(input_ptr + input_x_offset + input_y_offset);

    // Top left corner.
    float* output_ptr = &output_data[Offset(output_shape, batch, y, x, ic)];
    vst1q_f32(output_ptr, x0y0);

    // Top right corner.
    output_ptr += output_x_offset;
    float32x4_t tr = vaddq_f32(x0y0, x1y0);
    tr = vmulq_n_f32(tr, 0.5f);
    vst1q_f32(output_ptr, tr);

    // Bottom left corner.
    output_ptr += -output_x_offset + output_y_offset;
    float32x4_t bl = vaddq_f32(x0y0, x0y1);
    bl = vmulq_n_f32(bl, 0.5f);
    vst1q_f32(output_ptr, bl);

    // Bottom right corner.
    output_ptr += output_x_offset;
    float32x4_t br = vaddq_f32(x1y0, x1y1);
    br = vmlaq_n_f32(bl, br, 0.5f);
    br = vmulq_n_f32(br, 0.5f);
    vst1q_f32(output_ptr, br);
  }
  // Handle one input channel at a time.
  for (; ic < depth; ic++) {
    const int32 input_offset = Offset(input_shape, batch, y0, x0, ic);

    float x0y0 = input_data[input_offset];
    float x1y0 = input_data[input_offset + input_x_offset];
    float x0y1 = input_data[input_offset + input_y_offset];
    float x1y1 = input_data[input_offset + input_x_offset + input_y_offset];

    // Top left corner.
    const int32 output_offset = Offset(output_shape, batch, y, x, ic);
    output_data[output_offset] = x0y0;

    // Top right corner.
    output_data[output_offset + output_x_offset] = (x0y0 + x1y0) / 2;

    // Bottom left corner.
    float output = (x0y0 + x0y1) / 2;
    output_data[output_offset + output_y_offset] = output;

    // Bottom right corner.
    output_data[output_offset + output_x_offset + output_y_offset] =
        (output + ((x1y0 + x1y1) / 2)) / 2;
  }
#else
  for (int ch = 0; ch < depth; ch++) {
    const int32 input_offset = Offset(input_shape, batch, y0, x0, ch);

    float x0y0 = input_data[input_offset];
    float x1y0 = input_data[input_offset + input_x_offset];
    float x0y1 = input_data[input_offset + input_y_offset];
    float x1y1 = input_data[input_offset + input_x_offset + input_y_offset];

    // Top left corner.
    const int32 output_offset = Offset(output_shape, batch, y, x, ch);
    output_data[output_offset] = x0y0;

    // Top right corner.
    output_data[output_offset + output_x_offset] = (x0y0 + x1y0) / 2;

    // Bottom left corner.
    float output = (x0y0 + x0y1) / 2;
    output_data[output_offset + output_y_offset] = output;

    // Bottom right corner.
    output_data[output_offset + output_x_offset + output_y_offset] =
        (output + ((x1y0 + x1y1) / 2)) / 2;
  }
#endif
}

inline void ResizeBilinear2x2(int32 batches, int32 input_height,
                              int32 input_width, int32 depth,
                              int32 output_height, int32 output_width,
                              const RuntimeShape& input_shape,
                              const float* input_data,
                              const RuntimeShape& output_shape,
                              float* output_data) {
  for (int b = 0; b < batches; b++) {
    for (int y0 = 0, y = 0; y <= output_height - 2; y += 2, y0++) {
      for (int x0 = 0, x = 0; x <= output_width - 2; x += 2, x0++) {
        int32 x1 = std::min(x0 + 1, input_width - 1);
        int32 y1 = std::min(y0 + 1, input_height - 1);
        ResizeBilinearKernel2x2(x0, x1, y0, y1, x, y, depth, b, input_shape,
                                input_data, output_shape, output_data);
      }
    }
  }
}

inline void ResizeBilinearGeneric(
    int32 batches, int32 input_height, int32 input_width, int32 depth,
    int32 output_height, int32 output_width, float height_scale,
    float width_scale, const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& output_shape, float* output_data) {
  memset(output_data, 0,
         batches * output_height * output_width * depth * sizeof(float));

  int32 output_offset = 0;
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y = y * height_scale;
      int32 y0 = static_cast<int32>(std::floor(input_y));
      int32 y1 = std::min(y0 + 1, input_height - 1);
      for (int x = 0; x < output_width; ++x) {
        float input_x = x * width_scale;
        int32 x0 = static_cast<int32>(input_x);
        int32 x1 = std::min(x0 + 1, input_width - 1);
        float* output_ptr = &output_data[output_offset];

        // Run kernel on the 4 corners of the bilinear resize algorithm.
        int32 input_offset = Offset(input_shape, b, y0, x0, 0);
        float scale = (1 - (input_y - y0)) * (1 - (input_x - x0));
        const float* input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_shape, b, y0, x1, 0);
        scale = (1 - (input_y - y0)) * (input_x - x0);
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_shape, b, y1, x0, 0);
        scale = (input_y - y0) * (1 - (input_x - x0));
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        input_offset = Offset(input_shape, b, y1, x1, 0);
        scale = (input_y - y0) * (input_x - x0);
        input_ptr = &input_data[input_offset];
        ResizeBilinearKernel(input_ptr, depth, scale, output_ptr);

        output_offset += depth;
      }
    }
  }
}

template <typename T>
inline void ResizeBilinearGenericSmallChannel(
    int32 batches, int32 input_height, int32 input_width, int32 depth,
    int32 output_height, int32 output_width, float height_scale,
    float width_scale, const RuntimeShape& input_shape, const T* input_data,
    const RuntimeShape& output_shape, T* output_data) {
  T* output_ptr = &output_data[0];
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      float input_y = y * height_scale;
      int32 y0 = static_cast<int32>(std::floor(input_y));
      int32 y1 = std::min(y0 + 1, input_height - 1);
      for (int x = 0; x < output_width; ++x) {
        float input_x = x * width_scale;
        int32 x0 = static_cast<int32>(std::floor((input_x)));
        int32 x1 = std::min(x0 + 1, input_width - 1);

        int32 input_offset[4] = {Offset(input_shape, b, y0, x0, 0),
                                 Offset(input_shape, b, y0, x1, 0),
                                 Offset(input_shape, b, y1, x0, 0),
                                 Offset(input_shape, b, y1, x1, 0)};
        float scale[4] = {(1 - (input_y - y0)) * (1 - (input_x - x0)),
                          (1 - (input_y - y0)) * (input_x - x0),
                          (input_y - y0) * (1 - (input_x - x0)),
                          (input_y - y0) * (input_x - x0)};

        for (int d = 0; d < depth; d++) {
          const T* input_ptr = &input_data[d];
          *output_ptr++ = static_cast<T>(input_ptr[input_offset[0]] * scale[0] +
                                         input_ptr[input_offset[1]] * scale[1] +
                                         input_ptr[input_offset[2]] * scale[2] +
                                         input_ptr[input_offset[3]] * scale[3]);
        }
      }
    }
  }
}

inline void ResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           const RuntimeShape& unextended_input_shape,
                           const float* input_data,
                           const RuntimeShape& output_size_shape,
                           const int32* output_size_data,
                           const RuntimeShape& unextended_output_shape,
                           float* output_data) {
  gemmlowp::ScopedProfilingLabel label("ResizeBilinear");
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32 batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32 input_height = input_shape.Dims(1);
  int32 input_width = input_shape.Dims(2);
  int32 depth = MatchingDim(input_shape, 3, output_shape, 3);

  TFLITE_DCHECK_EQ(output_size_shape.FlatSize(), 2);
  int32 output_height = output_size_data[0];
  int32 output_width = output_size_data[1];

  // Specialize for 2x2 upsample.
  if (!op_params.align_corners && output_height == 2 * input_height &&
      output_width == 2 * input_width) {
    ResizeBilinear2x2(batches, input_height, input_width, depth, output_height,
                      output_width, input_shape, input_data, output_shape,
                      output_data);
  } else {
    float height_scale = static_cast<float>(input_height) / output_height;
    float width_scale = static_cast<float>(input_width) / output_width;
    if (op_params.align_corners && output_height > 1) {
      height_scale = static_cast<float>(input_height - 1) / (output_height - 1);
    }
    if (op_params.align_corners && output_width > 1) {
      width_scale = static_cast<float>(input_width - 1) / (output_width - 1);
    }

    ResizeBilinearGeneric(batches, input_height, input_width, depth,
                          output_height, output_width, height_scale,
                          width_scale, input_shape, input_data, output_shape,
                          output_data);
  }
}

// TODO(prabhumk): This is not a real quantized bilinear. It does not use int8
// or int16 arithmetic.
inline void ResizeBilinear(const tflite::ResizeBilinearParams& op_params,
                           const RuntimeShape& unextended_input_shape,
                           const uint8* input_data,
                           const RuntimeShape& output_size_shape,
                           const int32* output_size_data,
                           const RuntimeShape& unextended_output_shape,
                           uint8* output_data) {
  gemmlowp::ScopedProfilingLabel label("ResizeBilinear");
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32 batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32 input_height = input_shape.Dims(1);
  int32 input_width = input_shape.Dims(2);
  int32 depth = MatchingDim(input_shape, 3, output_shape, 3);

  TFLITE_DCHECK_EQ(output_size_shape.FlatSize(), 2);
  int32 output_height = output_size_data[0];
  int32 output_width = output_size_data[1];

  float height_scale =
      (op_params.align_corners && output_height > 1)
          ? (static_cast<float>(input_height - 1) / (output_height - 1))
          : (static_cast<float>(input_height) / output_height);

  float width_scale =
      (op_params.align_corners && output_width > 1)
          ? (static_cast<float>(input_width - 1) / (output_width - 1))
          : (static_cast<float>(input_width) / output_width);

  ResizeBilinearGenericSmallChannel<uint8>(
      batches, input_height, input_width, depth, output_height, output_width,
      height_scale, width_scale, input_shape, input_data, output_shape,
      output_data);
}

// Helper methods for BatchToSpaceND.
// `spatial_index_dim` specifies post-crop offset index in this spatial
// dimension, i.e. spatial offset introduced by flattening batch to spatial
// dimension minus the crop size at beginning. `block_shape_dim` is the block
// size in current dimension. `input_dim` and `output_dim` are input and output
// size of BatchToSpaceND operation in current dimension.
// Output start index is inclusive and end index is exclusive.
inline void GetIndexRange(int spatial_index_dim, int block_shape_dim,
                          int input_dim, int output_dim, int* start_index,
                          int* end_index) {
  // (*start_index) * block_shape_dim is effectively rounded up to the next
  // multiple of block_shape_dim by the integer division.
  *start_index =
      std::max(0, (-spatial_index_dim + block_shape_dim - 1) / block_shape_dim);
  // Similarly, (*end_index) * block_shape_dim is rounded up too (note that
  // end_index is exclusive).
  *end_index = std::min(
      input_dim,
      (output_dim - spatial_index_dim + block_shape_dim - 1) / block_shape_dim);
}

template <typename T>
inline void BatchToSpaceND(
    const RuntimeShape& unextended_input1_shape, const T* input1_data,
    const RuntimeShape& unextended_input2_shape, const int32* block_shape_data,
    const RuntimeShape& unextended_input3_shape, const int32* crops_data,
    const RuntimeShape& unextended_output_shape, T* output_data) {
  gemmlowp::ScopedProfilingLabel label("BatchToSpaceND");

  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input1_shape =
      RuntimeShape::ExtendedShape(4, unextended_input1_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  const int output_width = output_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_batch_size = output_shape.Dims(0);

  const int depth = input1_shape.Dims(3);
  const int input_width = input1_shape.Dims(2);
  const int input_height = input1_shape.Dims(1);
  const int input_batch_size = input1_shape.Dims(0);

  const int block_shape_width = block_shape_data[1];
  const int block_shape_height = block_shape_data[0];
  const int crops_top = crops_data[0];
  const int crops_left = crops_data[2];

  for (int in_batch = 0; in_batch < input_batch_size; ++in_batch) {
    const int out_batch = in_batch % output_batch_size;
    const int spatial_offset = in_batch / output_batch_size;

    int in_h_start = 0;
    int in_h_end = 0;
    // GetIndexRange ensures start and end indices are in [0, output_height).
    GetIndexRange(spatial_offset / block_shape_width - crops_top,
                  block_shape_height, input_height, output_height, &in_h_start,
                  &in_h_end);

    for (int in_h = in_h_start; in_h < in_h_end; ++in_h) {
      const int out_h = in_h * block_shape_height +
                        spatial_offset / block_shape_width - crops_top;
      TFLITE_DCHECK_GE(out_h, 0);
      TFLITE_DCHECK_LT(out_h, output_height);

      int in_w_start = 0;
      int in_w_end = 0;
      // GetIndexRange ensures start and end indices are in [0, output_width).
      GetIndexRange(spatial_offset % block_shape_width - crops_left,
                    block_shape_width, input_width, output_width, &in_w_start,
                    &in_w_end);

      for (int in_w = in_w_start; in_w < in_w_end; ++in_w) {
        const int out_w = in_w * block_shape_width +
                          spatial_offset % block_shape_width - crops_left;
        TFLITE_DCHECK_GE(out_w, 0);
        TFLITE_DCHECK_LT(out_w, output_width);
        T* out = output_data + Offset(output_shape, out_batch, out_h, out_w, 0);
        const T* in =
            input1_data + Offset(input1_shape, in_batch, in_h, in_w, 0);
        memcpy(out, in, depth * sizeof(T));
      }
    }
  }
}

template <typename T>
void TypedMemset(void* ptr, T value, size_t num) {
  // Optimization for common cases where memset() will suffice.
  if (value == 0 || std::is_same<T, uint8_t>::value) {
    memset(ptr, value, num * sizeof(T));
  } else {
    // Default implementation for cases where memset() will not preserve the
    // bytes, e.g., typically when sizeof(T) > sizeof(uint8_t).
    char* pos = static_cast<char*>(ptr);
    for (size_t i = 0; i < num; ++i) {
      memcpy(pos, &value, sizeof(T));
      pos = pos + sizeof(T);
    }
  }
}

// This makes heavy use of Offset, along with conditional branches. There may be
// opportunities for improvement.
//
// There are two versions of pad: Pad and PadV2.  In PadV2 there is a second
// scalar input that provides the padding value.  Therefore pad_value_ptr can be
// equivalent to a simple input1_data.  For Pad, it should point to a zero
// value.
//
// Note that two typenames are required, so that T=P=int32 is considered a
// specialization distinct from P=int32.
template <typename T, typename P>
inline void PadImpl(const tflite::PadParams& op_params,
                    const RuntimeShape& input_shape, const T* input_data,
                    const P* pad_value_ptr, const RuntimeShape& output_shape,
                    T* output_data) {
  gemmlowp::ScopedProfilingLabel label("Pad4DSlowImpl");
  const RuntimeShape ext_input_shape =
      RuntimeShape::ExtendedShape(4, input_shape);
  const RuntimeShape ext_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);
  TFLITE_DCHECK_LE(op_params.left_padding_count, 4);
  TFLITE_DCHECK_LE(op_params.right_padding_count, 4);

  // Pad kernels are limited to max 4 dimensions. Copy inputs so we can pad them
  // to 4 dims (yes, we are "padding the padding").
  std::vector<int> left_padding_copy(4, 0);
  const int left_padding_extend = 4 - op_params.left_padding_count;
  for (int i = 0; i < op_params.left_padding_count; ++i) {
    left_padding_copy[left_padding_extend + i] = op_params.left_padding[i];
  }
  std::vector<int> right_padding_copy(4, 0);
  const int right_padding_extend = 4 - op_params.right_padding_count;
  for (int i = 0; i < op_params.right_padding_count; ++i) {
    right_padding_copy[right_padding_extend + i] = op_params.right_padding[i];
  }

  const int output_batch = ext_output_shape.Dims(0);
  const int output_height = ext_output_shape.Dims(1);
  const int output_width = ext_output_shape.Dims(2);
  const int output_depth = ext_output_shape.Dims(3);

  const int left_b_padding = left_padding_copy[0];
  const int left_h_padding = left_padding_copy[1];
  const int left_w_padding = left_padding_copy[2];
  const int left_d_padding = left_padding_copy[3];

  const int right_b_padding = right_padding_copy[0];
  const int right_h_padding = right_padding_copy[1];
  const int right_w_padding = right_padding_copy[2];
  const int right_d_padding = right_padding_copy[3];

  const int input_depth = ext_input_shape.Dims(3);
  const T pad_value = *pad_value_ptr;

  if (left_b_padding != 0) {
    TypedMemset<T>(
        output_data, pad_value,
        left_b_padding * output_height * output_width * output_depth);
  }
  for (int out_b = left_b_padding; out_b < output_batch - right_b_padding;
       ++out_b) {
    if (left_h_padding != 0) {
      TypedMemset<T>(output_data + Offset(ext_output_shape, out_b, 0, 0, 0),
                     pad_value, left_h_padding * output_width * output_depth);
    }
    for (int out_h = left_h_padding; out_h < output_height - right_h_padding;
         ++out_h) {
      if (left_w_padding != 0) {
        TypedMemset<T>(
            output_data + Offset(ext_output_shape, out_b, out_h, 0, 0),
            pad_value, left_w_padding * output_depth);
      }
      for (int out_w = left_w_padding; out_w < output_width - right_w_padding;
           ++out_w) {
        if (left_d_padding != 0) {
          TypedMemset<T>(
              output_data + Offset(ext_output_shape, out_b, out_h, out_w, 0),
              pad_value, left_d_padding);
        }

        T* out = output_data +
                 Offset(ext_output_shape, out_b, out_h, out_w, left_d_padding);
        const T* in = input_data +
                      Offset(ext_input_shape, out_b - left_b_padding,
                             out_h - left_h_padding, out_w - left_w_padding, 0);
        memcpy(out, in, input_depth * sizeof(T));

        if (right_d_padding != 0) {
          TypedMemset<T>(
              output_data + Offset(ext_output_shape, out_b, out_h, out_w,
                                   output_depth - right_d_padding),
              pad_value, right_d_padding);
        }
      }
      if (right_w_padding != 0) {
        TypedMemset<T>(output_data + Offset(ext_output_shape, out_b, out_h,
                                            output_width - right_w_padding, 0),
                       pad_value, right_w_padding * output_depth);
      }
    }
    if (right_h_padding != 0) {
      TypedMemset<T>(
          output_data + Offset(ext_output_shape, out_b,
                               output_height - right_h_padding, 0, 0),
          pad_value, right_h_padding * output_width * output_depth);
    }
  }
  if (right_b_padding != 0) {
    TypedMemset<T>(
        output_data +
            Offset(ext_output_shape, output_batch - right_b_padding, 0, 0, 0),
        pad_value,
        right_b_padding * output_height * output_width * output_depth);
  }
}

template <typename T, typename P>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const T* input_data,
                const P* pad_value_ptr, const RuntimeShape& output_shape,
                T* output_data) {
  PadImpl(op_params, input_shape, input_data, pad_value_ptr, output_shape,
          output_data);
}

// The second (pad-value) input can be int32 when, say, the first is uint8.
template <typename T>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const T* input_data,
                const int32* pad_value_ptr, const RuntimeShape& output_shape,
                T* output_data) {
  const T converted_pad_value = static_cast<T>(*pad_value_ptr);
  PadImpl(op_params, input_shape, input_data, &converted_pad_value,
          output_shape, output_data);
}

// This version avoids conflicting template matching.
template <>
inline void Pad(const tflite::PadParams& op_params,
                const RuntimeShape& input_shape, const int32* input_data,
                const int32* pad_value_ptr, const RuntimeShape& output_shape,
                int32* output_data) {
  PadImpl(op_params, input_shape, input_data, pad_value_ptr, output_shape,
          output_data);
}

// TODO(b/117643175): Optimize. (This is an introductory copy of standard Pad.)
//
// This pad requires that (a) left and right paddings are in the 4D patterns
// {0, h_pad, w_pad, 0}, and (b) memset can be used: *pad_value_ptr == 0 and/or
// T is uint8.
//
// There are two versions of pad: Pad and PadV2.  In PadV2 there is a second
// scalar input that provides the padding value.  Therefore pad_value_ptr can be
// equivalent to a simple input1_data.  For Pad, it should point to a zero
// value.
//
// Note that two typenames are required, so that T=P=int32 is considered a
// specialization distinct from P=int32.
template <typename T, typename P>
inline void PadImageStyleMemset(const tflite::PadParams& op_params,
                                const RuntimeShape& input_shape,
                                const T* input_data, const P* pad_value_ptr,
                                const RuntimeShape& output_shape,
                                T* output_data) {
  gemmlowp::ScopedProfilingLabel label("PadImageStyle");
  const RuntimeShape ext_input_shape =
      RuntimeShape::ExtendedShape(4, input_shape);
  const RuntimeShape ext_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);
  TFLITE_DCHECK_LE(op_params.left_padding_count, 4);
  TFLITE_DCHECK_LE(op_params.right_padding_count, 4);

  // Pad kernels are limited to max 4 dimensions. Copy inputs so we can pad them
  // to 4 dims (yes, we are "padding the padding").
  std::vector<int> left_padding_copy(4, 0);
  const int left_padding_extend = 4 - op_params.left_padding_count;
  for (int i = 0; i < op_params.left_padding_count; ++i) {
    left_padding_copy[left_padding_extend + i] = op_params.left_padding[i];
  }
  std::vector<int> right_padding_copy(4, 0);
  const int right_padding_extend = 4 - op_params.right_padding_count;
  for (int i = 0; i < op_params.right_padding_count; ++i) {
    right_padding_copy[right_padding_extend + i] = op_params.right_padding[i];
  }
  // The following padding restrictions are contractual requirements, and
  // embody what it means for a padding op to be "image-style".
  TFLITE_DCHECK_EQ(left_padding_copy[0], 0);
  TFLITE_DCHECK_EQ(left_padding_copy[3], 0);
  TFLITE_DCHECK_EQ(right_padding_copy[0], 0);
  TFLITE_DCHECK_EQ(right_padding_copy[3], 0);

  const int batch = MatchingDim(ext_input_shape, 0, ext_output_shape, 0);
  const int output_height = ext_output_shape.Dims(1);
  const int output_width = ext_output_shape.Dims(2);
  const int input_height = ext_input_shape.Dims(1);
  const int input_width = ext_input_shape.Dims(2);
  const int depth = MatchingDim(ext_input_shape, 3, ext_output_shape, 3);

  const int left_h_padding = left_padding_copy[1];
  const int left_w_padding = left_padding_copy[2];
  const int right_h_padding = right_padding_copy[1];
  const int right_w_padding = right_padding_copy[2];

  TFLITE_DCHECK_EQ(output_height,
                   input_height + left_h_padding + right_h_padding);
  TFLITE_DCHECK_EQ(output_width,
                   input_width + left_w_padding + right_w_padding);

  const T pad_value = *pad_value_ptr;
  const int top_block_size = left_h_padding * output_width * depth;
  const size_t num_top_block_bytes = top_block_size * sizeof(T);
  const int bottom_block_size = right_h_padding * output_width * depth;
  const size_t num_bottom_block_bytes = bottom_block_size * sizeof(T);
  const int left_blocks_size = left_w_padding * depth;
  const size_t num_left_block_bytes = left_blocks_size * sizeof(T);
  const int right_blocks_size = right_w_padding * depth;
  const size_t num_right_block_bytes = right_blocks_size * sizeof(T);
  const int inner_line_size = input_width * depth;
  const size_t num_inner_line_bytes = inner_line_size * sizeof(T);

  if (input_height == 0) {
    memset(output_data, pad_value,
           num_top_block_bytes + num_bottom_block_bytes);
  } else {
    for (int i = 0; i < batch; ++i) {
      // For each image in the batch, apply the top padding, then iterate
      // through rows, then apply the bottom padding.
      //
      // By unwinding one iteration, we can combine the first left-margin
      // padding with the top padding, and the last right-margin padding with
      // the bottom padding.
      memset(output_data, pad_value,
             num_top_block_bytes + num_left_block_bytes);
      output_data += top_block_size + left_blocks_size;
      memcpy(output_data, input_data, num_inner_line_bytes);
      input_data += inner_line_size;
      output_data += inner_line_size;
      // One iteration unwound.
      // Unwinding this loop affords the opportunity to reorder the loop work
      // and hence combine memset() calls.
      //
      // Before unwinding:
      // for (int j = 0; j < input_height; ++j) {
      //   // Pad on left, copy central data, pad on right.
      //   memset(output_data, pad_value, num_left_block_bytes);
      //   output_data += left_blocks_size;
      //   memcpy(output_data, input_data, num_inner_line_bytes);
      //   input_data += inner_line_size;
      //   output_data += inner_line_size;
      //   memset(output_data, pad_value, num_right_block_bytes);
      //   output_data += right_blocks_size;
      // }
      for (int j = 1; j < input_height; ++j) {
        memset(output_data, pad_value,
               num_right_block_bytes + num_left_block_bytes);
        output_data += right_blocks_size + left_blocks_size;
        memcpy(output_data, input_data, num_inner_line_bytes);
        input_data += inner_line_size;
        output_data += inner_line_size;
      }
      memset(output_data, pad_value,
             num_right_block_bytes + num_bottom_block_bytes);
      output_data += right_blocks_size + bottom_block_size;
    }
  }
}

template <typename T, typename P>
inline void PadImageStyle(const tflite::PadParams& op_params,
                          const RuntimeShape& input_shape, const T* input_data,
                          const P* pad_value_ptr,
                          const RuntimeShape& output_shape, T* output_data) {
  TFLITE_ASSERT_FALSE;
}

template <typename P>
inline void PadImageStyle(const tflite::PadParams& op_params,
                          const RuntimeShape& input_shape,
                          const uint8* input_data, const P* pad_value_ptr,
                          const RuntimeShape& output_shape,
                          uint8* output_data) {
  PadImageStyleMemset(op_params, input_shape, input_data, pad_value_ptr,
                      output_shape, output_data);
}

template <typename P>
inline void PadImageStyle(const tflite::PadParams& op_params,
                          const RuntimeShape& input_shape,
                          const float* input_data, const P* pad_value_ptr,
                          const RuntimeShape& output_shape,
                          float* output_data) {
  const float converted_pad_value = static_cast<float>(*pad_value_ptr);
  if (converted_pad_value == 0.0f) {
    PadImageStyleMemset(op_params, input_shape, input_data, pad_value_ptr,
                        output_shape, output_data);
  } else {
    PadImpl(op_params, input_shape, input_data, pad_value_ptr, output_shape,
            output_data);
  }
}

template <typename T>
inline void Slice(const tflite::SliceParams& op_params,
                  const RuntimeShape& input_shape,
                  const RuntimeShape& output_shape,
                  SequentialTensorWriter<T>* writer) {
  gemmlowp::ScopedProfilingLabel label("Slice");
  const RuntimeShape ext_shape = RuntimeShape::ExtendedShape(4, input_shape);
  // TODO(dkalenichenko): This op only supports 4D tensors or smaller.
  TFLITE_DCHECK_LE(op_params.begin_count, 4);
  TFLITE_DCHECK_LE(op_params.size_count, 4);
  const int begin_count = op_params.begin_count;
  const int size_count = op_params.size_count;
  // We front-pad the begin and size vectors.
  const int start_b = 4 - begin_count > 0 ? 0 : op_params.begin[0];
  const int stop_b = (4 - size_count > 0 || op_params.size[0] == -1)
                         ? ext_shape.Dims(0)
                         : start_b + op_params.size[0];
  const int start_h = begin_count < 3 ? 0 : op_params.begin[begin_count - 3];
  const int stop_h = (size_count < 3 || op_params.size[size_count - 3] == -1)
                         ? ext_shape.Dims(1)
                         : start_h + op_params.size[size_count - 3];
  const int start_w = begin_count < 2 ? 0 : op_params.begin[begin_count - 2];
  const int stop_w = (size_count < 2 || op_params.size[size_count - 2] == -1)
                         ? ext_shape.Dims(2)
                         : start_w + op_params.size[size_count - 2];
  const int start_d = begin_count < 1 ? 0 : op_params.begin[begin_count - 1];
  const int stop_d = (size_count < 1 || op_params.size[size_count - 1] == -1)
                         ? ext_shape.Dims(3)
                         : start_d + op_params.size[size_count - 1];

  for (int in_b = start_b; in_b < stop_b; ++in_b) {
    for (int in_h = start_h; in_h < stop_h; ++in_h) {
      for (int in_w = start_w; in_w < stop_w; ++in_w) {
        const int len = stop_d - start_d;
        if (len > 0)
          writer->WriteN(Offset(ext_shape, in_b, in_h, in_w, start_d), len);
      }
    }
  }
}

template <typename T>
inline void Slice(const tflite::SliceParams& op_params,
                  const RuntimeShape& input_shape, const T* input_data,
                  const RuntimeShape& output_shape, T* output_data) {
  SequentialTensorWriter<T> writer(input_data, output_data);
  return Slice(op_params, input_shape, output_shape, &writer);
}

template <typename T>
inline void Slice(const tflite::SliceParams& op_params,
                  const RuntimeShape& input_shape, const TfLiteTensor* input,
                  const RuntimeShape& output_shape, TfLiteTensor* output) {
  SequentialTensorWriter<T> writer(input, output);
  return Slice(op_params, input_shape, output_shape, &writer);
}

template <typename T>
void Minimum(const RuntimeShape& input1_shape, const T* input1_data,
             const T* input2_data, const RuntimeShape& output_shape,
             T* output_data) {
  gemmlowp::ScopedProfilingLabel label("TensorFlowMinimum");
  auto input1_map = MapAsVector(input1_data, input1_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  auto min_value = input2_data[0];
  output_map.array() = input1_map.array().min(min_value);
}

// Convenience version that allows, for example, generated-code calls to be
// the same as other binary ops.
template <typename T>
inline void Minimum(const RuntimeShape& input1_shape, const T* input1_data,
                    const RuntimeShape&, const T* input2_data,
                    const RuntimeShape& output_shape, T* output_data) {
  // Drop shape of second input: not needed.
  Minimum(input1_shape, input1_data, input2_data, output_shape, output_data);
}

template <typename T>
void Maximum(const RuntimeShape& input1_shape, const T* input1_data,
             const T* input2_data, const RuntimeShape& output_shape,
             T* output_data) {
  gemmlowp::ScopedProfilingLabel label("TensorFlowMaximum");
  auto input1_map = MapAsVector(input1_data, input1_shape);
  auto output_map = MapAsVector(output_data, output_shape);
  auto max_value = input2_data[0];
  output_map.array() = input1_map.array().max(max_value);
}

// Convenience version that allows, for example, generated-code calls to be
// the same as other binary ops.
template <typename T>
inline void Maximum(const RuntimeShape& input1_shape, const T* input1_data,
                    const RuntimeShape&, const T* input2_data,
                    const RuntimeShape& output_shape, T* output_data) {
  // Drop shape of second input: not needed.
  Maximum(input1_shape, input1_data, input2_data, output_shape, output_data);
}

template <typename T>
void TransposeIm2col(const ConvParams& params, uint8 zero_byte,
                     const RuntimeShape& input_shape, const T* input_data,
                     const RuntimeShape& filter_shape,
                     const RuntimeShape& output_shape, T* im2col_data) {
  gemmlowp::ScopedProfilingLabel label("TransposeIm2col");
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  TFLITE_DCHECK(im2col_data);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  MatchingDim(output_shape, 3, filter_shape, 0);  // output_depth

  // Construct the MxN sized im2col matrix.
  // The rows M, are sub-ordered B x H x W
  const RuntimeShape row_shape({1, batches, output_height, output_width});
  // The columns, N, are sub-ordered Kh x Kw x Din
  const RuntimeShape col_shape({1, filter_height, filter_width, input_depth});
  // Use dimensions M and N to construct dims for indexing directly into im2col
  const RuntimeShape im2col_shape(
      {1, 1, row_shape.FlatSize(), col_shape.FlatSize()});

  // Build the im2col matrix by looping through all the input pixels,
  // computing their influence on the output, rather than looping through all
  // the output pixels. We therefore must initialize the im2col array to zero.
  // This is potentially inefficient because we subsequently overwrite bytes
  // set here. However, in practice memset is very fast and costs negligible.
  memset(im2col_data, zero_byte, im2col_shape.FlatSize() * sizeof(T));

  // Loop through the output batches
  for (int batch = 0; batch < batches; ++batch) {
    // Loop through input pixels one at a time.
    for (int in_y = 0; in_y < input_height; ++in_y) {
      for (int in_x = 0; in_x < input_width; ++in_x) {
        // Loop through the output pixels it will influence
        const int out_x_origin = (in_x * stride_width) - pad_width;
        const int out_y_origin = (in_y * stride_height) - pad_height;
        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
          const int out_y = out_y_origin + filter_y;
          // Is output pixel within height bounds?
          if ((out_y >= 0) && (out_y < output_height)) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int out_x = out_x_origin + filter_x;
              // Is output pixel within width bounds?
              if ((out_x >= 0) && (out_x < output_width)) {
                // Copy the input elements of this pixel
                T const* src =
                    input_data + Offset(input_shape, batch, in_y, in_x, 0);
                int row_offset = Offset(row_shape, 0, batch, out_y, out_x);
                int col_offset = Offset(col_shape, 0, filter_y, filter_x, 0);
                T* dst = im2col_data +
                         Offset(im2col_shape, 0, 0, row_offset, col_offset);
                memcpy(dst, src, input_depth * sizeof(T));
              }
            }
          }
        }
      }
    }
  }
}

// Returns in 'im_data' (assumes to be zero-initialized) image patch in storage
// order (height, width, depth), constructed from patches in 'col_data', which
// is required to be in storage order (out_height * out_width, filter_height,
// filter_width, in_depth).  Implementation by Yangqing Jia (jiayq).
// Copied from //tensorflow/core/kernels/conv_grad_input_ops.cc
template <typename T>
void Col2im(const T* col_data, const int depth, const int height,
            const int width, const int filter_h, const int filter_w,
            const int pad_t, const int pad_l, const int pad_b, const int pad_r,
            const int stride_h, const int stride_w, T* im_data) {
  int height_col = (height + pad_t + pad_b - filter_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - filter_w) / stride_w + 1;
  int h_pad = -pad_t;
  for (int h = 0; h < height_col; ++h) {
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      T* im_patch_data = im_data + (h_pad * width + w_pad) * depth;
      for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
        for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            // TODO(andydavis) Vectorize this loop (if compiler does not).
            for (int i = 0; i < depth; ++i) {
              im_patch_data[i] += col_data[i];
            }
          }
          im_patch_data += depth;
          col_data += depth;
        }
        // Jump over remaining number of depth.
        im_patch_data += depth * (width - filter_w);
      }
      w_pad += stride_w;
    }
    h_pad += stride_h;
  }
}

// TransposeConvV2 expect the weights in HWOI order.
inline void TransposeConvV2(
    const ConvParams& params, const RuntimeShape& input_shape,
    const float* input_data, const RuntimeShape& hwoi_ordered_filter_shape,
    const float* hwoi_ordered_filter_data, const RuntimeShape& output_shape,
    float* output_data, const RuntimeShape& col2im_shape, float* col2im_data,
    CpuBackendContext* cpu_backend_context) {
  gemmlowp::ScopedProfilingLabel label("TransposeConvV2");
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(hwoi_ordered_filter_shape.DimensionsCount(), 4);
  const int batch_size = input_shape.Dims(0);
  TFLITE_DCHECK(col2im_data);
  TFLITE_DCHECK(hwoi_ordered_filter_data);

  const int input_image_size = input_shape.Dims(1) * input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_image_size = output_height * output_width;
  const int input_depth =
      MatchingDim(input_shape, 3, hwoi_ordered_filter_shape, 3);
  const int output_depth =
      MatchingDim(output_shape, 3, hwoi_ordered_filter_shape, 2);
  const int input_offset = input_image_size * input_depth;
  const int output_offset = output_image_size * output_depth;

  const int filter_height = hwoi_ordered_filter_shape.Dims(0);
  const int filter_width = hwoi_ordered_filter_shape.Dims(1);
  const int padding_top = params.padding_values.height;
  const int padding_bottom =
      params.padding_values.height + params.padding_values.height_offset;
  const int padding_left = params.padding_values.width;
  const int padding_right =
      params.padding_values.width + params.padding_values.width_offset;
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;

  const int hwoi_ordered_filter_total_size =
      filter_height * filter_width * output_depth;

  cpu_backend_gemm::MatrixParams<float> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = hwoi_ordered_filter_total_size;
  lhs_params.cols = input_depth;
  float* output_data_p = output_data;
  std::fill_n(output_data, output_offset * batch_size, 0.0f);
  for (int i = 0; i < batch_size; ++i) {
    cpu_backend_gemm::MatrixParams<float> rhs_params;
    rhs_params.order = cpu_backend_gemm::Order::kColMajor;
    rhs_params.rows = input_depth;
    rhs_params.cols = input_image_size;
    cpu_backend_gemm::MatrixParams<float> dst_params;
    dst_params.order = cpu_backend_gemm::Order::kColMajor;
    dst_params.rows = hwoi_ordered_filter_total_size;
    dst_params.cols = input_image_size;
    cpu_backend_gemm::GemmParams<float, float> gemm_params;
    cpu_backend_gemm::Gemm(lhs_params, hwoi_ordered_filter_data, rhs_params,
                           input_data + input_offset * i, dst_params,
                           col2im_data, gemm_params, cpu_backend_context);

    Col2im(col2im_data, output_depth, output_height, output_width,
           filter_height, filter_width, padding_top, padding_left,
           padding_bottom, padding_right, stride_height, stride_width,
           output_data_p);
    output_data_p += output_offset;
  }
}

// Integer-only version of ResizeNearestNeighbor. Since scales are represented
// in fixed-point and thus approximated, |in_x| or |in_y| may differ from the
// reference version. Debug checks are in place to test if this occurs.
inline void ResizeNearestNeighbor(
    const tflite::ResizeNearestNeighborParams& op_params,
    const RuntimeShape& unextended_input_shape, const uint8* input_data,
    const RuntimeShape& output_size_shape, const int32* output_size_data,
    const RuntimeShape& unextended_output_shape, uint8* output_data) {
  // Align corners = true is not supported.
  TFLITE_DCHECK(!op_params.align_corners);
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);

  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32 batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32 input_height = input_shape.Dims(1);
  int32 input_width = input_shape.Dims(2);
  int32 depth = MatchingDim(input_shape, 3, output_shape, 3);

  // The Tensorflow version of this op allows resize on the width and height
  // axis only.
  TFLITE_DCHECK_EQ(output_size_shape.FlatSize(), 2);
  int32 output_height = output_size_data[0];
  int32 output_width = output_size_data[1];

  // Convert scales to fixed-point with 16 fractional bits. We add 1 as an
  // error factor and to avoid zero scales. For example, with input_height = 1,
  // output_height = 3, the float scaling factor would be non-zero at 1/3.
  // With fixed-point, this is zero.
  int32 height_scale = (input_height << 16) / output_height + 1;
  int32 width_scale = (input_width << 16) / output_width + 1;

  const int col_offset = input_shape.Dims(3);
  const int row_offset = input_shape.Dims(2) * col_offset;
  const int batch_offset = input_shape.Dims(1) * row_offset;

  const uint8* input_ptr = input_data;
  uint8* output_ptr = output_data;
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      int32 in_y = std::min((y * height_scale) >> 16, input_height - 1);
      // Check offset calculation is the same as the reference version. See
      // function comment for details. We check using a non-float version of:
      // TFLITE_DCHECK_EQ(in_y, std::floor(y * (static_cast<float>(input_height)
      //                                            / output_height)));
      TFLITE_DCHECK_LT(y * input_height, output_height + in_y * output_height);
      TFLITE_DCHECK_GE(y * input_height, in_y * output_height);
      const uint8* y_input_ptr = input_ptr + in_y * row_offset;
      for (int x = 0; x < output_width; ++x) {
        int32 in_x = std::min((x * width_scale) >> 16, input_width - 1);
        // Check offset calculation is the same as the reference version. See
        // function comment for details. We check using a non-float version of:
        // TFLITE_DCHECK_EQ(in_y,
        //                  std::floor(y * (static_cast<float>(input_width)
        //                                      / output_width)));
        TFLITE_DCHECK_LT(x * input_width, output_width + in_x * output_width);
        TFLITE_DCHECK_GE(x * input_width, in_x * output_width);
        const uint8* x_input_ptr = y_input_ptr + in_x * col_offset;
        memcpy(output_ptr, x_input_ptr, depth);
        output_ptr += depth;
      }
    }
    input_ptr += batch_offset;
  }
}

template <typename input_type, typename output_type>
inline void Requantize(const input_type* input_data, int32_t size,
                       int32_t effective_scale_multiplier,
                       int32_t effective_scale_shift, int32_t input_zeropoint,
                       int32_t output_zeropoint, output_type* output_data) {
  reference_ops::Requantize(input_data, size, effective_scale_multiplier,
                            effective_scale_shift, input_zeropoint,
                            output_zeropoint, output_data);
}

#ifdef USE_NEON

inline void MultiplyByQuantizedMultiplier4Rows(
    const int32x4_t input_val_1, const int32x4_t input_val_2,
    const int32x4_t input_val_3, const int32x4_t input_val_4,
    const int32_t multiplier, const int32_t left_shifted_one,
    const int32_t right_shift, int32x4_t* result_val_1, int32x4_t* result_val_2,
    int32x4_t* result_val_3, int32x4_t* result_val_4) {
  using gemmlowp::RoundingDivideByPOT;
  using gemmlowp::SaturatingRoundingDoublingHighMul;
  int32x4_t left_shifted_one_dup = vdupq_n_s32(left_shifted_one);
  *result_val_1 = RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(
          vmulq_s32(input_val_1, left_shifted_one_dup), multiplier),
      right_shift);
  *result_val_2 = RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(
          vmulq_s32(input_val_2, left_shifted_one_dup), multiplier),
      right_shift);
  *result_val_3 = RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(
          vmulq_s32(input_val_3, left_shifted_one_dup), multiplier),
      right_shift);
  *result_val_4 = RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(
          vmulq_s32(input_val_4, left_shifted_one_dup), multiplier),
      right_shift);
}

#endif

template <>
inline void Requantize<int8_t, uint8_t>(const int8_t* input_data, int32_t size,
                                        int32_t effective_scale_multiplier,
                                        int32_t effective_scale_shift,
                                        int32_t input_zeropoint,
                                        int32_t output_zeropoint,
                                        uint8_t* output_data) {
  gemmlowp::ScopedProfilingLabel label("Requantize/Int8ToUint8");

  static constexpr int32_t kMinOutput = std::numeric_limits<uint8_t>::min();
  static constexpr int32_t kMaxOutput = std::numeric_limits<uint8_t>::max();

  int i = 0;
#ifdef USE_NEON
  // Constants.
  const int32x4_t input_zero_point_dup = vdupq_n_s32(-input_zeropoint);
  const int32x4_t output_zero_point_dup = vdupq_n_s32(output_zeropoint);
  const int32x4_t min_val_dup = vdupq_n_s32(kMinOutput);
  const int32x4_t max_val_dup = vdupq_n_s32(kMaxOutput);

  // Left shift & right shift unconditionally.
  const int32_t left_shifted_one =
      effective_scale_shift > 0 ? 1 << effective_scale_shift : 1;
  const int32_t right_shift =
      effective_scale_shift > 0 ? 0 : -effective_scale_shift;

  for (; i <= size - 16; i += 16) {
    const int8x16_t input_vec = vld1q_s8(input_data + i);
    const int16x8_t first_half = vmovl_s8(vget_low_s8(input_vec));
    const int16x8_t second_half = vmovl_s8(vget_high_s8(input_vec));
    int32x4_t input_val_1 = vmovl_s16(vget_low_s16(first_half));
    int32x4_t input_val_2 = vmovl_s16(vget_high_s16(first_half));
    int32x4_t input_val_3 = vmovl_s16(vget_low_s16(second_half));
    int32x4_t input_val_4 = vmovl_s16(vget_high_s16(second_half));
    input_val_1 = vaddq_s32(input_val_1, input_zero_point_dup);
    input_val_2 = vaddq_s32(input_val_2, input_zero_point_dup);
    input_val_3 = vaddq_s32(input_val_3, input_zero_point_dup);
    input_val_4 = vaddq_s32(input_val_4, input_zero_point_dup);

    int32x4_t result_val_1, result_val_2, result_val_3, result_val_4;
    MultiplyByQuantizedMultiplier4Rows(
        input_val_1, input_val_2, input_val_3, input_val_4,
        effective_scale_multiplier, left_shifted_one, right_shift,
        &result_val_1, &result_val_2, &result_val_3, &result_val_4);

    result_val_1 = vaddq_s32(result_val_1, output_zero_point_dup);
    result_val_2 = vaddq_s32(result_val_2, output_zero_point_dup);
    result_val_3 = vaddq_s32(result_val_3, output_zero_point_dup);
    result_val_4 = vaddq_s32(result_val_4, output_zero_point_dup);
    result_val_1 = vmaxq_s32(vminq_s32(result_val_1, max_val_dup), min_val_dup);
    result_val_2 = vmaxq_s32(vminq_s32(result_val_2, max_val_dup), min_val_dup);
    result_val_3 = vmaxq_s32(vminq_s32(result_val_3, max_val_dup), min_val_dup);
    result_val_4 = vmaxq_s32(vminq_s32(result_val_4, max_val_dup), min_val_dup);

    const uint32x4_t result_val_1_unsigned =
        vreinterpretq_u32_s32(result_val_1);
    const uint32x4_t result_val_2_unsigned =
        vreinterpretq_u32_s32(result_val_2);
    const uint32x4_t result_val_3_unsigned =
        vreinterpretq_u32_s32(result_val_3);
    const uint32x4_t result_val_4_unsigned =
        vreinterpretq_u32_s32(result_val_4);

    const uint16x4_t narrowed_val_1 = vqmovn_u32(result_val_1_unsigned);
    const uint16x4_t narrowed_val_2 = vqmovn_u32(result_val_2_unsigned);
    const uint16x4_t narrowed_val_3 = vqmovn_u32(result_val_3_unsigned);
    const uint16x4_t narrowed_val_4 = vqmovn_u32(result_val_4_unsigned);
    const uint16x8_t output_first_half =
        vcombine_u16(narrowed_val_1, narrowed_val_2);
    const uint16x8_t output_second_half =
        vcombine_u16(narrowed_val_3, narrowed_val_4);
    const uint8x8_t narrowed_first_half = vqmovn_u16(output_first_half);
    const uint8x8_t narrowed_second_half = vqmovn_u16(output_second_half);
    const uint8x16_t result =
        vcombine_u8(narrowed_first_half, narrowed_second_half);
    vst1q_u8(output_data + i, result);
  }

#endif
  for (; i < size; ++i) {
    const int32_t input = input_data[i] - input_zeropoint;
    const int32_t output =
        MultiplyByQuantizedMultiplier(input, effective_scale_multiplier,
                                      effective_scale_shift) +
        output_zeropoint;
    const int32_t clamped_output =
        std::max(std::min(output, kMaxOutput), kMinOutput);
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

template <>
inline void Requantize<uint8_t, int8_t>(const uint8_t* input_data, int32_t size,
                                        int32_t effective_scale_multiplier,
                                        int32_t effective_scale_shift,
                                        int32_t input_zeropoint,
                                        int32_t output_zeropoint,
                                        int8_t* output_data) {
  gemmlowp::ScopedProfilingLabel label("Requantize/Uint8ToInt8");

  static constexpr int32_t kMinOutput = std::numeric_limits<int8_t>::min();
  static constexpr int32_t kMaxOutput = std::numeric_limits<int8_t>::max();

  int i = 0;
#ifdef USE_NEON
  // Constants.
  const int32x4_t input_zero_point_dup = vdupq_n_s32(-input_zeropoint);
  const int32x4_t output_zero_point_dup = vdupq_n_s32(output_zeropoint);
  const int32x4_t min_val_dup = vdupq_n_s32(kMinOutput);
  const int32x4_t max_val_dup = vdupq_n_s32(kMaxOutput);

  // Left shift & right shift unconditionally.
  const int32_t left_shifted_one =
      effective_scale_shift > 0 ? 1 << effective_scale_shift : 1;
  const int32_t right_shift =
      effective_scale_shift > 0 ? 0 : -effective_scale_shift;

  for (; i <= size - 16; i += 16) {
    const uint8x16_t input_vec = vld1q_u8(input_data + i);
    const uint16x8_t first_half = vmovl_u8(vget_low_u8(input_vec));
    const uint16x8_t second_half = vmovl_u8(vget_high_u8(input_vec));
    int32x4_t input_val_1 =
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(first_half)));
    int32x4_t input_val_2 =
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(first_half)));
    int32x4_t input_val_3 =
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(second_half)));
    int32x4_t input_val_4 =
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(second_half)));
    input_val_1 = vaddq_s32(input_val_1, input_zero_point_dup);
    input_val_2 = vaddq_s32(input_val_2, input_zero_point_dup);
    input_val_3 = vaddq_s32(input_val_3, input_zero_point_dup);
    input_val_4 = vaddq_s32(input_val_4, input_zero_point_dup);

    int32x4_t result_val_1, result_val_2, result_val_3, result_val_4;
    MultiplyByQuantizedMultiplier4Rows(
        input_val_1, input_val_2, input_val_3, input_val_4,
        effective_scale_multiplier, left_shifted_one, right_shift,
        &result_val_1, &result_val_2, &result_val_3, &result_val_4);

    result_val_1 = vaddq_s32(result_val_1, output_zero_point_dup);
    result_val_2 = vaddq_s32(result_val_2, output_zero_point_dup);
    result_val_3 = vaddq_s32(result_val_3, output_zero_point_dup);
    result_val_4 = vaddq_s32(result_val_4, output_zero_point_dup);
    result_val_1 = vmaxq_s32(vminq_s32(result_val_1, max_val_dup), min_val_dup);
    result_val_2 = vmaxq_s32(vminq_s32(result_val_2, max_val_dup), min_val_dup);
    result_val_3 = vmaxq_s32(vminq_s32(result_val_3, max_val_dup), min_val_dup);
    result_val_4 = vmaxq_s32(vminq_s32(result_val_4, max_val_dup), min_val_dup);

    const int16x4_t narrowed_val_1 = vqmovn_s32(result_val_1);
    const int16x4_t narrowed_val_2 = vqmovn_s32(result_val_2);
    const int16x4_t narrowed_val_3 = vqmovn_s32(result_val_3);
    const int16x4_t narrowed_val_4 = vqmovn_s32(result_val_4);
    const int16x8_t output_first_half =
        vcombine_s16(narrowed_val_1, narrowed_val_2);
    const int16x8_t output_second_half =
        vcombine_s16(narrowed_val_3, narrowed_val_4);
    const int8x8_t narrowed_first_half = vqmovn_s16(output_first_half);
    const int8x8_t narrowed_second_half = vqmovn_s16(output_second_half);
    const int8x16_t result =
        vcombine_s8(narrowed_first_half, narrowed_second_half);
    vst1q_s8(output_data + i, result);
  }

#endif
  for (; i < size; ++i) {
    const int32_t input = input_data[i] - input_zeropoint;
    const int32_t output =
        MultiplyByQuantizedMultiplier(input, effective_scale_multiplier,
                                      effective_scale_shift) +
        output_zeropoint;
    const int32_t clamped_output =
        std::max(std::min(output, kMaxOutput), kMinOutput);
    output_data[i] = static_cast<int8_t>(clamped_output);
  }
}

template <>
inline void Requantize<int8_t, int8_t>(const int8_t* input_data, int32_t size,
                                       int32_t effective_scale_multiplier,
                                       int32_t effective_scale_shift,
                                       int32_t input_zeropoint,
                                       int32_t output_zeropoint,
                                       int8_t* output_data) {
  gemmlowp::ScopedProfilingLabel label("Requantize/Int8ToInt8");

  static constexpr int32_t kMinOutput = std::numeric_limits<int8_t>::min();
  static constexpr int32_t kMaxOutput = std::numeric_limits<int8_t>::max();

  int i = 0;
#ifdef USE_NEON
  // Constants.
  const int32x4_t input_zero_point_dup = vdupq_n_s32(-input_zeropoint);
  const int32x4_t output_zero_point_dup = vdupq_n_s32(output_zeropoint);
  const int32x4_t min_val_dup = vdupq_n_s32(kMinOutput);
  const int32x4_t max_val_dup = vdupq_n_s32(kMaxOutput);

  // Left shift & right shift unconditionally.
  int32_t left_shifted_one =
      effective_scale_shift > 0 ? 1 << effective_scale_shift : 1;
  int32_t right_shift = effective_scale_shift > 0 ? 0 : -effective_scale_shift;

  for (; i <= size - 16; i += 16) {
    const int8x16_t input_vec = vld1q_s8(input_data + i);
    const int16x8_t first_half = vmovl_s8(vget_low_s8(input_vec));
    const int16x8_t second_half = vmovl_s8(vget_high_s8(input_vec));
    int32x4_t input_val_1 = vmovl_s16(vget_low_s16(first_half));
    int32x4_t input_val_2 = vmovl_s16(vget_high_s16(first_half));
    int32x4_t input_val_3 = vmovl_s16(vget_low_s16(second_half));
    int32x4_t input_val_4 = vmovl_s16(vget_high_s16(second_half));

    input_val_1 = vaddq_s32(input_val_1, input_zero_point_dup);
    input_val_2 = vaddq_s32(input_val_2, input_zero_point_dup);
    input_val_3 = vaddq_s32(input_val_3, input_zero_point_dup);
    input_val_4 = vaddq_s32(input_val_4, input_zero_point_dup);

    int32x4_t result_val_1, result_val_2, result_val_3, result_val_4;
    MultiplyByQuantizedMultiplier4Rows(
        input_val_1, input_val_2, input_val_3, input_val_4,
        effective_scale_multiplier, left_shifted_one, right_shift,
        &result_val_1, &result_val_2, &result_val_3, &result_val_4);

    result_val_1 = vaddq_s32(result_val_1, output_zero_point_dup);
    result_val_2 = vaddq_s32(result_val_2, output_zero_point_dup);
    result_val_3 = vaddq_s32(result_val_3, output_zero_point_dup);
    result_val_4 = vaddq_s32(result_val_4, output_zero_point_dup);
    result_val_1 = vmaxq_s32(vminq_s32(result_val_1, max_val_dup), min_val_dup);
    result_val_2 = vmaxq_s32(vminq_s32(result_val_2, max_val_dup), min_val_dup);
    result_val_3 = vmaxq_s32(vminq_s32(result_val_3, max_val_dup), min_val_dup);
    result_val_4 = vmaxq_s32(vminq_s32(result_val_4, max_val_dup), min_val_dup);

    const int16x4_t narrowed_val_1 = vqmovn_s32(result_val_1);
    const int16x4_t narrowed_val_2 = vqmovn_s32(result_val_2);
    const int16x4_t narrowed_val_3 = vqmovn_s32(result_val_3);
    const int16x4_t narrowed_val_4 = vqmovn_s32(result_val_4);
    const int16x8_t output_first_half =
        vcombine_s16(narrowed_val_1, narrowed_val_2);
    const int16x8_t output_second_half =
        vcombine_s16(narrowed_val_3, narrowed_val_4);
    const int8x8_t narrowed_first_half = vqmovn_s16(output_first_half);
    const int8x8_t narrowed_second_half = vqmovn_s16(output_second_half);
    const int8x16_t result =
        vcombine_s8(narrowed_first_half, narrowed_second_half);
    vst1q_s8(output_data + i, result);
  }

#endif
  for (; i < size; ++i) {
    const int32_t input = input_data[i] - input_zeropoint;
    const int32_t output =
        MultiplyByQuantizedMultiplier(input, effective_scale_multiplier,
                                      effective_scale_shift) +
        output_zeropoint;
    const int32_t clamped_output =
        std::max(std::min(output, kMaxOutput), kMinOutput);
    output_data[i] = static_cast<int8_t>(clamped_output);
  }
}

template <>
inline void Requantize<uint8_t, uint8_t>(
    const uint8_t* input_data, int32_t size, int32_t effective_scale_multiplier,
    int32_t effective_scale_shift, int32_t input_zeropoint,
    int32_t output_zeropoint, uint8_t* output_data) {
  gemmlowp::ScopedProfilingLabel label("Requantize/Uint8ToUint8");

  static constexpr int32_t kMinOutput = std::numeric_limits<uint8_t>::min();
  static constexpr int32_t kMaxOutput = std::numeric_limits<uint8_t>::max();

  int i = 0;
#ifdef USE_NEON
  // Constants.
  const int32x4_t input_zero_point_dup = vdupq_n_s32(-input_zeropoint);
  const int32x4_t output_zero_point_dup = vdupq_n_s32(output_zeropoint);
  const int32x4_t min_val_dup = vdupq_n_s32(kMinOutput);
  const int32x4_t max_val_dup = vdupq_n_s32(kMaxOutput);

  // Left shift & right shift unconditionally.
  int32_t left_shifted_one =
      effective_scale_shift > 0 ? 1 << effective_scale_shift : 1;
  int32_t right_shift = effective_scale_shift > 0 ? 0 : -effective_scale_shift;

  for (; i <= size - 16; i += 16) {
    const uint8x16_t input_vec = vld1q_u8(input_data + i);
    const uint16x8_t first_half = vmovl_u8(vget_low_u8(input_vec));
    const uint16x8_t second_half = vmovl_u8(vget_high_u8(input_vec));
    int32x4_t input_val_1 =
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(first_half)));
    int32x4_t input_val_2 =
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(first_half)));
    int32x4_t input_val_3 =
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(second_half)));
    int32x4_t input_val_4 =
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(second_half)));
    input_val_1 = vaddq_s32(input_val_1, input_zero_point_dup);
    input_val_2 = vaddq_s32(input_val_2, input_zero_point_dup);
    input_val_3 = vaddq_s32(input_val_3, input_zero_point_dup);
    input_val_4 = vaddq_s32(input_val_4, input_zero_point_dup);

    int32x4_t result_val_1, result_val_2, result_val_3, result_val_4;
    MultiplyByQuantizedMultiplier4Rows(
        input_val_1, input_val_2, input_val_3, input_val_4,
        effective_scale_multiplier, left_shifted_one, right_shift,
        &result_val_1, &result_val_2, &result_val_3, &result_val_4);

    result_val_1 = vaddq_s32(result_val_1, output_zero_point_dup);
    result_val_2 = vaddq_s32(result_val_2, output_zero_point_dup);
    result_val_3 = vaddq_s32(result_val_3, output_zero_point_dup);
    result_val_4 = vaddq_s32(result_val_4, output_zero_point_dup);
    result_val_1 = vmaxq_s32(vminq_s32(result_val_1, max_val_dup), min_val_dup);
    result_val_2 = vmaxq_s32(vminq_s32(result_val_2, max_val_dup), min_val_dup);
    result_val_3 = vmaxq_s32(vminq_s32(result_val_3, max_val_dup), min_val_dup);
    result_val_4 = vmaxq_s32(vminq_s32(result_val_4, max_val_dup), min_val_dup);

    const uint32x4_t result_val_1_unsigned =
        vreinterpretq_u32_s32(result_val_1);
    const uint32x4_t result_val_2_unsigned =
        vreinterpretq_u32_s32(result_val_2);
    const uint32x4_t result_val_3_unsigned =
        vreinterpretq_u32_s32(result_val_3);
    const uint32x4_t result_val_4_unsigned =
        vreinterpretq_u32_s32(result_val_4);

    const uint16x4_t narrowed_val_1 = vqmovn_u32(result_val_1_unsigned);
    const uint16x4_t narrowed_val_2 = vqmovn_u32(result_val_2_unsigned);
    const uint16x4_t narrowed_val_3 = vqmovn_u32(result_val_3_unsigned);
    const uint16x4_t narrowed_val_4 = vqmovn_u32(result_val_4_unsigned);
    const uint16x8_t output_first_half =
        vcombine_u16(narrowed_val_1, narrowed_val_2);
    const uint16x8_t output_second_half =
        vcombine_u16(narrowed_val_3, narrowed_val_4);
    const uint8x8_t narrowed_first_half = vqmovn_u16(output_first_half);
    const uint8x8_t narrowed_second_half = vqmovn_u16(output_second_half);
    const uint8x16_t result =
        vcombine_u8(narrowed_first_half, narrowed_second_half);
    vst1q_u8(output_data + i, result);
  }

#endif
  for (; i < size; ++i) {
    const int32_t input = input_data[i] - input_zeropoint;
    const int32_t output =
        MultiplyByQuantizedMultiplier(input, effective_scale_multiplier,
                                      effective_scale_shift) +
        output_zeropoint;
    const int32_t clamped_output =
        std::max(std::min(output, kMaxOutput), kMinOutput);
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

inline void HardSwish(const RuntimeShape& input_shape, const float* input_data,
                      const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("HardSwish/Float");
  auto size = MatchingFlatSize(input_shape, output_shape);
  int i = 0;
#ifdef USE_NEON
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t three = vdupq_n_f32(3.0f);
  const float32x4_t six = vdupq_n_f32(6.0f);
  const float32x4_t one_sixth = vdupq_n_f32(1.0f / 6.0f);

  for (; i <= size - 16; i += 16) {
    // 4x partially unrolled version of the loop below. Refer to its comments.
    const float32x4_t in_0 = vld1q_f32(input_data + i + 0);
    const float32x4_t in_1 = vld1q_f32(input_data + i + 4);
    const float32x4_t in_2 = vld1q_f32(input_data + i + 8);
    const float32x4_t in_3 = vld1q_f32(input_data + i + 12);
    const float32x4_t in_scaled_0 = vmulq_f32(in_0, one_sixth);
    const float32x4_t in_scaled_1 = vmulq_f32(in_1, one_sixth);
    const float32x4_t in_scaled_2 = vmulq_f32(in_2, one_sixth);
    const float32x4_t in_scaled_3 = vmulq_f32(in_3, one_sixth);
    const float32x4_t in_reluish_0 =
        vminq_f32(six, vmaxq_f32(zero, vaddq_f32(in_0, three)));
    const float32x4_t in_reluish_1 =
        vminq_f32(six, vmaxq_f32(zero, vaddq_f32(in_1, three)));
    const float32x4_t in_reluish_2 =
        vminq_f32(six, vmaxq_f32(zero, vaddq_f32(in_2, three)));
    const float32x4_t in_reluish_3 =
        vminq_f32(six, vmaxq_f32(zero, vaddq_f32(in_3, three)));
    const float32x4_t product_0 = vmulq_f32(in_scaled_0, in_reluish_0);
    const float32x4_t product_1 = vmulq_f32(in_scaled_1, in_reluish_1);
    const float32x4_t product_2 = vmulq_f32(in_scaled_2, in_reluish_2);
    const float32x4_t product_3 = vmulq_f32(in_scaled_3, in_reluish_3);
    vst1q_f32(output_data + i + 0, product_0);
    vst1q_f32(output_data + i + 4, product_1);
    vst1q_f32(output_data + i + 8, product_2);
    vst1q_f32(output_data + i + 12, product_3);
  }
  for (; i <= size - 4; i += 4) {
    // The expression to be computed is:
    //   out = one_sixth * in * min(six, max(zero, (in + three)))
    // We structure the AST to have two roughly balanced, independent branches:
    //  - Multiplication: in_scaled = one_sixth * in.
    //  - Addition and clamping: in_reluish = min(six, max(zero, (in + three))).
    // Then the remaining multiplication at the root of the tree.
    const float32x4_t in = vld1q_f32(input_data + i);
    const float32x4_t in_scaled = vmulq_f32(in, one_sixth);
    const float32x4_t in_reluish =
        vminq_f32(six, vmaxq_f32(zero, vaddq_f32(in, three)));
    const float32x4_t product = vmulq_f32(in_scaled, in_reluish);
    vst1q_f32(output_data + i, product);
  }
#endif
  for (; i < size; i++) {
    const float in = input_data[i];
    output_data[i] =
        in * std::min(6.0f, std::max(0.0f, in + 3.0f)) * (1.0f / 6.0f);
  }
}

#ifdef USE_NEON
inline void SaturateAndStore(int16x8_t src, std::uint8_t* dst) {
  // Narrow values down to 8 bit unsigned, saturating.
  uint8x8_t res8 = vqmovun_s16(src);
  // Store results to destination.
  vst1_u8(dst, res8);
}

inline void SaturateAndStore(int16x8_t src, std::int8_t* dst) {
  // Narrow values down to 8 bit unsigned, saturating.
  int8x8_t res8 = vqmovn_s16(src);
  // Store results to destination.
  vst1_s8(dst, res8);
}
#endif

template <typename T>
inline void HardSwish(const HardSwishParams& params,
                      const RuntimeShape& input_shape, const T* input_data,
                      const RuntimeShape& output_shape, T* output_data) {
  gemmlowp::ScopedProfilingLabel label("HardSwish/Quantized");

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  int i = 0;
  // This code heavily uses NEON saturating left shifts (vqshl*) with shift
  // amounts that can be zero, in which case we rely on the correct behavior
  // of a left shift by zero returning just its first operand unmodified.
  // Unfortunately, the Intel arm_neon_sse.h implementation of vqshl* is
  // buggy in the case of zero shift amounts, see b/137199585. That is why
  // this NEON code path is restricted to true ARM NEON, excluding
  // arm_neon_sse.h. Anyway, the arm_neon_sse.h implemenation of saturating
  // left shifts is slow scalar code, so there may not be much benefit in
  // running that over just plain reference code.
  //
  // TODO(b/137199585): revisit when this is fixed.
#ifdef __ARM_NEON
  const int16x8_t positive_reluish_multiplier_exponent_minus_one =
      vdupq_n_s16(std::max(0, params.reluish_multiplier_exponent - 1));
  const int16x8_t positive_reluish_multiplier_exponent_last_bit =
      vdupq_n_s16(params.reluish_multiplier_exponent > 0 ? 1 : 0);
  const int16x8_t negative_reluish_multiplier_exponent =
      vdupq_n_s16(std::min(0, params.reluish_multiplier_exponent));
  const int16x8_t constant_32767 = vdupq_n_s16(32767);
  const int16x8_t output_multiplier_exponent =
      vdupq_n_s16(params.output_multiplier_exponent);
  const int16x8_t output_zero_point = vdupq_n_s16(params.output_zero_point);
  // 4x unrolled version of the below NEON loop. Read that first.
  for (; i <= flat_size - 32; i += 32) {
    using cpu_backend_gemm::detail::Load16AndSubtractZeroPoint;
    const int16x8x2_t input_value_0_1 =
        Load16AndSubtractZeroPoint(input_data + i, params.input_zero_point);
    const int16x8x2_t input_value_2_3 = Load16AndSubtractZeroPoint(
        input_data + i + 16, params.input_zero_point);
    const int16x8_t input_value_on_hires_input_scale_0 =
        vshlq_n_s16(input_value_0_1.val[0], 7);
    const int16x8_t input_value_on_hires_input_scale_1 =
        vshlq_n_s16(input_value_0_1.val[1], 7);
    const int16x8_t input_value_on_hires_input_scale_2 =
        vshlq_n_s16(input_value_2_3.val[0], 7);
    const int16x8_t input_value_on_hires_input_scale_3 =
        vshlq_n_s16(input_value_2_3.val[1], 7);
    const int16x8_t input_value_on_preshift_output_scale_0 =
        vqrdmulhq_n_s16(input_value_on_hires_input_scale_0,
                        params.output_multiplier_fixedpoint_int16);
    const int16x8_t input_value_on_preshift_output_scale_1 =
        vqrdmulhq_n_s16(input_value_on_hires_input_scale_1,
                        params.output_multiplier_fixedpoint_int16);
    const int16x8_t input_value_on_preshift_output_scale_2 =
        vqrdmulhq_n_s16(input_value_on_hires_input_scale_2,
                        params.output_multiplier_fixedpoint_int16);
    const int16x8_t input_value_on_preshift_output_scale_3 =
        vqrdmulhq_n_s16(input_value_on_hires_input_scale_3,
                        params.output_multiplier_fixedpoint_int16);
    int16x8_t reluish_value_0 = input_value_on_hires_input_scale_0;
    int16x8_t reluish_value_1 = input_value_on_hires_input_scale_1;
    int16x8_t reluish_value_2 = input_value_on_hires_input_scale_2;
    int16x8_t reluish_value_3 = input_value_on_hires_input_scale_3;
    reluish_value_0 = vqshlq_s16(
        reluish_value_0, positive_reluish_multiplier_exponent_minus_one);
    reluish_value_1 = vqshlq_s16(
        reluish_value_1, positive_reluish_multiplier_exponent_minus_one);
    reluish_value_2 = vqshlq_s16(
        reluish_value_2, positive_reluish_multiplier_exponent_minus_one);
    reluish_value_3 = vqshlq_s16(
        reluish_value_3, positive_reluish_multiplier_exponent_minus_one);
    reluish_value_0 = vqrdmulhq_n_s16(
        reluish_value_0, params.reluish_multiplier_fixedpoint_int16);
    reluish_value_1 = vqrdmulhq_n_s16(
        reluish_value_1, params.reluish_multiplier_fixedpoint_int16);
    reluish_value_2 = vqrdmulhq_n_s16(
        reluish_value_2, params.reluish_multiplier_fixedpoint_int16);
    reluish_value_3 = vqrdmulhq_n_s16(
        reluish_value_3, params.reluish_multiplier_fixedpoint_int16);
    reluish_value_0 = vqshlq_s16(reluish_value_0,
                                 positive_reluish_multiplier_exponent_last_bit);
    reluish_value_1 = vqshlq_s16(reluish_value_1,
                                 positive_reluish_multiplier_exponent_last_bit);
    reluish_value_2 = vqshlq_s16(reluish_value_2,
                                 positive_reluish_multiplier_exponent_last_bit);
    reluish_value_3 = vqshlq_s16(reluish_value_3,
                                 positive_reluish_multiplier_exponent_last_bit);
    reluish_value_0 =
        vrshlq_s16(reluish_value_0, negative_reluish_multiplier_exponent);
    reluish_value_1 =
        vrshlq_s16(reluish_value_1, negative_reluish_multiplier_exponent);
    reluish_value_2 =
        vrshlq_s16(reluish_value_2, negative_reluish_multiplier_exponent);
    reluish_value_3 =
        vrshlq_s16(reluish_value_3, negative_reluish_multiplier_exponent);
    reluish_value_0 = vrhaddq_s16(reluish_value_0, constant_32767);
    reluish_value_1 = vrhaddq_s16(reluish_value_1, constant_32767);
    reluish_value_2 = vrhaddq_s16(reluish_value_2, constant_32767);
    reluish_value_3 = vrhaddq_s16(reluish_value_3, constant_32767);
    const int16x8_t preshift_output_value_0 =
        vqdmulhq_s16(reluish_value_0, input_value_on_preshift_output_scale_0);
    const int16x8_t preshift_output_value_1 =
        vqdmulhq_s16(reluish_value_1, input_value_on_preshift_output_scale_1);
    const int16x8_t preshift_output_value_2 =
        vqdmulhq_s16(reluish_value_2, input_value_on_preshift_output_scale_2);
    const int16x8_t preshift_output_value_3 =
        vqdmulhq_s16(reluish_value_3, input_value_on_preshift_output_scale_3);
    int16x8_t output_value_0 =
        vrshlq_s16(preshift_output_value_0, output_multiplier_exponent);
    int16x8_t output_value_1 =
        vrshlq_s16(preshift_output_value_1, output_multiplier_exponent);
    int16x8_t output_value_2 =
        vrshlq_s16(preshift_output_value_2, output_multiplier_exponent);
    int16x8_t output_value_3 =
        vrshlq_s16(preshift_output_value_3, output_multiplier_exponent);
    output_value_0 = vaddq_s16(output_value_0, output_zero_point);
    output_value_1 = vaddq_s16(output_value_1, output_zero_point);
    output_value_2 = vaddq_s16(output_value_2, output_zero_point);
    output_value_3 = vaddq_s16(output_value_3, output_zero_point);
    SaturateAndStore(output_value_0, output_data + i);
    SaturateAndStore(output_value_1, output_data + i + 8);
    SaturateAndStore(output_value_2, output_data + i + 16);
    SaturateAndStore(output_value_3, output_data + i + 24);
  }
  // NEON version of reference_ops::HardSwish. Read that first.
  for (; i <= flat_size - 8; i += 8) {
    using cpu_backend_gemm::detail::Load8AndSubtractZeroPoint;
    const int16x8_t input_value =
        Load8AndSubtractZeroPoint(input_data + i, params.input_zero_point);
    const int16x8_t input_value_on_hires_input_scale =
        vshlq_n_s16(input_value, 7);
    const int16x8_t input_value_on_preshift_output_scale =
        vqrdmulhq_n_s16(input_value_on_hires_input_scale,
                        params.output_multiplier_fixedpoint_int16);
    int16x8_t reluish_value = input_value_on_hires_input_scale;
    reluish_value = vqshlq_s16(reluish_value,
                               positive_reluish_multiplier_exponent_minus_one);
    reluish_value = vqrdmulhq_n_s16(reluish_value,
                                    params.reluish_multiplier_fixedpoint_int16);
    reluish_value = vqshlq_s16(reluish_value,
                               positive_reluish_multiplier_exponent_last_bit);
    reluish_value =
        vrshlq_s16(reluish_value, negative_reluish_multiplier_exponent);
    reluish_value = vrhaddq_s16(reluish_value, constant_32767);
    const int16x8_t preshift_output_value =
        vqdmulhq_s16(reluish_value, input_value_on_preshift_output_scale);
    int16x8_t output_value =
        vrshlq_s16(preshift_output_value, output_multiplier_exponent);
    output_value = vaddq_s16(output_value, output_zero_point);
    SaturateAndStore(output_value, output_data + i);
  }
#endif
  // TODO(b/137208495): revisit when unit tests cover reference code.
  // Fall back to reference_ops::HardSwish. In general we have preferred
  // to duplicate such scalar code rather than call reference code to handle
  // leftovers, thinking that code duplication was not a big concern.
  // However, most of our unit tests happen to test only optimized code,
  // and the quantized HardSwish implementation is nontrivial enough that
  // I really want test coverage for the reference code.
  if (i < flat_size) {
    const RuntimeShape leftover_shape{flat_size - i};
    reference_ops::HardSwish(params, leftover_shape, input_data + i,
                             leftover_shape, output_data + i);
  }
}

template <typename T>
inline void IntegerExponentPow(const ArithmeticParams& params,
                               const RuntimeShape& unextended_base_shape,
                               const T* base_data, const int exponent,
                               const RuntimeShape& unextended_output_shape,
                               T* output_data) {
  TFLITE_DCHECK_GE(exponent, 1);
  if (exponent == 1) {
    // copy data over.
    std::memcpy(output_data, base_data,
                unextended_base_shape.FlatSize() * sizeof(T));
  } else {
    IntegerExponentPow(params, unextended_base_shape, base_data, exponent / 2,
                       unextended_output_shape, output_data);
    Mul(params, unextended_base_shape, output_data, unextended_base_shape,
        output_data, unextended_output_shape, output_data);
    if (exponent % 2 == 1) {
      Mul(params, unextended_base_shape, base_data, unextended_base_shape,
          output_data, unextended_output_shape, output_data);
    }
  }
}

template <typename T>
inline void BroadcastPow4D(const RuntimeShape& unextended_input1_shape,
                           const T* input1_data,
                           const RuntimeShape& unextended_input2_shape,
                           const T* input2_data,
                           const RuntimeShape& unextended_output_shape,
                           T* output_data) {
  gemmlowp::ScopedProfilingLabel label("PowBroadcast");

  if (unextended_input2_shape.FlatSize() == 1) {
    static const float epsilon = 1e-5;
    const T exponent = input2_data[0];
    const int int_exponent = static_cast<int>(std::round(exponent));
    if ((std::abs(input2_data[0] - int_exponent) < epsilon) &&
        (int_exponent >= 1)) {
      ArithmeticParams params;
      if (std::is_same<T, float>::value) {
        params.float_activation_max = std::numeric_limits<float>::max();
        params.float_activation_min = std::numeric_limits<float>::lowest();
      } else if (std::is_same<T, int>::value) {
        params.quantized_activation_max = std::numeric_limits<int>::max();
        params.quantized_activation_min = std::numeric_limits<int>::lowest();
      }
      IntegerExponentPow(params, unextended_input1_shape, input1_data,
                         int_exponent, unextended_output_shape, output_data);
      return;
    }
  }
  reference_ops::BroadcastPow4DSlow(unextended_input1_shape, input1_data,
                                    unextended_input2_shape, input2_data,
                                    unextended_output_shape, output_data);
}

#ifdef USE_NEON

inline void ScaleWithNewZeroPoint(const int32x4_t input,
                                  const float32x4_t scale_dup,
                                  const float32x4_t zero_times_scale_dup,
                                  float32x4_t* output) {
#ifdef __ARM_FEATURE_FMA
  *output = vfmaq_f32(zero_times_scale_dup, vcvtq_f32_s32(input), scale_dup);
#else
  *output = vaddq_f32(vmulq_f32(vcvtq_f32_s32(input), scale_dup),
                      zero_times_scale_dup);
#endif
}

#endif  // USE_NEON

inline void Dequantize(const tflite::DequantizationParams& op_params,
                       const RuntimeShape& input_shape,
                       const uint8_t* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("Dequantize/Uint8");
  const int32 zero_point = op_params.zero_point;
  const double scale = op_params.scale;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  int i = 0;
#ifdef USE_NEON
  const float32x4_t scale_dup = vdupq_n_f32(static_cast<float>(scale));
  const float32x4_t zero_times_scale_dup =
      vdupq_n_f32(static_cast<float>(-zero_point * scale));
  for (; i <= flat_size - 8; i += 8) {
    const uint8x8_t input_u8 = vld1_u8(input_data + i);
    const uint16x8_t input_u16 = vmovl_u8(input_u8);
    const int16x8_t input_s16 = vreinterpretq_s16_u16(input_u16);
    const int16x4_t input_s16_low = vget_low_s16(input_s16);
    const int16x4_t input_s16_high = vget_high_s16(input_s16);
    const int32x4_t val_low = vmovl_s16(input_s16_low);
    const int32x4_t val_high = vmovl_s16(input_s16_high);

    float32x4_t result_low, result_high;
    ScaleWithNewZeroPoint(val_low, scale_dup, zero_times_scale_dup,
                          &result_low);
    ScaleWithNewZeroPoint(val_high, scale_dup, zero_times_scale_dup,
                          &result_high);

    vst1q_f32(output_data + i, result_low);
    vst1q_f32(output_data + i + 4, result_high);
  }
#endif  // NEON
  for (; i < flat_size; ++i) {
    const int32 val = input_data[i];
    const float result = static_cast<float>(scale * (val - zero_point));
    output_data[i] = result;
  }
}

inline void Dequantize(const tflite::DequantizationParams& op_params,
                       const RuntimeShape& input_shape,
                       const int8_t* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("Dequantize/Int8");
  const int32 zero_point = op_params.zero_point;
  const double scale = op_params.scale;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  int i = 0;
#ifdef USE_NEON
  const float32x4_t scale_dup = vdupq_n_f32(static_cast<float>(scale));
  const float32x4_t zero_times_scale_dup =
      vdupq_n_f32(static_cast<float>(-zero_point * scale));
  for (; i <= flat_size - 8; i += 8) {
    const int8x8_t input_s8 = vld1_s8(input_data + i);
    const int16x8_t input_s16 = vmovl_s8(input_s8);
    const int16x4_t input_s16_low = vget_low_s16(input_s16);
    const int16x4_t input_s16_high = vget_high_s16(input_s16);
    const int32x4_t val_low = vmovl_s16(input_s16_low);
    const int32x4_t val_high = vmovl_s16(input_s16_high);

    float32x4_t result_low, result_high;
    ScaleWithNewZeroPoint(val_low, scale_dup, zero_times_scale_dup,
                          &result_low);
    ScaleWithNewZeroPoint(val_high, scale_dup, zero_times_scale_dup,
                          &result_high);

    vst1q_f32(output_data + i, result_low);
    vst1q_f32(output_data + i + 4, result_high);
  }
#endif  // NEON
  for (; i < flat_size; ++i) {
    const int32 val = input_data[i];
    const float result = static_cast<float>(scale * (val - zero_point));
    output_data[i] = result;
  }
}

inline void Dequantize(const tflite::DequantizationParams& op_params,
                       const RuntimeShape& input_shape,
                       const int16_t* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  gemmlowp::ScopedProfilingLabel label("Dequantize/Int16");
  const int32 zero_point = op_params.zero_point;
  const double scale = op_params.scale;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  int i = 0;
#ifdef USE_NEON
  const float32x4_t scale_dup = vdupq_n_f32(static_cast<float>(scale));
  const float32x4_t zero_times_scale_dup =
      vdupq_n_f32(static_cast<float>(-zero_point * scale));
  for (; i <= flat_size - 8; i += 8) {
    const int16x4_t input_s16_low = vld1_s16(input_data + i);
    const int16x4_t input_s16_high = vld1_s16(input_data + i + 4);
    const int32x4_t val_low = vmovl_s16(input_s16_low);
    const int32x4_t val_high = vmovl_s16(input_s16_high);

    float32x4_t result_low, result_high;
    ScaleWithNewZeroPoint(val_low, scale_dup, zero_times_scale_dup,
                          &result_low);
    ScaleWithNewZeroPoint(val_high, scale_dup, zero_times_scale_dup,
                          &result_high);

    vst1q_f32(output_data + i, result_low);
    vst1q_f32(output_data + i + 4, result_high);
  }
#endif  // NEON
  for (; i < flat_size; ++i) {
    const int32 val = input_data[i];
    const float result = static_cast<float>(scale * (val - zero_point));
    output_data[i] = result;
  }
}

inline void Dequantize(const RuntimeShape& input_shape,
                       const Eigen::half* input_data,
                       const RuntimeShape& output_shape, float* output_data) {
  reference_ops::Dequantize(input_shape, input_data, output_shape, output_data);
}

template <typename T>
inline void AffineQuantize(const tflite::QuantizationParams& op_params,
                           const RuntimeShape& input_shape,
                           const float* input_data,
                           const RuntimeShape& output_shape, T* output_data) {
  reference_ops::AffineQuantize(op_params, input_shape, input_data,
                                output_shape, output_data);
}

template <>
inline void AffineQuantize(const tflite::QuantizationParams& op_params,
                           const RuntimeShape& input_shape,
                           const float* input_data,
                           const RuntimeShape& output_shape,
                           int8_t* output_data) {
  gemmlowp::ScopedProfilingLabel label("Quantize/Int8");
  const int32 zero_point = op_params.zero_point;
  const double scale = static_cast<double>(op_params.scale);
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static constexpr int32 min_val = std::numeric_limits<int8_t>::min();
  static constexpr int32 max_val = std::numeric_limits<int8_t>::max();

  int i = 0;
#ifdef USE_NEON
  const float32x4_t reverse_scale_dup = vdupq_n_f32(1.0f / scale);
  const int32x4_t zero_point_dup = vdupq_n_s32(zero_point);
  const int32x4_t min_val_dup = vdupq_n_s32(min_val);
  const int32x4_t max_val_dup = vdupq_n_s32(max_val);

  for (; i <= flat_size - 8; i += 8) {
    const float* src_data_ptr = input_data + i;
    float32x4_t input_val_0 = vld1q_f32(src_data_ptr);
    float32x4_t input_val_1 = vld1q_f32(src_data_ptr + 4);

    input_val_0 = vmulq_f32(input_val_0, reverse_scale_dup);
    input_val_1 = vmulq_f32(input_val_1, reverse_scale_dup);

    int32x4_t casted_val_0 = RoundToNearest(input_val_0);
    int32x4_t casted_val_1 = RoundToNearest(input_val_1);

    casted_val_0 = vaddq_s32(casted_val_0, zero_point_dup);
    casted_val_1 = vaddq_s32(casted_val_1, zero_point_dup);

    // Clamp the values to fit the target type's range.
    casted_val_0 = vmaxq_s32(casted_val_0, min_val_dup);
    casted_val_1 = vmaxq_s32(casted_val_1, min_val_dup);
    casted_val_0 = vminq_s32(casted_val_0, max_val_dup);
    casted_val_1 = vminq_s32(casted_val_1, max_val_dup);

    const int16x4_t narrowed_val_0 = vmovn_s32(casted_val_0);
    const int16x4_t narrowed_val_1 = vmovn_s32(casted_val_1);
    const int16x8_t combined_val = vcombine_s16(narrowed_val_0, narrowed_val_1);
    const int8x8_t combined_val_narrowed = vmovn_s16(combined_val);
    vst1_s8(output_data + i, combined_val_narrowed);
  }
#endif  // NEON

  for (; i < flat_size; ++i) {
    const float val = input_data[i];
    const int32 unclamped =
        static_cast<int32>(TfLiteRound(val / scale)) + zero_point;
    const int32 clamped = std::min(std::max(unclamped, min_val), max_val);
    output_data[i] = clamped;
  }
}

template <>
inline void AffineQuantize(const tflite::QuantizationParams& op_params,
                           const RuntimeShape& input_shape,
                           const float* input_data,
                           const RuntimeShape& output_shape,
                           uint8_t* output_data) {
  gemmlowp::ScopedProfilingLabel label("Quantize/Uint8");
  const int32 zero_point = op_params.zero_point;
  const double scale = static_cast<double>(op_params.scale);
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static constexpr int32 min_val = std::numeric_limits<uint8_t>::min();
  static constexpr int32 max_val = std::numeric_limits<uint8_t>::max();

  int i = 0;
#ifdef USE_NEON
  const float32x4_t reverse_scale_dup = vdupq_n_f32(1.0f / scale);
  const int32x4_t zero_point_dup = vdupq_n_s32(zero_point);
  const int32x4_t min_val_dup = vdupq_n_s32(min_val);
  const int32x4_t max_val_dup = vdupq_n_s32(max_val);

  for (; i <= flat_size - 8; i += 8) {
    const float* src_data_ptr = input_data + i;
    float32x4_t input_val_0 = vld1q_f32(src_data_ptr);
    float32x4_t input_val_1 = vld1q_f32(src_data_ptr + 4);

    input_val_0 = vmulq_f32(input_val_0, reverse_scale_dup);
    input_val_1 = vmulq_f32(input_val_1, reverse_scale_dup);

    int32x4_t casted_val_0 = RoundToNearest(input_val_0);
    int32x4_t casted_val_1 = RoundToNearest(input_val_1);

    casted_val_0 = vaddq_s32(casted_val_0, zero_point_dup);
    casted_val_1 = vaddq_s32(casted_val_1, zero_point_dup);

    // Clamp the values to fit the target type's range.
    casted_val_0 = vmaxq_s32(casted_val_0, min_val_dup);
    casted_val_1 = vmaxq_s32(casted_val_1, min_val_dup);
    casted_val_0 = vminq_s32(casted_val_0, max_val_dup);
    casted_val_1 = vminq_s32(casted_val_1, max_val_dup);

    const uint16x4_t narrowed_val_0 = vqmovun_s32(casted_val_0);
    const uint16x4_t narrowed_val_1 = vqmovun_s32(casted_val_1);
    const uint16x8_t combined_val =
        vcombine_u16(narrowed_val_0, narrowed_val_1);
    const uint8x8_t combined_val_narrowed = vmovn_u16(combined_val);
    vst1_u8(output_data + i, combined_val_narrowed);
  }
#endif  // NEON

  for (; i < flat_size; ++i) {
    const float val = input_data[i];
    const int32 unclamped =
        static_cast<int32>(TfLiteRound(val / scale)) + zero_point;
    const int32 clamped = std::min(std::max(unclamped, min_val), max_val);
    output_data[i] = clamped;
  }
}

template <>
inline void AffineQuantize(const tflite::QuantizationParams& op_params,
                           const RuntimeShape& input_shape,
                           const float* input_data,
                           const RuntimeShape& output_shape,
                           int16_t* output_data) {
  gemmlowp::ScopedProfilingLabel label("Quantize/Int16");
  const int32 zero_point = op_params.zero_point;
  const double scale = static_cast<double>(op_params.scale);
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static constexpr int32 min_val = std::numeric_limits<int16_t>::min();
  static constexpr int32 max_val = std::numeric_limits<int16_t>::max();

  int i = 0;
#ifdef USE_NEON
  const float32x4_t reverse_scale_dup = vdupq_n_f32(1.0f / scale);
  const int32x4_t zero_point_dup = vdupq_n_s32(zero_point);
  const int32x4_t min_val_dup = vdupq_n_s32(min_val);
  const int32x4_t max_val_dup = vdupq_n_s32(max_val);

  for (; i <= flat_size - 8; i += 8) {
    const float* src_data_ptr = input_data + i;
    float32x4_t input_val_0 = vld1q_f32(src_data_ptr);
    float32x4_t input_val_1 = vld1q_f32(src_data_ptr + 4);

    input_val_0 = vmulq_f32(input_val_0, reverse_scale_dup);
    input_val_1 = vmulq_f32(input_val_1, reverse_scale_dup);

    int32x4_t casted_val_0 = RoundToNearest(input_val_0);
    int32x4_t casted_val_1 = RoundToNearest(input_val_1);

    casted_val_0 = vaddq_s32(casted_val_0, zero_point_dup);
    casted_val_1 = vaddq_s32(casted_val_1, zero_point_dup);

    // Clamp the values to fit the target type's range.
    casted_val_0 = vmaxq_s32(casted_val_0, min_val_dup);
    casted_val_1 = vmaxq_s32(casted_val_1, min_val_dup);
    casted_val_0 = vminq_s32(casted_val_0, max_val_dup);
    casted_val_1 = vminq_s32(casted_val_1, max_val_dup);

    const int16x4_t narrowed_val_0 = vmovn_s32(casted_val_0);
    const int16x4_t narrowed_val_1 = vmovn_s32(casted_val_1);
    vst1_s16(output_data + i, narrowed_val_0);
    vst1_s16(output_data + i + 4, narrowed_val_1);
  }
#endif  // NEON

  for (; i < flat_size; ++i) {
    const float val = input_data[i];
    const int32 unclamped =
        static_cast<int32>(TfLiteRound(val / scale)) + zero_point;
    const int32 clamped = std::min(std::max(unclamped, min_val), max_val);
    output_data[i] = clamped;
  }
}

// TODO(b/139252020): Replace GEMMLOWP_NEON with USE_NEON when the bug is fixed.
// The converted versions of gemmlowp::tanh and gemmlowp::logistic, done by
// arm_sse_2_neon.h, produce incorrect results with int16x8_t data types.
#ifdef GEMMLOWP_NEON

inline int16x8x4_t SaturatingRounding(
    int16x8_t input_val_0, int16x8_t input_val_1, int16x8_t input_val_2,
    int16x8_t input_val_3, int input_left_shift, int input_multiplier) {
  // This performs what is expressed in the scalar code as
  // const int16 input_val_rescaled = SaturatingRoundingDoublingHighMul(
  //      static_cast<int16>(input_val_centered * (1 << input_left_shift)),
  //      static_cast<int16>(input_multiplier));
  const int16x8_t left_shift_dup = vdupq_n_s16(input_left_shift);
  const int16x8_t input_val_shifted_0 = vshlq_s16(input_val_0, left_shift_dup);
  const int16x8_t input_val_shifted_1 = vshlq_s16(input_val_1, left_shift_dup);
  const int16x8_t input_val_shifted_2 = vshlq_s16(input_val_2, left_shift_dup);
  const int16x8_t input_val_shifted_3 = vshlq_s16(input_val_3, left_shift_dup);
  int16x8x4_t result;
  result.val[0] = vqrdmulhq_n_s16(input_val_shifted_0, input_multiplier);
  result.val[1] = vqrdmulhq_n_s16(input_val_shifted_1, input_multiplier);
  result.val[2] = vqrdmulhq_n_s16(input_val_shifted_2, input_multiplier);
  result.val[3] = vqrdmulhq_n_s16(input_val_shifted_3, input_multiplier);
  return result;
}

// 4-bit fixed point is enough for tanh since tanh(16) is almost same with one,
// considering 7 digits under zero.
inline int16x8x4_t FixedPoint4Logistic(int16x8x4_t input_val) {
  // Invoke gemmlowp::logistic on FixedPoint wrapping int16x8_t
  using FixedPoint4 = gemmlowp::FixedPoint<int16x8_t, 4>;
  using FixedPoint0 = gemmlowp::FixedPoint<int16x8_t, 0>;
  const FixedPoint4 input_val_f4_0 = FixedPoint4::FromRaw(input_val.val[0]);
  const FixedPoint4 input_val_f4_1 = FixedPoint4::FromRaw(input_val.val[1]);
  const FixedPoint4 input_val_f4_2 = FixedPoint4::FromRaw(input_val.val[2]);
  const FixedPoint4 input_val_f4_3 = FixedPoint4::FromRaw(input_val.val[3]);

  // TODO(b/134622898) Implement a low accuracy version of logistic. In this
  // method, gemmlowp::tanh spends about 80% of the execution times. The
  // current implementation is rougly 12-bit accurate in the 16-bit fixed
  // point case. Until reaching to error bounds, there are rooms for
  // improvements.
  const FixedPoint0 output_val_f0_0 = gemmlowp::logistic(input_val_f4_0);
  const FixedPoint0 output_val_f0_1 = gemmlowp::logistic(input_val_f4_1);
  const FixedPoint0 output_val_f0_2 = gemmlowp::logistic(input_val_f4_2);
  const FixedPoint0 output_val_f0_3 = gemmlowp::logistic(input_val_f4_3);

  // Divide by 2^7 as in the scalar code
  int16x8x4_t result;
  result.val[0] = vrshrq_n_s16(output_val_f0_0.raw(), 7);
  result.val[1] = vrshrq_n_s16(output_val_f0_1.raw(), 7);
  result.val[2] = vrshrq_n_s16(output_val_f0_2.raw(), 7);
  result.val[3] = vrshrq_n_s16(output_val_f0_3.raw(), 7);
  return result;
}

// 4-bit fixed point is enough for tanh since tanh(16) is almost same with one,
// considering 11 digits under zero at least.
inline int16x8x4_t FixedPoint4Tanh(int16x8x4_t input_val) {
  // Invoke gemmlowp::logistic on FixedPoint wrapping int16x8_t
  using FixedPoint4 = gemmlowp::FixedPoint<int16x8_t, 4>;
  using FixedPoint0 = gemmlowp::FixedPoint<int16x8_t, 0>;
  const FixedPoint4 input_val_f4_0 = FixedPoint4::FromRaw(input_val.val[0]);
  const FixedPoint4 input_val_f4_1 = FixedPoint4::FromRaw(input_val.val[1]);
  const FixedPoint4 input_val_f4_2 = FixedPoint4::FromRaw(input_val.val[2]);
  const FixedPoint4 input_val_f4_3 = FixedPoint4::FromRaw(input_val.val[3]);

  // TODO(b/134622898) Implement a low accuracy version of logistic. In this
  // method, gemmlowp::tanh spends about 80% of the execution times. The
  // current implementation is rougly 12-bit accurate in the 16-bit fixed
  // point case. Until reaching to error bounds, there are rooms for
  // improvements.
  const FixedPoint0 output_val_f0_0 = gemmlowp::tanh(input_val_f4_0);
  const FixedPoint0 output_val_f0_1 = gemmlowp::tanh(input_val_f4_1);
  const FixedPoint0 output_val_f0_2 = gemmlowp::tanh(input_val_f4_2);
  const FixedPoint0 output_val_f0_3 = gemmlowp::tanh(input_val_f4_3);

  // Divide by 2^7 as in the scalar code
  int16x8x4_t result;
  result.val[0] = vrshrq_n_s16(output_val_f0_0.raw(), 8);
  result.val[1] = vrshrq_n_s16(output_val_f0_1.raw(), 8);
  result.val[2] = vrshrq_n_s16(output_val_f0_2.raw(), 8);
  result.val[3] = vrshrq_n_s16(output_val_f0_3.raw(), 8);
  return result;
}

inline uint8x16x2_t CalculateUnsignedClampingWithRangeBitMasks(
    int16x8x2_t input_val, int16x8_t range_radius_dup,
    int16x8_t neg_range_radius_dup) {
  const uint16x8_t mask_rightclamp_0 =
      vcgtq_s16(input_val.val[0], range_radius_dup);
  const uint16x8_t mask_rightclamp_1 =
      vcgtq_s16(input_val.val[1], range_radius_dup);

  const uint16x8_t mask_leftclamp_0 =
      vcgeq_s16(input_val.val[0], neg_range_radius_dup);
  const uint16x8_t mask_leftclamp_1 =
      vcgeq_s16(input_val.val[1], neg_range_radius_dup);

  uint8x16x2_t result;
  result.val[0] = vcombine_u8(vshrn_n_u16(mask_leftclamp_0, 8),
                              vshrn_n_u16(mask_leftclamp_1, 8));
  result.val[1] = vcombine_u8(vshrn_n_u16(mask_rightclamp_0, 8),
                              vshrn_n_u16(mask_rightclamp_1, 8));
  return result;
}

inline uint8x16x2_t CalculateSignedClampingWithRangeBitMasks(
    int16x8x2_t input_val, int16x8_t range_radius_dup,
    int16x8_t neg_range_radius_dup) {
  const uint16x8_t mask_rightclamp_0 =
      vcgtq_s16(input_val.val[0], range_radius_dup);
  const uint16x8_t mask_rightclamp_1 =
      vcgtq_s16(input_val.val[1], range_radius_dup);

  const uint16x8_t mask_leftclamp_0 =
      vcltq_s16(input_val.val[0], neg_range_radius_dup);
  const uint16x8_t mask_leftclamp_1 =
      vcltq_s16(input_val.val[1], neg_range_radius_dup);

  uint8x16x2_t result;
  result.val[0] = vcombine_u8(vshrn_n_u16(mask_leftclamp_0, 8),
                              vshrn_n_u16(mask_leftclamp_1, 8));
  result.val[1] = vcombine_u8(vshrn_n_u16(mask_rightclamp_0, 8),
                              vshrn_n_u16(mask_rightclamp_1, 8));
  return result;
}

inline void ClampWithRangeAndStore(uint8_t* output_dst, uint8x16_t input_val,
                                   uint8x16x2_t masks_clamp) {
  // Store back to memory
  vst1q_u8(output_dst, vandq_u8(vorrq_u8(input_val, masks_clamp.val[1]),
                                masks_clamp.val[0]));
}

inline void ClampWithRangeAndStore(int8_t* output_dst, int8x16_t input_val,
                                   uint8x16x2_t masks_clamp) {
  static const int8x16_t max_dup = vdupq_n_s8(127);
  static const int8x16_t min_dup = vdupq_n_s8(-128);
  // Store back to memory
  vst1q_s8(output_dst,
           vbslq_s8(masks_clamp.val[1], max_dup,
                    vbslq_s8(masks_clamp.val[0], min_dup, input_val)));
}

#endif  // GEMMLOWP_NEON

inline void Tanh16bitPercision(const TanhParams& params,
                               const RuntimeShape& input_shape,
                               const uint8* input_data,
                               const RuntimeShape& output_shape,
                               uint8* output_data) {
  // Note that this is almost the exact same code as in Logistic().
  gemmlowp::ScopedProfilingLabel label("Tanh/Uint8");
  const int32 input_zero_point = params.input_zero_point;
  const int32 input_range_radius = params.input_range_radius;
  const int16 input_multiplier = static_cast<int16>(params.input_multiplier);
  const int16 input_left_shift = static_cast<int16>(params.input_left_shift);
  const int size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
  int16_t output_zero_point = 128;

// TODO(b/139252020): Replace GEMMLOWP_NEON with USE_NEON when the bug is fixed.
// The converted versions of gemmlowp::tanh and gemmlowp::logistic, done by
// arm_sse_2_neon.h, produce incorrect results with int16x8_t data types.
#ifdef GEMMLOWP_NEON
  const int16x8_t range_radius_dup = vdupq_n_s16(input_range_radius);
  const int16x8_t neg_range_radius_dup = vdupq_n_s16(-input_range_radius);
  const int16x8_t output_zero_point_s16 = vdupq_n_s16(output_zero_point);

  // Handle 32 values at a time
  for (; c <= size - 32; c += 32) {
    // Read input uint8 values, cast to int16 and subtract input_zero_point
    using cpu_backend_gemm::detail::Load16AndSubtractZeroPoint;
    const int16x8x2_t input_val_centered_0_1 =
        Load16AndSubtractZeroPoint(input_data + c, input_zero_point);
    const int16x8x2_t input_val_centered_2_3 =
        Load16AndSubtractZeroPoint(input_data + c + 16, input_zero_point);

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = 0;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 255;
    //   } else {
    //     ...
    uint8x16x2_t masks_clamp_0_1 = CalculateUnsignedClampingWithRangeBitMasks(
        input_val_centered_0_1, range_radius_dup, neg_range_radius_dup);
    uint8x16x2_t masks_clamp_2_3 = CalculateUnsignedClampingWithRangeBitMasks(
        input_val_centered_2_3, range_radius_dup, neg_range_radius_dup);

    int16x8x4_t input_val_rescaled = SaturatingRounding(
        input_val_centered_0_1.val[0], input_val_centered_0_1.val[1],
        input_val_centered_2_3.val[0], input_val_centered_2_3.val[1],
        input_left_shift, input_multiplier);

    int16x8x4_t output_val_s16 = FixedPoint4Tanh(input_val_rescaled);

    // Add the output zero point
    output_val_s16.val[0] =
        vaddq_s16(output_val_s16.val[0], output_zero_point_s16);
    output_val_s16.val[1] =
        vaddq_s16(output_val_s16.val[1], output_zero_point_s16);
    output_val_s16.val[2] =
        vaddq_s16(output_val_s16.val[2], output_zero_point_s16);
    output_val_s16.val[3] =
        vaddq_s16(output_val_s16.val[3], output_zero_point_s16);

    // Cast output values to uint8, saturating
    uint8x16_t output_val_u8_0_1 = vcombine_u8(
        vqmovun_s16(output_val_s16.val[0]), vqmovun_s16(output_val_s16.val[1]));
    uint8x16_t output_val_u8_2_3 = vcombine_u8(
        vqmovun_s16(output_val_s16.val[2]), vqmovun_s16(output_val_s16.val[3]));

    ClampWithRangeAndStore(output_data + c, output_val_u8_0_1, masks_clamp_0_1);
    ClampWithRangeAndStore(output_data + c + 16, output_val_u8_2_3,
                           masks_clamp_2_3);
  }
#endif  // GEMMLOWP_NEON
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const uint8 input_val_u8 = input_data[c];
    const int16 input_val_centered =
        static_cast<int16>(input_val_u8) - input_zero_point;
    uint8 output_val;
    if (input_val_centered < -input_range_radius) {
      output_val = 0;
    } else if (input_val_centered > input_range_radius) {
      output_val = 255;
    } else {
      using gemmlowp::SaturatingRoundingDoublingHighMul;
      const int16 input_val_rescaled = SaturatingRoundingDoublingHighMul(
          static_cast<int16>(input_val_centered * (1 << input_left_shift)),
          static_cast<int16>(input_multiplier));
      using FixedPoint4 = gemmlowp::FixedPoint<int16, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int16, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::tanh(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int16 output_val_s16 = RoundingDivideByPOT(output_val_f0.raw(), 8);
      output_val_s16 += output_zero_point;
      if (output_val_s16 == 256) {
        output_val_s16 = 255;
      }
      TFLITE_DCHECK_GE(output_val_s16, 0);
      TFLITE_DCHECK_LE(output_val_s16, 255);
      output_val = static_cast<uint8>(output_val_s16);
    }
    output_data[c] = output_val;
  }
}

inline void Tanh16bitPercision(const TanhParams& params,
                               const RuntimeShape& input_shape,
                               const int8* input_data,
                               const RuntimeShape& output_shape,
                               int8* output_data) {
  // Note that this is almost the exact same code as in Logistic().
  gemmlowp::ScopedProfilingLabel label("Tanh/Int8");
  const int32 input_zero_point = params.input_zero_point;
  const int32 input_range_radius = params.input_range_radius;
  const int16 input_multiplier = static_cast<int16>(params.input_multiplier);
  const int16 input_left_shift = static_cast<int16>(params.input_left_shift);
  const int size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
// TODO(b/139252020): Replace GEMMLOWP_NEON with USE_NEON when the bug is fixed.
// The converted versions of gemmlowp::tanh and gemmlowp::logistic, done by
// arm_sse_2_neon.h, produce incorrect results with int16x8_t data types.
#ifdef GEMMLOWP_NEON
  const int16x8_t range_radius_dup = vdupq_n_s16(input_range_radius);
  const int16x8_t neg_range_radius_dup = vdupq_n_s16(-input_range_radius);

  // Handle 32 values at a time
  for (; c <= size - 32; c += 32) {
    // Read input int8 values, cast to int16 and subtract input_zero_point
    using cpu_backend_gemm::detail::Load16AndSubtractZeroPoint;
    const int16x8x2_t input_val_centered_0_1 =
        Load16AndSubtractZeroPoint(input_data + c, input_zero_point);
    const int16x8x2_t input_val_centered_2_3 =
        Load16AndSubtractZeroPoint(input_data + c + 16, input_zero_point);

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = -128;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 127;
    //   } else {
    //     ...
    uint8x16x2_t masks_clamp_0_1 = CalculateSignedClampingWithRangeBitMasks(
        input_val_centered_0_1, range_radius_dup, neg_range_radius_dup);
    uint8x16x2_t masks_clamp_2_3 = CalculateSignedClampingWithRangeBitMasks(
        input_val_centered_2_3, range_radius_dup, neg_range_radius_dup);

    int16x8x4_t input_val_rescaled = SaturatingRounding(
        input_val_centered_0_1.val[0], input_val_centered_0_1.val[1],
        input_val_centered_2_3.val[0], input_val_centered_2_3.val[1],
        input_left_shift, input_multiplier);

    int16x8x4_t output_val_s16 = FixedPoint4Tanh(input_val_rescaled);

    // Cast output values to uint8, saturating
    int8x16_t output_val_s8_0_1 = vcombine_s8(
        vqmovn_s16(output_val_s16.val[0]), vqmovn_s16(output_val_s16.val[1]));
    int8x16_t output_val_s8_2_3 = vcombine_s8(
        vqmovn_s16(output_val_s16.val[2]), vqmovn_s16(output_val_s16.val[3]));

    ClampWithRangeAndStore(output_data + c, output_val_s8_0_1, masks_clamp_0_1);
    ClampWithRangeAndStore(output_data + c + 16, output_val_s8_2_3,
                           masks_clamp_2_3);
  }
#endif  // GEMMLOWP_NEON
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const int8 input_val_s8 = input_data[c];
    const int16 input_val_centered =
        static_cast<int16>(input_val_s8) - input_zero_point;
    int8 output_val;
    if (input_val_centered <= -input_range_radius) {
      output_val = -128;
    } else if (input_val_centered >= input_range_radius) {
      output_val = 127;
    } else {
      using gemmlowp::SaturatingRoundingDoublingHighMul;
      const int16 input_val_rescaled = SaturatingRoundingDoublingHighMul(
          static_cast<int16>(input_val_centered * (1 << input_left_shift)),
          static_cast<int16>(input_multiplier));
      using FixedPoint4 = gemmlowp::FixedPoint<int16, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int16, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::tanh(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int16 output_val_s16 = RoundingDivideByPOT(output_val_f0.raw(), 8);
      if (output_val_s16 == 128) {
        output_val_s16 = 127;
      }
      TFLITE_DCHECK_GE(output_val_s16, -128);
      TFLITE_DCHECK_LE(output_val_s16, 127);
      output_val = static_cast<int8>(output_val_s16);
    }
    output_data[c] = output_val;
  }
}

inline void Logistic16bitPercision(const LogisticParams& params,
                                   const RuntimeShape& input_shape,
                                   const uint8* input_data,
                                   const RuntimeShape& output_shape,
                                   uint8* output_data) {
  gemmlowp::ScopedProfilingLabel label("Logistic/Uint8");
  const int32 input_zero_point = params.input_zero_point;
  const int32 input_range_radius = params.input_range_radius;
  const int32 input_multiplier = params.input_multiplier;
  const int16 input_left_shift = static_cast<int16>(params.input_left_shift);
  const int size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
// TODO(b/139252020): Replace GEMMLOWP_NEON with USE_NEON when the bug is fixed.
// The converted versions of gemmlowp::tanh and gemmlowp::logistic, done by
// arm_sse_2_neon.h, produce incorrect results with int16x8_t data types.
#ifdef GEMMLOWP_NEON
  const int16x8_t range_radius_dup = vdupq_n_s16(input_range_radius);
  const int16x8_t neg_range_radius_dup = vdupq_n_s16(-input_range_radius);

  // Handle 32 values at a time
  for (; c <= size - 32; c += 32) {
    // Read input uint8 values, cast to int16 and subtract input_zero_point
    using cpu_backend_gemm::detail::Load16AndSubtractZeroPoint;
    const int16x8x2_t input_val_centered_0_1 =
        Load16AndSubtractZeroPoint(input_data + c, input_zero_point);
    const int16x8x2_t input_val_centered_2_3 =
        Load16AndSubtractZeroPoint(input_data + c + 16, input_zero_point);

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = 0;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 255;
    //   } else {
    //     ...
    uint8x16x2_t masks_clamp_0_1 = CalculateUnsignedClampingWithRangeBitMasks(
        input_val_centered_0_1, range_radius_dup, neg_range_radius_dup);
    uint8x16x2_t masks_clamp_2_3 = CalculateUnsignedClampingWithRangeBitMasks(
        input_val_centered_2_3, range_radius_dup, neg_range_radius_dup);

    int16x8x4_t input_val_rescaled = SaturatingRounding(
        input_val_centered_0_1.val[0], input_val_centered_0_1.val[1],
        input_val_centered_2_3.val[0], input_val_centered_2_3.val[1],
        input_left_shift, input_multiplier);

    int16x8x4_t output_val_s16 = FixedPoint4Logistic(input_val_rescaled);

    // Cast output values to uint8, saturating
    uint8x16_t output_val_u8_0_1 = vcombine_u8(
        vqmovun_s16(output_val_s16.val[0]), vqmovun_s16(output_val_s16.val[1]));
    uint8x16_t output_val_u8_2_3 = vcombine_u8(
        vqmovun_s16(output_val_s16.val[2]), vqmovun_s16(output_val_s16.val[3]));

    ClampWithRangeAndStore(output_data + c, output_val_u8_0_1, masks_clamp_0_1);
    ClampWithRangeAndStore(output_data + c + 16, output_val_u8_2_3,
                           masks_clamp_2_3);
  }
#endif  // GEMMLOWP_NEON
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const uint8 input_val_u8 = input_data[c];
    const int16 input_val_centered =
        static_cast<int16>(input_val_u8) - input_zero_point;
    uint8 output_val;
    if (input_val_centered < -input_range_radius) {
      output_val = 0;
    } else if (input_val_centered > input_range_radius) {
      output_val = 255;
    } else {
      using gemmlowp::SaturatingRoundingDoublingHighMul;
      const int16 input_val_rescaled = SaturatingRoundingDoublingHighMul(
          static_cast<int16>(input_val_centered * (1 << input_left_shift)),
          static_cast<int16>(input_multiplier));
      using FixedPoint4 = gemmlowp::FixedPoint<int16, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int16, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::logistic(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int16 output_val_s16 = RoundingDivideByPOT(output_val_f0.raw(), 7);
      if (output_val_s16 == 256) {
        output_val_s16 = 255;
      }
      TFLITE_DCHECK_GE(output_val_s16, 0);
      TFLITE_DCHECK_LE(output_val_s16, 255);
      output_val = static_cast<uint8>(output_val_s16);
    }
    output_data[c] = output_val;
  }
}

inline void Logistic16bitPercision(const LogisticParams& params,
                                   const RuntimeShape& input_shape,
                                   const int8* input_data,
                                   const RuntimeShape& output_shape,
                                   int8* output_data) {
  gemmlowp::ScopedProfilingLabel label("Logistic/Int8");
  const int32 input_zero_point = params.input_zero_point;
  const int32 input_range_radius = params.input_range_radius;
  const int32 input_multiplier = params.input_multiplier;
  const int16 input_left_shift = static_cast<int16>(params.input_left_shift);
  const int size = MatchingFlatSize(input_shape, output_shape);

  int c = 0;
  const int16 output_zero_point = 128;
// TODO(b/139252020): Replace GEMMLOWP_NEON with USE_NEON when the bug is fixed.
// The converted versions of gemmlowp::tanh and gemmlowp::logistic, done by
// arm_sse_2_neon.h, produce incorrect results with int16x8_t data types.
#ifdef GEMMLOWP_NEON
  const int16x8_t range_radius_dup = vdupq_n_s16(input_range_radius);
  const int16x8_t neg_range_radius_dup = vdupq_n_s16(-input_range_radius);
  const int16x8_t output_zero_point_dup = vdupq_n_s16(output_zero_point);

  // Handle 32 values at a time
  for (; c <= size - 32; c += 32) {
    // Read input int8 values, cast to int16 and subtract input_zero_point
    using cpu_backend_gemm::detail::Load16AndSubtractZeroPoint;
    const int16x8x2_t input_val_centered_0_1 =
        Load16AndSubtractZeroPoint(input_data + c, input_zero_point);
    const int16x8x2_t input_val_centered_2_3 =
        Load16AndSubtractZeroPoint(input_data + c + 16, input_zero_point);

    // Prepare the bit masks that we will use at the end to implement the logic
    // that was expressed in the scalar code with branching:
    //   if (input_val_centered < -input_range_radius) {
    //     output_val = -128;
    //   } else if (input_val_centered > input_range_radius) {
    //     output_val = 127;
    //   } else {
    //     ...
    uint8x16x2_t masks_clamp_0_1 = CalculateSignedClampingWithRangeBitMasks(
        input_val_centered_0_1, range_radius_dup, neg_range_radius_dup);
    uint8x16x2_t masks_clamp_2_3 = CalculateSignedClampingWithRangeBitMasks(
        input_val_centered_2_3, range_radius_dup, neg_range_radius_dup);

    int16x8x4_t input_val_rescaled = SaturatingRounding(
        input_val_centered_0_1.val[0], input_val_centered_0_1.val[1],
        input_val_centered_2_3.val[0], input_val_centered_2_3.val[1],
        input_left_shift, input_multiplier);

    int16x8x4_t output_val_s16 = FixedPoint4Logistic(input_val_rescaled);

    // Substract output zero point.
    output_val_s16.val[0] =
        vsubq_s16(output_val_s16.val[0], output_zero_point_dup);
    output_val_s16.val[1] =
        vsubq_s16(output_val_s16.val[1], output_zero_point_dup);
    output_val_s16.val[2] =
        vsubq_s16(output_val_s16.val[2], output_zero_point_dup);
    output_val_s16.val[3] =
        vsubq_s16(output_val_s16.val[3], output_zero_point_dup);

    // Cast output values to int8, saturating
    int8x16_t output_val_s8_0_1 = vcombine_s8(
        vqmovn_s16(output_val_s16.val[0]), vqmovn_s16(output_val_s16.val[1]));
    int8x16_t output_val_s8_2_3 = vcombine_s8(
        vqmovn_s16(output_val_s16.val[2]), vqmovn_s16(output_val_s16.val[3]));

    ClampWithRangeAndStore(output_data + c, output_val_s8_0_1, masks_clamp_0_1);
    ClampWithRangeAndStore(output_data + c + 16, output_val_s8_2_3,
                           masks_clamp_2_3);
  }
#endif  // GEMMLOWP_NEON
  // Leftover loop: handle one value at a time with scalar code.
  for (; c < size; ++c) {
    const int8 input_val_s8 = input_data[c];
    const int16 input_val_centered =
        static_cast<int16>(input_val_s8) - input_zero_point;
    int8 output_val;
    if (input_val_centered < -input_range_radius) {
      output_val = -128;
    } else if (input_val_centered > input_range_radius) {
      output_val = 127;
    } else {
      using gemmlowp::SaturatingRoundingDoublingHighMul;
      const int16 input_val_rescaled = SaturatingRoundingDoublingHighMul(
          static_cast<int16>(input_val_centered * (1 << input_left_shift)),
          static_cast<int16>(input_multiplier));
      using FixedPoint4 = gemmlowp::FixedPoint<int16, 4>;
      using FixedPoint0 = gemmlowp::FixedPoint<int16, 0>;
      const FixedPoint4 input_val_f4 = FixedPoint4::FromRaw(input_val_rescaled);
      const FixedPoint0 output_val_f0 = gemmlowp::logistic(input_val_f4);
      using gemmlowp::RoundingDivideByPOT;
      int16 output_val_s16 = RoundingDivideByPOT(output_val_f0.raw(), 7);
      output_val_s16 -= output_zero_point;
      if (output_val_s16 == 128) {
        output_val_s16 = 127;
      }
      TFLITE_DCHECK_GE(output_val_s16, -128);
      TFLITE_DCHECK_LE(output_val_s16, 127);
      output_val = static_cast<int8>(output_val_s16);
    }
    output_data[c] = output_val;
  }
}

}  // namespace optimized_ops
}  // namespace tflite

#if defined OPTIMIZED_OPS_H__IGNORE_DEPRECATED_DECLARATIONS
#undef OPTIMIZED_OPS_H__IGNORE_DEPRECATED_DECLARATIONS
#pragma GCC diagnostic pop
#endif

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_OPTIMIZED_OPS_H_
