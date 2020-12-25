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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SSE_TENSOR_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SSE_TENSOR_UTILS_H_

// Note: This file is a copy-paste version of neon_tensor_utils.h, only
// difference is in MatrixBatchVectorMultiplyAccumulate and
// SparseMatrixBatchVectorMultiplyAccumulate (other functions do not have SSE
// implementation yet).

// Note: Most of the functions below use NEON_OR_PORTABLE, through the Intel
// NEON_2_SSE translator library. If a native SSE version of a function is
// implemented, replace the appropriate one to SSE_OR_PORTABLE.

// TODO(ghodrat): Remove this header file and the dependency to internal data
// structure.
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_tensor_utils_impl.h"
#include "tensorflow/lite/kernels/internal/optimized/sse_check.h"
#include "tensorflow/lite/kernels/internal/optimized/sse_tensor_utils_impl.h"
#include "tensorflow/lite/kernels/internal/reference/portable_tensor_utils_impl.h"

namespace tflite {
namespace tensor_utils {

void MatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                         int m_cols, const float* vector,
                                         int n_batch, float* result,
                                         int result_stride) {
  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols,
                   vector, n_batch, result, result_stride);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, int result_stride) {
  SSE_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols,
                  vectors, scaling_factors, n_batch, result, result_stride);
}

void SparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result, int result_stride) {
  NEON_OR_PORTABLE(SparseMatrixBatchVectorMultiplyAccumulate, matrix, ledger,
                   m_rows, m_cols, vector, n_batch, result, result_stride);
}

void SparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result,
    int result_stride) {
  SSE_OR_PORTABLE(SparseMatrixBatchVectorMultiplyAccumulate, matrix, ledger,
                  m_rows, m_cols, vectors, scaling_factors, n_batch, result,
                  result_stride);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, int32_t input_zeropoint,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    const int32_t* gate_bias, int32_t n_batch, int32_t n_input,
    int32_t n_output, int32_t output_zp, int16_t* output) {
  PortableMatrixBatchVectorMultiplyAccumulate(
      input, input_zeropoint, input_to_gate_weights, multiplier, shift,
      gate_bias, n_batch, n_input, n_output, output_zp, output);
}

void MatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, int32_t input_zeropoint,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    const int32_t* gate_bias, int32_t n_batch, int32_t n_input,
    int32_t n_output, int32_t output_zp, int8_t* output) {
  PortableMatrixBatchVectorMultiplyAccumulate(
      input, input_zeropoint, input_to_gate_weights, multiplier, shift,
      gate_bias, n_batch, n_input, n_output, output_zp, output);
}

void ApplyLayerNorm(const int16_t* input, const int16_t* layer_norm_weights,
                    const int32_t* bias, int32_t layer_norm_scale_a,
                    int32_t layer_norm_scale_b, int32_t variance_limit,
                    int n_batch, int n_input, int16_t* output) {
  PortableApplyLayerNorm(input, layer_norm_weights, bias, layer_norm_scale_a,
                         layer_norm_scale_b, variance_limit, n_batch, n_input,
                         output);
}

void ApplySigmoid(const int16_t* input, int32_t n_batch, int32_t n_input,
                  int16_t* output) {
  PortableApplySigmoid(input, n_batch, n_input, output);
}

void ApplyTanh3(const int16_t* input, int32_t n_batch, int32_t n_input,
                int16_t* output) {
  PortableApplyTanh3(input, n_batch, n_input, output);
}

void ApplyTanh4(const int16_t* input, int32_t n_batch, int32_t n_input,
                int16_t* output) {
  PortableApplyTanh4(input, n_batch, n_input, output);
}

void CwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch,
              int n_input, int shift, int16_t* output) {
  PortableCwiseMul(input_1, input_2, n_batch, n_input, shift, output);
}

void CwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch,
              int n_input, int shift, int8_t* output) {
  PortableCwiseMul(input_1, input_2, n_batch, n_input, shift, output);
}

void CwiseAdd(const int16_t* input_1, const int16_t* input_2, int n_batch,
              int n_input, int16_t* output) {
  PortableCwiseAdd(input_1, input_2, n_batch, n_input, output);
}

void CwiseClipping(int16_t* input, const int16_t clipping_value,
                   int32_t n_batch, int32_t n_input) {
  PortableCwiseClipping(input, clipping_value, n_batch, n_input);
}

void CwiseClipping(int8_t* input, const int8_t clipping_value, int32_t n_batch,
                   int32_t n_input) {
  PortableCwiseClipping(input, clipping_value, n_batch, n_input);
}

void VectorVectorCwiseProduct(const float* vector1, const float* vector2,
                              int v_size, float* result) {
  NEON_OR_PORTABLE(VectorVectorCwiseProduct, vector1, vector2, v_size, result);
}

void VectorVectorCwiseProductAccumulate(const float* vector1,
                                        const float* vector2, int v_size,
                                        float* result) {
  NEON_OR_PORTABLE(VectorVectorCwiseProductAccumulate, vector1, vector2, v_size,
                   result);
}

void VectorBatchVectorCwiseProduct(const float* vector, int v_size,
                                   const float* batch_vector, int n_batch,
                                   float* result) {
  NEON_OR_PORTABLE(VectorBatchVectorCwiseProduct, vector, v_size, batch_vector,
                   n_batch, result);
}

void VectorBatchVectorCwiseProductAccumulate(const float* vector, int v_size,
                                             const float* batch_vector,
                                             int n_batch, float* result) {
  NEON_OR_PORTABLE(VectorBatchVectorCwiseProductAccumulate, vector, v_size,
                   batch_vector, n_batch, result);
}

float VectorVectorDotProduct(const float* vector1, const float* vector2,
                             int v_size) {
  return NEON_OR_PORTABLE(VectorVectorDotProduct, vector1, vector2, v_size);
}

void BatchVectorBatchVectorDotProduct(const float* vector1,
                                      const float* vector2, int v_size,
                                      int n_batch, float* result,
                                      int result_stride) {
  NEON_OR_PORTABLE(BatchVectorBatchVectorDotProduct, vector1, vector2, v_size,
                   n_batch, result, result_stride);
}

void VectorBatchVectorAdd(const float* vector, int v_size, int n_batch,
                          float* batch_vector) {
  PortableVectorBatchVectorAdd(vector, v_size, n_batch, batch_vector);
}

void ApplySigmoidToVector(const float* vector, int v_size, float* result) {
  PortableApplySigmoidToVector(vector, v_size, result);
}

void ApplyActivationToVector(const float* vector, int v_size,
                             TfLiteFusedActivation activation, float* result) {
  PortableApplyActivationToVector(vector, v_size, activation, result);
}

void Sub1Vector(const float* vector, int v_size, float* result) {
  NEON_OR_PORTABLE(Sub1Vector, vector, v_size, result);
}

float Clip(float f, float abs_limit) { return PortableClip(f, abs_limit); }

// Check if all entries of a vector are zero.
bool IsZeroVector(const float* vector, int v_size) {
  return NEON_OR_PORTABLE(IsZeroVector, vector, v_size);
}

void VectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                          float* result) {
  NEON_OR_PORTABLE(VectorScalarMultiply, vector, v_size, scale, result);
}
void ClipVector(const float* vector, int v_size, float abs_limit,
                float* result) {
  NEON_OR_PORTABLE(ClipVector, vector, v_size, abs_limit, result);
}

void SymmetricQuantizeFloats(const float* values, const int size,
                             int8_t* quantized_values, float* min_value,
                             float* max_value, float* scaling_factor) {
  NEON_OR_PORTABLE(SymmetricQuantizeFloats, values, size, quantized_values,
                   min_value, max_value, scaling_factor);
}

void ReductionSumVector(const float* input_vector, float* output_vector,
                        int output_size, int reduction_size) {
  NEON_OR_PORTABLE(ReductionSumVector, input_vector, output_vector, output_size,
                   reduction_size);
}

void MeanStddevNormalization(const float* input_vector, float* output_vector,
                             int v_size, int n_batch,
                             float normalization_epsilon) {
  PortableMeanStddevNormalization(input_vector, output_vector, v_size, n_batch,
                                  normalization_epsilon);
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SSE_TENSOR_UTILS_H_
