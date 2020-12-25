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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_TENSOR_UTILS_IMPL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_TENSOR_UTILS_IMPL_H_

// TODO(ghodrat): Remove this header file and the dependency to internal data
// structure.
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {
namespace tensor_utils {

#ifdef USE_NEON

// Multiply a matrix by a batch vector, and store results in a batch-size
// vector.
void NeonMatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                             int m_cols, const float* vector,
                                             int n_batch, float* result,
                                             int result_stride);

// Matrix multiplication for quantized values using symmetric quantization.
void NeonMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, int result_stride);

// Multiply a matrix by a batch vector, and store results in a batch-size
// vector. Sparse version.
void NeonSparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result, int result_stride);

// Matrix multiplication for quantized values using symmetric quantization.
// Sparse version.
void NeonSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result,
    int result_stride);

// Cwise product of two vectors.
void NeonVectorVectorCwiseProduct(const float* vector1, const float* vector2,
                                  int v_size, float* result);

// Cwise product and accumulate of two vectors. Since it's a MAC operation, the
// assumption here is that result array is initialized to valid values.
void NeonVectorVectorCwiseProductAccumulate(const float* vector1,
                                            const float* vector2, int v_size,
                                            float* result);

// Dot product of two vectors.
float NeonVectorVectorDotProduct(const float* vector1, const float* vector2,
                                 int v_size);

// Dot product of two batch vectors.
void NeonBatchVectorBatchVectorDotProduct(const float* vector1,
                                          const float* vector2, int v_size,
                                          int n_batch, float* result,
                                          int result_stride);

// Cwise product of a vector and a batch-vector.
void NeonVectorBatchVectorCwiseProduct(const float* vector, int v_size,
                                       const float* batch_vector, int n_batch,
                                       float* result);

// Cwise product and accumulate of a vector and a batch-vector. Since it's a MAC
// operation, the assumption here is that result array is initialized to valid
// values.
void NeonVectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                 int v_size,
                                                 const float* batch_vector,
                                                 int n_batch, float* result);

// Compute "1.0f - elements of vector" (used in CIFG).
void NeonSub1Vector(const float* vector, int v_size, float* result);

// Clip elements of a vector using a abs_limit value.
void NeonClipVector(const float* vector, int v_size, float abs_limit,
                    float* result);

// Multiply all elements of vector with a scalar.
void NeonVectorScalarMultiply(const int8_t* vector, int v_size, float scale,
                              float* result);

// Check if all entries of a vector are zero.
bool NeonIsZeroVector(const float* vector, int v_size);

// Symmetric quantizer.
void NeonSymmetricQuantizeFloats(const float* values, const int size,
                                 int8_t* quantized_values, float* min,
                                 float* max, float* scaling_factor);

// Shift left a vector in place with v_size size.
void NeonVectorShiftLeft(float* vector, int v_size, float shift_value);

// Reduce-sum on a float input vector:
// input_vector: float pointer to input vector.
// output_vector: float pointer to vector.
// output_size: output vector size.
// reduction_size: number of consecutive elements from input vector which are
// added to get one element of output.
void NeonReductionSumVector(const float* input_vector, float* output_vector,
                            int output_size, int reduction_size);

#endif  // USE_NEON

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_NEON_TENSOR_UTILS_IMPL_H_
