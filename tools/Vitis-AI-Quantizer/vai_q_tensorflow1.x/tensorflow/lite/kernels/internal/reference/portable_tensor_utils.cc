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
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/activation_functor.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/reference/portable_tensor_utils_impl.h"
#include "tensorflow/lite/kernels/internal/round.h"
#include "tensorflow/lite/kernels/op_macros.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {
namespace tensor_utils {

float PortableClip(float f, float abs_limit) {
  float result = (abs_limit < f) ? abs_limit : f;
  result = (-abs_limit > result) ? -abs_limit : result;
  return result;
}

bool PortableIsZeroVector(const float* vector, int v_size) {
  for (int i = 0; i < v_size; ++i) {
    if (*vector++ != 0.0f) return false;
  }
  return true;
}

void PortableSymmetricQuantizeFloats(const float* values, const int size,
                                     int8_t* quantized_values, float* min_value,
                                     float* max_value, float* scaling_factor) {
  auto minmax = std::minmax_element(values, values + size);
  *min_value = *minmax.first;
  *max_value = *minmax.second;
  const int kScale = 127;
  const float range = std::max(std::abs(*min_value), std::abs(*max_value));
  if (range == 0) {
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1;
    return;
  }
  *scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;
  for (int i = 0; i < size; ++i) {
    const int32_t quantized_value =
        static_cast<int32_t>(TfLiteRound(values[i] * scaling_factor_inv));
    // Clamp: just in case some odd numeric offset.
    quantized_values[i] = std::min(kScale, std::max(-kScale, quantized_value));
  }
}

void PortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride) {
  float* result_in_batch = result;
  for (int b = 0; b < n_batch; b++) {
    const float* matrix_ptr = matrix;
    for (int r = 0; r < m_rows; r++) {
      float dot_prod = 0.0f;
      const float* vector_in_batch = vector + b * m_cols;
      for (int c = 0; c < m_cols; c++) {
        dot_prod += *matrix_ptr++ * *vector_in_batch++;
      }
      *result_in_batch += dot_prod;
      result_in_batch += result_stride;
    }
  }
}

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, int result_stride) {
  for (int batch = 0; batch < n_batch; ++batch, vectors += m_cols) {
    const float batch_scaling_factor = scaling_factors[batch];
    // Get the address of the first row.
    const int8_t* row_ptr = matrix;
    for (int row = 0; row < m_rows; ++row, result += result_stride) {
      // Initialize the dot product sum for the row to 0.
      int32_t dotprod = 0;
#if defined(__GNUC__)
      // Prefetch the row to cache.
      __builtin_prefetch(row_ptr, 0 /* prefetch for read */,
                         3 /* temporal locality */);
#endif
      for (int col = 0; col < m_cols; ++col, ++row_ptr) {
        dotprod += (*row_ptr) * (vectors[col]);
      }  // for col
      *result += dotprod * batch_scaling_factor;
    }  // for row
  }    // for batch
}

void PortableSparseMatrixBatchVectorMultiplyAccumulate(
    const float* __restrict__ matrix, const uint8_t* __restrict__ ledger,
    int m_rows, int m_cols, const float* __restrict__ vector, int n_batch,
    float* __restrict__ result, int result_stride) {
  const int kBlockSize = 16;
  TFLITE_DCHECK_EQ(  // NOLINT
      m_cols % kBlockSize, 0);
  float* result_in_batch = result;
  for (int b = 0; b < n_batch; b++) {
    const float* matrix_ptr = matrix;
    const uint8_t* ledger_ptr = ledger;
    for (int r = 0; r < m_rows; r++) {
      float dot_prod = 0.0f;
      int num_nonzero_blocks = *ledger_ptr++;
      if (num_nonzero_blocks > 0) {
        const float* vector_in_batch = vector + b * m_cols;
        for (int i = 0; i < num_nonzero_blocks; i++) {
          const int block_start_index = *ledger_ptr++ * kBlockSize;
          const float* vector_block_in_batch_ptr =
              vector_in_batch + block_start_index;
          for (int c = 0; c < kBlockSize; c++) {
            dot_prod += *matrix_ptr++ * *vector_block_in_batch_ptr++;
          }
        }
      }
      *result_in_batch += dot_prod;
      result_in_batch += result_stride;
    }
  }
}

void PortableSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result,
    int result_stride) {
  static const int kBlockSize = 16;
  TFLITE_DCHECK_EQ(  // NOLINT
      m_cols % kBlockSize, 0);
  for (int batch = 0; batch < n_batch; ++batch, vectors += m_cols) {
    const float batch_scaling_factor = scaling_factors[batch];
    const uint8_t* ledger_ptr = ledger;
    // Get the address of the first row.
    const int8_t* row_ptr = matrix;
    for (int row = 0; row < m_rows; ++row, result += result_stride) {
      // Initialize the dot product sum for the row to 0.
      int32_t dotprod = 0;
#if defined(__GNUC__)
      // Prefetch the row to cache.
      __builtin_prefetch(row_ptr, 0 /* prefetch for read */,
                         3 /* temporal locality */);
#endif
      int num_nonzero_blocks = *ledger_ptr++;
      for (int i = 0; i < num_nonzero_blocks; i++) {
        const int block_start_index = *ledger_ptr++ * kBlockSize;
        const int8_t* vector_block_ptr = vectors + block_start_index;
        for (int c = 0; c < kBlockSize; c++) {
          dotprod += (*row_ptr++) * (*vector_block_ptr++);
        }  // for block
      }    // for num_nonzero_blocks
      *result += dotprod * batch_scaling_factor;
    }  // for row
  }    // for batch
}

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, int32_t input_zeropoint,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    const int32_t* gate_bias, int32_t n_batch, int32_t n_input,
    int32_t n_output, int32_t output_zp, int16_t* output) {
  const int16_t output_max = std::numeric_limits<int16_t>::max();
  const int16_t output_min = std::numeric_limits<int16_t>::min();
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int row = 0; row < n_output; ++row) {
      int32_t acc = gate_bias == nullptr ? 0 : gate_bias[row];
      for (int col = 0; col < n_input; ++col) {
        int8 input_val = input[batch * n_input + col];
        int8 weights_val = input_to_gate_weights[row * n_input + col];
        acc += (input_val - input_zeropoint) * weights_val;
      }
      acc = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
      acc += output_zp;
      acc += output[batch * n_output + row];
      if (acc > output_max) {
        acc = output_max;
      }
      if (acc < output_min) {
        acc = output_min;
      }
      output[batch * n_output + row] = static_cast<int16_t>(acc);
    }
  }
}

void PortableMatrixBatchVectorMultiplyAccumulate(
    const int8_t* input, int32_t input_zeropoint,
    const int8_t* input_to_gate_weights, int32_t multiplier, int32_t shift,
    const int32_t* gate_bias, int32_t n_batch, int32_t n_input,
    int32_t n_output, int32_t output_zp, int8_t* output) {
  const int8_t output_max = std::numeric_limits<int8_t>::max();
  const int8_t output_min = std::numeric_limits<int8_t>::min();
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int row = 0; row < n_output; ++row) {
      int32_t acc = gate_bias == nullptr ? 0 : gate_bias[row];
      for (int col = 0; col < n_input; ++col) {
        int8 input_val = input[batch * n_input + col];
        int8 weights_val = input_to_gate_weights[row * n_input + col];
        acc += (input_val - input_zeropoint) * weights_val;
      }
      acc = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
      acc += output_zp;
      acc += output[batch * n_output + row];
      if (acc > output_max) {
        acc = output_max;
      }
      if (acc < output_min) {
        acc = output_min;
      }
      output[batch * n_output + row] = static_cast<int8_t>(acc);
    }
  }
}

void PortableApplyLayerNorm(const int16_t* input,
                            const int16_t* layer_norm_weights,
                            const int32_t* bias, int32_t layer_norm_scale_a,
                            int32_t layer_norm_scale_b, int32_t variance_limit,
                            int n_batch, int n_input, int16_t* output) {
  const int32_t int16_max = std::numeric_limits<int16_t>::max();
  const int32_t int16_min = std::numeric_limits<int16_t>::min();
  static const int kOverflowGuard = 1 << 20;
  for (int i = 0; i < n_batch; ++i) {
    int64_t sum = 0;
    int64_t sum_sq = 0;
    for (int j = 0; j < n_input; ++j) {
      const int32_t index = i * n_input + j;
      int32_t val = static_cast<int32_t>(input[index]);
      sum += val;
      sum_sq += val * val;
    }
    int32_t mean =
        static_cast<int32_t>(static_cast<int64_t>(sum) * 1024 / n_input);
    // TODO(jianlijianli): Avoids overflow but only works for POT n_input.
    int32 temp = kOverflowGuard / n_input;
    int64_t variance =
        sum_sq * temp - static_cast<int64_t>(mean) * static_cast<int64_t>(mean);
    int32_t variance2 = static_cast<int32>(variance / kOverflowGuard);
    if (variance2 < 1) {
      variance2 = variance_limit;
    }
    int32_t stddev_inverse_a;
    int stddev_inverse_b;
    GetInvSqrtQuantizedMultiplierExp(variance2, /*reverse_shift*/ -1,
                                     &stddev_inverse_a, &stddev_inverse_b);

    for (int j = 0; j < n_input; ++j) {
      const int32 index = i * n_input + j;
      int32 val = static_cast<int32_t>(input[index]);
      int32 shifted = 1024 * val - mean;
      int32 rescaled = MultiplyByQuantizedMultiplier(shifted, stddev_inverse_a,
                                                     stddev_inverse_b);
      // TODO(jianlijianli): Saturate this.
      int64_t val3 = rescaled * layer_norm_weights[j] + bias[j];
      int32 val4 =
          static_cast<int32>((val3 > 0 ? val3 + 512 : val3 - 512) / 1024);
      int32 val5 = MultiplyByQuantizedMultiplier(val4, layer_norm_scale_a,
                                                 layer_norm_scale_b + 12);
      val5 = std::min(std::max(int16_min, val5), int16_max);
      output[index] = static_cast<int16_t>(val5);
    }
  }
}

void PortableApplySigmoid(const int16_t* input, int32_t n_batch,
                          int32_t n_input, int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int c = 0; c < n_input; c++) {
      using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;
      using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
      const int index = batch * n_input + c;
      F3 sigmoid_input = F3::FromRaw(input[index]);
      F0 sigmoid_output = gemmlowp::logistic(sigmoid_input);
      output[index] = sigmoid_output.raw();
    }
  }
}

void PortableApplyTanh3(const int16_t* input, int32_t n_batch, int32_t n_input,
                        int16_t* output) {
  using FX = gemmlowp::FixedPoint<std::int16_t, 3>;
  using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      FX tanh_input = FX::FromRaw(input[index]);
      F0 tanh_output = gemmlowp::tanh(tanh_input);
      output[index] = tanh_output.raw();
    }
  }
}

void PortableApplyTanh4(const int16_t* input, int32_t n_batch, int32_t n_input,
                        int16_t* output) {
  using FX = gemmlowp::FixedPoint<std::int16_t, 4>;
  using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      FX tanh_input = FX::FromRaw(input[index]);
      F0 tanh_output = gemmlowp::tanh(tanh_input);
      output[index] = tanh_output.raw();
    }
  }
}

void PortableCwiseMul(const int16_t* input_1, const int16_t* input_2,
                      int n_batch, int n_input, int shift, int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      const int16_t a = input_1[index];
      const int16_t b = input_2[index];
      int64_t x = a * b;
      if (x > std::numeric_limits<std::int32_t>::max()) {
        x = std::numeric_limits<std::int32_t>::max();
      }
      const int32_t value = static_cast<int32_t>(x);
      output[index] =
          static_cast<int16_t>(gemmlowp::RoundingDivideByPOT(value, shift));
    }
  }
}

void PortableCwiseMul(const int16_t* input_1, const int16_t* input_2,
                      int n_batch, int n_input, int shift, int8_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      const int16_t a = input_1[index];
      const int16_t b = input_2[index];
      int64_t x = a * b;
      if (x > std::numeric_limits<std::int32_t>::max()) {
        x = std::numeric_limits<std::int32_t>::max();
      }
      const int32_t value = static_cast<int32_t>(x);
      output[index] =
          static_cast<int8_t>(gemmlowp::RoundingDivideByPOT(value, shift));
    }
  }
}

void PortableCwiseAdd(const int16_t* input_1, const int16_t* input_2,
                      int n_batch, int n_input, int16_t* output) {
  const int32 int16_max = std::numeric_limits<int16>::max();
  const int32 int16_min = std::numeric_limits<int16>::min();
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      int32_t sum = input_1[index] + input_2[index];
      const int32 sum_clamped = std::min(int16_max, std::max(int16_min, sum));
      output[index] = static_cast<int16_t>(sum_clamped);
    }
  }
}

void PortableCwiseClipping(int16_t* input, const int16_t clipping_value,
                           int32_t n_batch, int32_t n_input) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      if (input[index] > clipping_value) {
        input[index] = clipping_value;
      }
      if (input[index] < -clipping_value) {
        input[index] = -clipping_value;
      }
    }
  }
}

void PortableCwiseClipping(int8_t* input, const int8_t clipping_value,
                           int32_t n_batch, int32_t n_input) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      if (input[index] > clipping_value) {
        input[index] = clipping_value;
      }
      if (input[index] < -clipping_value) {
        input[index] = -clipping_value;
      }
    }
  }
}

void PortableVectorVectorCwiseProduct(const float* vector1,
                                      const float* vector2, int v_size,
                                      float* result) {
  for (int v = 0; v < v_size; v++) {
    *result++ = *vector1++ * *vector2++;
  }
}

float PortableVectorVectorDotProduct(const float* vector1, const float* vector2,
                                     int v_size) {
  float result = 0.0;
  for (int v = 0; v < v_size; v++) {
    result += *vector1++ * *vector2++;
  }
  return result;
}

void PortableBatchVectorBatchVectorDotProduct(const float* vector1,
                                              const float* vector2, int v_size,
                                              int n_batch, float* result,
                                              int result_stride) {
  float* result_ptr = result;
  const float* vector1_ptr = vector1;
  const float* vector2_ptr = vector2;
  for (int b = 0; b < n_batch; b++) {
    *result_ptr =
        PortableVectorVectorDotProduct(vector1_ptr, vector2_ptr, v_size);
    vector1_ptr += v_size;
    vector2_ptr += v_size;
    result_ptr += result_stride;
  }
}

void PortableVectorVectorCwiseProductAccumulate(const float* vector1,
                                                const float* vector2,
                                                int v_size, float* result) {
  for (int v = 0; v < v_size; v++) {
    *result++ += *vector1++ * *vector2++;
  }
}

void PortableVectorBatchVectorCwiseProduct(const float* vector, int v_size,
                                           const float* batch_vector,
                                           int n_batch, float* result) {
  for (int b = 0; b < n_batch; b++) {
    for (int v = 0; v < v_size; v++) {
      *result++ = vector[v] * *batch_vector++;
    }
  }
}

void PortableVectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  for (int b = 0; b < n_batch; b++) {
    for (int v = 0; v < v_size; v++) {
      *result++ += vector[v] * *batch_vector++;
    }
  }
}

void PortableVectorBatchVectorAdd(const float* vector, int v_size, int n_batch,
                                  float* batch_vector) {
  for (int b = 0; b < n_batch; b++) {
    for (int i = 0; i < v_size; ++i) {
      batch_vector[i] += vector[i];
    }
    batch_vector += v_size;
  }
}

void PortableApplySigmoidToVector(const float* vector, int v_size,
                                  float* result) {
  auto sigmoid_func = ActivationFunctor(kTfLiteActSigmoid);
  for (int v = 0; v < v_size; v++) {
    *result++ = (sigmoid_func)(*vector++);
  }
}

void PortableApplyActivationToVector(const float* vector, int v_size,
                                     TfLiteFusedActivation activation,
                                     float* result) {
  auto activation_func = ActivationFunctor(activation);
  for (int v = 0; v < v_size; v++) {
    *result++ = (activation_func)(*vector++);
  }
}

void PortableSub1Vector(const float* vector, int v_size, float* result) {
  for (int v = 0; v < v_size; v++) {
    *result++ = 1.0f - *vector++;
  }
}

void PortableVectorScalarMultiply(const int8_t* vector, const int v_size,
                                  const float scale, float* result) {
  for (int v = 0; v < v_size; ++v) {
    *result++ = scale * *vector++;
  }
}

void PortableClipVector(const float* vector, int v_size, float abs_limit,
                        float* result) {
  for (int v = 0; v < v_size; v++) {
    *result++ = PortableClip(*vector++, abs_limit);
  }
}

void PortableReductionSumVector(const float* input_vector, float* output_vector,
                                int output_size, int reduction_size) {
  const float* input_vector_ptr = input_vector;
  for (int o = 0; o < output_size; o++) {
    for (int r = 0; r < reduction_size; r++) {
      output_vector[o] += *input_vector_ptr++;
    }
  }
}

void PortableMeanStddevNormalization(const float* input_vector,
                                     float* output_vector, int v_size,
                                     int n_batch, float normalization_epsilon) {
  for (int batch = 0; batch < n_batch; ++batch) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < v_size; ++i) {
      sum += input_vector[i];
      sum_sq += input_vector[i] * input_vector[i];
    }
    const float mean = sum / v_size;
    float stddev_inv = 0.0f;
    const float variance = sum_sq / v_size - mean * mean;
    if (variance == 0) {
      stddev_inv = 1.0f / std::sqrt(normalization_epsilon);
    } else {
      stddev_inv = 1.0f / std::sqrt(variance);
    }
    for (int i = 0; i < v_size; ++i) {
      output_vector[i] = (input_vector[i] - mean) * stddev_inv;
    }
    input_vector += v_size;
    output_vector += v_size;
  }
}

}  // namespace tensor_utils
}  // namespace tflite
