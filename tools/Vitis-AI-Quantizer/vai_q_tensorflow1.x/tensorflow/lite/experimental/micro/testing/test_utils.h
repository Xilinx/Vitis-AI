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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_TESTING_TEST_UTILS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_TESTING_TEST_UTILS_H_

#include <cstdarg>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

// How many elements are in the array with this shape.
inline int ElementCount(const TfLiteIntArray& dims) {
  int result = 1;
  for (int i = 0; i < dims.size; ++i) {
    result *= dims.data[i];
  }
  return result;
}

// Wrapper to forward kernel errors to the interpreter's error reporter.
inline void ReportOpError(struct TfLiteContext* context, const char* format,
                          ...) {
  ErrorReporter* error_reporter = static_cast<ErrorReporter*>(context->impl_);
  va_list args;
  va_start(args, format);
  error_reporter->Report(format, args);
  va_end(args);
}

// Derives the quantization scaling factor from a min and max range.
template <typename T>
inline float ScaleFromMinMax(const float min, const float max) {
  return (max - min) / ((std::numeric_limits<T>::max() * 1.0) -
                        std::numeric_limits<T>::min());
}

// Derives the quantization zero point from a min and max range.
template <typename T>
inline int ZeroPointFromMinMax(const float min, const float max) {
  return static_cast<int>((-min / ScaleFromMinMax<T>(min, max)) + 0.5f);
}

// Converts a float value into an unsigned eight-bit quantized value.
inline uint8_t F2Q(const float value, const float min, const float max) {
  int32_t result = ZeroPointFromMinMax<uint8_t>(min, max) +
                   (value / ScaleFromMinMax<uint8_t>(min, max)) + 0.5f;
  if (result < 0) {
    result = 0;
  }
  if (result > 256) {
    result = 256;
  }
  return result;
}

// Converts a float value into a signed eight-bit quantized value.
inline int8_t F2QS(const float value, const float min, const float max) {
  return F2Q(value, min, max) + std::numeric_limits<int8_t>::min();
}

// Converts a float value into a signed thirty-two-bit quantized value.
inline int32_t F2Q32(const float value, const float min, const float max) {
  return static_cast<int32_t>((value - ZeroPointFromMinMax<int32_t>(min, max)) /
                              ScaleFromMinMax<int32_t>(min, max));
}

inline void PopulateContext(TfLiteTensor* tensors, int tensors_size,
                            TfLiteContext* context) {
  context->tensors_size = tensors_size;
  context->tensors = tensors;
  context->impl_ = static_cast<void*>(micro_test::reporter);
  context->GetExecutionPlan = nullptr;
  context->ResizeTensor = nullptr;
  context->ReportError = ReportOpError;
  context->AddTensors = nullptr;
  context->GetNodeAndRegistration = nullptr;
  context->ReplaceNodeSubsetsWithDelegateKernels = nullptr;
  context->recommended_num_threads = 1;
  context->GetExternalContext = nullptr;
  context->SetExternalContext = nullptr;
}

inline TfLiteIntArray* IntArrayFromInts(const int* int_array) {
  return const_cast<TfLiteIntArray*>(
      reinterpret_cast<const TfLiteIntArray*>(int_array));
}

inline TfLiteIntArray* IntArrayFromInitializer(
    std::initializer_list<int> int_initializer) {
  return IntArrayFromInts(int_initializer.begin());
}

inline TfLiteTensor CreateFloatTensor(const float* data, TfLiteIntArray* dims,
                                      const char* name) {
  TfLiteTensor result;
  result.type = kTfLiteFloat32;
  result.data.f = const_cast<float*>(data);
  result.dims = dims;
  result.params = {};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(float);
  result.allocation = nullptr;
  result.name = name;
  return result;
}

inline TfLiteTensor CreateFloatTensor(std::initializer_list<float> data,
                                      TfLiteIntArray* dims, const char* name) {
  return CreateFloatTensor(data.begin(), dims, name);
}

inline TfLiteTensor CreateInt32Tensor(const int32_t* data, TfLiteIntArray* dims,
                                      const char* name) {
  TfLiteTensor result;
  result.type = kTfLiteInt32;
  result.data.i32 = const_cast<int32_t*>(data);
  result.dims = dims;
  result.params = {};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(int32_t);
  result.allocation = nullptr;
  result.name = name;
  return result;
}

inline TfLiteTensor CreateInt32Tensor(std::initializer_list<int32_t> data,
                                      TfLiteIntArray* dims, const char* name) {
  return CreateInt32Tensor(data.begin(), dims, name);
}

inline TfLiteTensor CreateBoolTensor(const bool* data, TfLiteIntArray* dims,
                                     const char* name) {
  TfLiteTensor result;
  result.type = kTfLiteBool;
  result.data.b = const_cast<bool*>(data);
  result.dims = dims;
  result.params = {};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(bool);
  result.allocation = nullptr;
  result.name = name;
  return result;
}

inline TfLiteTensor CreateBoolTensor(std::initializer_list<bool> data,
                                     TfLiteIntArray* dims, const char* name) {
  return CreateBoolTensor(data.begin(), dims, name);
}

inline TfLiteTensor CreateQuantizedTensor(const uint8_t* data,
                                          TfLiteIntArray* dims,
                                          const char* name, float min,
                                          float max) {
  TfLiteTensor result;
  result.type = kTfLiteUInt8;
  result.data.uint8 = const_cast<uint8_t*>(data);
  result.dims = dims;
  result.params = {ScaleFromMinMax<uint8_t>(min, max),
                   ZeroPointFromMinMax<uint8_t>(min, max)};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(uint8_t);
  result.allocation = nullptr;
  result.name = name;
  return result;
}

inline TfLiteTensor CreateQuantizedTensor(std::initializer_list<uint8_t> data,
                                          TfLiteIntArray* dims,
                                          const char* name, float min,
                                          float max) {
  return CreateQuantizedTensor(data.begin(), dims, name, min, max);
}

inline TfLiteTensor CreateQuantizedInt8Tensor(const int8_t* data,
                                              TfLiteIntArray* dims,
                                              const char* name, float min,
                                              float max) {
  TfLiteTensor result;
  result.type = kTfLiteInt8;
  result.data.int8 = const_cast<int8_t*>(data);
  result.dims = dims;
  result.params = {ScaleFromMinMax<int8_t>(min, max),
                   ZeroPointFromMinMax<int8_t>(min, max)};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(int8_t);
  result.allocation = nullptr;
  result.name = name;
  return result;
}

inline TfLiteTensor CreateQuantizedInt8Tensor(
    std::initializer_list<int8_t> data, TfLiteIntArray* dims, const char* name,
    float min, float max) {
  return CreateQuantizedInt8Tensor(data.begin(), dims, name, min, max);
}

inline TfLiteTensor CreateQuantized32Tensor(const int32_t* data,
                                            TfLiteIntArray* dims,
                                            const char* name, float min,
                                            float max) {
  TfLiteTensor result;
  result.type = kTfLiteInt32;
  result.data.i32 = const_cast<int32_t*>(data);
  result.dims = dims;
  result.params = {ScaleFromMinMax<int32_t>(min, max),
                   ZeroPointFromMinMax<int32_t>(min, max)};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(int32_t);
  result.allocation = nullptr;
  result.name = name;
  return result;
}

inline TfLiteTensor CreateQuantized32Tensor(std::initializer_list<int32_t> data,
                                            TfLiteIntArray* dims,
                                            const char* name, float min,
                                            float max) {
  return CreateQuantized32Tensor(data.begin(), dims, name, min, max);
}

template <typename input_type = int32_t,
          TfLiteType tensor_input_type = kTfLiteInt32>
inline TfLiteTensor CreateTensor(const input_type* data, TfLiteIntArray* dims,
                                 const char* name) {
  TfLiteTensor result;
  result.type = tensor_input_type;
  result.data.raw = reinterpret_cast<char*>(const_cast<input_type*>(data));
  result.dims = dims;
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(input_type);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = true;
  return result;
}

template <typename input_type = int32_t,
          TfLiteType tensor_input_type = kTfLiteInt32>
inline TfLiteTensor CreateTensor(std::initializer_list<input_type> data,
                                 TfLiteIntArray* dims, const char* name) {
  return CreateTensor<input_type, tensor_input_type>(data.begin(), dims, name);
}

// Do a simple string comparison for testing purposes, without requiring the
// standard C library.
inline int TestStrcmp(const char* a, const char* b) {
  if ((a == nullptr) || (b == nullptr)) {
    return -1;
  }
  while ((*a != 0) && (*a == *b)) {
    a++;
    b++;
  }
  return *(const unsigned char*)a - *(const unsigned char*)b;
}

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_TESTING_TEST_UTILS_H_
