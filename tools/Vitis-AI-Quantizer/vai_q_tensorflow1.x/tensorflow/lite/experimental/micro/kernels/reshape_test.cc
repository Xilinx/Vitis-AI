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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/simple_tensor_allocator.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

namespace tflite {
namespace testing {
namespace {

TfLiteReshapeParams create_params(int* shape_data) {
  TfLiteReshapeParams op_params = {};
  op_params.num_dimensions = shape_data[0];
  for (int i = 0; i < shape_data[0]; ++i)
    op_params.shape[i] = shape_data[i + 1];
  return op_params;
}

// If expected output is empty, the test is expected to fail.
template <typename T>
void TestReshapeImpl(TfLiteTensor* input_tensor, TfLiteTensor* shape_tensor,
                     TfLiteTensor* output_tensor,
                     std::initializer_list<T> expected_output,
                     std::initializer_list<int> expected_dims) {
  TfLiteContext context;
  TfLiteTensor tensors[3];
  TfLiteNode node;
  if (shape_tensor == nullptr) {
    constexpr int inputs_size = 1;
    constexpr int outputs_size = 1;
    constexpr int tensors_size = inputs_size + outputs_size;
    tensors[0] = *input_tensor;
    tensors[1] = *output_tensor,
    PopulateContext(tensors, tensors_size, &context);
    node.inputs = IntArrayFromInitializer({1, 0});
    node.outputs = IntArrayFromInitializer({1, 1});
  } else {
    constexpr int inputs_size = 2;
    constexpr int outputs_size = 1;
    constexpr int tensors_size = inputs_size + outputs_size;
    tensors[0] = *input_tensor;
    tensors[1] = *shape_tensor;
    tensors[2] = *output_tensor;
    PopulateContext(tensors, tensors_size, &context);
    node.inputs = IntArrayFromInitializer({2, 0, 1});
    node.outputs = IntArrayFromInitializer({1, 2});
  }

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_RESHAPE, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);
  TfLiteReshapeParams builtin_data =
      create_params(reinterpret_cast<int*>(output_tensor->dims));
  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  size_t init_data_size = 0;
  void* user_data = nullptr;
  node.temporaries = nullptr;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }
  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  if (expected_output.size() == 0) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                            registration->invoke(&context, &node));
    return;
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }

  const int output_dims_count = ElementCount(*output_tensor->dims);
  const T* output_data = GetTensorData<T>(output_tensor);
  for (int i = 0; i < expected_output.size(); ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output.begin()[i], output_data[i],
                              1e-5f);
  }
  TF_LITE_MICRO_EXPECT_EQ(expected_dims.size(), output_tensor->dims->size);
  for (int i = 0; i < expected_dims.size(); ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_dims.begin()[i],
                              output_tensor->dims->data[i], 1e-5f);
  }
}

template <typename T = float, TfLiteType tensor_input_type = kTfLiteFloat32>
void TestReshape(std::initializer_list<int> input_dims_data,
                 std::initializer_list<T> input_data,
                 std::initializer_list<int> shape_dims_data,
                 std::initializer_list<int32_t> shape_data,
                 int* output_dims_data, uint8_t* output_data_raw,
                 std::initializer_list<T> expected_output,
                 std::initializer_list<int> expected_dims) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  TfLiteTensor input_tensor = CreateTensor<T, tensor_input_type>(
      input_data, input_dims, "input_tensor");
  T* output_data = reinterpret_cast<T*>(output_data_raw);
  TfLiteTensor output_tensor = CreateTensor<T, tensor_input_type>(
      output_data, output_dims, "input_tensor");
  // Reshape param is passed as op's param.
  TestReshapeImpl<T>(&input_tensor, nullptr, &output_tensor, expected_output,
                     expected_dims);
  // Reshape param is passed as a tensor.
  TfLiteIntArray* shape_dims = IntArrayFromInitializer(shape_dims_data);
  auto shape_tensor = CreateTensor<int32_t, kTfLiteInt32>(
      shape_data, shape_dims, "shape_tensor");
  TestReshapeImpl<T>(&input_tensor, &shape_tensor, &output_tensor,
                     expected_output, expected_dims);
}
}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

#define TEST_RESHAPE(...)                                           \
  using tflite::testing::TestReshape;                               \
  tflite::testing::TestReshape<float, kTfLiteFloat32>(__VA_ARGS__); \
  tflite::testing::TestReshape<uint8_t, kTfLiteUInt8>(__VA_ARGS__); \
  tflite::testing::TestReshape<int8_t, kTfLiteInt8>(__VA_ARGS__);

TF_LITE_MICRO_TEST(MismatchedDimensions) {
  uint8_t output_data[32];
  int output_dims[3] = {2, 2, 1};
  TEST_RESHAPE({4, 1, 2, 4, 1},  // input_dims
               {3},              // input_data
               {1, 2},           // shape_dims
               {2, 1},           // shape_data
               output_dims,      // output_dims
               output_data, {},  // expected_output
               {}                // expected_dims
  );
}

TF_LITE_MICRO_TEST(TooManyDimensions) {
  uint8_t output_data[32];
  int output_dims[10] = {9, 1, 1, 1, 1, 1, 1, 1, 1, 2};
  TEST_RESHAPE({9, 1, 1, 2, 1, 1, 1, 1, 1, 1},  // input_dims
               {3, 2},                          // input_data
               {1, 9},                          // shape_dims
               {1, 1, 1, 1, 1, 1, 1, 1, 2},     // shape_data
               output_dims,                     // output_dims
               output_data, {3, 2},             // expected_output
               {1, 1, 1, 1, 1, 1, 1, 1, 2}      // expected_dims
  );
}

// Number of dimensions > 8 is accepted in micro since it does not use
// TfLiteReshapeParams.
TF_LITE_MICRO_TEST(TooManySpecialDimensions) {
  uint8_t output_data[32];
  int output_dims[5] = {4, -1, -1, 2, 4};
  TEST_RESHAPE({4, 1, 2, 4, 1},  // input_dims
               {3},              // input_data
               {1, 4},           // shape_dims
               {-1, -1, 2, 4},   // shape_data
               output_dims,      // output_dims
               output_data, {},  // expected_output
               {}                // expected_dims
  );
}

// Create the model with a 2x2 shape. Processing still works because the new
// shape ends up being hardcoded as a flat vector.
TF_LITE_MICRO_TEST(InvalidShape) {
  using tflite::testing::CreateFloatTensor;
  using tflite::testing::IntArrayFromInitializer;
  using tflite::testing::IntArrayFromInts;
  TfLiteIntArray* input_dims = IntArrayFromInitializer({3, 1, 2, 2});
  auto input_data = {3.0f};
  auto input_tensor = CreateFloatTensor(input_data, input_dims, "input_tensor");
  float output_data[4];
  int output_dims_data[6] = {2, 2, 1, 2, 2, 1};
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  auto output_tensor =
      CreateFloatTensor(output_data, output_dims, "input_tensor");
  tflite::testing::TestReshapeImpl<float>(&input_tensor,   // input_tensor
                                          nullptr,         // shape_tensor
                                          &output_tensor,  // output_tensor
                                          {},              // expected_output
                                          {}               // expected_dims
  );
}

TF_LITE_MICRO_TEST(RegularShapes) {
  uint8_t output_data[32];
  int output_dims[4] = {3, 2, 2, 2};
  TEST_RESHAPE({4, 1, 2, 4, 1},                        // input_dims
               {1, 2, 3, 4, 5, 6, 7, 8},               // input_data
               {1, 3},                                 // shape_dims
               {2, 2, 2},                              // shape_data
               output_dims,                            // output_dims
               output_data, {1, 2, 3, 4, 5, 6, 7, 8},  // expected_output
               {2, 2, 2}                               // expected_dims
  );
}

TF_LITE_MICRO_TEST(WithStretchDimension) {
  uint8_t output_data[32];
  int output_dims[4] = {3, 2, 1, -1};
  TEST_RESHAPE({4, 1, 2, 4, 1},                        // input_dims
               {1, 2, 3, 4, 5, 6, 7, 8},               // input_data
               {1, 3},                                 // shape_dims
               {2, 1, -1},                             // shape_data
               output_dims,                            // output_dims
               output_data, {1, 2, 3, 4, 5, 6, 7, 8},  // expected_output
               {2, 1, 4}                               // expected_dims
  );
}

// Shape is specified as '[]', which is the modern way to represent scalar
// input and output.
TF_LITE_MICRO_TEST(ScalarOutput) {
  uint8_t output_data[4];
  int output_dims[1] = {0};
  TEST_RESHAPE({1, 1},            // input_dims
               {3},               // input_data
               {0},               // shape_dims
               {},                // shape_data
               output_dims,       // output_dims
               output_data, {3},  // expected_output
               {}                 // expected_dims
  );
}

// Some old models specify '[0]' as the new shape, indicating that both input
// and output are scalars.
TF_LITE_MICRO_TEST(LegacyScalarOutput) {
  using tflite::testing::CreateFloatTensor;
  using tflite::testing::IntArrayFromInitializer;
  using tflite::testing::IntArrayFromInts;
  TfLiteIntArray* input_dims = IntArrayFromInitializer({1, 1});
  auto input_data = {3.0f};
  auto input_tensor = CreateFloatTensor(input_data, input_dims, "input_tensor");
  float output_data[1];
  int output_dims_data[2] = {1, 0};
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  auto output_tensor =
      CreateFloatTensor(output_data, output_dims, "input_tensor");
  TfLiteIntArray* shape_dims = tflite::testing::IntArrayFromInitializer({1, 0});
  auto shape_tensor = tflite::testing::CreateTensor<int32_t, kTfLiteInt32>(
      {0}, shape_dims, "shape_tensor");
  tflite::testing::TestReshapeImpl<float>(&input_tensor,   // input_tensor
                                          &shape_tensor,   // shape_tensor
                                          &output_tensor,  // output_tensor
                                          {},              // expected_output
                                          {}               // expected_dims
  );
  tflite::testing::TestReshapeImpl<float>(&input_tensor,   // input_tensor
                                          nullptr,         // shape_tensor
                                          &output_tensor,  // output_tensor
                                          {3},             // expected_output
                                          {}               // expected_dims
  );
}

#undef TEST_RESHAPE

TF_LITE_MICRO_TESTS_END
