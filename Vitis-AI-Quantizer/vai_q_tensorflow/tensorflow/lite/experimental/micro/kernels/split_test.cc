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
#include "tensorflow/lite/experimental/micro/debug_log.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

namespace tflite {
namespace testing {

void TestSplitTwoOutputsFloat(
    std::initializer_list<int> input_dims_data,
    std::initializer_list<float> input_data,
    std::initializer_list<int> axis_dims_data,
    std::initializer_list<int32_t> axis_data,
    std::initializer_list<int> output1_dims_data,
    std::initializer_list<float> expected_output1_data,
    std::initializer_list<int> output2_dims_data,
    std::initializer_list<float> expected_output2_data, float* output1_data,
    float* output2_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInitializer(axis_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInitializer(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInitializer(output2_dims_data);
  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 2;
  constexpr int axis_size = 1;
  constexpr int tensors_size = input_size + output_size + axis_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantized32Tensor(axis_data, axis_dims, "axis_tensor", 0, 5),
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateFloatTensor(output1_data, output1_dims, "output1_tensor"),
      CreateFloatTensor(output2_data, output2_dims, "output2_tensor")};

  // Currently only support constant axis tensor.
  tensors[0].allocation_type = kTfLiteMmapRo;
  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output1_dims_count; ++i) {
    output1_data[i] = 23;
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    output2_data[i] = 23;
  }

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);
  tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_SPLIT, /* version= */ 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteSplitParams builtin_data = {
      .num_splits = 2,
  };

  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, 0);
  }
  TfLiteIntArray* inputs_array = IntArrayFromInitializer({2, 0, 1});
  TfLiteIntArray* outputs_array = IntArrayFromInitializer({2, 2, 3});
  TfLiteIntArray* temporaries_array = IntArrayFromInitializer({0});

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output1_data.begin()[i], output1_data[i],
                              1e-5f);
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output2_data.begin()[i], output2_data[i],
                              1e-5f);
  }
}

void TestSplitFourOutputsFloat(
    std::initializer_list<int> input_dims_data,
    std::initializer_list<float> input_data,
    std::initializer_list<int> axis_dims_data,
    std::initializer_list<int32_t> axis_data,
    std::initializer_list<int> output1_dims_data,
    std::initializer_list<float> expected_output1_data,
    std::initializer_list<int> output2_dims_data,
    std::initializer_list<float> expected_output2_data,
    std::initializer_list<int> output3_dims_data,
    std::initializer_list<float> expected_output3_data,
    std::initializer_list<int> output4_dims_data,
    std::initializer_list<float> expected_output4_data, float* output1_data,
    float* output2_data, float* output3_data, float* output4_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInitializer(axis_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInitializer(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInitializer(output2_dims_data);
  TfLiteIntArray* output3_dims = IntArrayFromInitializer(output3_dims_data);
  TfLiteIntArray* output4_dims = IntArrayFromInitializer(output4_dims_data);
  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);
  const int output3_dims_count = ElementCount(*output3_dims);
  const int output4_dims_count = ElementCount(*output4_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 4;
  constexpr int axis_size = 1;
  constexpr int tensors_size = input_size + output_size + axis_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantized32Tensor(axis_data, axis_dims, "axis_tensor", 0, 5),
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateFloatTensor(output1_data, output1_dims, "output1_tensor"),
      CreateFloatTensor(output2_data, output2_dims, "output2_tensor"),
      CreateFloatTensor(output3_data, output1_dims, "output3_tensor"),
      CreateFloatTensor(output4_data, output1_dims, "output4_tensor")};

  // Currently only support constant axis tensor.
  tensors[0].allocation_type = kTfLiteMmapRo;
  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output1_dims_count; ++i) {
    output1_data[i] = 23;
  }
  for (int i = 0; i < output2_dims_count; ++i) {
    output2_data[i] = 23;
  }
  for (int i = 0; i < output3_dims_count; ++i) {
    output3_data[i] = 23;
  }
  for (int i = 0; i < output4_dims_count; ++i) {
    output4_data[i] = 23;
  }

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);
  tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_SPLIT, /* version= */ 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteSplitParams builtin_data = {
      .num_splits = 4,
  };

  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, 0);
  }
  TfLiteIntArray* inputs_array = IntArrayFromInitializer({2, 0, 1});
  TfLiteIntArray* outputs_array = IntArrayFromInitializer({4, 2, 3, 4, 5});
  TfLiteIntArray* temporaries_array = IntArrayFromInitializer({0});

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output1_data.begin()[i], output1_data[i],
                              1e-5f);
  }
  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output2_data.begin()[i], output2_data[i],
                              1e-5f);
  }
  for (int i = 0; i < output3_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output3_data.begin()[i], output3_data[i],
                              1e-5f);
  }
  for (int i = 0; i < output4_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output4_data.begin()[i], output4_data[i],
                              1e-5f);
  }
}

void TestSplitTwoOutputsQuantized(
    std::initializer_list<int> input_dims_data,
    std::initializer_list<uint8_t> input_data,
    std::initializer_list<int> axis_dims_data,
    std::initializer_list<int32_t> axis_data,
    std::initializer_list<int> output1_dims_data,
    std::initializer_list<uint8_t> expected_output1_data,
    std::initializer_list<int> output2_dims_data,
    std::initializer_list<uint8_t> expected_output2_data, uint8_t* output1_data,
    uint8_t* output2_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInitializer(axis_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInitializer(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInitializer(output2_dims_data);
  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 2;
  constexpr int axis_size = 1;
  constexpr int tensors_size = input_size + output_size + axis_size;
  TfLiteTensor tensors[tensors_size] = {
      // CreateQuantizedTensor needs min/max values as input, but these values
      // don't matter as to the functionality of SPLIT, so just set as 0 and 10.
      CreateQuantized32Tensor(axis_data, axis_dims, "axis_tensor", 0, 10),
      CreateQuantizedTensor(input_data, input_dims, "input_tensor", 0, 10),
      CreateQuantizedTensor(output1_data, output1_dims, "output1_tensor", 0,
                            10),
      CreateQuantizedTensor(output2_data, output2_dims, "output2_tensor", 0,
                            10)};

  // Currently only support constant axis tensor.
  tensors[0].allocation_type = kTfLiteMmapRo;

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output1_dims_count; ++i) {
    output1_data[i] = 23;
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    output2_data[i] = 23;
  }

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);
  tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_SPLIT, /* version= */ 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteSplitParams builtin_data = {
      .num_splits = 2,
  };

  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, 0);
  }
  TfLiteIntArray* inputs_array = IntArrayFromInitializer({2, 0, 1});
  TfLiteIntArray* outputs_array = IntArrayFromInitializer({2, 2, 3});
  TfLiteIntArray* temporaries_array = IntArrayFromInitializer({0});

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output1_data.begin()[i], output1_data[i]);
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output2_data.begin()[i], output2_data[i]);
  }
}

void TestSplitTwoOutputsQuantized32(
    std::initializer_list<int> input_dims_data,
    std::initializer_list<int32_t> input_data,
    std::initializer_list<int> axis_dims_data,
    std::initializer_list<int32_t> axis_data,
    std::initializer_list<int> output1_dims_data,
    std::initializer_list<int32_t> expected_output1_data,
    std::initializer_list<int> output2_dims_data,
    std::initializer_list<int32_t> expected_output2_data, int32_t* output1_data,
    int32_t* output2_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInitializer(axis_dims_data);
  TfLiteIntArray* output1_dims = IntArrayFromInitializer(output1_dims_data);
  TfLiteIntArray* output2_dims = IntArrayFromInitializer(output2_dims_data);
  const int output1_dims_count = ElementCount(*output1_dims);
  const int output2_dims_count = ElementCount(*output2_dims);

  constexpr int input_size = 1;
  constexpr int output_size = 2;
  constexpr int axis_size = 1;
  constexpr int tensors_size = input_size + output_size + axis_size;
  TfLiteTensor tensors[tensors_size] = {
      // CreateQuantizedTensor needs min/max values as input, but these values
      // don't matter as to the functionality of SPLIT, so just set as 0 and 10.
      CreateQuantized32Tensor(axis_data, axis_dims, "axis_tensor", 0, 10),
      CreateQuantized32Tensor(input_data, input_dims, "input_tensor", 0, 10),
      CreateQuantized32Tensor(output1_data, output1_dims, "output1_tensor", 0,
                              10),
      CreateQuantized32Tensor(output2_data, output2_dims, "output2_tensor", 0,
                              10)};

  // Currently only support constant axis tensor.
  tensors[0].allocation_type = kTfLiteMmapRo;

  // Place a unique value in the uninitialized output buffer.
  for (int i = 0; i < output1_dims_count; ++i) {
    output1_data[i] = 23;
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    output2_data[i] = 23;
  }

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);
  tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_SPLIT, /* version= */ 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteSplitParams builtin_data = {
      .num_splits = 2,
  };

  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, 0);
  }
  TfLiteIntArray* inputs_array = IntArrayFromInitializer({2, 0, 1});
  TfLiteIntArray* outputs_array = IntArrayFromInitializer({2, 2, 3});
  TfLiteIntArray* temporaries_array = IntArrayFromInitializer({0});

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }

  for (int i = 0; i < output1_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output1_data.begin()[i], output1_data[i]);
  }

  for (int i = 0; i < output2_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output2_data.begin()[i], output2_data[i]);
  }
}

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TwoSplitFourDimensionalAxisZero) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsFloat(
      {4, 2, 2, 2, 2},                                          // Input shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // Input values
      {1, 1},                                                   // Axis shape
      {0},                                                      // Axis value
      {4, 1, 2, 2, 2},                                          // Output1 shape
      {1, 2, 3, 4, 5, 6, 7, 8},         // Output1 values
      {4, 1, 2, 2, 2},                  // Output2 shape
      {9, 10, 11, 12, 13, 14, 15, 16},  // Output2 values
      output1_data, output2_data);
}

TF_LITE_MICRO_TEST(TwoSplitFourDimensionalAxisOne) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsFloat(
      {4, 2, 2, 2, 2},                                          // Input shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // Input values
      {1, 1},                                                   // Axis shape
      {1},                                                      // Axis value
      {4, 2, 1, 2, 2},                                          // Output1 shape
      {1, 2, 3, 4, 9, 10, 11, 12},   // Output1 values
      {4, 2, 1, 2, 2},               // Output2 shape
      {5, 6, 7, 8, 13, 14, 15, 16},  // Output2 values
      output1_data, output2_data);
}

TF_LITE_MICRO_TEST(TwoSplitFourDimensionalAxisTwo) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsFloat(
      {4, 2, 2, 2, 2},                                          // Input shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // Input values
      {1, 1},                                                   // Axis shape
      {2},                                                      // Axis value
      {4, 2, 2, 1, 2},                                          // Output1 shape
      {1, 2, 5, 6, 9, 10, 13, 14},   // Output1 values
      {4, 2, 2, 1, 2},               // Output2 shape
      {3, 4, 7, 8, 11, 12, 15, 16},  // Output2 values
      output1_data, output2_data);
}

TF_LITE_MICRO_TEST(TwoSplitFourDimensionalAxisThree) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsFloat(
      {4, 2, 2, 2, 2},                                          // Input shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // Input values
      {1, 1},                                                   // Axis shape
      {3},                                                      // Axis value
      {4, 2, 2, 2, 1},                                          // Output1 shape
      {1, 3, 5, 7, 9, 11, 13, 15},   // Output1 values
      {4, 2, 2, 2, 1},               // Output2 shape
      {2, 4, 6, 8, 10, 12, 14, 16},  // Output2 values
      output1_data, output2_data);
}

TF_LITE_MICRO_TEST(TwoSplitFourDimensionalNegativeAxis) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsFloat(
      {4, 2, 2, 2, 2},                                          // Input shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // Input values
      {1, 1},                                                   // Axis shape
      {-4},                                                     // Axis value
      {4, 1, 2, 2, 2},                                          // Output1 shape
      {1, 2, 3, 4, 5, 6, 7, 8},         // Output1 values
      {4, 1, 2, 2, 2},                  // Output2 shape
      {9, 10, 11, 12, 13, 14, 15, 16},  // Output2 values
      output1_data, output2_data);
}

TF_LITE_MICRO_TEST(FourSplit) {
  constexpr int output1_dims_count = 1;
  constexpr int output2_dims_count = 1;
  constexpr int output3_dims_count = 1;
  constexpr int output4_dims_count = 1;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  float output3_data[output3_dims_count];
  float output4_data[output4_dims_count];
  tflite::testing::TestSplitFourOutputsFloat({1, 4},        // Input shape
                                             {1, 2, 3, 4},  // Input values
                                             {1, 1},        // Axis shape
                                             {0},           // Axis value
                                             {1, 1},        // Output1 shape
                                             {1},           // Output1 values
                                             {1, 1},        // Output2 shape
                                             {2},           // Output2 values
                                             {1, 1},        // Output3 shape
                                             {3},           // Output3 values
                                             {1, 1},        // Output4 shape
                                             {4},           // Output4 values
                                             output1_data, output2_data,
                                             output3_data, output4_data);
}

TF_LITE_MICRO_TEST(TwoSplitOneDimensional) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  float output1_data[output1_dims_count];
  float output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsFloat({1, 2},  // Input shape
                                            {1, 2},  // Input values
                                            {1, 1},  // Axis shape
                                            {0},     // Axis value
                                            {1, 1},  // Output1 shape
                                            {1},     // Output1 values
                                            {1, 1},  // Output2 shape
                                            {2},     // Output2 values
                                            output1_data, output2_data);
}

TF_LITE_MICRO_TEST(TwoSplitFourDimensionalQuantized) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  uint8_t output1_data[output1_dims_count];
  uint8_t output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsQuantized(
      {4, 2, 2, 2, 2},                                          // Input shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // Input values
      {1, 1},                                                   // Axis shape
      {0},                                                      // Axis value
      {4, 1, 2, 2, 2},                                          // Output1 shape
      {1, 2, 3, 4, 5, 6, 7, 8},         // Output1 values
      {4, 1, 2, 2, 2},                  // Output2 shape
      {9, 10, 11, 12, 13, 14, 15, 16},  // Output2 values
      output1_data, output2_data);
}

TF_LITE_MICRO_TEST(TwoSplitFourDimensionalQuantized32) {
  constexpr int output1_dims_count = 8;
  constexpr int output2_dims_count = 8;
  int32_t output1_data[output1_dims_count];
  int32_t output2_data[output2_dims_count];
  tflite::testing::TestSplitTwoOutputsQuantized32(
      {4, 2, 2, 2, 2},                                          // Input shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},  // Input values
      {1, 1},                                                   // Axis shape
      {0},                                                      // Axis value
      {4, 1, 2, 2, 2},                                          // Output1 shape
      {1, 2, 3, 4, 5, 6, 7, 8},         // Output1 values
      {4, 1, 2, 2, 2},                  // Output2 shape
      {9, 10, 11, 12, 13, 14, 15, 16},  // Output2 values
      output1_data, output2_data);
}

TF_LITE_MICRO_TESTS_END
