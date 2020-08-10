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
#include <cstdint>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Matcher;

template <typename RegularInputOuput>
class PadOpModel : public SingleOpModel {
 public:
  void SetInput(std::initializer_list<RegularInputOuput> data) {
    PopulateTensor<RegularInputOuput>(input_, data);
  }

  template <typename QuantizedInputOutput>
  void SetQuantizedInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<QuantizedInputOutput>(input_, data);
  }

  template <typename QuantizedInputOutput>
  void SetQuantizedPadValue(float data) {
    QuantizeAndPopulate<QuantizedInputOutput>(constant_values_, {data});
  }

  void SetPaddings(std::initializer_list<int> paddings) {
    PopulateTensor<int>(paddings_, paddings);
  }

  std::vector<RegularInputOuput> GetOutput() {
    return ExtractVector<RegularInputOuput>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  template <typename QuantizedInputOutput>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<QuantizedInputOutput>(
        ExtractVector<QuantizedInputOutput>(output_), GetScale(output_),
        GetZeroPoint(output_));
  }

 protected:
  int input_;
  int output_;
  int paddings_;
  int constant_values_;
};

// Tests case where paddings is a const tensor. Type T is the dtype.
template <typename T1>
class PadV2OpConstModel : public PadOpModel<T1> {
 public:
  PadV2OpConstModel(const TensorData& input,
                    std::initializer_list<int> paddings_shape,
                    std::initializer_list<int> paddings, T1 constant_values,
                    const TensorData& output) {
    this->input_ = this->AddInput(input);
    this->paddings_ =
        this->AddConstInput(TensorType_INT32, paddings, paddings_shape);
    this->constant_values_ =
        this->AddConstInput(GetTensorType<T1>(), {constant_values}, {1});

    this->output_ = this->AddOutput(output);

    this->SetBuiltinOp(BuiltinOperator_PADV2, BuiltinOptions_PadV2Options,
                       CreatePadV2Options(this->builder_).Union());
    this->BuildInterpreter({input.shape});
  }

  PadV2OpConstModel(const TensorData& input,
                    std::initializer_list<int> paddings_shape,
                    std::initializer_list<int> paddings,
                    const TensorData& constant_values,
                    const TensorData& output) {
    this->input_ = this->AddInput(input);
    this->paddings_ =
        this->AddConstInput(TensorType_INT32, paddings, paddings_shape);
    this->constant_values_ = this->AddInput(constant_values);

    this->output_ = this->AddOutput(output);

    this->SetBuiltinOp(BuiltinOperator_PADV2, BuiltinOptions_PadV2Options,
                       CreatePadV2Options(this->builder_).Union());
    this->BuildInterpreter({input.shape});
  }
};

// Tests case where paddings is a const tensor.
//
// Example usage is as follows:
//    PadOpDynamicModel m(input_shape, paddings_shape, paddings_data);
//    m.SetInput(input_data);
//    m.Invoke();
class PadOpConstModel : public PadOpModel<float> {
 public:
  PadOpConstModel(const TensorData& input,
                  std::initializer_list<int> paddings_shape,
                  std::initializer_list<int> paddings,
                  const TensorData& output) {
    this->input_ = AddInput(input);
    paddings_ = AddConstInput(TensorType_INT32, paddings, paddings_shape);
    constant_values_ = AddNullInput();
    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_PAD, BuiltinOptions_PadOptions,
                 CreatePadOptions(builder_).Union());
    BuildInterpreter({input.shape});
  }
};

// Test case where paddings is a non-const tensor.
template <typename RegularInputOuput>
class PadV2OpDynamicModel : public PadOpModel<RegularInputOuput> {
 public:
  PadV2OpDynamicModel(const TensorData& input,
                      std::initializer_list<int> paddings_shape,
                      RegularInputOuput constant_values,
                      const TensorData& output) {
    this->input_ = this->AddInput(input);
    this->paddings_ = this->AddInput(TensorType_INT32);
    this->constant_values_ = this->AddConstInput(
        GetTensorType<RegularInputOuput>(), {constant_values}, {1});
    this->output_ = this->AddOutput(output);

    this->SetBuiltinOp(BuiltinOperator_PADV2, BuiltinOptions_PadV2Options,
                       CreatePadV2Options(this->builder_).Union());
    this->BuildInterpreter({input.shape, paddings_shape});
  }
  PadV2OpDynamicModel(const TensorData& input,
                      std::initializer_list<int> paddings_shape,
                      const TensorData& constant_values,
                      const TensorData& output) {
    this->input_ = this->AddInput(input);
    this->paddings_ = this->AddInput(TensorType_INT32);
    this->constant_values_ = this->AddInput(constant_values);
    this->output_ = this->AddOutput(output);

    this->SetBuiltinOp(BuiltinOperator_PADV2, BuiltinOptions_PadV2Options,
                       CreatePadV2Options(this->builder_).Union());
    this->BuildInterpreter({input.shape, paddings_shape});
  }
};

// Test case where paddings is a non-const tensor.
//
// Example usage is as follows:
//    PadOpDynamicModel m(input_shape, paddings_shape);
//    m.SetInput(input_data);
//    m.SetPaddings(paddings_data);
//    m.Invoke();
class PadOpDynamicModel : public PadOpModel<float> {
 public:
  PadOpDynamicModel(const TensorData& input,
                    std::initializer_list<int> paddings_shape,
                    const TensorData& output) {
    this->input_ = this->AddInput(input);
    this->paddings_ = this->AddInput(TensorType_INT32);
    this->constant_values_ = this->AddNullInput();
    this->output_ = this->AddOutput(output);

    this->SetBuiltinOp(BuiltinOperator_PAD, BuiltinOptions_PadOptions,
                       CreatePadOptions(this->builder_).Union());
    this->BuildInterpreter({input.shape, paddings_shape});
  }
};

#ifdef GTEST_HAS_DEATH_TEST
TEST(PadOpTest, TooManyDimensions) {
  EXPECT_DEATH(
      PadOpConstModel({TensorType_FLOAT32, {1, 2, 3, 4, 5, 6, 7, 8, 9}}, {9, 2},
                      {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9},
                      {TensorType_FLOAT32}),
      "dims <= 4");
}

TEST(PadOpTest, UnequalDimensions) {
  EXPECT_DEATH(PadOpConstModel({TensorType_FLOAT32, {1, 1, 2, 1}}, {3, 2},
                               {1, 1, 2, 2, 3, 3}, {TensorType_FLOAT32}),
               "3 != 4");
}

TEST(PadOpTest, InvalidPadValue) {
  EXPECT_DEATH(
      PadOpConstModel({TensorType_FLOAT32, {1, 1, 2, 1}}, {4, 2},
                      {0, 0, 1, -1, 2, -1, 0, 0}, {TensorType_FLOAT32}),
      "Pad value has to be greater than equal to 0.");
}
#endif

TEST(PadOpTest, SimpleConstTest) {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadOpConstModel m({TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2},
                    {1, 1, 0, 0, 1, 1, 0, 0}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0,
                                0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({3, 2, 4, 1}));
}

TEST(PadOpTest, SimpleConstImageStyleTest) {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadOpConstModel m({TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2},
                    {0, 0, 1, 1, 1, 1, 0, 0}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
                                               0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

// Optimized versions may choose to handle zero-sized images differently.
TEST(PadOpTest, ZeroHeightConstImageStyleTest) {
  PadOpConstModel m({TensorType_FLOAT32, {1, 0, 2, 1}}, {4, 2},
                    {0, 0, 1, 1, 1, 1, 0, 0}, {TensorType_FLOAT32});
  // Nothing to SetInput().
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 4, 1}));
}

// Optimized versions may choose to handle zero-sized images differently.
TEST(PadOpTest, ZeroWidthConstImageStyleTest) {
  PadOpConstModel m({TensorType_FLOAT32, {1, 2, 0, 1}}, {4, 2},
                    {0, 0, 1, 1, 1, 1, 0, 0}, {TensorType_FLOAT32});
  // Nothing to SetInput().
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 2, 1}));
}

TEST(PadOpTest, SimpleConst1DTest) {
  PadOpConstModel m({TensorType_FLOAT32, {2}}, {1, 2}, {1, 2},
                    {TensorType_FLOAT32});
  m.SetInput({2, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 2, 3, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({5}));
}

TEST(PadOpTest, SimpleDynamicTest) {
  PadOpDynamicModel m({TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2},
                      {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
                                               0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST(PadOpTest, AdvancedConstTest) {
  PadOpConstModel m({TensorType_FLOAT32, {1, 2, 3, 1}}, {4, 2},
                    {1, 0, 0, 2, 0, 3, 0, 0}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 4, 5,
                        6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 4, 6, 1}));
}

TEST(PadOpTest, AdvancedConstImageStyleTest) {
  PadOpConstModel m({TensorType_FLOAT32, {1, 2, 3, 1}}, {4, 2},
                    {0, 0, 0, 2, 1, 3, 0, 0}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST(PadOpTest, AdvancedDynamicTest) {
  PadOpDynamicModel m({TensorType_FLOAT32, {1, 2, 3, 1}}, {4, 2},
                      {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetPaddings({0, 0, 0, 2, 1, 3, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

std::vector<Matcher<float>> DequantizedArrayNear(
    const std::vector<float>& values, const float min, const float max) {
  const float quantization_tolerance = (max - min) / 255.0;
  return ArrayFloatNear(values, quantization_tolerance);
}

class QuantizedPadOpTest : public ::testing::Test {};

#ifdef GTEST_HAS_DEATH_TEST
template <typename integer_type, TensorType tensor_dtype>
void ZeroNotInQuantizationRange() {
  // The test_util and actual quantization code currently ensure that the range
  // must include zero, but if that ever changes, this test will catch it.
  EXPECT_DEATH(
      PadOpConstModel m({tensor_dtype, {1, 2, 2, 1}, 1.0, 2.0}, {4, 2},
                        {0, 0, 1, 1, 1, 1, 0, 0}, {tensor_dtype, {}, 1.0, 2.0}),
      ".*Check failed: f_min <= 0.*");
}

TEST_F(QuantizedPadOpTest, UInt8ZeroNotInQuantizationRange) {
  ZeroNotInQuantizationRange<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadOpTest, Int8ZeroNotInQuantizationRange) {
  ZeroNotInQuantizationRange<int8_t, TensorType_INT8>();
}
#endif

template <typename integer_type, TensorType tensor_dtype>
void SimpleConstTest() {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadOpConstModel m({tensor_dtype, {1, 2, 2, 1}, -1.0, 1.0}, {4, 2},
                    {0, 0, 1, 1, 1, 1, 0, 0}, {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7});
  m.Invoke();
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, 0, 0, 0, 0, -0.8, 0.2, 0, 0, 0.9, 0.7, 0, 0, 0, 0, 0},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(QuantizedPadOpTest, UInt8SimpleConstTest) {
  SimpleConstTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadOpTest, Int8SimpleConstTest) {
  SimpleConstTest<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void SimpleDynamicTest() {
  PadOpDynamicModel m({tensor_dtype, {1, 2, 2, 1}, -1.0, 1.0}, {4, 2},
                      {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7});
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, 0, 0, 0, 0, -0.8, 0.2, 0, 0, 0.9, 0.7, 0, 0, 0, 0, 0},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(QuantizedPadOpTest, UInt8SimpleDynamicTest) {
  SimpleDynamicTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadOpTest, Int8SimpleDynamicTest) {
  SimpleDynamicTest<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void AdvancedConstTest() {
  PadOpConstModel m({tensor_dtype, {1, 2, 3, 1}, -1.0, 1.0}, {4, 2},
                    {0, 0, 0, 2, 1, 3, 0, 0}, {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7, 0.1, -0.3});
  m.Invoke();
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, -0.8, 0.2, 0.9, 0, 0, 0, 0, 0.7, 0.1, -0.3, 0, 0, 0,
                   0, 0,    0,   0,   0, 0, 0, 0, 0,   0,   0,    0, 0, 0},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(QuantizedPadOpTest, UInt8AdvancedConstTest) {
  AdvancedConstTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadOpTest, Int8AdvancedConstTest) {
  AdvancedConstTest<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void AdvancedDynamicTest() {
  PadOpDynamicModel m({tensor_dtype, {1, 2, 3, 1}, -1.0, 1.0}, {4, 2},
                      {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7, 0.1, -0.3});
  m.SetPaddings({0, 0, 0, 2, 1, 3, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, -0.8, 0.2, 0.9, 0, 0, 0, 0, 0.7, 0.1, -0.3, 0, 0, 0,
                   0, 0,    0,   0,   0, 0, 0, 0, 0,   0,   0,    0, 0, 0},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(QuantizedPadOpTest, UInt8AdvancedDynamicTest) {
  AdvancedDynamicTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadOpTest, Int8AdvancedDynamicTest) {
  AdvancedDynamicTest<int8_t, TensorType_INT8>();
}

#ifdef GTEST_HAS_DEATH_TEST
TEST(PadV2OpTest, TooManyDimensions) {
  typedef PadV2OpConstModel<float> f;
  EXPECT_DEATH(f({TensorType_FLOAT32, {1, 2, 3, 4, 5, 6, 7, 8, 9}}, {9, 2},
                 {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9}, 0.0,
                 {TensorType_FLOAT32}),
               "dims <= 4");
}

TEST(PadV2OpTest, UnequalDimensions) {
  typedef PadV2OpConstModel<float> f;
  EXPECT_DEATH(f({TensorType_FLOAT32, {1, 1, 2, 1}}, {3, 2}, {1, 1, 2, 2, 3, 3},
                 0.0, {TensorType_FLOAT32}),
               "3 != 4");
}

TEST(PadV2OpTest, InvalidPadValue) {
  typedef PadV2OpConstModel<float> f;
  EXPECT_DEATH(f({TensorType_FLOAT32, {1, 1, 2, 1}}, {4, 2},
                 {0, 0, 1, -1, 2, -1, 0, 0}, 0.0, {TensorType_FLOAT32}),
               "Pad value has to be greater than equal to 0.");
}
#endif

TEST(PadV2OpTest, SimpleConstTestUint8) {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<float> m({TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2},
                             {0, 0, 1, 1, 1, 1, 0, 0}, 0.0,
                             {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
                                               0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST(PadV2OpTest, SimpleConstTestInt8) {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<float> m({TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2},
                             {0, 0, 1, 1, 1, 1, 0, 0}, 0.0,
                             {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
                                               0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST(PadV2OpTest, SimpleConstFloat32ValuedTestUint8) {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<float> m({TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2},
                             {0, 0, 1, 1, 1, 1, 0, 0}, 5, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 1, 2, 5, 5, 3, 4,
                                               5, 5, 5, 5, 5}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST(PadV2OpTest, SimpleConstFloat32ValuedTestInt8) {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<float> m({TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2},
                             {0, 0, 1, 1, 1, 1, 0, 0}, 5, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 1, 2, 5, 5, 3, 4,
                                               5, 5, 5, 5, 5}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST(PadV2OpTest, Simple4DConstFloat32ValuedTest) {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<float> m({TensorType_FLOAT32, {1, 1, 2, 1}}, {4, 2},
                             {0, 1, 0, 0, 0, 0, 0, 1}, 5, {TensorType_FLOAT32});
  m.SetInput({3, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3, 5, 3, 5, 5, 5, 5, 5}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 2, 2}));
}

TEST(PadV2OpTest, SimpleConstInt32ValuedTest) {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<int32_t> m({TensorType_INT32, {1, 2, 2, 1}}, {4, 2},
                               {0, 0, 1, 1, 1, 1, 0, 0}, 5, {TensorType_INT32});
  m.SetInput({1, 2, 3, 4});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 1, 2, 5, 5, 3, 4,
                                               5, 5, 5, 5, 5}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST(PadV2OpTest, SimpleDynamicTest) {
  PadV2OpDynamicModel<float> m({TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2}, 0.0,
                               {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
                                               0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST(PadV2OpTest, SimpleDynamicValuedTest) {
  PadV2OpDynamicModel<float> m({TensorType_FLOAT32, {1, 2, 2, 1}}, {4, 2}, 5,
                               {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4});
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({5, 5, 5, 5, 5, 1, 2, 5, 5, 3, 4,
                                               5, 5, 5, 5, 5}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST(PadV2OpTest, AdvancedConstTest) {
  PadV2OpConstModel<float> m({TensorType_FLOAT32, {1, 2, 3, 1}}, {4, 2},
                             {0, 0, 0, 2, 1, 3, 0, 0}, 0, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST(PadV2OpTest, AdvancedDynamicTest) {
  PadV2OpDynamicModel<float> m({TensorType_FLOAT32, {1, 2, 3, 1}}, {4, 2}, 0,
                               {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetPaddings({0, 0, 0, 2, 1, 3, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

class QuantizedPadV2OpTest : public ::testing::Test {
 protected:
  std::vector<Matcher<float>> DequantizedArrayNear(
      const std::vector<float>& values, const float min, const float max) {
    const float quantization_tolerance = (max - min) / 255.0;
    return ArrayFloatNear(values, quantization_tolerance);
  }
};

#ifdef GTEST_HAS_DEATH_TEST
template <TensorType tensor_dtype>
void ZeroNotInQuantizationRangeV2() {
  // The test_util and actual quantization code currently ensure that the range
  // must include zero, but if that ever changes, this test will catch it.
  typedef PadV2OpConstModel<float> f;
  EXPECT_DEATH(f({tensor_dtype, {1, 2, 2, 1}, 1.0, 2.0}, {4, 2},
                 {0, 0, 1, 1, 1, 1, 0, 0}, 0, {tensor_dtype, {}, 1.0, 2.0}),
               ".*Check failed: f_min <= 0.*");
}

TEST_F(QuantizedPadV2OpTest, UInt8ZeroNotInQuantizationRange) {
  ZeroNotInQuantizationRangeV2<TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8ZeroNotInQuantizationRange) {
  ZeroNotInQuantizationRangeV2<TensorType_INT8>();
}
#endif

template <typename integer_type, TensorType tensor_dtype>
void SimpleConstTestV2() {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<integer_type> m(
      {tensor_dtype, {1, 2, 2, 1}, -1.0, 1.0}, {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0},
      {tensor_dtype, {1}, -1.0, 1.0}, {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7});
  m.template SetQuantizedPadValue<integer_type>(0);
  m.Invoke();
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, 0, 0, 0, 0, -0.8, 0.2, 0, 0, 0.9, 0.7, 0, 0, 0, 0, 0},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8SimpleConstTest) {
  SimpleConstTestV2<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8SimpleConstTest) {
  SimpleConstTestV2<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void SimpleDynamicTestV2() {
  PadV2OpDynamicModel<integer_type> m({tensor_dtype, {1, 2, 2, 1}, -1.0, 1.0},
                                      {4, 2}, {tensor_dtype, {1}, -1.0, 1.0},
                                      {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7});
  m.template SetQuantizedPadValue<integer_type>(0);
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, 0, 0, 0, 0, -0.8, 0.2, 0, 0, 0.9, 0.7, 0, 0, 0, 0, 0},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8SimpleDynamicTest) {
  SimpleDynamicTestV2<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8SimpleDynamicTest) {
  SimpleDynamicTestV2<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void AdvancedConstTestV2() {
  PadV2OpConstModel<integer_type> m(
      {tensor_dtype, {1, 2, 3, 1}, -1.0, 1.0}, {4, 2}, {0, 0, 0, 2, 1, 3, 0, 0},
      {tensor_dtype, {1}, -1.0, 1.0}, {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7, 0.1, -0.3});
  m.template SetQuantizedPadValue<integer_type>(0);
  m.Invoke();
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, -0.8, 0.2, 0.9, 0, 0, 0, 0, 0.7, 0.1, -0.3, 0, 0, 0,
                   0, 0,    0,   0,   0, 0, 0, 0, 0,   0,   0,    0, 0, 0},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8AdvancedConstTest) {
  AdvancedConstTestV2<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8AdvancedConstTest) {
  AdvancedConstTestV2<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void AdvancedDynamicTestV2() {
  PadV2OpDynamicModel<integer_type> m({tensor_dtype, {1, 2, 3, 1}, -1.0, 1.0},
                                      {4, 2}, {tensor_dtype, {1}, -1.0, 1.0},
                                      {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7, 0.1, -0.3});
  m.template SetQuantizedPadValue<integer_type>(0);
  m.SetPaddings({0, 0, 0, 2, 1, 3, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, -0.8, 0.2, 0.9, 0, 0, 0, 0, 0.7, 0.1, -0.3, 0, 0, 0,
                   0, 0,    0,   0,   0, 0, 0, 0, 0,   0,   0,    0, 0, 0},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8AdvancedDynamicTest) {
  AdvancedDynamicTestV2<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8AdvancedDynamicTest) {
  AdvancedDynamicTestV2<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void SimpleConstValuedTest() {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadV2OpConstModel<integer_type> m(
      {tensor_dtype, {1, 2, 2, 1}, -1.0, 1.0}, {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0},
      {tensor_dtype, {1}, -1.0, 1.0}, {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7});
  m.template SetQuantizedPadValue<integer_type>(-0.5);
  m.Invoke();
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {-0.5, -0.5, -0.5, -0.5, -0.5, -0.8, 0.2, -0.5, -0.5, 0.9,
                   0.7, -0.5, -0.5, -0.5, -0.5, -0.5},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8SimpleConstValuedTest) {
  SimpleConstValuedTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8SimpleConstValuedTest) {
  SimpleConstValuedTest<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void SimpleDynamicValuedTest() {
  PadV2OpDynamicModel<integer_type> m({tensor_dtype, {1, 2, 2, 1}, -1.0, 1.0},
                                      {4, 2}, {tensor_dtype, {1}, -1.0, 1.0},
                                      {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7});
  m.template SetQuantizedPadValue<integer_type>(-0.5);
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {-0.5, -0.5, -0.5, -0.5, -0.5, -0.8, 0.2, -0.5, -0.5, 0.9,
                   0.7, -0.5, -0.5, -0.5, -0.5, -0.5},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8SimpleDynamicValuedTest) {
  SimpleDynamicValuedTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8SimpleDynamicValuedTest) {
  SimpleDynamicValuedTest<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void AdvancedConstValuedTest() {
  PadV2OpConstModel<integer_type> m(
      {tensor_dtype, {1, 2, 3, 1}, -1.0, 1.0}, {4, 2}, {0, 0, 0, 2, 1, 3, 0, 0},
      {tensor_dtype, {1}, -1.0, 1.0}, {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7, 0.1, -0.3});
  m.template SetQuantizedPadValue<integer_type>(-0.5);
  m.Invoke();
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {-0.5, -0.8, 0.2,  0.9,  -0.5, -0.5, -0.5, -0.5, 0.7,  0.1,
                   -0.3, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                   -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8AdvancedConstValuedTest) {
  AdvancedConstValuedTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8AdvancedConstValuedTest) {
  AdvancedConstValuedTest<int8_t, TensorType_INT8>();
}

template <typename integer_type, TensorType tensor_dtype>
void AdvancedDynamicValuedTest() {
  PadV2OpDynamicModel<integer_type> m({tensor_dtype, {1, 2, 3, 1}, -1.0, 1.0},
                                      {4, 2}, {tensor_dtype, {1}, -1.0, 1.0},
                                      {tensor_dtype, {}, -1.0, 1.0});
  m.template SetQuantizedInput<integer_type>({-0.8, 0.2, 0.9, 0.7, 0.1, -0.3});
  m.template SetQuantizedPadValue<integer_type>(-0.5);
  m.SetPaddings({0, 0, 0, 2, 1, 3, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.template GetDequantizedOutput<integer_type>(),
              ElementsAreArray(DequantizedArrayNear(
                  {-0.5, -0.8, 0.2,  0.9,  -0.5, -0.5, -0.5, -0.5, 0.7,  0.1,
                   -0.3, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                   -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5},
                  -1.0, 1.0)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST_F(QuantizedPadV2OpTest, UInt8AdvancedDynamicValuedTest) {
  AdvancedDynamicValuedTest<uint8_t, TensorType_UINT8>();
}
TEST_F(QuantizedPadV2OpTest, Int8AdvancedDynamicValuedTest) {
  AdvancedDynamicValuedTest<int8_t, TensorType_INT8>();
}

}  // namespace
}  // namespace tflite
