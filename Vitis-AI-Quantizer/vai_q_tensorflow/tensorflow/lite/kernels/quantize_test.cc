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
#include <cstdint>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class QuantizeOpModel : public SingleOpModel {
 public:
  QuantizeOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_QUANTIZE, BuiltinOptions_QuantizeOptions,
                 CreateQuantizeOptions(builder_).Union());

    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  template <typename T>
  void SetInputAndQuantize(std::initializer_list<float> data) {
    QuantizeAndPopulate<T>(input_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 private:
  int input_;
  int output_;
};

TEST(QuantizeOpTest, UINT8) {
  // [-63.5, 64] -> scale=0.5 zero_point=127 for UINT8
  QuantizeOpModel m({TensorType_FLOAT32, {2, 5}},
                    {TensorType_UINT8, {2, 5}, 0, 0, 0.5, 127});

  m.SetInput({-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<uint8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 251, 252, 253, 254, 255}));
}

TEST(QuantizeOpTest, INT8) {
  // [-63.5, 64] -> scale=0.5, zero_point=1 for INT8
  QuantizeOpModel m({TensorType_FLOAT32, {2, 5}},
                    {TensorType_INT8, {2, 5}, 0, 0, 0.5, -1});

  m.SetInput({-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(
                  {-128, -127, -126, -125, -124, 123, 124, 125, 126, 127}));
}

TEST(QuantizeOpTest, INT16) {
  QuantizeOpModel m({TensorType_FLOAT32, {2, 5}},
                    {TensorType_INT16, {2, 5}, 0, 0, 0.005, 0});

  m.SetInput({-63.5, -63, -3, -2, -1, 1, 2, 3, 63.5, 64});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int16_t>(),
              ElementsAreArray({-12700, -12600, -600, -400, -200, 200, 400, 600,
                                12700, 12800}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint -1, output
// zeropoint -1
TEST(QuantizeOpTest, Int8Int8SameScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -63.5, 64},
                    {TensorType_INT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {1,3,5,7,9,11,13,15,17,19}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Input scale 0.500000, output scale 1.000000, input zeropoint -1, output
// zeropoint -1
TEST(QuantizeOpTest, Int8Int8LargerScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -63.5, 64},
                    {TensorType_INT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {1,3,5,7,9,11,13,15,17,19}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Input scale 1.000000, output scale 0.500000, input zeropoint -1, output
// zeropoint -1
TEST(QuantizeOpTest, Int8Int8SmallerScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_INT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Int8Int8SmallerScaleNeonPath) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_INT8, {1, 1, 4, 5}, -63.5, 64});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,0}.
  m.SetInputAndQuantize<int8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1,  3,  5,  7,  9,  11, 13, 15, 17, 19,
                                19, 17, 15, 13, 11, 9,  7,  5,  3,  1}));
}

// Input scale 0.500000, output scale 0.500000, input zeropoint 127, output
// zeropoint 127
TEST(QuantizeOpTest, UInt8UInt8SameScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64},
                    {TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {129,131,133,135,137,139,141,143,145,147}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147}));
}

// Input scale 0.500000, output scale 1.000000, input zeropoint 127, output
// zeropoint 127
TEST(QuantizeOpTest, Uint8Uint8LargerScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64},
                    {TensorType_UINT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {129,131,133,135,137,139,141,143,145,147}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 129, 130, 131, 132, 133, 134, 135, 136, 137}));
}

// Input scale 1.000000, output scale 0.500000, input zeropoint 127, output
// zeropoint 127
TEST(QuantizeOpTest, Uint8Uint8SmallerScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {128, 129, 130, 131, 132, 133, 134, 135, 136, 137}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Uint8Uint8SmallerScaleNeonPath) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 4, 5}, -63.5, 64});

  // Input will quantized to {128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
  // 137, 136, 135, 134, 133, 132, 131, 130, 129, 128}.
  m.SetInputAndQuantize<uint8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147,
                        147, 145, 143, 141, 139, 137, 135, 133, 131, 129}));
}

// Input scale 1.000000, output scale 1.000000, input zeropoint -1, output
// zeropoint 127
TEST(QuantizeOpTest, Int8Uint8SameScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 129, 130, 131, 132, 133, 134, 135, 136, 137}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Int8UInt8SameScaleNeonPath) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 4, 5}, -127, 128});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,0}.
  m.SetInputAndQuantize<int8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
                        137, 136, 135, 134, 133, 132, 131, 130, 129, 128}));
}

//  Input scale 1.000000, output scale 0.500000, input zeropoint -1, output
//  zeropoint 127
TEST(QuantizeOpTest, Int8Uint8SmallerScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Int8Uint8SmallerScaleNeonPath) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 4, 5}, -63.5, 64});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,0}.
  m.SetInputAndQuantize<int8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({129, 131, 133, 135, 137, 139, 141, 143, 145, 147,
                        147, 145, 143, 141, 139, 137, 135, 133, 131, 129}));
}

//  Input scale 1.000000, output scale 2.000000, input zeropoint -1, output
//  zeropoint 127
TEST(QuantizeOpTest, Int8Uint8LargerScale) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 2, 5}, -254, 256});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9}.
  m.SetInputAndQuantize<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 128, 129, 129, 130, 130, 131, 131, 132, 132}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Int8Uint8LargerScaleNeonPath) {
  QuantizeOpModel m({TensorType_INT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_UINT8, {1, 1, 4, 5}, -254, 256});

  // Input will quantized to {0,1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,0}.
  m.SetInputAndQuantize<int8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput<uint8_t>(),
      ElementsAreArray({128, 128, 129, 129, 130, 130, 131, 131, 132, 132,
                        132, 132, 131, 131, 130, 130, 129, 129, 128, 128}));
}

// input scale 0.500000, output scale 0.500000, input zeropoint 127, output
// zeropoint -1
TEST(QuantizeOpTest, UInt8Int8SameScale128Diff) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, -127, 128},
                    {TensorType_INT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {128, 129, 130, 131, 132, 133, 134, 135, 136, 137}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, UInt8Int8SameScale128DiffNeonPath) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 4, 5}, -127, 128},
                    {TensorType_INT8, {1, 1, 4, 5}, -127, 128});

  // Input will quantized to {128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
  // 137, 136, 135, 134, 133, 132, 131, 130, 129, 128}.
  m.SetInputAndQuantize<uint8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                9, 8, 7, 6, 5, 4, 3, 2, 1, 0}));
}

// input scale 0.500000, output scale 0.500000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Uint8Int8SameScaleArbitraryDiff) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, 0, 127.5},
                    {TensorType_INT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {2,4,6,8,10,12,14,16,18,20}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Uint8Int8SameScaleArbitraryDiffNeonPath) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 4, 5}, 0, 127.5},
                    {TensorType_INT8, {1, 1, 4, 5}, -63.5, 64});

  // Input will quantized to
  // {2,4,6,8,10,12,14,16,18,20,20,18,16,14,12,10,8,6,4,2}.
  m.SetInputAndQuantize<uint8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1,  3,  5,  7,  9,  11, 13, 15, 17, 19,
                                19, 17, 15, 13, 11, 9,  7,  5,  3,  1}));
}

// input scale 0.500000, output scale 1.000000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Uint8Int8LargerScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, 0, 127.5},
                    {TensorType_INT8, {1, 1, 2, 5}, -127, 128});

  // Input will quantized to {2,4,6,8,10,12,14,16,18,20}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// Same as previous test, except more data to hit the neon path.
TEST(QuantizeOpTest, Uint8Int8LargerScaleNeonPath) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 4, 5}, 0, 127.5},
                    {TensorType_INT8, {1, 1, 4, 5}, -127, 128});

  // Input will quantized to
  // {2,4,6,8,10,12,14,16,18,20,20,18,16,14,12,10,8,6,4,2}.
  m.SetInputAndQuantize<uint8_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                9, 8, 7, 6, 5, 4, 3, 2, 1, 0}));
}

// input scale 1.000000, output scale 0.500000, input zeropoint 0, output
// zeropoint -1
TEST(QuantizeOpTest, Uint8Int8SmallerScale) {
  QuantizeOpModel m({TensorType_UINT8, {1, 1, 2, 5}, 0, 255},
                    {TensorType_INT8, {1, 1, 2, 5}, -63.5, 64});

  // Input will quantized to {1,2,3,4,5,6,7,8,9,10}.
  m.SetInputAndQuantize<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray({1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}

}  // namespace
}  // namespace tflite
