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
#include <cstdint>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {

namespace ops {
namespace builtin {

TfLiteRegistration* Register_DEQUANTIZE();

}  // namespace builtin
}  // namespace ops

namespace {

using ::testing::ElementsAreArray;

class DequantizeOpModel : public SingleOpModel {
 public:
  DequantizeOpModel(TensorType type, std::initializer_list<int> shape,
                    float scale, int32_t zero_point, int version) {
    const TensorData input_tensor_data = {type, shape, 0, 0, scale, zero_point};
    input_ = AddInput(input_tensor_data);
    output_ = AddOutput({TensorType_FLOAT32, shape});
    SetBuiltinOp(BuiltinOperator_DEQUANTIZE, BuiltinOptions_DequantizeOptions,
                 CreateDequantizeOptions(builder_).Union());

    resolver_ = absl::make_unique<SingleOpResolver>(
        BuiltinOperator_DEQUANTIZE, ops::builtin::Register_DEQUANTIZE(),
        version);

    BuildInterpreter({GetShape(input_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

TEST(DequantizeOpTest, Uint8) {
  // [-63.5, 64] -> scale=0.5 zero_point=127 for UINT8
  DequantizeOpModel m(TensorType_UINT8, {2, 5}, 0.5, 127, 1);

  m.SetInput<uint8_t>({0, 1, 2, 3, 4, 251, 252, 253, 254, 255});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64})));
}

TEST(DequantizeOpTest, Int8) {
  // [-63.5, 64] -> scale=0.5, zero_point=1 for INT8
  DequantizeOpModel m(TensorType_INT8, {2, 5}, 0.5, -1, 2);

  m.SetInput<int8_t>({-128, -127, -126, -125, -124, 123, 124, 125, 126, 127});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64})));
}

TEST(DequantizeOpTest, Float16) {
  DequantizeOpModel m(TensorType_FLOAT16, {2, 3}, 1.0f, 0, 3);

  std::vector<Eigen::half> half{Eigen::half{-535.54f}, Eigen::half{-100.0f},
                                Eigen::half{-1.0f},    Eigen::half{0.f},
                                Eigen::half{1.0f},     Eigen::half{100.32f}};
  m.PopulateTensor(0, 0, reinterpret_cast<TfLiteFloat16*>(half.data()),
                   reinterpret_cast<TfLiteFloat16*>(half.data()) + half.size());
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(
                                 {-535.54f, -100.0f, -1.0f, 0.f, 1.0f, 100.32f},
                                 /*max_abs_error=*/0.1f)));
}

TEST(DequantizeOpTest, Int16) {
  DequantizeOpModel m(TensorType_INT16, {2, 5}, 0.5, -1, 4);
  m.SetInput<int16_t>({-130, -127, -126, -125, -124, 123, 124, 125, 126, 130});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {-64.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 65.5})));
}

}  // namespace
}  // namespace tflite
