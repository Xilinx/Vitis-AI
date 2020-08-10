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

#include <initializer_list>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class RankOpModel : public SingleOpModel {
 public:
  RankOpModel(std::initializer_list<int> input_shape, TensorType input_type) {
    TensorType output_type = TensorType_INT32;
    input_ = AddInput(input_type);
    output_ = AddOutput(output_type);
    SetBuiltinOp(BuiltinOperator_RANK, BuiltinOptions_RankOptions,
                 CreateRankOptions(builder_).Union());
    BuildInterpreter({input_shape});
  }

  TfLiteStatus InvokeWithResult() { return interpreter_->Invoke(); }

  int input() { return input_; }

  std::vector<int32_t> GetOutput() { return ExtractVector<int32_t>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

TEST(RankOpTest, InputTypeFloat) {
  RankOpModel model({1, 3, 1, 3, 5}, TensorType_FLOAT32);
  model.Invoke();

  EXPECT_THAT(model.GetOutput(), ElementsAreArray({5}));
  EXPECT_TRUE(model.GetOutputShape().empty());
}

TEST(RankOpTest, InputTypeInt) {
  RankOpModel model({1, 3, 1, 3, 5}, TensorType_INT32);
  model.Invoke();

  EXPECT_THAT(model.GetOutput(), ElementsAreArray({5}));
  EXPECT_TRUE(model.GetOutputShape().empty());
}

TEST(RankOpTest, ScalarTensor) {
  RankOpModel model({}, TensorType_FLOAT32);
  model.Invoke();

  EXPECT_THAT(model.GetOutput(), ElementsAreArray({0}));
  EXPECT_TRUE(model.GetOutputShape().empty());
}

TEST(RankOpTest, EmptyTensor) {
  RankOpModel model({1, 0}, TensorType_FLOAT32);
  model.Invoke();

  EXPECT_THAT(model.GetOutput(), ElementsAreArray({2}));
  EXPECT_TRUE(model.GetOutputShape().empty());
}

}  // namespace
}  // namespace tflite
