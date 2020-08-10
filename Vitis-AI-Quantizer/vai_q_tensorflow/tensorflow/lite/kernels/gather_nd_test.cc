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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class GatherNdOpModel : public SingleOpModel {
 public:
  GatherNdOpModel(const TensorData& params, const TensorData& indices) {
    params_ = AddInput(params);
    indices_ = AddInput(indices);
    output_ = AddOutput(params.type);
    SetBuiltinOp(BuiltinOperator_GATHER_ND, BuiltinOptions_GatherNdOptions,
                 CreateGatherNdOptions(builder_).Union());
    BuildInterpreter({GetShape(params_), GetShape(indices_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(params_, data);
  }

  template <typename T>
  void SetPositions(std::initializer_list<T> data) {
    PopulateTensor<T>(indices_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int params_;
  int indices_;
  int output_;
};

TEST(GatherNdOpTest, ElementIndexingIntoMatrix) {
  GatherNdOpModel m({TensorType_FLOAT32, {2, 2}}, {TensorType_INT32, {2, 2}});
  m.SetInput<float>({1.1, 1.2, 2.1, 2.2});
  m.SetPositions<int32_t>({0, 0, 1, 1});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({1.1, 2.2}));
}

TEST(GatherNdOpTest, SliceIndexingIntoMatrix) {
  GatherNdOpModel m({TensorType_FLOAT32, {2, 2}}, {TensorType_INT32, {2, 1}});
  m.SetInput<float>({1.1, 1.2, 2.1, 2.2});
  m.SetPositions<int32_t>({1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({2.1, 2.2, 1.1, 1.2}));
}

TEST(GatherNdOpTest, BatchedIndexingIntoMatrix1) {
  GatherNdOpModel m({TensorType_FLOAT32, {2, 2}},
                    {TensorType_INT32, {2, 1, 1}});
  m.SetInput<float>({1.1, 1.2, 2.1, 2.2});
  m.SetPositions<int32_t>({1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({2.1, 2.2, 1.1, 1.2}));
}

TEST(GatherNdOpTest, BatchedIndexingIntoMatrix2) {
  GatherNdOpModel m({TensorType_FLOAT32, {2, 2}},
                    {TensorType_INT32, {2, 1, 2}});
  m.SetInput<float>({1.1, 1.2, 2.1, 2.2});
  m.SetPositions<int32_t>({0, 0, 1, 1});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({1.1, 2.2}));
}

TEST(GatherNdOpTest, DuplicateIndexingIntoMatrix) {
  GatherNdOpModel m({TensorType_FLOAT32, {2, 2}}, {TensorType_INT32, {2, 2}});
  m.SetInput<float>({1.1, 1.2, 2.1, 2.2});
  m.SetPositions<int32_t>({0, 0, 0, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({1.1, 1.1}));
}

TEST(GatherNdOpTest, ElementIndexingIntoRank3Tensor) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {1, 2, 3}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({0, 0, 1, 1, 1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({-1.2, -4.1}));
}

TEST(GatherNdOpTest, SliceIndexingIntoRank3Tensor) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {2, 1}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({0, 2});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({1.1, -1.2, 1.3, -2.1, 2.2, 2.3, 5.1, -5.2, 5.3,
                                6.1, -6.2, 6.3}));
}

TEST(GatherNdOpTest, BatchedIndexingIntoRank3Tensor1) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {2, 1, 3}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({0, 0, 1, 1, 1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({-1.2, -4.1}));
}

TEST(GatherNdOpTest, BatchedIndexingIntoRank3Tensor2) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {2, 1, 1}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({3.1, 3.2, -3.3, -4.1, -4.2, 4.3, 1.1, -1.2, 1.3,
                                -2.1, 2.2, 2.3}));
}

TEST(GatherNdOpTest, BatchedIndexingIntoRank3Tensor3) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {2, 2, 2}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({0, 1, 1, 0, 0, 0, 2, 1});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({-2.1, 2.2, 2.3, 3.1, 3.2, -3.3, 1.1, -1.2, 1.3,
                                6.1, -6.2, 6.3}));
}

TEST(GatherNdOpTest, BatchedIndexingIntoRank3Tensor4) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {2, 2, 3}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({0, 0, 1, 1, 0, 1, 1, 1, 2, 2, 1, 2});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({-1.2, 3.2, 4.3, 6.3}));
}

TEST(GatherNdOpTest, DuplicateIndexingIntoRank3Tensor) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {2, 2}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({0, 1, 0, 1});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({-2.1, 2.2, 2.3, -2.1, 2.2, 2.3}));
}

TEST(GatherNdOpTest, Float32Int32) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT32, {2, 2}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int32_t>({0, 1, 1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({-2.1, 2.2, 2.3, 3.1, 3.2, -3.3}));
}

TEST(GatherNdOpTest, Float32Int64) {
  GatherNdOpModel m({TensorType_FLOAT32, {3, 2, 3}},
                    {TensorType_INT64, {2, 2}});
  m.SetInput<float>({1.1, -1.2, 1.3, -2.1, 2.2, 2.3,   //
                     3.1, 3.2, -3.3, -4.1, -4.2, 4.3,  //
                     5.1, -5.2, 5.3, 6.1, -6.2, 6.3});
  m.SetPositions<int64_t>({0LL, 1LL, 1LL, 0LL});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({-2.1, 2.2, 2.3, 3.1, 3.2, -3.3}));
}

TEST(GatherNdOpTest, Int32Int32) {
  GatherNdOpModel m({TensorType_INT32, {3, 2, 3}}, {TensorType_INT32, {2, 2}});
  m.SetInput<int32_t>({1, -1, 1, -2, 2, 2,   //
                       3, 3, -3, -4, -4, 4,  //
                       5, -5, 5, 6, -6, 6});
  m.SetPositions<int32_t>({0, 1, 1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({-2, 2, 2, 3, 3, -3}));
}

TEST(GatherNdOpTest, Int32Int64) {
  GatherNdOpModel m({TensorType_INT32, {3, 2, 3}}, {TensorType_INT64, {2, 2}});
  m.SetInput<int32_t>({1, -1, 1, -2, 2, 2,   //
                       3, 3, -3, -4, -4, 4,  //
                       5, -5, 5, 6, -6, 6});
  m.SetPositions<int64_t>({0LL, 1LL, 1LL, 0LL});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({-2, 2, 2, 3, 3, -3}));
}

TEST(GatherNdOpTest, Uint8Int32) {
  GatherNdOpModel m({TensorType_UINT8, {3, 2, 3}}, {TensorType_INT32, {2, 2}});
  m.SetInput<uint8_t>({1, 1, 1, 2, 2, 2,  //
                       3, 3, 3, 4, 4, 4,  //
                       5, 5, 5, 6, 6, 6});
  m.SetPositions<int32_t>({0, 1, 1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({2, 2, 2, 3, 3, 3}));
}

TEST(GatherNdOpTest, Uint8Int64) {
  GatherNdOpModel m({TensorType_UINT8, {3, 2, 3}}, {TensorType_INT64, {2, 2}});
  m.SetInput<uint8_t>({1, 1, 1, 2, 2, 2,  //
                       3, 3, 3, 4, 4, 4,  //
                       5, 5, 5, 6, 6, 6});
  m.SetPositions<int64_t>({0, 1, 1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({2, 2, 2, 3, 3, 3}));
}

TEST(GatherNdOpTest, Int8Int32) {
  GatherNdOpModel m({TensorType_INT8, {3, 2, 3}}, {TensorType_INT32, {2, 2}});
  m.SetInput<int8_t>({1, -1, 1, -2, 2, 2,   //
                      3, 3, -3, -4, -4, 4,  //
                      5, -5, 5, 6, -6, 6});
  m.SetPositions<int32_t>({0, 1, 1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({-2, 2, 2, 3, 3, -3}));
}

TEST(GatherNdOpTest, Int8Int64) {
  GatherNdOpModel m({TensorType_INT8, {3, 2, 3}}, {TensorType_INT64, {2, 2}});
  m.SetInput<int8_t>({1, -1, 1, -2, 2, 2,   //
                      3, 3, -3, -4, -4, 4,  //
                      5, -5, 5, 6, -6, 6});
  m.SetPositions<int64_t>({0LL, 1LL, 1LL, 0LL});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({-2, 2, 2, 3, 3, -3}));
}

TEST(GatherNdOpTest, Int64Int32) {
  GatherNdOpModel m({TensorType_INT64, {3, 2, 3}}, {TensorType_INT32, {2, 2}});
  m.SetInput<int64_t>({1LL, -1LL, 1LL, -2LL, 2LL, 2LL,   //
                       3LL, 3LL, -3LL, -4LL, -4LL, 4LL,  //
                       5LL, -5LL, 5LL, 6LL, -6LL, 6LL});
  m.SetPositions<int32_t>({0, 1, 1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<int64_t>(),
              ElementsAreArray({-2LL, 2LL, 2LL, 3LL, 3LL, -3LL}));
}

TEST(GatherNdOpTest, Int64Int64) {
  GatherNdOpModel m({TensorType_INT64, {3, 2, 3}}, {TensorType_INT64, {2, 2}});
  m.SetInput<int64_t>({1LL, -1LL, 1LL, -2LL, 2LL, 2LL,   //
                       3LL, 3LL, -3LL, -4LL, -4LL, 4LL,  //
                       5LL, -5LL, 5LL, 6LL, -6LL, 6LL});
  m.SetPositions<int64_t>({0LL, 1LL, 1LL, 0LL});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<int64_t>(),
              ElementsAreArray({-2LL, 2LL, 2LL, 3LL, 3LL, -3LL}));
}

}  // namespace
}  // namespace tflite
