/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"  // NOLINT
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

#if GOOGLE_CUDA
// These tests are copied from non_max_suppression_op_test.cc file and modified
// to use GPU ops. See other file for test details.

class NonMaxSuppressionV2GPUOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));

    TF_EXPECT_OK(
        NodeDefBuilder("non_max_suppression_op_gpu", "NonMaxSuppressionV2")
            .Input(FakeInput(DT_FLOAT))
            .Input(FakeInput(DT_FLOAT))
            .Input(FakeInput(DT_INT32))
            .Input(FakeInput(DT_FLOAT))
            .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(NonMaxSuppressionV2GPUOpTest, TestSelectFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2GPUOpTest,
       TestSelectFromThreeClustersFlippedCoordinates) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({6, 4}),
                           {1, 1,     0, 0,      // score= 0.9
                            0, 0.1f,  1, 1.1f,   // score= 0.75
                            0, .9f,   1, -0.1f,  // score= 0.6
                            0, 10,    1, 11,     // score= 0.95
                            1, 10.1f, 0, 11.1f,  // score=  0.5
                            1, 101,   0, 100});  // score=0.3

  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2GPUOpTest,
       TestSelectAtMostTwoBoxesFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {2});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected, {3, 0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2GPUOpTest,
       TestSelectAtMostThirtyBoxesFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2GPUOpTest, TestSelectSingleBox) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2GPUOpTest, TestSelectFromTenIdenticalBoxes) {
  MakeOp();

  int num_boxes = 10;
  std::vector<float> corners(num_boxes * 4);
  std::vector<float> scores(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    corners[i * 4 + 0] = 0;
    corners[i * 4 + 1] = 0;
    corners[i * 4 + 2] = 1;
    corners[i * 4 + 3] = 1;
    scores[i] = .9;
  }
  AddInputFromArray<float>(TensorShape({num_boxes, 4}), corners);
  AddInputFromArray<float>(TensorShape({num_boxes}), scores);
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV2GPUOpTest, TestInconsistentBoxAndScoreShapes) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({5}), {.9f, .75f, .6f, .95f, .5f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      str_util::StrContains(s.ToString(), "scores has incompatible shape"))
      << s;
}

TEST_F(NonMaxSuppressionV2GPUOpTest, TestInvalidIOUThreshold) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {1.2f});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      str_util::StrContains(s.ToString(), "iou_threshold must be in [0, 1]"))
      << s;
}

TEST_F(NonMaxSuppressionV2GPUOpTest, TestEmptyInput) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({0, 4}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({0}));
  test::FillValues<int>(&expected, {});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

//
// NonMaxSuppressionV3GPUOp Tests
// Copied from CPU tests

class NonMaxSuppressionV3GPUOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    SetDevice(DEVICE_GPU,
              std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                  "GPU", {}, "/job:a/replica:0/task:0")));

    TF_EXPECT_OK(NodeDefBuilder("non_max_suppression_op", "NonMaxSuppressionV3")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
  }
};

TEST_F(NonMaxSuppressionV3GPUOpTest, TestSelectFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest,
       TestSelectFromThreeClustersWithScoreThreshold) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {0.5f});
  AddInputFromArray<float>(TensorShape({}), {0.4f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected, {3, 0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest,
       TestSelectFromThreeClustersWithScoreThresholdZeroScores) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.1, 0, 0, .3, .2, -5.0});
  // If we ask for more boxes than we actually expect to get back;
  // should still only get 2 boxes back.
  AddInputFromArray<int>(TensorShape({}), {6});
  AddInputFromArray<float>(TensorShape({}), {0.5f});
  AddInputFromArray<float>(TensorShape({}), {-3.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected, {3, 0});

  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest,
       TestSelectFromThreeClustersFlippedCoordinates) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({6, 4}),
                           {1, 1,  0, 0,  0, 0.1f,  1, 1.1f,  0, .9f, 1, -0.1f,
                            0, 10, 1, 11, 1, 10.1f, 0, 11.1f, 1, 101, 0, 100});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest,
       TestSelectAtMostTwoBoxesFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {2});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({2}));
  test::FillValues<int>(&expected, {3, 0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest,
       TestSelectAtMostThirtyBoxesFromThreeClusters) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({6}), {.9f, .75f, .6f, .95f, .5f, .3f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({3}));
  test::FillValues<int>(&expected, {3, 0, 5});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest, TestSelectSingleBox) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest, TestSelectFromTenIdenticalBoxes) {
  MakeOp();

  int num_boxes = 10;
  std::vector<float> corners(num_boxes * 4);
  std::vector<float> scores(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    corners[i * 4 + 0] = 0;
    corners[i * 4 + 1] = 0;
    corners[i * 4 + 2] = 1;
    corners[i * 4 + 3] = 1;
    scores[i] = .9;
  }
  AddInputFromArray<float>(TensorShape({num_boxes, 4}), corners);
  AddInputFromArray<float>(TensorShape({num_boxes}), scores);
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int>(&expected, {0});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

TEST_F(NonMaxSuppressionV3GPUOpTest, TestInconsistentBoxAndScoreShapes) {
  MakeOp();
  AddInputFromArray<float>(
      TensorShape({6, 4}),
      {0, 0,  1, 1,  0, 0.1f,  1, 1.1f,  0, -0.1f, 1, 0.9f,
       0, 10, 1, 11, 0, 10.1f, 1, 11.1f, 0, 100,   1, 101});
  AddInputFromArray<float>(TensorShape({5}), {.9f, .75f, .6f, .95f, .5f});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(), "scores has incompatible shape"))
      << s;
}

TEST_F(NonMaxSuppressionV3GPUOpTest, TestInvalidIOUThreshold) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({1, 4}), {0, 0, 1, 1});
  AddInputFromArray<float>(TensorShape({1}), {.9f});
  AddInputFromArray<int>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {1.2f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  Status s = RunOpKernel();

  ASSERT_FALSE(s.ok());
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "iou_threshold must be in [0, 1]"))
      << s;
}

TEST_F(NonMaxSuppressionV3GPUOpTest, TestEmptyInput) {
  MakeOp();
  AddInputFromArray<float>(TensorShape({0, 4}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<int>(TensorShape({}), {30});
  AddInputFromArray<float>(TensorShape({}), {.5f});
  AddInputFromArray<float>(TensorShape({}), {0.0f});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_INT32, TensorShape({0}));
  test::FillValues<int>(&expected, {});
  test::ExpectTensorEqual<int>(expected, *GetOutput(0));
}

#endif

}  // namespace tensorflow
