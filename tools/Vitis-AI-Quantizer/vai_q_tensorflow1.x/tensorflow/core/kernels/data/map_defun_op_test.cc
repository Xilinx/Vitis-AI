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
#include "tensorflow/core/kernels/data/map_defun_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "map_defun";
constexpr char kOpName[] = "MapDefun";

class MapDefunOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `MapDefun` op kernel
  Status CreateMapDefunOpKernel(
      const DataTypeVector& t_arguments, const DataTypeVector& t_captured,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      const FunctionDefHelper::AttrValueWrapper& func,
      int max_intra_op_parallelism,
      std::unique_ptr<OpKernel>* map_defun_kernel) {
    std::vector<string> input_placeholders;
    input_placeholders.reserve(t_arguments.size() + t_captured.size());
    for (int i = 0; i < t_arguments.size(); ++i) {
      input_placeholders.emplace_back(
          strings::StrCat(MapDefunOp::kArguments, "_", i));
    }
    for (int i = 0; i < t_captured.size(); ++i) {
      input_placeholders.emplace_back(
          strings::StrCat(MapDefunOp::kCapturedInputs, "_", i));
    }

    NodeDef node_def = test::function::NDef(
        kNodeName, kOpName, input_placeholders,
        {{MapDefunOp::kTarguments, t_arguments},
         {MapDefunOp::kTcaptured, t_captured},
         {MapDefunOp::kOutputTypes, output_types},
         {MapDefunOp::kOutputShapes, output_shapes},
         {MapDefunOp::kFunc, func},
         {MapDefunOp::kMaxIntraOpParallelism, max_intra_op_parallelism}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, map_defun_kernel));
    return Status::OK();
  }

  // Creates a new `MapDefun` op kernel context.
  Status CreateMapDefunContext(OpKernel* const op_kernel,
                               gtl::InlinedVector<TensorValue, 4>* const inputs,
                               std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct TestCase {
  std::vector<Tensor> arguments;
  std::vector<Tensor> captured_inputs;
  DataTypeVector t_arguments;
  DataTypeVector t_captured;
  FunctionDefHelper::AttrValueWrapper func;
  std::vector<FunctionDef> func_lib;
  int max_intra_op_parallelism;
  DataTypeVector output_dtypes;
  std::vector<PartialTensorShape> output_shapes;
  std::vector<Tensor> expected_outputs;
};

// Test case 1: one input for the map function with no captured inputs.
TestCase TestCase1() {
  return {
      /*arguments*/ {
          CreateTensor<int64>(TensorShape({3, 2}), {0, 1, 2, 3, 4, 5})},
      /*captured_inputs*/ {},
      /*t_arguments*/ {DT_INT64},
      /*t_captured*/ {},
      /*func*/ {FunctionDefHelper::FunctionRef("XTimesTwo", {{"T", DT_INT64}})},
      /*func_lib*/ {test::function::XTimesTwo()},
      /*max_intra_op_parallelism*/ 2,
      /*output_dtypes*/ {DT_INT64},
      /*output_shapes*/ {PartialTensorShape({2})},
      /*expected_outputs*/
      {CreateTensor<int64>(TensorShape({3, 2}), {0, 2, 4, 6, 8, 10})}};
}

// Test case 2: two inputs for the map function with no captured inputs.
TestCase TestCase2() {
  return {
      /*arguments*/ {
          CreateTensor<int64>(TensorShape({3, 2}), {0, 1, 2, 3, 4, 5}),
          CreateTensor<int64>(TensorShape({3, 2}), {0, 10, 20, 30, 40, 50})},
      /*captured_inputs*/ {},
      /*t_arguments*/ {DT_INT64, DT_INT64},
      /*t_captured*/ {},
      /*func*/ {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
      /*func_lib*/ {test::function::XAddY()},
      /*max_intra_op_parallelism*/ 2,
      /*output_dtypes*/ {DT_INT64},
      /*output_shapes*/ {PartialTensorShape({2})},
      /*expected_outputs*/
      {CreateTensor<int64>(TensorShape({3, 2}), {0, 11, 22, 33, 44, 55})}};
}

// Test case 3: two inputs for the map function with one captured input.
TestCase TestCase3() {
  return {
      /*arguments*/ {
          CreateTensor<int64>(TensorShape({3, 2}), {0, 1, 2, 3, 4, 5})},
      /*captured_inputs*/
      {CreateTensor<int64>(TensorShape({2}), {10, 100})},
      /*t_arguments*/ {DT_INT64},
      /*t_captured*/ {DT_INT64},
      /*func*/ {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
      /*func_lib*/ {test::function::XAddY()},
      /*max_intra_op_parallelism*/ 2,
      /*output_dtypes*/ {DT_INT64},
      /*output_shapes*/ {PartialTensorShape({2})},
      /*expected_outputs*/
      {CreateTensor<int64>(TensorShape({3, 2}), {10, 101, 12, 103, 14, 105})}};
}

TestCase InvalidOutputTypes() {
  return {
      /*arguments*/ {
          CreateTensor<int64>(TensorShape({3, 2}), {0, 1, 2, 3, 4, 5})},
      /*captured_inputs*/
      {CreateTensor<int64>(TensorShape({2}), {10, 100})},
      /*t_arguments*/ {DT_INT64},
      /*t_captured*/ {DT_INT64},
      /*func*/ {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
      /*func_lib*/ {test::function::XAddY()},
      /*max_intra_op_parallelism*/ 2,
      /*output_dtypes*/ {DT_FLOAT},
      /*output_shapes*/ {PartialTensorShape({2})},
      /*expected_outputs*/
      {CreateTensor<int64>(TensorShape({3, 2}), {10, 101, 12, 103, 14, 105})}};
}

TestCase InvalidOutputShapes() {
  return {
      /*arguments*/ {
          CreateTensor<int64>(TensorShape({3, 2}), {0, 1, 2, 3, 4, 5})},
      /*captured_inputs*/
      {CreateTensor<int64>(TensorShape({2}), {10, 100})},
      /*t_arguments*/ {DT_INT64},
      /*t_captured*/ {DT_INT64},
      /*func*/ {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
      /*func_lib*/ {test::function::XAddY()},
      /*max_intra_op_parallelism*/ 2,
      /*output_dtypes*/ {DT_INT64},
      /*output_shapes*/ {PartialTensorShape({2, 2})},
      /*expected_outputs*/
      {CreateTensor<int64>(TensorShape({3, 2}), {10, 101, 12, 103, 14, 105})}};
}

TestCase InvalidInputs() {
  return {
      /*arguments*/ {
          CreateTensor<int64>(TensorShape({3, 2}), {0, 1, 2, 3, 4, 5}),
          CreateTensor<int64>(TensorShape({2, 2}), {0, 1, 2, 3})},
      /*captured_inputs*/
      {CreateTensor<int64>(TensorShape({2}), {10, 100})},
      /*t_arguments*/ {DT_INT64, DT_INT64},
      /*t_captured*/ {DT_INT64},
      /*func*/ {FunctionDefHelper::FunctionRef("XAddY", {{"T", DT_INT64}})},
      /*func_lib*/ {test::function::XAddY()},
      /*max_intra_op_parallelism*/ 2,
      /*output_dtypes*/ {DT_INT64},
      /*output_shapes*/ {PartialTensorShape({2})},
      /*expected_outputs*/
      {CreateTensor<int64>(TensorShape({3, 2}), {10, 101, 12, 103, 14, 105})}};
}

class ParameterizedMapDefunOpTest
    : public MapDefunOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedMapDefunOpTest, NormalTests) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

  std::unique_ptr<OpKernel> map_defun_kernel;
  TF_ASSERT_OK(CreateMapDefunOpKernel(
      test_case.t_arguments, test_case.t_captured, test_case.output_dtypes,
      test_case.output_shapes, test_case.func,
      test_case.max_intra_op_parallelism, &map_defun_kernel));
  gtl::InlinedVector<TensorValue, 4> inputs;
  for (auto& arg : test_case.arguments) {
    inputs.emplace_back(&arg);
  }
  for (auto& captured_input : test_case.captured_inputs) {
    inputs.emplace_back(&captured_input);
  }
  std::unique_ptr<OpKernelContext> context;
  TF_ASSERT_OK(
      CreateMapDefunContext(map_defun_kernel.get(), &inputs, &context));
  TF_ASSERT_OK(RunOpKernel(map_defun_kernel.get(), context.get()));

  EXPECT_EQ(context->num_outputs(), test_case.expected_outputs.size());
  for (int i = 0; i < context->num_outputs(); ++i) {
    TF_EXPECT_OK(ExpectEqual(*context->mutable_output(i),
                             test_case.expected_outputs[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(MapDefunOpTest, ParameterizedMapDefunOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3()})));

TEST_F(MapDefunOpTest, InvalidArguments) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  std::vector<TestCase> test_cases = {InvalidOutputTypes(),
                                      InvalidOutputShapes(), InvalidInputs()};
  for (auto& test_case : test_cases) {
    TF_ASSERT_OK(InitFunctionLibraryRuntime(test_case.func_lib, cpu_num));

    std::unique_ptr<OpKernel> map_defun_kernel;
    TF_ASSERT_OK(CreateMapDefunOpKernel(
        test_case.t_arguments, test_case.t_captured, test_case.output_dtypes,
        test_case.output_shapes, test_case.func,
        test_case.max_intra_op_parallelism, &map_defun_kernel));
    gtl::InlinedVector<TensorValue, 4> inputs;
    for (auto& arg : test_case.arguments) {
      inputs.emplace_back(&arg);
    }
    for (auto& captured_input : test_case.captured_inputs) {
      inputs.emplace_back(&captured_input);
    }
    std::unique_ptr<OpKernelContext> context;
    TF_ASSERT_OK(
        CreateMapDefunContext(map_defun_kernel.get(), &inputs, &context));
    EXPECT_EQ(RunOpKernel(map_defun_kernel.get(), context.get()).code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
