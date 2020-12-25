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
#include "tensorflow/core/kernels/data/text_line_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "text_line_dataset";
constexpr char kIteratorPrefix[] = "Iterator";

class TextLineDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Create a new `TextLineDataset` op kernel.
  Status CreateTextLineDatasetOpKernel(
      std::unique_ptr<OpKernel>* text_line_dataset_op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(TextLineDatasetOp::kDatasetType),
        {TextLineDatasetOp::kFileNames, TextLineDatasetOp::kCompressionType,
         TextLineDatasetOp::kBufferSize},
        {});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, text_line_dataset_op_kernel));
    return Status::OK();
  }

  // Create a new `TextLineDataset` op kernel context
  Status CreateTextLineDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

struct TestCase {
  std::vector<tstring> filenames;
  std::vector<string> texts;
  CompressionType compression_type;
  int64 buffer_size;
  std::vector<Tensor> expected_outputs;
  DataTypeVector expected_output_dtypes;
  std::vector<PartialTensorShape> expected_output_shapes;
  int64 expected_cardinality;
  std::vector<int> breakpoints;
};

Status CreateTestFiles(const TestCase& test_case) {
  if (test_case.filenames.size() != test_case.texts.size()) {
    return tensorflow::errors::InvalidArgument(
        "The number of files does not match with the contents");
  }
  if (test_case.compression_type == CompressionType::UNCOMPRESSED) {
    for (int i = 0; i < test_case.filenames.size(); ++i) {
      TF_RETURN_IF_ERROR(
          WriteDataToFile(test_case.filenames[i], test_case.texts[i].data()));
    }
  } else {
    CompressionParams params;
    params.compression_type = test_case.compression_type;
    params.input_buffer_size = test_case.buffer_size;
    params.output_buffer_size = test_case.buffer_size;
    for (int i = 0; i < test_case.filenames.size(); ++i) {
      TF_RETURN_IF_ERROR(WriteDataToFile(test_case.filenames[i],
                                         test_case.texts[i].data(), params));
    }
  }
  return Status::OK();
}

// Test case 1: multiple text files with ZLIB compression.
TestCase TestCase1() {
  return {/*filenames*/ {absl::StrCat(testing::TmpDir(), "/text_line_ZLIB_1"),
                         absl::StrCat(testing::TmpDir(), "/text_line_ZLIB_2")},
          /*texts*/
          {absl::StrCat("hello world\n", "11223334455\n"),
           absl::StrCat("abcd, EFgH\n", "           \n", "$%^&*()\n")},
          /*compression_type*/ CompressionType::ZLIB,
          /*buffer_size*/ 10,
          /*expected_outputs*/
          {CreateTensor<tstring>(TensorShape({}), {"hello world"}),
           CreateTensor<tstring>(TensorShape({}), {"11223334455"}),
           CreateTensor<tstring>(TensorShape({}), {"abcd, EFgH"}),
           CreateTensor<tstring>(TensorShape({}), {"           "}),
           CreateTensor<tstring>(TensorShape({}), {"$%^&*()"})},
          /*expected_output_dtypes*/ {DT_STRING},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0, 2, 6}};
}

// Test case 2: multiple text files with GZIP compression.
TestCase TestCase2() {
  return {/*filenames*/ {absl::StrCat(testing::TmpDir(), "/text_line_GZIP_1"),
                         absl::StrCat(testing::TmpDir(), "/text_line_GZIP_2")},
          /*texts*/
          {absl::StrCat("hello world\n", "11223334455\n"),
           absl::StrCat("abcd, EFgH\n", "           \n", "$%^&*()\n")},
          /*compression_type*/ CompressionType::GZIP,
          /*buffer_size*/ 10,
          /*expected_outputs*/
          {CreateTensor<tstring>(TensorShape({}), {"hello world"}),
           CreateTensor<tstring>(TensorShape({}), {"11223334455"}),
           CreateTensor<tstring>(TensorShape({}), {"abcd, EFgH"}),
           CreateTensor<tstring>(TensorShape({}), {"           "}),
           CreateTensor<tstring>(TensorShape({}), {"$%^&*()"})},
          /*expected_output_dtypes*/ {DT_STRING},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0, 2, 6}};
}

// Test case 3: multiple text files without compression.
TestCase TestCase3() {
  return {/*filenames*/ {
              absl::StrCat(testing::TmpDir(), "/text_line_UNCOMPRESSED_1"),
              absl::StrCat(testing::TmpDir(), "/text_line_UNCOMPRESSED_2")},
          /*texts*/
          {absl::StrCat("hello world\n", "11223334455\n"),
           absl::StrCat("abcd, EFgH\n", "           \n", "$%^&*()\n")},
          /*compression_type*/ CompressionType::UNCOMPRESSED,
          /*buffer_size*/ 10,
          /*expected_outputs*/
          {CreateTensor<tstring>(TensorShape({}), {"hello world"}),
           CreateTensor<tstring>(TensorShape({}), {"11223334455"}),
           CreateTensor<tstring>(TensorShape({}), {"abcd, EFgH"}),
           CreateTensor<tstring>(TensorShape({}), {"           "}),
           CreateTensor<tstring>(TensorShape({}), {"$%^&*()"})},
          /*expected_output_dtypes*/ {DT_STRING},
          /*expected_output_shapes*/ {PartialTensorShape({})},
          /*expected_cardinality*/ kUnknownCardinality,
          /*breakpoints*/ {0, 2, 6}};
}

class ParameterizedTextLineDatasetOpTest
    : public TextLineDatasetOpTest,
      public ::testing::WithParamInterface<TestCase> {};

TEST_P(ParameterizedTextLineDatasetOpTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(CreateTextLineDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<tstring>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<tstring>(
      TensorShape({}), {ToString(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateTextLineDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(text_line_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(text_line_dataset->MakeIterator(iterator_ctx.get(),
                                               kIteratorPrefix, &iterator));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order*/ true));
}

TEST_F(TextLineDatasetOpTest, DatasetNodeName) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(CreateTextLineDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<tstring>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<tstring>(
      TensorShape({}), {ToString(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateTextLineDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);
  EXPECT_EQ(text_line_dataset->node_name(), kNodeName);
}

TEST_F(TextLineDatasetOpTest, DatasetTypeString) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = TestCase1();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(CreateTextLineDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<tstring>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<tstring>(
      TensorShape({}), {ToString(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateTextLineDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);
  EXPECT_EQ(text_line_dataset->type_string(),
            name_utils::OpName(TextLineDatasetOp::kDatasetType));
}

TEST_P(ParameterizedTextLineDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(CreateTextLineDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<tstring>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<tstring>(
      TensorShape({}), {ToString(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateTextLineDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);
  TF_EXPECT_OK(VerifyTypesMatch(text_line_dataset->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedTextLineDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(CreateTextLineDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<tstring>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<tstring>(
      TensorShape({}), {ToString(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateTextLineDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);
  TF_EXPECT_OK(VerifyShapesCompatible(text_line_dataset->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedTextLineDatasetOpTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(CreateTextLineDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<tstring>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<tstring>(
      TensorShape({}), {ToString(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateTextLineDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);
  EXPECT_EQ(text_line_dataset->Cardinality(), test_case.expected_cardinality);
}

TEST_P(ParameterizedTextLineDatasetOpTest, IteratorOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(CreateTextLineDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<tstring>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<tstring>(
      TensorShape({}), {ToString(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateTextLineDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(text_line_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(text_line_dataset->MakeIterator(iterator_ctx.get(),
                                               kIteratorPrefix, &iterator));

  TF_EXPECT_OK(VerifyTypesMatch(iterator->output_dtypes(),
                                test_case.expected_output_dtypes));
}

TEST_P(ParameterizedTextLineDatasetOpTest, IteratorOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(CreateTextLineDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<tstring>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<tstring>(
      TensorShape({}), {ToString(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateTextLineDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(text_line_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(text_line_dataset->MakeIterator(iterator_ctx.get(),
                                               kIteratorPrefix, &iterator));

  TF_EXPECT_OK(VerifyShapesCompatible(iterator->output_shapes(),
                                      test_case.expected_output_shapes));
}

TEST_P(ParameterizedTextLineDatasetOpTest, IteratorOutputPrefix) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(CreateTextLineDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<tstring>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<tstring>(
      TensorShape({}), {ToString(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateTextLineDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(text_line_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(text_line_dataset->MakeIterator(iterator_ctx.get(),
                                               kIteratorPrefix, &iterator));

  EXPECT_EQ(iterator->prefix(),
            name_utils::IteratorPrefix(TextLineDatasetOp::kDatasetType,
                                       kIteratorPrefix));
}

TEST_P(ParameterizedTextLineDatasetOpTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  TestCase test_case = GetParam();
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(CreateTestFiles(test_case));

  std::unique_ptr<OpKernel> text_line_dataset_kernel;
  TF_ASSERT_OK(CreateTextLineDatasetOpKernel(&text_line_dataset_kernel));

  int64 num_files = test_case.filenames.size();
  Tensor filenames =
      CreateTensor<tstring>(TensorShape({num_files}), test_case.filenames);
  Tensor compression_type = CreateTensor<tstring>(
      TensorShape({}), {ToString(test_case.compression_type)});
  Tensor buffer_size =
      CreateTensor<int64>(TensorShape({}), {test_case.buffer_size});
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&filenames),
                                            TensorValue(&compression_type),
                                            TensorValue(&buffer_size)};
  std::unique_ptr<OpKernelContext> text_line_dataset_context;
  TF_ASSERT_OK(CreateTextLineDatasetContext(
      text_line_dataset_kernel.get(), &inputs, &text_line_dataset_context));

  DatasetBase* text_line_dataset;
  TF_ASSERT_OK(CreateDataset(text_line_dataset_kernel.get(),
                             text_line_dataset_context.get(),
                             &text_line_dataset));
  core::ScopedUnref scoped_unref(text_line_dataset);

  std::unique_ptr<IteratorContext> iterator_ctx;
  TF_ASSERT_OK(
      CreateIteratorContext(text_line_dataset_context.get(), &iterator_ctx));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(text_line_dataset->MakeIterator(iterator_ctx.get(),
                                               kIteratorPrefix, &iterator));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  const std::vector<int>& breakpoints = test_case.breakpoints;
  for (int breakpoint : breakpoints) {
    VariantTensorData data;
    VariantTensorDataWriter writer(&data);
    TF_EXPECT_OK(iterator->Save(serialization_ctx.get(), &writer));
    TF_EXPECT_OK(writer.Flush());
    VariantTensorDataReader reader(&data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx.get(), &reader, kIteratorPrefix,
                                 *text_line_dataset, &iterator));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_EXPECT_OK(
          iterator->GetNext(iterator_ctx.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order*/ true));
}

INSTANTIATE_TEST_SUITE_P(TextLineDatasetOpTest,
                         ParameterizedTextLineDatasetOpTest,
                         ::testing::ValuesIn(std::vector<TestCase>(
                             {TestCase1(), TestCase2(), TestCase3()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
