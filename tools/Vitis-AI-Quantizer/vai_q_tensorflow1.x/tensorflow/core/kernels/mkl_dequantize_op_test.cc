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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

class MklDequantizeOpTest : public OpsTestBase {};

static const uint8 dummy_tensor[] = {0, 0, 0, 0, 0, 0, 0, 0};
static const TensorShape dummy_shape({8});

TEST_F(MklDequantizeOpTest, small) {
  TF_ASSERT_OK(NodeDefBuilder("dequantize_op", "_MklDequantize")
                   .Input(FakeInput(DT_QUINT8))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_UINT8))  // MKL second tensor
                   .Input(FakeInput(DT_UINT8))  // MKL second tensor
                   .Input(FakeInput(DT_UINT8))  // MKL second tensor
                   .Attr("T", DataTypeToEnum<quint8>::v())
                   .Attr("mode", "SCALED")
                   .Attr("_kernel", "QuantizedMklOp")
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  AddInputFromArray<quint8>(TensorShape({1, 2, 2, 2}),
                            {0, 10, 50, 40, 25, 115, 190, 255});
  // min_range = 0
  AddInputFromArray<float>(TensorShape({1}), {0});
  // max_range = 200
  AddInputFromArray<float>(TensorShape({1}), {200.0f});
  AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
  AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
  AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1, 2, 2, 2}));
  test::FillValues<float>(&expected,
                          {0.0, 7.84, 39.21, 31.37, 19.6, 90.2, 149.0, 200});
  const Tensor& output = *GetOutput(0);
  test::ExpectTensorNear<float>(expected, output, 0.1);
}

}  // namespace tensorflow
