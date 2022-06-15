/*Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/contrib/decent_q/utils/separate_shared_constants.h"
#include "tensorflow/contrib/decent_q/utils/transform_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace decent_q {

class SeparateSharedConstantsTest : public ::testing::Test {
 protected:
  void TestSeparateSharedConstants() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));
    Output input_op_1 =
        Const(root.WithOpName("input_op_1"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");
    Output conv_op_1 = Conv2D(root.WithOpName("conv_op_1"), input_op_1,
                              weights_op, {1, 1, 1, 1}, "VALID");

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    // get original output
    GraphDef out_graph_def;
    TF_ASSERT_OK(SeparateSharedConstants(original_graph_def, &out_graph_def));
    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"conv_op"}, {}, &original_outputs));

    std::unique_ptr<Session> transformed_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(transformed_session->Create(out_graph_def));
    std::vector<Tensor> transformed_outputs;
    TF_ASSERT_OK(
        transformed_session->Run({}, {"conv_op_1"}, {}, &transformed_outputs));

    test::ExpectTensorNear<float>(original_outputs[0], transformed_outputs[0],
                                  1e-5);

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(out_graph_def, &node_lookup);
    EXPECT_EQ(6, node_lookup.size());
    EXPECT_EQ(1, node_lookup.count("input_op"));
    EXPECT_EQ(1, node_lookup.count("input_op_1"));
    EXPECT_EQ(1, node_lookup.count("weights_op"));
    EXPECT_EQ(1, node_lookup.count("weights_op1"));
    EXPECT_EQ(1, node_lookup.count("conv_op"));
    EXPECT_EQ(1, node_lookup.count("conv_op_1"));
  }

};  // namespace decent_q

TEST_F(SeparateSharedConstantsTest, TestSeparateSharedConstants) {
  TestSeparateSharedConstants();
}

}  // namespace decent_q
}  // namespace tensorflow
