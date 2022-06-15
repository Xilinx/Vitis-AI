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
#include "tensorflow/contrib/decent_q/utils/fold_batch_norms.h"
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

class FoldBatchNormsTest : public ::testing::Test {
 protected:
  void TestUpdateOldBatchNorms() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    Tensor scale_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&scale_data, {1.5f, 2.5f});
    Tensor offset_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&offset_data, {3.5f, 4.5f});
    Tensor mean_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&mean_data, {5.5f, 6.5f});
    Tensor variance_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&variance_data, {7.5f, 8.5f});
    FusedBatchNorm::Attrs attr;
    attr = attr.IsTraining(false);
    auto with_fused_batch_norm = FusedBatchNorm(
        root.WithOpName("with_fused_batch_norm"), conv_op,
        Const(root.WithOpName("scale"), Input::Initializer(scale_data)),
        Const(root.WithOpName("offset"), Input::Initializer(offset_data)),
        Const(root.WithOpName("mean"), Input::Initializer(mean_data)),
        Const(root.WithOpName("var"), Input::Initializer(variance_data)), attr);

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    // get original output
    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"with_fused_batch_norm"}, {},
                                       &original_outputs));
    GraphDef out_graph_def;
    TF_ASSERT_OK(UpdateOldBatchNorms(original_graph_def, &out_graph_def));

    std::unique_ptr<Session> transformed_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(transformed_session->Create(out_graph_def));
    std::vector<Tensor> transformed_outputs;
    TF_ASSERT_OK(transformed_session->Run({}, {"with_fused_batch_norm/add"}, {},
                                          &transformed_outputs));

    test::ExpectTensorNear<float>(original_outputs[0], transformed_outputs[0],
                                  1e-5);

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(out_graph_def, &node_lookup);
    EXPECT_EQ(7, node_lookup.size());
    EXPECT_EQ(1, node_lookup.count("input_op"));
    EXPECT_EQ(1, node_lookup.count("weights_op"));
    EXPECT_EQ(1, node_lookup.count("conv_op"));
    EXPECT_EQ(1, node_lookup.count("with_fused_batch_norm/mul"));
    EXPECT_EQ(1, node_lookup.count("with_fused_batch_norm/scale"));
    EXPECT_EQ(1, node_lookup.count("with_fused_batch_norm/offset"));
    EXPECT_EQ(1, node_lookup.count("with_fused_batch_norm/add"));
  }

  void TestGetMergedConvWeights() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");
    Tensor scale_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&scale_data, {2.f, -1.f});
    Output scale =
        Const(root.WithOpName("scale"), Input::Initializer(scale_data));

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(original_graph_def, &node_lookup);

    NodeDef scaled_weights_node;
    GetMergedConvWeights(*(node_lookup.at("conv_op")),
                         *(node_lookup.at("weights_op")),
                         *(node_lookup.at("scale")), &scaled_weights_node);
    Tensor scaled_weights = GetNodeTensorAttr(scaled_weights_node, "value");

    Tensor expected(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&expected, {2., -2, 6, -4, 0.2, -0.2, 0.6, -0.4});
    test::ExpectTensorNear<float>(scaled_weights, expected, 1e-5);
  }

  void TestGetMergedConvBiases() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor bias_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&bias_data, {1.f, 2.f});
    Output bias_op =
        Const(root.WithOpName("bias_op"), Input::Initializer(bias_data));
    Tensor mul_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&mul_data, {3.f, 4.f});
    Output mul_op =
        Const(root.WithOpName("mul_op"), Input::Initializer(mul_data));
    Tensor add_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&add_data, {5.f, 6.f});
    Output add_op =
        Const(root.WithOpName("add_op"), Input::Initializer(add_data));

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(original_graph_def, &node_lookup);
    NodeDef scaled_bias_node;
    GetMergedConvBiases(*(node_lookup.at("bias_op")),
                        *(node_lookup.at("mul_op")),
                        *(node_lookup.at("add_op")), &scaled_bias_node);
    Tensor scaled_bias = GetNodeTensorAttr(scaled_bias_node, "value");

    auto scaled_bias_vector = scaled_bias.flat<float>();

    Tensor expected(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&expected, {8., 14.});
    test::ExpectTensorNear<float>(scaled_bias, expected, 1e-5);
  }

  void TestFoldBatchNormsTraining() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    auto weights_op =
        Variable(root.WithOpName("weights/kernel"), {1, 2, 2, 2}, DT_FLOAT);
    auto w_assigned = Assign(
        root.WithOpName("weights/assign"), weights_op,
        Const(root.WithOpName("weights/init"),
              {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f}, {1, 2, 2, 2}));
    auto w_read = Identity(root.WithOpName("weights/read"), weights_op);
    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, w_read,
                            {1, 1, 1, 1}, "VALID");

    auto scale_op = Variable(root.WithOpName("scale/kernel"), {2}, DT_FLOAT);
    auto scale_assigned =
        Assign(root.WithOpName("scale/assign"), scale_op,
               Const(root.WithOpName("scale/init"), {1.0f, 1.0f}, {2}));
    auto scale_read = Identity(root.WithOpName("scale/read"), scale_op);

    auto offset_op = Variable(root.WithOpName("offset/kernel"), {2}, DT_FLOAT);
    auto offset_assigned =
        Assign(root.WithOpName("offset/assign"), offset_op,
               Const(root.WithOpName("offset/init"), {0.0f, 0.0f}, {2}));
    auto offset_read = Identity(root.WithOpName("offset/read"), offset_op);

    auto moving_mean_op =
        Variable(root.WithOpName("moving_mean/kernel"), {2}, DT_FLOAT);
    auto moving_mean_assigned =
        Assign(root.WithOpName("moving_mean/assign"), moving_mean_op,
               Const(root.WithOpName("moving_mean/init"), {0.0f, 0.0f}, {2}));
    auto moving_mean_read =
        Identity(root.WithOpName("moving_mean/read"), moving_mean_op);

    auto moving_var_op =
        Variable(root.WithOpName("moving_var/kernel"), {2}, DT_FLOAT);
    auto moving_var_assigned =
        Assign(root.WithOpName("moving_var/assign"), moving_var_op,
               Const(root.WithOpName("moving_var/init"), {0.0f, 0.0f}, {2}));
    auto moving_var_read =
        Identity(root.WithOpName("moving_var/read"), moving_var_op);
    Tensor mean_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&mean_data, {0.f, 0.f});
    Tensor var_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&var_data, {1.f, 1.f});
    FusedBatchNorm::Attrs attr;
    attr = attr.IsTraining(true);
    auto fused_batch_norm =
        FusedBatchNorm(root.WithOpName("fused_batch_norm"), conv_op, scale_read,
                       offset_read, {}, {}, attr);
    auto assign_sub_mm = Sub(root.WithOpName("AssignMovingAvg/sub"),
                             moving_mean_read, fused_batch_norm.batch_mean);
    auto assign_sub_mv = Sub(root.WithOpName("AssignMovingAvg_1/sub"),
                             moving_var_read, fused_batch_norm.batch_variance);

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    GraphDef out_graph_def;
    TF_ASSERT_OK(FoldBatchNormsTraining(original_graph_def, &out_graph_def));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(out_graph_def, &node_lookup);

    EXPECT_EQ(1, node_lookup.count("conv_op/biasadd"));
    EXPECT_EQ("conv_op_Fold", node_lookup.at("conv_op/biasadd")->input(0));
    EXPECT_EQ("conv_op/BatchNorm_Fold/sub",
              node_lookup.at("conv_op/biasadd")->input(1));
    EXPECT_EQ(1, node_lookup.count("conv_op_Fold"));
    EXPECT_EQ("input_op", node_lookup.at("conv_op_Fold")->input(0));
    EXPECT_EQ("conv_op/BatchNorm_Fold/mul_1",
              node_lookup.at("conv_op_Fold")->input(1));
    EXPECT_EQ(1, node_lookup.count("input_op"));

    EXPECT_EQ(1, node_lookup.count("conv_op/BatchNorm_Fold/mul_1"));
    EXPECT_EQ("conv_op/BatchNorm_Fold/mul_0",
              node_lookup.at("conv_op/BatchNorm_Fold/mul_1")->input(0));
    EXPECT_EQ("weights/read",
              node_lookup.at("conv_op/BatchNorm_Fold/mul_1")->input(1));
    EXPECT_EQ(1, node_lookup.count("conv_op/BatchNorm_Fold/sub"));
    EXPECT_EQ("offset/read",
              node_lookup.at("conv_op/BatchNorm_Fold/sub")->input(0));
    EXPECT_EQ("conv_op/BatchNorm_Fold/mul_2",
              node_lookup.at("conv_op/BatchNorm_Fold/sub")->input(1));
    EXPECT_EQ(1, node_lookup.count("conv_op/BatchNorm_Fold/mul_2"));
    EXPECT_EQ("conv_op/BatchNorm_Fold/mul_0",
              node_lookup.at("conv_op/BatchNorm_Fold/mul_2")->input(0));
    EXPECT_EQ("moving_mean/read",
              node_lookup.at("conv_op/BatchNorm_Fold/mul_2")->input(1));
    EXPECT_EQ(1, node_lookup.count("conv_op/BatchNorm_Fold/mul_0"));
    EXPECT_EQ("conv_op/BatchNorm_Fold/rsqrt",
              node_lookup.at("conv_op/BatchNorm_Fold/mul_0")->input(0));
    EXPECT_EQ("scale/read",
              node_lookup.at("conv_op/BatchNorm_Fold/mul_0")->input(1));
    EXPECT_EQ(1, node_lookup.count("conv_op/BatchNorm_Fold/rsqrt"));
    EXPECT_EQ("conv_op/BatchNorm_Fold/add",
              node_lookup.at("conv_op/BatchNorm_Fold/rsqrt")->input(0));
    EXPECT_EQ(1, node_lookup.count("conv_op/BatchNorm_Fold/add"));
    EXPECT_EQ("conv_op/BatchNorm_Fold/eps",
              node_lookup.at("conv_op/BatchNorm_Fold/add")->input(0));
    EXPECT_EQ("moving_var/read",
              node_lookup.at("conv_op/BatchNorm_Fold/add")->input(1));
  }

  void TestFoldBatchNormsInference() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    Tensor scale_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&scale_data, {1.5f, 2.5f});
    Output scale_op =
        Const(root.WithOpName("scale"), Input::Initializer(scale_data));
    auto mul_op = Mul(root.WithOpName("mul"), conv_op, scale_op);
    Tensor bias_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&bias_data, {5.5f, 6.5f});
    Output add_op =
        Const(root.WithOpName("add_op"), Input::Initializer(bias_data));
    Output bias_add_op = BiasAdd(root.WithOpName("bias_add"), mul_op, add_op);

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    GraphDef out_graph_def;
    TF_ASSERT_OK(FoldBatchNormsInference(original_graph_def, &out_graph_def));

    // get original output;
    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(
        original_session->Run({}, {"bias_add"}, {}, &original_outputs));

    std::unique_ptr<Session> transformed_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(transformed_session->Create(out_graph_def));
    std::vector<Tensor> transformed_outputs;
    TF_ASSERT_OK(
        transformed_session->Run({}, {"bias_add"}, {}, &transformed_outputs));

    test::ExpectTensorNear<float>(original_outputs[0], transformed_outputs[0],
                                  1e-5);

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(out_graph_def, &node_lookup);
    EXPECT_EQ(5, node_lookup.size());
    EXPECT_EQ(1, node_lookup.count("input_op"));
    EXPECT_EQ(1, node_lookup.count("weights_op"));
    EXPECT_EQ(1, node_lookup.count("conv_op"));
    EXPECT_EQ(1, node_lookup.count("bias_add"));
    EXPECT_EQ(1, node_lookup.count("weights_op/biases"));
  }

  void TestFoldConvMulInference() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    Tensor bias_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&bias_data, {5.5f, 6.5f});
    Output bias_op =
        Const(root.WithOpName("bias_op"), Input::Initializer(bias_data));
    Output bias_add_op = BiasAdd(root.WithOpName("bias_add"), conv_op, bias_op);

    Tensor scale_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&scale_data, {1.5f, 2.5f});
    Output scale_op =
        Const(root.WithOpName("scale"), Input::Initializer(scale_data));
    auto mul_op = Mul(root.WithOpName("mul"), bias_add_op, scale_op);

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    GraphDef out_graph_def;
    TF_ASSERT_OK(FoldConvMulInference(original_graph_def, &out_graph_def));
    // get original output;
    std::unique_ptr<Session> original_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(original_session->Create(original_graph_def));
    std::vector<Tensor> original_outputs;
    TF_ASSERT_OK(original_session->Run({}, {"mul"}, {}, &original_outputs));

    std::unique_ptr<Session> transformed_session(NewSession(SessionOptions()));
    TF_ASSERT_OK(transformed_session->Create(out_graph_def));
    std::vector<Tensor> transformed_outputs;
    TF_ASSERT_OK(
        transformed_session->Run({}, {"bias_add"}, {}, &transformed_outputs));

    test::ExpectTensorNear<float>(original_outputs[0], transformed_outputs[0],
                                  1e-5);

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(out_graph_def, &node_lookup);
    EXPECT_EQ(5, node_lookup.size());
    EXPECT_EQ(1, node_lookup.count("input_op"));
    EXPECT_EQ(1, node_lookup.count("weights_op"));
    EXPECT_EQ(1, node_lookup.count("conv_op"));
    EXPECT_EQ(1, node_lookup.count("bias_add"));
    EXPECT_EQ(1, node_lookup.count("bias_op"));
  }

};  // namespace decent_q

TEST_F(FoldBatchNormsTest, TestUpdateOldBatchNorms) {
  TestUpdateOldBatchNorms();
}

TEST_F(FoldBatchNormsTest, TestGetMergedConvWeights) {
  TestGetMergedConvWeights();
}

TEST_F(FoldBatchNormsTest, TestGetMergedConvBiases) {
  TestGetMergedConvBiases();
}

TEST_F(FoldBatchNormsTest, TestFoldBatchNormsTraining) {
  TestFoldBatchNormsTraining();
}

TEST_F(FoldBatchNormsTest, TestFoldBatchNormsInference) {
  TestFoldBatchNormsInference();
}

TEST_F(FoldBatchNormsTest, TestFoldConvMulInference) {
  TestFoldConvMulInference();
}

}  // namespace decent_q
}  // namespace tensorflow
