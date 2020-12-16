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

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/contrib/decent_q/utils/deploy_quantized_graph.h"
#include "tensorflow/contrib/decent_q/utils/graph_quantizer.h"
#include "tensorflow/contrib/decent_q/utils/ops/fix_neuron_ops.h"
#include "tensorflow/contrib/decent_q/utils/transform_utils.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"

namespace tensorflow {
namespace decent_q {

class DeployQuantizedGraphTest : public ::testing::Test {
 protected:
  void TestFoldFixNeuron() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(&input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f,
                                          0.1f, 0.4f, 0.2f, 0.5f, 0.3f, 0.6f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    FixNeuron::Attrs attrs;
    attrs.bit_width_ = 8;
    attrs.method_ = 1;
    attrs.output_dir_ = testing::TmpDir();
    attrs.quantize_pos_ = 3;
    Output input_fix_op =
        FixNeuron(root.WithOpName("input_fix_op"), input_op, 1, 1, attrs);
    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));
    attrs.quantize_pos_ = 4;
    Output weights_fix_op =
        FixNeuron(root.WithOpName("weights_fix_op"), weights_op, 0, 1, attrs);

    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_fix_op,
                            weights_fix_op, {1, 1, 1, 1}, "VALID");
    Output relu_op = Relu(root.WithOpName("relu_op"), conv_op);
    attrs.quantize_pos_ = 5;
    Output relu_fix_op =
        FixNeuron(root.WithOpName("conv_fix_op"), relu_op, 1, 1, attrs);
    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    // get original output
    GraphDef out_graph_def;
    TF_ASSERT_OK(FoldFixNeuron(original_graph_def, &out_graph_def));
    for (const auto& node : out_graph_def.node()) {
      ASSERT_NE("FixNeuron", node.op());
    }
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(out_graph_def, &node_lookup);
    ASSERT_EQ(4, node_lookup.size());
    EXPECT_EQ(1, node_lookup.count("input_op"));
    EXPECT_EQ(1, node_lookup.count("weights_op"));
    EXPECT_EQ(1, node_lookup.count("conv_op"));
    EXPECT_EQ(1, node_lookup.count("relu_op"));

    EXPECT_EQ(8, node_lookup.at("input_op")->attr().at("opos").list().i(0));
    EXPECT_EQ(8, node_lookup.at("weights_op")->attr().at("wpos").list().i(0));
    EXPECT_EQ(8, node_lookup.at("relu_op")->attr().at("opos").list().i(0));

    EXPECT_EQ(3, node_lookup.at("input_op")->attr().at("opos").list().i(1));
    EXPECT_EQ(4, node_lookup.at("weights_op")->attr().at("wpos").list().i(1));
    EXPECT_EQ(5, node_lookup.at("relu_op")->attr().at("opos").list().i(1));
  }

  void TestPolishActivationInfo() {
    GraphDef original_graph_def;

    NodeDef* relu_node = original_graph_def.add_node();
    relu_node->set_name("relu_op");
    relu_node->set_op("Relu");
    relu_node->add_input("conv_op");
    SetNodeAttr("opos", std::vector<int>({8, 5}), relu_node);

    NodeDef* conv_node = original_graph_def.add_node();
    conv_node->set_name("conv_op");
    conv_node->set_op("Conv2D");
    conv_node->add_input("input_op");
    SetNodeAttr("opos", std::vector<int>({8, 5}), conv_node);

    NodeDef* input_node = original_graph_def.add_node();
    input_node->set_name("input_op");
    input_node->set_op("Const");
    SetNodeAttr("opos", std::vector<int>({8, 3}), input_node);

    GraphDef out_graph_def, sorted_graph_def;
    SortByExecutionOrder(original_graph_def, &sorted_graph_def);
    TF_ASSERT_OK(PolishActivationInfo(sorted_graph_def, &out_graph_def));
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(out_graph_def, &node_lookup);

    EXPECT_EQ(8, node_lookup.at("conv_op")->attr().at("ipos").list().i(0));
    EXPECT_EQ(3, node_lookup.at("conv_op")->attr().at("ipos").list().i(1));

    EXPECT_EQ(8, node_lookup.at("relu_op")->attr().at("ipos").list().i(0));
    EXPECT_EQ(5, node_lookup.at("relu_op")->attr().at("ipos").list().i(1));
  }

  void TestFoldOpParams() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(&input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f,
                                          0.1f, 0.4f, 0.2f, 0.5f, 0.3f, 0.6f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));
    Output input_op_1 =
        Const(root.WithOpName("input_op_1"), Input::Initializer(input_data));

    Tensor axis_data(DT_INT32, TensorShape({}));
    test::FillValues<int>(&axis_data, {1});
    Output axis_op =
        Const(root.WithOpName("axis_op"), Input::Initializer(axis_data));

    Output concat_op =
        Concat(root.WithOpName("concat_op"), {input_op, input_op_1}, axis_op);
    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    GraphDef out_graph_def;
    TF_ASSERT_OK(FoldOpParams(original_graph_def, &out_graph_def));
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(out_graph_def, &node_lookup);
    ASSERT_EQ(3, node_lookup.size());
    EXPECT_EQ(1, node_lookup.count("input_op"));
    EXPECT_EQ(1, node_lookup.count("input_op_1"));
    EXPECT_EQ(1, node_lookup.count("concat_op"));
    EXPECT_EQ(1, node_lookup.at("concat_op")->attr().count("axis"));
  }

  void TestConvertPatterns() {
    GraphDef original_graph_def;

    NodeDef* relu_node = original_graph_def.add_node();
    relu_node->set_name("relu");
    relu_node->set_op("Relu");
    relu_node->add_input("bias_add");
    SetNodeAttr("opos", std::vector<int>({8, 5}), relu_node);

    NodeDef* bias_add_node = original_graph_def.add_node();
    bias_add_node->set_name("bias_add");
    bias_add_node->set_op("BiasAdd");
    bias_add_node->add_input("conv");
    bias_add_node->add_input("bias");

    NodeDef* bias_node = original_graph_def.add_node();
    bias_node->set_name("bias");
    bias_node->set_op("Const");
    SetNodeAttr("wpos", std::vector<int>({8, 4}), bias_node);
    Tensor bias_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&bias_data, {3.5f, 4.5f});
    SetNodeTensorAttr<float>("value", bias_data, bias_node);

    NodeDef* conv_node = original_graph_def.add_node();
    conv_node->set_name("conv");
    conv_node->set_op("Conv2D");
    conv_node->add_input("input");
    conv_node->add_input("weights");

    NodeDef* weights_node = original_graph_def.add_node();
    weights_node->set_name("weights");
    weights_node->set_op("Const");
    SetNodeAttr("wpos", std::vector<int>({8, 3}), weights_node);
    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    SetNodeTensorAttr<float>("value", weights_data, weights_node);

    NodeDef* input_node = original_graph_def.add_node();
    input_node->set_name("input");
    input_node->set_op("Const");
    SetNodeAttr("opos", std::vector<int>({8, 2}), input_node);

    const string config_string =
        "input_nodes,input_op,output_nodes,relu_op/aquant,input_shapes,"
        "1*1*6*2,weight_bit,8,activation_bit,8,method,0,calib_iter,"
        "10,output_dir," +
        testing::TmpDir() + ",align_concat,0,simulate_dpu,1,";
    QuantizeConfig config;
    GraphDef out_graph_def;
    TF_ASSERT_OK(ConvertPatterns(original_graph_def, &out_graph_def, config));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(out_graph_def, &node_lookup);
    ASSERT_EQ(3, node_lookup.size());
    EXPECT_EQ(1, node_lookup.count("input"));
    EXPECT_EQ(1, node_lookup.count("conv"));
    EXPECT_EQ(1, node_lookup.count("relu"));

    EXPECT_EQ(8, node_lookup.at("input")->attr().at("opos").list().i(0));
    EXPECT_EQ(8, node_lookup.at("conv")->attr().at("wpos").list().i(0));
    EXPECT_EQ(8, node_lookup.at("conv")->attr().at("bpos").list().i(0));
    EXPECT_EQ(8, node_lookup.at("relu")->attr().at("opos").list().i(0));

    EXPECT_EQ(2, node_lookup.at("input")->attr().at("opos").list().i(1));
    EXPECT_EQ(3, node_lookup.at("conv")->attr().at("wpos").list().i(1));
    EXPECT_EQ(4, node_lookup.at("conv")->attr().at("bpos").list().i(1));
    EXPECT_EQ(5, node_lookup.at("relu")->attr().at("opos").list().i(1));

    Tensor w = GetNodeTensorAttr(*node_lookup.at("conv"), "weights");
    test::ExpectTensorEqual<float>(weights_data, w);
    Tensor b = GetNodeTensorAttr(*node_lookup.at("conv"), "bias");
    test::ExpectTensorEqual<float>(bias_data, b);
  }

  void TestDeployQuantizedGraph() {
    GraphDef original_graph_def;

    NodeDef* relu_node = original_graph_def.add_node();
    relu_node->set_name("relu");
    relu_node->set_op("Relu");
    relu_node->add_input("bias_add");
    SetNodeAttr("opos", std::vector<int>({8, 5}), relu_node);

    NodeDef* bias_add_node = original_graph_def.add_node();
    bias_add_node->set_name("bias_add");
    bias_add_node->set_op("BiasAdd");
    bias_add_node->add_input("conv");
    bias_add_node->add_input("bias");

    NodeDef* bias_node = original_graph_def.add_node();
    bias_node->set_name("bias");
    bias_node->set_op("Const");
    SetNodeAttr("wpos", std::vector<int>({8, 4}), bias_node);
    Tensor bias_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&bias_data, {3.5f, 4.5f});
    SetNodeTensorAttr<float>("value", bias_data, bias_node);

    NodeDef* conv_node = original_graph_def.add_node();
    conv_node->set_name("conv");
    conv_node->set_op("Conv2D");
    conv_node->add_input("input");
    conv_node->add_input("weights");

    NodeDef* weights_node = original_graph_def.add_node();
    weights_node->set_name("weights");
    weights_node->set_op("Const");
    SetNodeAttr("wpos", std::vector<int>({8, 3}), weights_node);
    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    SetNodeTensorAttr<float>("value", weights_data, weights_node);

    NodeDef* input_node = original_graph_def.add_node();
    input_node->set_name("input");
    input_node->set_op("Const");
    SetNodeAttr("opos", std::vector<int>({8, 2}), input_node);

    const string config_string =
        "input_nodes,input_op,output_nodes,relu_op/aquant,input_shapes,"
        "1*1*6*2,weight_bit,8,activation_bit,8,method,0,calib_iter,"
        "10,output_dir," +
        testing::TmpDir() + ",align_concat,0,simulate_dpu,1,";
    QuantizeConfig config;
    GraphDef out_graph_def;
    TF_ASSERT_OK(
        DeployQuantizedGraph(original_graph_def, &out_graph_def, config));
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(out_graph_def, &node_lookup);
    ASSERT_EQ(3, node_lookup.size());
    EXPECT_EQ(1, node_lookup.count("input"));
    EXPECT_EQ(1, node_lookup.count("conv"));
    EXPECT_EQ(1, node_lookup.count("relu"));

    EXPECT_EQ(8, node_lookup.at("input")->attr().at("opos").list().i(0));
    EXPECT_EQ(8, node_lookup.at("conv")->attr().at("wpos").list().i(0));
    EXPECT_EQ(8, node_lookup.at("conv")->attr().at("bpos").list().i(0));
    EXPECT_EQ(8, node_lookup.at("relu")->attr().at("opos").list().i(0));

    EXPECT_EQ(2, node_lookup.at("input")->attr().at("opos").list().i(1));
    EXPECT_EQ(3, node_lookup.at("conv")->attr().at("wpos").list().i(1));
    EXPECT_EQ(4, node_lookup.at("conv")->attr().at("bpos").list().i(1));
    EXPECT_EQ(5, node_lookup.at("relu")->attr().at("opos").list().i(1));

    Tensor w = GetNodeTensorAttr(*node_lookup.at("conv"), "weights");
    test::ExpectTensorEqual<float>(weights_data, w);
    Tensor b = GetNodeTensorAttr(*node_lookup.at("conv"), "bias");
    test::ExpectTensorEqual<float>(bias_data, b);
  }
};  // namespace decent_q

TEST_F(DeployQuantizedGraphTest, TestFoldFixNeuron) { TestFoldFixNeuron(); }

TEST_F(DeployQuantizedGraphTest, TestPolishActivationInfo) {
  TestPolishActivationInfo();
}

TEST_F(DeployQuantizedGraphTest, TestFoldOpParams) { TestFoldOpParams(); }

TEST_F(DeployQuantizedGraphTest, TestConvertPatterns) { TestConvertPatterns(); }

TEST_F(DeployQuantizedGraphTest, TestDeployQuantizedGraph) {
  TestDeployQuantizedGraph();
}

}  // namespace decent_q
}  // namespace tensorflow
