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

#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/contrib/decent_q/utils/ops/fix_neuron_ops.h"
#include "tensorflow/contrib/decent_q/utils/quantize_utils.h"
#include "tensorflow/contrib/decent_q/utils/transform_utils.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace decent_q {

class QuantizeUtilsTest : public ::testing::Test {
 protected:
  void TestQuantizeKernelCpu() {
    float x = 0.1;
    float step = std::pow(2, -5);
    float lower_bound = -std::pow(2, 8 - 1) * step;
    float upper_bound = std::pow(2, 8 - 1) * step - step;
    float y = quantize_kernel_cpu(x, step, lower_bound, upper_bound);
    EXPECT_NEAR(y, x, step);
  }

  void TestQuantizeCpu() {
    Tensor x(DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&x, {0.1f, 0.2f, 0.4f});
    auto flat_x = x.flat<float>();
    Tensor y(DT_FLOAT, x.shape());
    auto flat_y = y.flat<float>();
    int n = 3, bit_width = 8, pos = 5;
    float step = std::pow(2, -5);
    quantize_cpu<float>(n, flat_x.data(), flat_y.data(), bit_width, pos);

    for (int i = 0; i < n; ++i) {
      EXPECT_NEAR(flat_y.data()[i], flat_y.data()[i], step);
    }
  }

  void TestGetNodeTypeByName() {
    GraphDef original_graph_def;

    NodeDef* relu_node = original_graph_def.add_node();
    relu_node->set_name("relu");
    relu_node->set_op("Relu");
    relu_node->add_input("conv");

    NodeDef* conv_node = original_graph_def.add_node();
    conv_node->set_name("conv");
    conv_node->set_op("Conv2D");
    conv_node->add_input("input");
    string type_1, type_2;
    TF_ASSERT_OK(GetNodeTypeByName(original_graph_def, "relu", &type_1));
    TF_ASSERT_OK(GetNodeTypeByName(original_graph_def, "conv", &type_2));
    ASSERT_EQ(type_1, "Relu");
    ASSERT_EQ(type_2, "Conv2D");
  }

  void TestRecordMatchedPatterns() {
    NodeMatch second;
    second.node.set_name("second");

    NodeMatch first;
    first.node.set_name("first");
    first.inputs.push_back(second);

    std::vector<std::tuple<int, NodeMatch>> matched_node_patterns;
    int pattern_id = 1;
    TF_ASSERT_OK(
        RecordMatchedPatterns(matched_node_patterns, pattern_id, first));
    ASSERT_EQ(1, std::get<0>(matched_node_patterns[0]));
    ASSERT_EQ("first", std::get<1>(matched_node_patterns[0]).node.name());
    ASSERT_EQ("second",
              std::get<1>(matched_node_patterns[0]).inputs[0].node.name());
    // ASSERT_EQ("second",
    // matched_node_patterns[0].second.inputs[0].node.name());
  }

  void TestRecordMatchedNodes() {
    NodeMatch second;
    second.node.set_name("second");

    NodeMatch first;
    first.node.set_name("first");
    first.inputs.push_back(second);

    std::unordered_map<string, int> matched_nodes;
    const int match_id = 3;
    const std::set<string> irrelevant_nodes({"first"});
    int pattern_id = 1;
    TF_ASSERT_OK(
        RecordMatchedNodes(matched_nodes, first, match_id, irrelevant_nodes));
    ASSERT_EQ(1, matched_nodes.count("second"));
    EXPECT_EQ(3, matched_nodes["second"]);
  }

  void TestCheckAnyMatchedNodes() {
    NodeMatch fifth;
    fifth.node.set_name("fifth");

    NodeMatch fourth;
    fourth.node.set_name("fourth");
    fourth.inputs.push_back(fifth);

    NodeMatch second;
    second.node.set_name("second");
    second.inputs.push_back(fourth);

    NodeMatch third;
    third.node.set_name("third");
    third.inputs.push_back(fourth);

    NodeMatch first;
    first.node.set_name("first");
    first.inputs.push_back(second);
    first.inputs.push_back(third);

    std::unordered_map<string, int> matched_nodes;
    matched_nodes.insert(std::pair<string, int>("second", 3));
    matched_nodes.insert(std::pair<string, int>("fourth", 2));

    const std::set<string> irrelevant_nodes({"fourth"});

    bool check_rst_1 =
        CheckAnyMatchedNodes(matched_nodes, first, irrelevant_nodes);
    bool check_rst_2 =
        CheckAnyMatchedNodes(matched_nodes, second, irrelevant_nodes);
    bool check_rst_3 =
        CheckAnyMatchedNodes(matched_nodes, third, irrelevant_nodes);
    bool check_rst_4 =
        CheckAnyMatchedNodes(matched_nodes, fourth, irrelevant_nodes);
    bool check_rst_5 =
        CheckAnyMatchedNodes(matched_nodes, fifth, irrelevant_nodes);
    EXPECT_EQ(true, check_rst_1);
    EXPECT_EQ(true, check_rst_2);
    EXPECT_EQ(false, check_rst_3);
    EXPECT_EQ(false, check_rst_4);
    EXPECT_EQ(false, check_rst_5);
  }

  void TestCheckAnyIgnoredNodes() {
    NodeMatch fifth;
    fifth.node.set_name("fifth");

    NodeMatch fourth;
    fourth.node.set_name("fourth");
    fourth.inputs.push_back(fifth);

    NodeMatch second;
    second.node.set_name("second");
    second.inputs.push_back(fourth);

    NodeMatch third;
    third.node.set_name("third");
    third.inputs.push_back(fourth);

    NodeMatch first;
    first.node.set_name("first");
    first.inputs.push_back(second);
    first.inputs.push_back(third);

    std::set<string> ignore_nodes;
    ignore_nodes.insert("second");
    ignore_nodes.insert("fourth");

    const std::set<string> irrelevant_nodes({"fourth"});

    bool check_rst_1 =
        CheckAnyIgnoredNodes(ignore_nodes, first, irrelevant_nodes);
    bool check_rst_2 =
        CheckAnyIgnoredNodes(ignore_nodes, second, irrelevant_nodes);
    bool check_rst_3 =
        CheckAnyIgnoredNodes(ignore_nodes, third, irrelevant_nodes);
    bool check_rst_4 =
        CheckAnyIgnoredNodes(ignore_nodes, fourth, irrelevant_nodes);
    bool check_rst_5 =
        CheckAnyIgnoredNodes(ignore_nodes, fifth, irrelevant_nodes);
    EXPECT_EQ(true, check_rst_1);
    EXPECT_EQ(true, check_rst_2);
    EXPECT_EQ(false, check_rst_3);
    EXPECT_EQ(false, check_rst_4);
    EXPECT_EQ(false, check_rst_5);
  }

  void TestConvertConstantsToVariables() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));
    GraphDef converted_graph_def;
    TF_ASSERT_OK(
        _ConvertConstantsToVariables(original_graph_def, &converted_graph_def));
    for (const NodeDef& node : converted_graph_def.node()) {
      EXPECT_NE("Const", node.op());
    }
  }

  void TestParseGraph() {
    const string config_string =
        "input_nodes,input,output_nodes,conv,input_shapes,"
        "1*1*6*2,weight_bit,8,activation_bit,8,method,0,calib_iter,"
        "10,output_dir,./quantize_results,align_concat,0,simulate_dpu,1,";
    QuantizeConfig config;
    config.FromString(config_string);
    GraphQuantizer graph_quantizer(config);

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    Tensor bias_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&bias_data, {1.0f, 2.0f});
    Output bias_op =
        Const(root.WithOpName("bias"), Input::Initializer(bias_data));
    Output bias_add_op = BiasAdd(root.WithOpName("bias_add"), conv_op, bias_op);
    Output relu_op = Relu(root.WithOpName("relu"), bias_add_op);
    Output id_op = Identity(root.WithOpName("identity"), relu_op);

    Tensor weights_data_2(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data_2,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op_2 =
        Const(root.WithOpName("weights_2"), Input::Initializer(weights_data_2));

    Output conv_op_2 = Conv2D(root.WithOpName("conv"), id_op, weights_op_2,
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
    auto fused_bn_op = FusedBatchNorm(
        root.WithOpName("FusedBatchNorm"), conv_op_2,
        Const(root.WithOpName("scale"), Input::Initializer(scale_data)),
        Const(root.WithOpName("offset"), Input::Initializer(offset_data)),
        Const(root.WithOpName("mean"), Input::Initializer(mean_data)),
        Const(root.WithOpName("var"), Input::Initializer(variance_data)), attr);

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    std::vector<std::tuple<int, NodeMatch>> matched_node_patterns;
    std::unordered_map<string, int> matched_nodes;
    std::set<string> ignore_nodes;
    std::unordered_set<string> unmatched_nodes;
    TF_ASSERT_OK(ParseGraph(original_graph_def, matched_node_patterns,
                            matched_nodes, ignore_nodes, unmatched_nodes));
    EXPECT_EQ(2, matched_node_patterns.size());
    EXPECT_EQ(9, std::get<0>(matched_node_patterns[0]));
    EXPECT_EQ("relu", std::get<1>(matched_node_patterns[0]).node.name());
    EXPECT_EQ(11, std::get<0>(matched_node_patterns[1]));
    EXPECT_EQ("conv_1", std::get<1>(matched_node_patterns[1]).node.name());

    ASSERT_EQ(1, matched_nodes.count("weights_2"));
    EXPECT_EQ(1, matched_nodes.at("weights_2"));
    ASSERT_EQ(1, matched_nodes.count("conv_1"));
    EXPECT_EQ(1, matched_nodes.at("conv_1"));

    ASSERT_EQ(1, matched_nodes.count("relu"));
    ASSERT_EQ(1, matched_nodes.count("bias_add"));
    ASSERT_EQ(1, matched_nodes.count("bias"));
    ASSERT_EQ(1, matched_nodes.count("conv"));
    ASSERT_EQ(1, matched_nodes.count("weights"));
    EXPECT_EQ(0, matched_nodes.at("relu"));
    EXPECT_EQ(0, matched_nodes.at("bias_add"));
    EXPECT_EQ(0, matched_nodes.at("bias"));
    EXPECT_EQ(0, matched_nodes.at("conv"));
    EXPECT_EQ(0, matched_nodes.at("weights"));

    ASSERT_EQ(7, unmatched_nodes.size());
    EXPECT_EQ(1, unmatched_nodes.count("FusedBatchNorm"));
    EXPECT_EQ(1, unmatched_nodes.count("scale"));
    EXPECT_EQ(1, unmatched_nodes.count("offset"));
    EXPECT_EQ(1, unmatched_nodes.count("input"));
    EXPECT_EQ(1, unmatched_nodes.count("identity"));
    EXPECT_EQ(1, unmatched_nodes.count("mean"));
    EXPECT_EQ(1, unmatched_nodes.count("var"));

    ASSERT_EQ(1, ignore_nodes.size());
    EXPECT_EQ(1, ignore_nodes.count("conv_1"));
  }

  void TestInsertIdForNodes() {
    GraphDef original_graph_def;

    NodeDef* relu_node = original_graph_def.add_node();
    relu_node->set_name("relu");
    relu_node->set_op("Relu");
    relu_node->add_input("conv");
    SetNodeAttr("T", DT_FLOAT, relu_node);

    NodeDef* conv_node = original_graph_def.add_node();
    conv_node->set_name("conv");
    conv_node->set_op("Conv2D");
    conv_node->add_input("input");
    SetNodeAttr("T", DT_FLOAT, conv_node);

    NodeDef* input_node = original_graph_def.add_node();
    input_node->set_name("input");
    input_node->set_op("Placeholder");
    SetNodeAttr("T", DT_FLOAT, input_node);

    GraphDef output_graph_def;
    std::vector<string> node_names({"conv"});
    TF_ASSERT_OK(
        InsertIdForNodes(original_graph_def, &output_graph_def, node_names));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(output_graph_def, &node_lookup);
    ASSERT_EQ(4, node_lookup.size());
    EXPECT_EQ(1, node_lookup.count("input"));
    EXPECT_EQ(1, node_lookup.count("conv_float"));
    EXPECT_EQ(1, node_lookup.count("conv"));
    EXPECT_EQ(1, node_lookup.count("relu"));
  }

  void TestGetNodeDataType() {
    GraphDef original_graph_def;

    NodeDef* node_1 = original_graph_def.add_node();
    node_1->set_name("node_1");
    node_1->set_op("Cast");
    SetNodeAttr("DstT", DT_INT32, node_1);

    NodeDef* node_2 = original_graph_def.add_node();
    node_2->set_name("node_2");
    node_2->set_op("Const");
    SetNodeAttr("dtype", DT_FLOAT, node_2);

    NodeDef* node_3 = original_graph_def.add_node();
    node_3->set_name("node_3");
    node_3->set_op("Const");
    SetNodeAttr("T", DT_INT64, node_3);

    NodeDef* node_4 = original_graph_def.add_node();
    node_4->set_name("node_4");
    node_4->set_op("Const");
    SetNodeAttr("type", DT_DOUBLE, node_4);

    DataType data_type;
    TF_ASSERT_OK(GetNodeDataType(*node_1, &data_type));
    EXPECT_EQ(data_type, DT_INT32);
    TF_ASSERT_OK(GetNodeDataType(*node_2, &data_type));
    EXPECT_EQ(data_type, DT_FLOAT);
    TF_ASSERT_OK(GetNodeDataType(*node_3, &data_type));
    EXPECT_EQ(data_type, DT_INT64);
    TF_ASSERT_OK(GetNodeDataType(*node_4, &data_type));
    EXPECT_EQ(data_type, DT_DOUBLE);
  }

  void TestGetDataTypeOfNodesef() {
    GraphDef original_graph_def;

    NodeDef* node_1 = original_graph_def.add_node();
    node_1->set_name("node_1");
    node_1->set_op("Cast");
    SetNodeAttr("DstT", DT_INT32, node_1);

    NodeDef* node_2 = original_graph_def.add_node();
    node_2->set_name("node_2");
    node_2->set_op("Const");
    SetNodeAttr("dtype", DT_FLOAT, node_2);

    NodeDef* node_3 = original_graph_def.add_node();
    node_3->set_name("node_3");
    node_3->set_op("Const");
    SetNodeAttr("T", DT_INT64, node_3);

    NodeDef* node_4 = original_graph_def.add_node();
    node_4->set_name("node_4");
    node_4->set_op("Const");
    SetNodeAttr("type", DT_DOUBLE, node_4);

    std::vector<std::string> node_names(
        {"node_1", "node_2", "node_3", "node_4"});
    std::unordered_map<string, DataType> data_type_of_nodes;
    DataType data_type;
    TF_ASSERT_OK(GetDataTypeOfNodes(original_graph_def, node_names,
                                    &data_type_of_nodes));
    EXPECT_EQ(DT_INT32, data_type_of_nodes.at("node_1"));
    EXPECT_EQ(DT_FLOAT, data_type_of_nodes.at("node_2"));
    EXPECT_EQ(DT_INT64, data_type_of_nodes.at("node_3"));
    EXPECT_EQ(DT_DOUBLE, data_type_of_nodes.at("node_4"));
  }

  void TestGetShapeOfNodes() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    Output input_op =
        Placeholder(root.WithOpName("input"), DT_FLOAT,
                    Placeholder::Shape(PartialTensorShape({-1, 8, 8, 2})));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    Tensor bias_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&bias_data, {1.0f, 2.0f});
    Output bias_op =
        Const(root.WithOpName("bias"), Input::Initializer(bias_data));
    Output bias_add_op = BiasAdd(root.WithOpName("bias_add"), conv_op, bias_op);
    Output relu_op = Relu(root.WithOpName("relu"), bias_add_op);
    Tensor shape_data(DT_INT32, TensorShape({3}));
    test::FillValues<int>(&shape_data, {1, 4, -1});
    Output shape_op =
        Const(root.WithOpName("shape"), Input::Initializer(shape_data));
    Output reshape_op = Reshape(root.WithOpName("reshape"), relu_op, shape_op);

    GraphDef input_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&input_graph_def));

    std::unordered_map<string, std::vector<int>> shape_of_nodes;
    const std::vector<std::string> node_names({"conv", "relu", "reshape"});
    const int batch_size = 1;
    TF_ASSERT_OK(GetShapeOfNodes(input_graph_def, node_names, batch_size,
                                 &shape_of_nodes));

    ASSERT_EQ(3, shape_of_nodes.size());
    EXPECT_EQ(std::vector<int>({1, 8, 7, 2}), shape_of_nodes.at("conv"));
    EXPECT_EQ(std::vector<int>({1, 8, 7, 2}), shape_of_nodes.at("relu"));
    EXPECT_EQ(std::vector<int>({1, 4, 28}), shape_of_nodes.at("reshape"));
  }

  void TestInferenceShape() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Output input_op =
        Placeholder(root.WithOpName("input"), DT_FLOAT,
                    Placeholder::Shape(PartialTensorShape({-1, 8, 8, 2})));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    Tensor bias_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&bias_data, {1.0f, 2.0f});
    Output bias_op =
        Const(root.WithOpName("bias"), Input::Initializer(bias_data));
    Output bias_add_op = BiasAdd(root.WithOpName("bias_add"), conv_op, bias_op);
    Output relu_op = Relu(root.WithOpName("relu"), bias_add_op);
    Tensor shape_data(DT_INT32, TensorShape({3}));
    test::FillValues<int>(&shape_data, {1, 4, -1});
    Output shape_op =
        Const(root.WithOpName("shape"), Input::Initializer(shape_data));
    Output reshape_op = Reshape(root.WithOpName("reshape"), relu_op, shape_op);

    GraphDef input_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&input_graph_def));

    const int batch_size = 1;
    GraphDef output_graph_def;
    TF_ASSERT_OK(
        InferenceShape(input_graph_def, &output_graph_def, batch_size));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(output_graph_def, &node_lookup);
    ASSERT_EQ(8, node_lookup.size());
    ASSERT_EQ(1, node_lookup.count("reshape/shape"));
    Tensor inferenced_shape =
        GetNodeTensorAttr(*node_lookup.at("reshape/shape"), "value");
    Tensor expected_shape_data(DT_INT32, TensorShape({3}));
    test::FillValues<int>(&expected_shape_data, {1, 4, 28});
    test::ExpectTensorEqual<int>(inferenced_shape, expected_shape_data);
  }

  void TestConvertMeanToAvgpool() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Output input_op =
        Placeholder(root.WithOpName("input"), DT_FLOAT,
                    Placeholder::Shape(PartialTensorShape({-1, 8, 8, 2})));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    Tensor bias_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&bias_data, {1.0f, 2.0f});
    Output bias_op =
        Const(root.WithOpName("bias"), Input::Initializer(bias_data));
    Output bias_add_op = BiasAdd(root.WithOpName("bias_add"), conv_op, bias_op);
    Output relu_op = Relu(root.WithOpName("relu"), bias_add_op);
    Tensor axis_data(DT_INT32, TensorShape({2}));
    test::FillValues<int>(&axis_data, {1, 2});
    Output axis_op =
        Const(root.WithOpName("axis"), Input::Initializer(axis_data));
    Mean::Attrs attrs;
    attrs.keep_dims_ = true;
    Output mean_op = Mean(root.WithOpName("mean"), relu_op, axis_op, attrs);
    GraphDef input_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&input_graph_def));

    GraphDef output_graph_def;
    TF_ASSERT_OK(ConvertMeanToAvgpool(input_graph_def, &output_graph_def));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(output_graph_def, &node_lookup);
    ASSERT_EQ(1, node_lookup.count("mean"));
    ASSERT_EQ("AvgPool", node_lookup.at("mean")->op());
    EXPECT_EQ(1, node_lookup.at("mean")->attr().at("ksize").list().i(0));
    EXPECT_EQ(8, node_lookup.at("mean")->attr().at("ksize").list().i(1));
    EXPECT_EQ(7, node_lookup.at("mean")->attr().at("ksize").list().i(2));
    EXPECT_EQ(1, node_lookup.at("mean")->attr().at("ksize").list().i(3));
    EXPECT_EQ(1, node_lookup.at("mean")->attr().at("strides").list().i(0));
    EXPECT_EQ(1, node_lookup.at("mean")->attr().at("strides").list().i(1));
    EXPECT_EQ(1, node_lookup.at("mean")->attr().at("strides").list().i(2));
    EXPECT_EQ(1, node_lookup.at("mean")->attr().at("strides").list().i(3));
    EXPECT_EQ("VALID", node_lookup.at("mean")->attr().at("padding").s());
    EXPECT_EQ("NHWC", node_lookup.at("mean")->attr().at("data_format").s());
  }

  void TestSimulateDPU() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Output input_op =
        Placeholder(root.WithOpName("input"), DT_FLOAT,
                    Placeholder::Shape(PartialTensorShape({-1, 7, 8, 2})));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    Tensor bias_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&bias_data, {1.0f, 2.0f});
    Output bias_op =
        Const(root.WithOpName("bias"), Input::Initializer(bias_data));
    Output bias_add_op = BiasAdd(root.WithOpName("bias_add"), conv_op, bias_op);
    Output relu_op = Relu(root.WithOpName("relu"), bias_add_op);

    Output mean_op = AvgPool(root.WithOpName("mean"), relu_op, {1, 7, 7, 1},
                             {1, 1, 1, 1}, "VALID");

    // leaky relu op pattern 1
    Output mean_idty_op = Identity(root.WithOpName("mean_idty"), mean_op);
    Tensor alpha_data = test::AsTensor<float>({0.1f}, {});
    Output alpha_op =
        Const(root.WithOpName("alpha"), Input::Initializer(alpha_data));
    Output leaky_mul_op =
        Mul(root.WithOpName("leaky_mul"), alpha_op, mean_idty_op);
    Output leaky_max_op =
        Maximum(root.WithOpName("leaky_max"), leaky_mul_op, mean_idty_op);

    // leaky relu op pattern 2
    Output leaky_idty_op = Identity(root.WithOpName("relu_idty"), leaky_max_op);
    Output relu_op_2 = Relu(root.WithOpName("leaky_relu_2"), leaky_idty_op);
    Output neg_op = Negate(root.WithOpName("leaky_neg"), leaky_idty_op);
    Output relu_op_3 = Relu(root.WithOpName("leaky_relu_3"), neg_op);
    Tensor alpha_data_2 = test::AsTensor<float>({0.1f}, {});
    Output alpha_op_2 =
        Const(root.WithOpName("alpha_2"), Input::Initializer(alpha_data_2));
    Output leaky_mul_op_2 =
        Mul(root.WithOpName("leaky_mul_2"), alpha_op_2, relu_op_3);
    Output sub_op =
        Sub(root.WithOpName("leaky_sub"), relu_op_2, leaky_mul_op_2);

    GraphDef input_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&input_graph_def));

    GraphDef output_graph_def;
    TF_ASSERT_OK(SimulateDPU(input_graph_def, &output_graph_def));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(output_graph_def, &node_lookup);

    // test avg pool
    ASSERT_EQ(1, node_lookup.count("mean/mul"));
    ASSERT_EQ(1, node_lookup.count("mean/scale_value"));

    Tensor scale_value =
        GetNodeTensorAttr(*node_lookup.at("mean/scale_value"), "value");
    Tensor expected_scale_value =
        test::AsTensor<float>({49.f * 21.f / 1024}, {});
    test::ExpectTensorNear<float>(scale_value, expected_scale_value, 1e-6);

    Tensor alpha_value = GetNodeTensorAttr(*node_lookup.at("alpha"), "value");
    Tensor expected_alpha_value = test::AsTensor<float>({26.f / 256}, {});
    test::ExpectTensorNear<float>(alpha_value, expected_alpha_value, 1e-6);

    Tensor alpha_2_value =
        GetNodeTensorAttr(*node_lookup.at("alpha_2"), "value");
    Tensor expected_alpha_2_value = test::AsTensor<float>({26.f / 256}, {});
    test::ExpectTensorNear<float>(alpha_2_value, expected_alpha_2_value, 1e-6);
  }

  void TestLoadQuantizeInfoFromFile() {
    // build graph
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));
    Output input_aquant =
        FixNeuron(root.WithOpName("input_op/aquant"), input_op, 1, 1);
    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));
    Output weights_wquant =
        FixNeuron(root.WithOpName("weights_op/wquant"), weights_op, 0, 1);
    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");
    Tensor bias_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&bias_data, {1.5f, 2.5f});
    Output bias_op =
        Const(root.WithOpName("bias_op"), Input::Initializer(bias_data));
    Output bias_wquant =
        FixNeuron(root.WithOpName("bias_op/wquant"), bias_op, 0, 1);
    Output bias_add_op =
        BiasAdd(root.WithOpName("bias_add_op"), conv_op, bias_op);
    Output relu_op = Relu(root.WithOpName("relu_op"), bias_add_op);
    Output relu_aquant =
        FixNeuron(root.WithOpName("relu_op/aquant"), relu_op, 1, 1);
    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    const string output_dir = testing::TmpDir();
    // svae pos file
    string make_temp_dir = "mkdir -p " + io::JoinPath(output_dir, "temp");
    system(make_temp_dir.c_str());
    std::vector<string> quantize_ops = {"input_op/aquant", "weights_op/wquant",
                                        "bias_op/wquant", "relu_op/aquant"};

    for (const auto& op_name : quantize_ops) {
      string pos_filename;
      int bit_width = 8, pos = 5;
      pos_filename = output_dir + "/temp/" +
                     str_util::StringReplace(op_name, "/", "_", true);
      std::ofstream ofile(pos_filename);
      ofile << op_name << " " << bit_width << " " << pos << std::endl;
      ofile.close();
    }
    std::unordered_map<string, std::vector<int>> quantize_info_map;
    TF_ASSERT_OK(LoadQuantizeInfoFromFile(original_graph_def,
                                          &quantize_info_map, output_dir));

    ASSERT_EQ(4, quantize_info_map.size());
    EXPECT_EQ(8, quantize_info_map.at("input_op/aquant")[0]);
    EXPECT_EQ(8, quantize_info_map.at("bias_op/wquant")[0]);
    EXPECT_EQ(8, quantize_info_map.at("weights_op/wquant")[0]);
    EXPECT_EQ(8, quantize_info_map.at("relu_op/aquant")[0]);

    EXPECT_EQ(5, quantize_info_map.at("input_op/aquant")[1]);
    EXPECT_EQ(5, quantize_info_map.at("bias_op/wquant")[1]);
    EXPECT_EQ(5, quantize_info_map.at("weights_op/wquant")[1]);
    EXPECT_EQ(5, quantize_info_map.at("relu_op/aquant")[1]);
  }

  void TestSaveQuantizeInfoForDebugging() {
    GraphDef original_graph_def;

    NodeDef* relu_node = original_graph_def.add_node();
    relu_node->set_name("relu_op");
    relu_node->set_op("Relu");
    relu_node->add_input("conv_op");
    SetNodeAttr("opos", std::vector<int>({8, 6}), relu_node);
    SetNodeAttr("ipos", std::vector<int>({8, 3}), relu_node);

    NodeDef* conv_node = original_graph_def.add_node();
    conv_node->set_name("conv_op");
    conv_node->set_op("Conv2D");
    conv_node->add_input("input_op");
    SetNodeAttr("bpos", std::vector<int>({8, 5}), conv_node);
    SetNodeAttr("wpos", std::vector<int>({8, 4}), conv_node);
    SetNodeAttr("opos", std::vector<int>({8, 3}), conv_node);
    SetNodeAttr("ipos", std::vector<int>({8, 2}), conv_node);

    NodeDef* input_node = original_graph_def.add_node();
    input_node->set_name("input_op");
    input_node->set_op("Const");
    SetNodeAttr("opos", std::vector<int>({8, 2}), input_node);

    setenv("DECENT_DEBUG", "1", 1);
    const string output_dir = testing::TmpDir();
    GraphDef sorted_graph_def;
    SortByExecutionOrder(original_graph_def, &sorted_graph_def);
    TF_ASSERT_OK(SaveQuantizeInfoForDebugging(sorted_graph_def, output_dir));

    const string filename = output_dir + "/decent_debug/quantize_info";
    std::ifstream ifile(filename);
    ASSERT_NE(0, ifile.is_open());
    string op_name;
    int in_bit_width, in_pos, out_bit_width, out_pos, w_bit_width, w_pos,
        b_bit_width, b_pos;

    ifile >> op_name >> out_bit_width >> out_pos;
    EXPECT_EQ(op_name, "input_op");
    EXPECT_EQ(out_bit_width, 8);
    EXPECT_EQ(out_pos, 2);

    ifile >> op_name >> in_bit_width >> in_pos >> out_bit_width >> out_pos >>
        w_bit_width >> w_pos >> b_bit_width >> b_pos;
    EXPECT_EQ(op_name, "conv_op");
    EXPECT_EQ(in_bit_width, 8);
    EXPECT_EQ(in_pos, 2);
    EXPECT_EQ(out_bit_width, 8);
    EXPECT_EQ(out_pos, 3);
    EXPECT_EQ(w_bit_width, 8);
    EXPECT_EQ(w_pos, 4);
    EXPECT_EQ(b_bit_width, 8);
    EXPECT_EQ(b_pos, 5);

    ifile >> op_name >> in_bit_width >> in_pos >> out_bit_width >> out_pos;
    EXPECT_EQ(op_name, "relu_op");
    EXPECT_EQ(in_bit_width, 8);
    EXPECT_EQ(in_pos, 3);
    EXPECT_EQ(out_bit_width, 8);
    EXPECT_EQ(out_pos, 6);
    ifile.close();
  }
};  // namespace decent_q

TEST_F(QuantizeUtilsTest, TestQuantizeKernelCpu) { TestQuantizeKernelCpu(); }

TEST_F(QuantizeUtilsTest, TestQuantizeCpu) { TestQuantizeCpu(); }

TEST_F(QuantizeUtilsTest, TestGetNodeTypeByName) { TestGetNodeTypeByName(); }

TEST_F(QuantizeUtilsTest, TestRecordMatchedPatterns) {
  TestRecordMatchedPatterns();
}

TEST_F(QuantizeUtilsTest, TestRecordMatchedNodes) { TestRecordMatchedNodes(); }

TEST_F(QuantizeUtilsTest, TestCheckAnyMatchedNodes) {
  TestCheckAnyMatchedNodes();
}

TEST_F(QuantizeUtilsTest, TestCheckAnyIgnoredNodes) {
  TestCheckAnyIgnoredNodes();
}

TEST_F(QuantizeUtilsTest, TestConvertConstantsToVariables) {
  TestConvertConstantsToVariables();
}

TEST_F(QuantizeUtilsTest, TestParseGraph) { TestParseGraph(); }

TEST_F(QuantizeUtilsTest, TestInsertIdForNodes) { TestInsertIdForNodes(); }

TEST_F(QuantizeUtilsTest, TestGetNodeDataType) { TestGetNodeDataType(); }

TEST_F(QuantizeUtilsTest, TestGetDataTypeOfNodesef) {
  TestGetDataTypeOfNodesef();
}

TEST_F(QuantizeUtilsTest, TestInferenceShape) { TestInferenceShape(); }

TEST_F(QuantizeUtilsTest, TestConvertMeanToAvgpool) {
  TestConvertMeanToAvgpool();
}

TEST_F(QuantizeUtilsTest, TestSimulateDPU) { TestSimulateDPU(); }

TEST_F(QuantizeUtilsTest, TestLoadQuantizeInfoFromFile) {
  TestLoadQuantizeInfoFromFile();
}

TEST_F(QuantizeUtilsTest, TestSaveQuantizeInfoForDebugging) {
  TestSaveQuantizeInfoForDebugging();
}
// TEST_F(QuantizeUtilsTest, ) { (); }
}  // namespace decent_q
}  // namespace tensorflow
