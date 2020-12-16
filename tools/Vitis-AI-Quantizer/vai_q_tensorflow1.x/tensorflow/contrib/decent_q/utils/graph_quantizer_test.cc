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

#include <fstream>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/contrib/decent_q/utils/graph_quantizer.h"
#include "tensorflow/contrib/decent_q/utils/ops/fix_neuron_ops.h"
#include "tensorflow/contrib/decent_q/utils/transform_graph.h"
#include "tensorflow/contrib/decent_q/utils/transform_utils.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace decent_q {

class GraphQuantizerTest : public ::testing::Test {
 protected:
  void TestQuantizeConfigFromString() {
    const string config_string =
        "input_nodes,input_1,input_nodes,input_2,output_nodes,relu,input_"
        "shapes,"
        "-1*4*4*3,weight_bit,8,activation_bit,8,method,0,calib_iter,"
        "10,output_dir,./quantize_results,align_concat,0,simulate_dpu,1,";
    QuantizeConfig config;
    config.FromString(config_string);
    EXPECT_EQ(config.phase, QuantizePhase::CALIB);
    EXPECT_EQ(config.input_nodes, std::vector<string>({"input_1", "input_2"}));
    EXPECT_EQ(config.output_nodes, std::vector<string>({"relu"}));
    EXPECT_EQ(config.input_shapes, std::vector<string>({"-1,4,4,3"}));
    EXPECT_EQ(config.weight_bit, 8);
    EXPECT_EQ(config.activation_bit, 8);
    EXPECT_EQ(config.method, 0);
    EXPECT_EQ(config.mode, QuantizeMode::WEIGHT);
    EXPECT_EQ(config.calib_iter, 10);
    EXPECT_EQ(config.output_dir, "./quantize_results");
    EXPECT_EQ(config.align_concat, 0);
    EXPECT_EQ(config.simulate_dpu, 1);
  }

  void TestConvertConstantsToVariables() {
    const string config_string =
        "input_nodes,input_op,output_nodes,conv_op,input_shapes,"
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
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");

    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));
    GraphDef converted_graph_def;
    TF_ASSERT_OK(graph_quantizer.ConvertConstantsToVariables(
        original_graph_def, converted_graph_def));
    for (const NodeDef& node : converted_graph_def.node()) {
      EXPECT_NE("Const", node.op());
    }
  }

  void TestCreateOptimizedGraph() {
    const string config_string =
        "input_nodes,input_op,output_nodes,conv_op,input_shapes,"
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
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));
    Output identity_op = Identity(root.WithOpName("expect_removed"), input_op);
    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv_op"), identity_op, weights_op,
                            {1, 1, 1, 1}, "VALID");
    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));
    GraphDef output_graph_def;
    TF_ASSERT_OK(graph_quantizer.CreateOptimizedGraph(original_graph_def,
                                                      output_graph_def));
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(output_graph_def, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("input_op"));
    EXPECT_EQ(1, node_lookup.count("weights_op"));
    EXPECT_EQ(1, node_lookup.count("conv_op"));
    EXPECT_EQ(0, node_lookup.count("expect_removed"));
  }

  void TestPartitionGraph() {
    const string config_string =
        "input_nodes,input_op,output_nodes,conv_op,input_shapes,"
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
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));
    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output identity_op =
        Identity(root.WithOpName("expect_removed_id"), input_op);
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");
    Tensor input_data_2(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data_2, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                        -5.0f, -3.0f, -6.0f});
    Output input_op_2 = Const(root.WithOpName("expect_removed_const_op"),
                              Input::Initializer(input_data_2));
    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));
    GraphDef main_graph_def;
    GraphDef aux_graph_def;
    std::map<string, NodeDef> origin_input_nodes;
    TF_ASSERT_OK(graph_quantizer.PartitionGraph(
        original_graph_def, main_graph_def, aux_graph_def, origin_input_nodes));

    std::map<string, const NodeDef*> main_node_lookup;
    MapNamesToNodes(main_graph_def, &main_node_lookup);
    EXPECT_EQ(3, main_node_lookup.size());
    EXPECT_EQ(1, main_node_lookup.count("input_op"));
    EXPECT_EQ(1, main_node_lookup.count("weights_op"));
    EXPECT_EQ(1, main_node_lookup.count("conv_op"));

    std::map<string, const NodeDef*> aux_node_lookup;
    MapNamesToNodes(aux_graph_def, &aux_node_lookup);
    EXPECT_EQ(2, aux_node_lookup.size());
    EXPECT_EQ(1, aux_node_lookup.count("expect_removed_id"));
    EXPECT_EQ(1, aux_node_lookup.count("expect_removed_const_op"));
  }

  void TestMergeGraph() {
    const string config_string =
        "input_nodes,input_op,output_nodes,conv_op,input_shapes,"
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
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));
    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output identity_op =
        Identity(root.WithOpName("expect_removed_id"), input_op);
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");
    Tensor input_data_2(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data_2, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                        -5.0f, -3.0f, -6.0f});
    Output input_op_2 = Const(root.WithOpName("expect_removed_const_op"),
                              Input::Initializer(input_data_2));
    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));
    GraphDef main_graph_def;
    GraphDef aux_graph_def;
    std::map<string, NodeDef> origin_input_nodes;
    TF_ASSERT_OK(graph_quantizer.PartitionGraph(
        original_graph_def, main_graph_def, aux_graph_def, origin_input_nodes));

    GraphDef output_graph_def;
    TF_ASSERT_OK(graph_quantizer.MergeGraph(
        main_graph_def, aux_graph_def, origin_input_nodes, output_graph_def));

    std::map<string, const NodeDef*> output_node_lookup;
    MapNamesToNodes(output_graph_def, &output_node_lookup);
    EXPECT_EQ(5, output_node_lookup.size());
    EXPECT_EQ(1, output_node_lookup.count("input_op"));
    EXPECT_EQ(1, output_node_lookup.count("weights_op"));
    EXPECT_EQ(1, output_node_lookup.count("conv_op"));
    EXPECT_EQ(1, output_node_lookup.count("expect_removed_id"));
    EXPECT_EQ(1, output_node_lookup.count("expect_removed_const_op"));
  }

  void TestConvertFoldedBatchnorms() {
    const string config_string =
        "input_nodes,input_op,output_nodes,relu_op,input_shapes,"
        "1*1*6*2,weight_bit,8,activation_bit,8,method,0,calib_iter,"
        "10,output_dir," +
        testing::TmpDir() + ",align_concat,0,simulate_dpu,1,";
    QuantizeConfig config;
    config.FromString(config_string);
    GraphQuantizer graph_quantizer(config);

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Tensor eps_data(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&eps_data, {1e-6});
    Output eps_op = Const(root.WithOpName("eps"), Input::Initializer(eps_data));
    Tensor var_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&var_data, {7.5f, 8.5f});
    Output var_op = Const(root.WithOpName("var"), Input::Initializer(var_data));
    Output add_op = Add(root.WithOpName("add"), eps_op, var_op);
    Output rsqrt_op = Rsqrt(root.WithOpName("rsqrt"), add_op);
    Tensor scale_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&scale_data, {1.5f, 2.5f});
    Output scale_op =
        Const(root.WithOpName("scale"), Input::Initializer(scale_data));
    Output mul_op_0 = Mul(root.WithOpName("mul_0"), rsqrt_op, scale_op);

    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input"), Input::Initializer(input_data));
    Output input_fix_op =
        FixNeuron(root.WithOpName("input_aquant"), input_op, 1, 1);

    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights"), Input::Initializer(weights_data));
    Output mul_op_1 = Mul(root.WithOpName("mul_1"), mul_op_0, weights_op);
    Output weights_fix_op =
        FixNeuron(root.WithOpName("weights_wquant"), mul_op_1, 0, 1);
    Output conv_op = Conv2D(root.WithOpName("conv"), input_fix_op,
                            weights_fix_op, {1, 1, 1, 1}, "VALID");

    Tensor mean_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&mean_data, {5.5f, 6.5f});
    Output mean_op =
        Const(root.WithOpName("mean"), Input::Initializer(mean_data));
    Output mul_op_2 = Mul(root.WithOpName("mul_2"), mul_op_0, mean_op);
    Tensor offset_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&offset_data, {3.5f, 4.5f});
    Output offset_op =
        Const(root.WithOpName("offset"), Input::Initializer(offset_data));
    Output sub_op = Sub(root.WithOpName("sub"), offset_op, mul_op_2);
    Output bias_fix_op =
        FixNeuron(root.WithOpName("bias_wquant"), sub_op, 0, 1);

    Output bias_add_op =
        BiasAdd(root.WithOpName("bias_add"), conv_op, bias_fix_op);
    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));

    GraphDef out_graph_def;
    TF_ASSERT_OK(graph_quantizer.ConvertFoldedBatchnorms(original_graph_def,
                                                         out_graph_def));
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(out_graph_def, &node_lookup);
    ASSERT_EQ(8, node_lookup.size());
    EXPECT_EQ(1, node_lookup.count("input"));
    EXPECT_EQ(1, node_lookup.count("weights"));
    EXPECT_EQ(1, node_lookup.count("conv"));
    EXPECT_EQ(1, node_lookup.count("bias_add"));
    EXPECT_EQ(1, node_lookup.count("input_aquant"));
    EXPECT_EQ(1, node_lookup.count("weights_wquant"));
    EXPECT_EQ(1, node_lookup.count("bias_wquant"));
  }

  void TestCreateQuantizeCalibrationGraph() {
    const string config_string =
        "input_nodes,input_op,output_nodes,relu_op,input_shapes,"
        "1*1*6*2,weight_bit,8,activation_bit,8,method,0,calib_iter,"
        "10,output_dir," +
        testing::TmpDir() + ",align_concat,0,simulate_dpu,1,";
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
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));
    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");
    Tensor bias_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&bias_data, {1.5f, 2.5f});
    Output bias_op =
        Const(root.WithOpName("bias_op"), Input::Initializer(bias_data));
    Output bias_add_op =
        BiasAdd(root.WithOpName("bias_add_op"), conv_op, bias_op);
    Output relu_op = Relu(root.WithOpName("relu_op"), bias_add_op);
    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));
    GraphDef output_graph_def;
    TF_ASSERT_OK(graph_quantizer.CreateQuantizeCalibrationGraph(
        original_graph_def, output_graph_def));
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(output_graph_def, &node_lookup);
    EXPECT_EQ(10, node_lookup.size());
    EXPECT_EQ(1, node_lookup.count("input_op"));
    EXPECT_EQ(1, node_lookup.count("weights_op"));
    EXPECT_EQ(1, node_lookup.count("conv_op"));
    EXPECT_EQ(1, node_lookup.count("bias_op"));
    EXPECT_EQ(1, node_lookup.count("bias_add_op"));
    EXPECT_EQ(1, node_lookup.count("relu_op"));
    EXPECT_EQ(1, node_lookup.count("input_op/aquant"));
    EXPECT_EQ(1, node_lookup.count("bias_op/wquant"));
    EXPECT_EQ(1, node_lookup.count("weights_op/wquant"));
    EXPECT_EQ(1, node_lookup.count("relu_op/aquant"));
  }

  void TestCreateQuantizeTrainingGraph() {
    const string config_string =
        "input_nodes,input_op,output_nodes,relu_op,input_shapes,"
        "1*1*6*2,weight_bit,8,activation_bit,8,method,0,calib_iter,"
        "10,output_dir," +
        testing::TmpDir() + ",align_concat,0,simulate_dpu,1,";
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
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));
    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));

    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_op, weights_op,
                            {1, 1, 1, 1}, "VALID");
    Tensor bias_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&bias_data, {1.5f, 2.5f});
    Output bias_op =
        Const(root.WithOpName("bias_op"), Input::Initializer(bias_data));
    Output bias_add_op =
        BiasAdd(root.WithOpName("bias_add_op"), conv_op, bias_op);
    Output relu_op = Relu(root.WithOpName("relu_op"), bias_add_op);
    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));
    GraphDef output_graph_def;
    TF_ASSERT_OK(graph_quantizer.CreateQuantizeTrainingGraph(original_graph_def,
                                                             output_graph_def));
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(output_graph_def, &node_lookup);
    EXPECT_EQ(10, node_lookup.size());
    EXPECT_EQ(1, node_lookup.count("input_op"));
    EXPECT_EQ(1, node_lookup.count("weights_op"));
    EXPECT_EQ(1, node_lookup.count("conv_op"));
    EXPECT_EQ(1, node_lookup.count("bias_op"));
    EXPECT_EQ(1, node_lookup.count("bias_add_op"));
    EXPECT_EQ(1, node_lookup.count("relu_op"));
    EXPECT_EQ(1, node_lookup.count("input_op/aquant"));
    EXPECT_EQ(1, node_lookup.count("bias_op/wquant"));
    EXPECT_EQ(1, node_lookup.count("weights_op/wquant"));
    EXPECT_EQ(1, node_lookup.count("relu_op/aquant"));
  }

  void TestCreateQuantizeEvaluationGraph() {
    const string config_string =
        "input_nodes,input_op,output_nodes,relu_op,input_shapes,"
        "1*1*6*2,weight_bit,8,activation_bit,8,method,0,calib_iter,"
        "10,output_dir," +
        testing::TmpDir() + ",align_concat,0,simulate_dpu,1,";
    QuantizeConfig config;
    config.FromString(config_string);

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
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(original_graph_def, &node_lookup);
    EXPECT_EQ(0, node_lookup["input_op/aquant"]->attr().at("quantize_pos").i());
    EXPECT_EQ(0,
              node_lookup["weights_op/wquant"]->attr().at("quantize_pos").i());
    EXPECT_EQ(0, node_lookup["bias_op/wquant"]->attr().at("quantize_pos").i());
    EXPECT_EQ(0, node_lookup["relu_op/aquant"]->attr().at("quantize_pos").i());

    // svae pos file
    string make_temp_dir =
        "mkdir -p " + io::JoinPath(config.output_dir, "temp");
    system(make_temp_dir.c_str());
    std::vector<string> quantize_ops = {"input_op/aquant", "weights_op/wquant",
                                        "bias_op/wquant", "relu_op/aquant"};
    for (const auto& op_name : quantize_ops) {
      string pos_filename;
      int bit_width = 8, pos = 5;
      pos_filename = config.output_dir + "/temp/" +
                     str_util::StringReplace(op_name, "/", "_", true);
      std::ofstream ofile(pos_filename);
      ofile << op_name << " " << bit_width << " " << pos << std::endl;
      ofile.close();
    }

    // create quantize eval graph
    GraphDef output_graph_def;
    GraphQuantizer graph_quantizer(config);
    TF_ASSERT_OK(graph_quantizer.CreateQuantizeEvaluationGraph(
        original_graph_def, output_graph_def));
    MapNamesToNodes(output_graph_def, &node_lookup);
    EXPECT_EQ(5, node_lookup["input_op/aquant"]->attr().at("quantize_pos").i());
    EXPECT_EQ(5,
              node_lookup["weights_op/wquant"]->attr().at("quantize_pos").i());
    EXPECT_EQ(5, node_lookup["bias_op/wquant"]->attr().at("quantize_pos").i());
    EXPECT_EQ(5, node_lookup["relu_op/aquant"]->attr().at("quantize_pos").i());
  }

  void TestCreateQuantizeDeployGraph() {
    const string config_string =
        "input_nodes,input_op,output_nodes,relu_op/aquant,input_shapes,"
        "1*1*6*2,weight_bit,8,activation_bit,8,method,0,calib_iter,"
        "10,output_dir," +
        testing::TmpDir() + ",align_concat,0,simulate_dpu,1,";
    QuantizeConfig config;
    config.FromString(config_string);

    // build graph
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    Tensor input_data(DT_FLOAT, TensorShape({1, 1, 6, 2}));
    test::FillValues<float>(
        &input_data, {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f, -1.0f, -4.0f, -2.0f,
                      -5.0f, -3.0f, -6.0f});
    Output input_op =
        Const(root.WithOpName("input_op"), Input::Initializer(input_data));
    // bit_width, mehtod, output_dir, quantize_pos
    FixNeuron::Attrs attrs;
    attrs.bit_width_ = 8;
    attrs.method_ = 1;
    attrs.output_dir_ = testing::TmpDir();
    attrs.quantize_pos_ = 2;
    Output input_aquant =
        FixNeuron(root.WithOpName("input_op/aquant"), input_op, 1, 1, attrs);
    Tensor weights_data(DT_FLOAT, TensorShape({1, 2, 2, 2}));
    test::FillValues<float>(&weights_data,
                            {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f});
    Output weights_op =
        Const(root.WithOpName("weights_op"), Input::Initializer(weights_data));
    attrs.quantize_pos_ = 3;
    Output weights_wquant = FixNeuron(root.WithOpName("weights_op/wquant"),
                                      weights_op, 0, 1, attrs);
    Output conv_op = Conv2D(root.WithOpName("conv_op"), input_aquant,
                            weights_wquant, {1, 1, 1, 1}, "VALID");
    Tensor bias_data(DT_FLOAT, TensorShape({2}));
    test::FillValues<float>(&bias_data, {1.5f, 2.5f});
    Output bias_op =
        Const(root.WithOpName("bias_op"), Input::Initializer(bias_data));
    attrs.quantize_pos_ = 4;
    Output bias_wquant =
        FixNeuron(root.WithOpName("bias_op/wquant"), bias_op, 0, 1, attrs);
    Output bias_add_op =
        BiasAdd(root.WithOpName("bias_add_op"), conv_op, bias_wquant);
    Output relu_op = Relu(root.WithOpName("relu_op"), bias_add_op);
    attrs.quantize_pos_ = 5;
    Output relu_aquant =
        FixNeuron(root.WithOpName("relu_op/aquant"), relu_op, 1, 1, attrs);
    GraphDef original_graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&original_graph_def));
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(original_graph_def, &node_lookup);
    EXPECT_EQ(2, node_lookup["input_op/aquant"]->attr().at("quantize_pos").i());
    EXPECT_EQ(3,
              node_lookup["weights_op/wquant"]->attr().at("quantize_pos").i());
    EXPECT_EQ(4, node_lookup["bias_op/wquant"]->attr().at("quantize_pos").i());
    EXPECT_EQ(5, node_lookup["relu_op/aquant"]->attr().at("quantize_pos").i());

    // create create deploy graph
    GraphDef output_graph_def;
    GraphQuantizer graph_quantizer(config);
    TF_ASSERT_OK(graph_quantizer.CreateQuantizeDeployGraph(original_graph_def,
                                                           output_graph_def));

    node_lookup.clear();
    MapNamesToNodes(output_graph_def, &node_lookup);
    EXPECT_EQ(3, node_lookup.size());
    ASSERT_EQ(1, node_lookup.count("input_op"));
    ASSERT_EQ(1, node_lookup.count("conv_op"));
    ASSERT_EQ(1, node_lookup.count("relu_op"));
    EXPECT_EQ(8, node_lookup["conv_op"]->attr().at("ipos").list().i(0));
    EXPECT_EQ(2, node_lookup["conv_op"]->attr().at("ipos").list().i(1));
    EXPECT_EQ(8, node_lookup["conv_op"]->attr().at("wpos").list().i(0));
    EXPECT_EQ(3, node_lookup["conv_op"]->attr().at("wpos").list().i(1));
    EXPECT_EQ(8, node_lookup["conv_op"]->attr().at("bpos").list().i(0));
    EXPECT_EQ(4, node_lookup["conv_op"]->attr().at("bpos").list().i(1));
    EXPECT_EQ(8, node_lookup["conv_op"]->attr().at("opos").list().i(0));
    EXPECT_EQ(5, node_lookup["conv_op"]->attr().at("opos").list().i(1));
    EXPECT_EQ(8, node_lookup["relu_op"]->attr().at("ipos").list().i(0));
    EXPECT_EQ(5, node_lookup["relu_op"]->attr().at("ipos").list().i(1));
    EXPECT_EQ(8, node_lookup["relu_op"]->attr().at("opos").list().i(0));
    EXPECT_EQ(5, node_lookup["relu_op"]->attr().at("opos").list().i(1));

    // test quantized tensor with float tensor
    Tensor bias_deploy;
    bias_deploy.FromProto(node_lookup["conv_op"]->attr().at("bias").tensor());
    test::ExpectTensorNear<float>(bias_data, bias_deploy, 5e-2);
    Tensor w_deploy;
    w_deploy.FromProto(node_lookup["conv_op"]->attr().at("weights").tensor());
    test::ExpectTensorNear<float>(weights_data, w_deploy, 5e-1);
  }
};

TEST_F(GraphQuantizerTest, TestQuantizeConfigFromString) {
  TestQuantizeConfigFromString();
}

TEST_F(GraphQuantizerTest, TestConvertConstantsToVariables) {
  TestConvertConstantsToVariables();
}

TEST_F(GraphQuantizerTest, TestCreateOptimizedGraph) {
  TestCreateOptimizedGraph();
}

TEST_F(GraphQuantizerTest, TestPartitionGraph) { TestPartitionGraph(); }

TEST_F(GraphQuantizerTest, TestMergeGraph) { TestMergeGraph(); }

TEST_F(GraphQuantizerTest, TestCreateQuantizeCalibrationGraph) {
  TestCreateQuantizeCalibrationGraph();
}

TEST_F(GraphQuantizerTest, TestCreateQuantizeTrainingGraph) {
  TestCreateQuantizeTrainingGraph();
}

TEST_F(GraphQuantizerTest, TestCreateQuantizeEvaluationGraph) {
  TestCreateQuantizeEvaluationGraph();
}

TEST_F(GraphQuantizerTest, TestConvertFoldedBatchnorms) {
  TestConvertFoldedBatchnorms();
}

TEST_F(GraphQuantizerTest, TestCreateQuantizeDeployGraph) {
  TestCreateQuantizeDeployGraph();
}

}  // namespace decent_q
}  // namespace tensorflow
