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

#include "quantize_utils.h"
#include "replace_softmax.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/platform/str_util.h"

namespace tensorflow {
namespace decent_q {

Status ReplaceSoftmaxWithDPUSoftmax(const GraphDef &input_graph_def,
                                    GraphDef *output_graph_def) {
  GraphDef current_graph_def, processed_graph_def;
  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> nodes_to_ignore;

  current_graph_def = input_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      current_graph_def,  // clang-format off
      {"Softmax"
      },  // clang-format on
      [&inputs_to_rename, &nodes_to_ignore](
          const NodeMatch &match, const std::set<string> &input_nodes,
          const std::set<string> &output_nodes,
          std::vector<NodeDef> *new_nodes) {
        const NodeDef &softmax_node = match.node;
        const string input_node_name = softmax_node.input(0);

        const string softmax_name = softmax_node.name();
        const auto &softmax_shape_list =
            softmax_node.attr().at("_output_shapes").list().shape();
        const auto softmax_shape = softmax_shape_list[0];
        const int col_num = softmax_shape.dim(1).size();
        DLOG_WARNING << "replace Softmax: " << softmax_name
                     << " with DPU Softmax node, with column number: "
                     << col_num;

        // exp poly
        const string exp_poly_namescope = softmax_name + "/exp_poly";

        const string round_namescope = exp_poly_namescope + "/round";

        NodeDef cast_0_node;
        cast_0_node.set_name(round_namescope + "/cast");
        cast_0_node.set_op("Cast");
        SetNodeAttr("DstT", DT_BFLOAT16, &cast_0_node);
        SetNodeAttr("SrcT", DT_FLOAT, &cast_0_node);
        SetNodeAttr("Truncate", false, &cast_0_node);
        AddNodeInput(input_node_name, &cast_0_node);
        new_nodes->push_back(cast_0_node);

        NodeDef mul_0_node;
        mul_0_node.set_name(round_namescope + "/mul");
        mul_0_node.set_op("Mul");
        SetNodeAttr("T", DT_BFLOAT16, &mul_0_node);

        NodeDef rcp_ln2_node;
        rcp_ln2_node.set_op("Const");
        rcp_ln2_node.set_name(round_namescope + "/rcp_ln2");
        SetNodeAttr("dtype", DT_BFLOAT16, &rcp_ln2_node);
        Tensor rcp_ln2_tensor(DT_BFLOAT16, TensorShape({}));
        rcp_ln2_tensor.flat<bfloat16>()(0) = (bfloat16)1.4426950408889634;
        SetNodeTensorAttr<bfloat16>("value", rcp_ln2_tensor, &rcp_ln2_node);
        new_nodes->push_back(rcp_ln2_node);

        AddNodeInput(cast_0_node.name(), &mul_0_node);
        AddNodeInput(rcp_ln2_node.name(), &mul_0_node);
        new_nodes->push_back(mul_0_node);

        NodeDef round_0_node;
        round_0_node.set_name(round_namescope + "/round");
        round_0_node.set_op("Floor");
        SetNodeAttr("T", DT_BFLOAT16, &round_0_node);
        AddNodeInput(mul_0_node.name(), &round_0_node);
        new_nodes->push_back(round_0_node);

        const string modulo_namescope = exp_poly_namescope + "/modulo";

        NodeDef mul_1_node;
        mul_1_node.set_name(modulo_namescope + "/mul");
        mul_1_node.set_op("Mul");
        SetNodeAttr("T", DT_BFLOAT16, &mul_1_node);

        NodeDef ln2_node;
        ln2_node.set_op("Const");
        ln2_node.set_name(modulo_namescope + "/ln2");
        SetNodeAttr("dtype", DT_BFLOAT16, &ln2_node);
        Tensor ln2_tensor(DT_BFLOAT16, TensorShape({}));
        ln2_tensor.flat<bfloat16>()(0) = (bfloat16)0.6931471805599453;
        SetNodeTensorAttr<bfloat16>("value", ln2_tensor, &ln2_node);
        new_nodes->push_back(ln2_node);

        AddNodeInput(round_0_node.name(), &mul_1_node);
        AddNodeInput(ln2_node.name(), &mul_1_node);
        new_nodes->push_back(mul_1_node);

        NodeDef sub_1_node;
        sub_1_node.set_name(modulo_namescope + "/sub");
        sub_1_node.set_op("Sub");
        SetNodeAttr("T", DT_BFLOAT16, &sub_1_node);
        AddNodeInput(cast_0_node.name(), &sub_1_node);
        AddNodeInput(mul_1_node.name(), &sub_1_node);
        new_nodes->push_back(sub_1_node);

        const string poly_approx_namescope =
            exp_poly_namescope + "/poly_approx";

        NodeDef cast_1_node;
        cast_1_node.set_name(poly_approx_namescope + "/cast_1");
        cast_1_node.set_op("Cast");
        SetNodeAttr("DstT", DT_FLOAT, &cast_1_node);
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_1_node);
        SetNodeAttr("Truncate", false, &cast_1_node);
        AddNodeInput(sub_1_node.name(), &cast_1_node);
        new_nodes->push_back(cast_1_node);

        NodeDef alpha_3_node;
        alpha_3_node.set_op("Const");
        alpha_3_node.set_name(poly_approx_namescope + "/alpha_3");
        SetNodeAttr("dtype", DT_BFLOAT16, &alpha_3_node);
        Tensor alpha_3_tensor(DT_BFLOAT16, TensorShape({}));
        alpha_3_tensor.flat<bfloat16>()(0) = (bfloat16)0.21875;
        SetNodeTensorAttr<bfloat16>("value", alpha_3_tensor, &alpha_3_node);
        new_nodes->push_back(alpha_3_node);

        NodeDef cast_2_node;
        cast_2_node.set_name(poly_approx_namescope + "/cast_2");
        cast_2_node.set_op("Cast");
        SetNodeAttr("DstT", DT_FLOAT, &cast_2_node);
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_2_node);
        SetNodeAttr("Truncate", false, &cast_2_node);
        AddNodeInput(alpha_3_node.name(), &cast_2_node);
        new_nodes->push_back(cast_2_node);

        NodeDef mul_2_node;
        mul_2_node.set_name(poly_approx_namescope + "/mul_2");
        mul_2_node.set_op("Mul");
        SetNodeAttr("T", DT_FLOAT, &mul_2_node);
        AddNodeInput(cast_1_node.name(), &mul_2_node);
        AddNodeInput(cast_2_node.name(), &mul_2_node);
        new_nodes->push_back(mul_2_node);

        NodeDef alpha_2_node;
        alpha_2_node.set_op("Const");
        alpha_2_node.set_name(poly_approx_namescope + "/alpha_2");
        SetNodeAttr("dtype", DT_BFLOAT16, &alpha_2_node);
        Tensor alpha_2_tensor(DT_BFLOAT16, TensorShape({}));
        alpha_2_tensor.flat<bfloat16>()(0) = (bfloat16)0.486328125;
        SetNodeTensorAttr<bfloat16>("value", alpha_2_tensor, &alpha_2_node);
        new_nodes->push_back(alpha_2_node);

        NodeDef cast_3_node;
        cast_3_node.set_name(poly_approx_namescope + "/cast_3");
        cast_3_node.set_op("Cast");
        SetNodeAttr("DstT", DT_FLOAT, &cast_3_node);
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_3_node);
        SetNodeAttr("Truncate", false, &cast_3_node);
        AddNodeInput(alpha_2_node.name(), &cast_3_node);
        new_nodes->push_back(cast_3_node);

        NodeDef add_0_node;
        add_0_node.set_name(poly_approx_namescope + "/add");
        add_0_node.set_op("AddV2");
        SetNodeAttr("T", DT_FLOAT, &add_0_node);
        AddNodeInput(mul_2_node.name(), &add_0_node);
        AddNodeInput(cast_3_node.name(), &add_0_node);
        new_nodes->push_back(add_0_node);

        NodeDef cast_4_node;
        cast_4_node.set_name(poly_approx_namescope + "/cast_4");
        cast_4_node.set_op("Cast");
        SetNodeAttr("SrcT", DT_FLOAT, &cast_4_node);
        SetNodeAttr("DstT", DT_BFLOAT16, &cast_4_node);
        SetNodeAttr("Truncate", false, &cast_4_node);
        AddNodeInput(add_0_node.name(), &cast_4_node);
        new_nodes->push_back(cast_4_node);

        NodeDef cast_5_node;
        cast_5_node.set_name(poly_approx_namescope + "/cast_5");
        cast_5_node.set_op("Cast");
        SetNodeAttr("DstT", DT_FLOAT, &cast_5_node);
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_5_node);
        SetNodeAttr("Truncate", false, &cast_5_node);
        AddNodeInput(cast_4_node.name(), &cast_5_node);
        new_nodes->push_back(cast_5_node);

        NodeDef mul_3_node;
        mul_3_node.set_name(poly_approx_namescope + "/mul_3");
        mul_3_node.set_op("Mul");
        SetNodeAttr("T", DT_FLOAT, &mul_3_node);
        AddNodeInput(cast_1_node.name(), &mul_3_node);
        AddNodeInput(cast_5_node.name(), &mul_3_node);
        new_nodes->push_back(mul_3_node);

        NodeDef alpha_1_node;
        alpha_1_node.set_op("Const");
        alpha_1_node.set_name(poly_approx_namescope + "/alpha_1");
        SetNodeAttr("dtype", DT_BFLOAT16, &alpha_1_node);
        Tensor alpha_1_tensor(DT_BFLOAT16, TensorShape({}));
        alpha_1_tensor.flat<bfloat16>()(0) = (bfloat16)1.;
        SetNodeTensorAttr<bfloat16>("value", alpha_1_tensor, &alpha_1_node);
        new_nodes->push_back(alpha_1_node);

        NodeDef cast_6_node;
        cast_6_node.set_name(poly_approx_namescope + "/cast_6");
        cast_6_node.set_op("Cast");
        SetNodeAttr("DstT", DT_FLOAT, &cast_6_node);
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_6_node);
        SetNodeAttr("Truncate", false, &cast_6_node);
        AddNodeInput(alpha_1_node.name(), &cast_6_node);
        new_nodes->push_back(cast_6_node);

        NodeDef add_1_node;
        add_1_node.set_name(poly_approx_namescope + "/add_1");
        add_1_node.set_op("AddV2");
        SetNodeAttr("T", DT_FLOAT, &add_1_node);
        AddNodeInput(mul_3_node.name(), &add_1_node);
        AddNodeInput(cast_6_node.name(), &add_1_node);
        new_nodes->push_back(add_1_node);

        NodeDef cast_7_node;
        cast_7_node.set_name(poly_approx_namescope + "/cast_7");
        cast_7_node.set_op("Cast");
        SetNodeAttr("SrcT", DT_FLOAT, &cast_7_node);
        SetNodeAttr("DstT", DT_BFLOAT16, &cast_7_node);
        SetNodeAttr("Truncate", false, &cast_7_node);
        AddNodeInput(add_1_node.name(), &cast_7_node);
        new_nodes->push_back(cast_7_node);

        NodeDef cast_8_node;
        cast_8_node.set_name(poly_approx_namescope + "/cast_8");
        cast_8_node.set_op("Cast");
        SetNodeAttr("DstT", DT_FLOAT, &cast_8_node);
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_8_node);
        SetNodeAttr("Truncate", false, &cast_8_node);
        AddNodeInput(cast_7_node.name(), &cast_8_node);
        new_nodes->push_back(cast_8_node);

        NodeDef mul_4_node;
        mul_4_node.set_name(poly_approx_namescope + "/mul_4");
        mul_4_node.set_op("Mul");
        SetNodeAttr("T", DT_FLOAT, &mul_4_node);
        AddNodeInput(cast_1_node.name(), &mul_4_node);
        AddNodeInput(cast_8_node.name(), &mul_4_node);
        new_nodes->push_back(mul_4_node);

        NodeDef alpha_0_node;
        alpha_0_node.set_op("Const");
        alpha_0_node.set_name(poly_approx_namescope + "/alpha_0");
        SetNodeAttr("dtype", DT_BFLOAT16, &alpha_0_node);
        Tensor alpha_0_tensor(DT_BFLOAT16, TensorShape({}));
        alpha_0_tensor.flat<bfloat16>()(0) = (bfloat16)1.;
        SetNodeTensorAttr<bfloat16>("value", alpha_0_tensor, &alpha_0_node);
        new_nodes->push_back(alpha_0_node);

        NodeDef cast_9_node;
        cast_9_node.set_name(poly_approx_namescope + "/cast_9");
        cast_9_node.set_op("Cast");
        SetNodeAttr("DstT", DT_FLOAT, &cast_9_node);
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_9_node);
        SetNodeAttr("Truncate", false, &cast_9_node);
        AddNodeInput(alpha_0_node.name(), &cast_9_node);
        new_nodes->push_back(cast_9_node);

        NodeDef add_2_node;
        add_2_node.set_name(poly_approx_namescope + "/add_2");
        add_2_node.set_op("AddV2");
        SetNodeAttr("T", DT_FLOAT, &add_2_node);
        AddNodeInput(mul_4_node.name(), &add_2_node);
        AddNodeInput(cast_9_node.name(), &add_2_node);
        new_nodes->push_back(add_2_node);

        NodeDef cast_10_node;
        cast_10_node.set_name(poly_approx_namescope + "/cast_10");
        cast_10_node.set_op("Cast");
        SetNodeAttr("SrcT", DT_FLOAT, &cast_10_node);
        SetNodeAttr("DstT", DT_BFLOAT16, &cast_10_node);
        SetNodeAttr("Truncate", false, &cast_10_node);
        AddNodeInput(add_2_node.name(), &cast_10_node);
        new_nodes->push_back(cast_10_node);

        const string pow_namescope = exp_poly_namescope + "/pow";

        NodeDef pow_x_node;
        pow_x_node.set_op("Const");
        pow_x_node.set_name(pow_namescope + "/pow/x");
        SetNodeAttr("dtype", DT_BFLOAT16, &pow_x_node);
        Tensor pow_x_tensor(DT_BFLOAT16, TensorShape({}));
        pow_x_tensor.flat<bfloat16>()(0) = (bfloat16)2.0;
        SetNodeTensorAttr<bfloat16>("value", pow_x_tensor, &pow_x_node);
        new_nodes->push_back(pow_x_node);

        NodeDef pow_node;
        pow_node.set_name(pow_namescope + "/pow");
        pow_node.set_op("Pow");
        SetNodeAttr("T", DT_BFLOAT16, &pow_node);
        AddNodeInput(pow_x_node.name(), &pow_node);
        AddNodeInput(round_0_node.name(), &pow_node);
        new_nodes->push_back(pow_node);

        NodeDef exp_x_node;
        exp_x_node.set_name(exp_poly_namescope + "/exp_x");
        exp_x_node.set_op("Mul");
        SetNodeAttr("T", DT_BFLOAT16, &exp_x_node);
        AddNodeInput(pow_node.name(), &exp_x_node);
        AddNodeInput(cast_10_node.name(), &exp_x_node);
        new_nodes->push_back(exp_x_node);

        const string exp_sum_namescope = softmax_name + "/exp_sum";

        // // // softmax exp norm reduce sum
        // NodeDef sum_axis_node;
        // sum_axis_node.set_op("Const");
        // sum_axis_node.set_name(exp_sum_namescope +
        // "/sum/reduction_indices"); SetNodeAttr("dtype", DT_INT32,
        // &sum_axis_node); Tensor axis_tensor(DT_INT32, TensorShape({}));
        // axis_tensor.flat<int>()(0) = -1;
        // SetNodeTensorAttr<int>("value", axis_tensor, &sum_axis_node);
        // new_nodes->push_back(sum_axis_node);

        // NodeDef sum_node;
        // sum_node.set_name(exp_sum_namescope + "/sum");
        // sum_node.set_op("Sum");
        // SetNodeAttr("keep_dims", true, &sum_node);
        // SetNodeAttr("T", DT_BFLOAT16, &sum_node);
        // SetNodeAttr("Tidx", DT_INT32, &sum_node);
        // AddNodeInput(exp_x_node.name(), &sum_node);
        // AddNodeInput(sum_axis_node.name(), &sum_node);
        // new_nodes->push_back(sum_node);
        ////////////////////////////////////////////////////////////
        // softmax exp parallel vector sum
        NodeDef cast_vsum_in;
        cast_vsum_in.set_name(exp_sum_namescope + "/cast_input");
        cast_vsum_in.set_op("Cast");
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_vsum_in);
        SetNodeAttr("DstT", DT_FLOAT, &cast_vsum_in);
        SetNodeAttr("Truncate", false, &cast_vsum_in);
        AddNodeInput(exp_x_node.name(), &cast_vsum_in);
        new_nodes->push_back(cast_vsum_in);

        NodeDef in_transpose_perm;
        in_transpose_perm.set_op("Const");
        in_transpose_perm.set_name(exp_sum_namescope + "/in_transpose/perm");
        SetNodeAttr("dtype", DT_INT32, &in_transpose_perm);
        Tensor perm_tensor(DT_INT32, TensorShape({2}));
        auto perm_tensor_flattern = perm_tensor.flat<int>();
        perm_tensor_flattern(0) = 1;
        perm_tensor_flattern(1) = 0;
        SetNodeTensorAttr<int>("value", perm_tensor, &in_transpose_perm);
        new_nodes->push_back(in_transpose_perm);

        NodeDef in_transpose_node;
        in_transpose_node.set_name(exp_sum_namescope + "/in_transpose");
        in_transpose_node.set_op("Transpose");
        SetNodeAttr("T", DT_FLOAT, &in_transpose_node);
        SetNodeAttr("Tperm", DT_INT32, &in_transpose_node);
        AddNodeInput(cast_vsum_in.name(), &in_transpose_node);
        AddNodeInput(in_transpose_perm.name(), &in_transpose_node);
        new_nodes->push_back(in_transpose_node);

        const int parallel_num = 16;
        // const int col_num = 1001;
        NodeDef segment_ids;
        segment_ids.set_op("Const");
        segment_ids.set_name(exp_sum_namescope + "/vector_sum/segment_ids");
        SetNodeAttr("dtype", DT_INT32, &segment_ids);
        Tensor seg_ids_tensor(DT_INT32, TensorShape({col_num}));
        auto seg_ids_tensor_flattern = seg_ids_tensor.flat<int>();
        for (int i = 0; i < col_num; ++i) {
          seg_ids_tensor_flattern(i) = i % parallel_num;
        }
        SetNodeTensorAttr<int>("value", seg_ids_tensor, &segment_ids);
        new_nodes->push_back(segment_ids);

        NodeDef num_seg_node;
        num_seg_node.set_op("Const");
        num_seg_node.set_name(exp_sum_namescope + "/vector_sum/num_segments");
        SetNodeAttr("dtype", DT_INT32, &num_seg_node);
        Tensor num_seg_tensor(DT_INT32, TensorShape({}));
        num_seg_tensor.flat<int>()(0) = col_num;
        SetNodeTensorAttr<int>("value", num_seg_tensor, &num_seg_node);
        new_nodes->push_back(num_seg_node);

        NodeDef vec_sum_node;
        vec_sum_node.set_name(exp_sum_namescope + "/vector_sum");
        vec_sum_node.set_op("UnsortedSegmentSum");
        SetNodeAttr("T", DT_FLOAT, &vec_sum_node);
        SetNodeAttr("Tindices", DT_INT32, &vec_sum_node);
        SetNodeAttr("Tnumsegments", DT_INT32, &vec_sum_node);
        AddNodeInput(in_transpose_node.name(), &vec_sum_node);
        AddNodeInput(segment_ids.name(), &vec_sum_node);
        AddNodeInput(num_seg_node.name(), &vec_sum_node);
        new_nodes->push_back(vec_sum_node);

        NodeDef out_transpose_perm;
        out_transpose_perm.set_op("Const");
        out_transpose_perm.set_name(exp_sum_namescope + "/out_transpose/perm");
        SetNodeAttr("dtype", DT_INT32, &out_transpose_perm);
        SetNodeTensorAttr<int>("value", perm_tensor, &out_transpose_perm);
        new_nodes->push_back(out_transpose_perm);

        NodeDef out_transpose_node;
        out_transpose_node.set_name(exp_sum_namescope + "/out_transpose");
        out_transpose_node.set_op("Transpose");
        SetNodeAttr("T", DT_FLOAT, &out_transpose_node);
        SetNodeAttr("Tperm", DT_INT32, &out_transpose_node);
        AddNodeInput(vec_sum_node.name(), &out_transpose_node);
        AddNodeInput(out_transpose_perm.name(), &out_transpose_node);
        new_nodes->push_back(out_transpose_node);

        NodeDef cast_vsum_out_16;
        cast_vsum_out_16.set_name(exp_sum_namescope + "/cast_out_16");
        cast_vsum_out_16.set_op("Cast");
        SetNodeAttr("SrcT", DT_FLOAT, &cast_vsum_out_16);
        SetNodeAttr("DstT", DT_BFLOAT16, &cast_vsum_out_16);
        SetNodeAttr("Truncate", false, &cast_vsum_out_16);
        AddNodeInput(out_transpose_node.name(), &cast_vsum_out_16);
        new_nodes->push_back(cast_vsum_out_16);

        NodeDef cast_vsum_out_32;
        cast_vsum_out_32.set_name(exp_sum_namescope + "/cast_out_32");
        cast_vsum_out_32.set_op("Cast");
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_vsum_out_32);
        SetNodeAttr("DstT", DT_FLOAT, &cast_vsum_out_32);
        SetNodeAttr("Truncate", false, &cast_vsum_out_32);
        AddNodeInput(cast_vsum_out_16.name(), &cast_vsum_out_32);
        new_nodes->push_back(cast_vsum_out_32);

        NodeDef sum_axis_node;
        sum_axis_node.set_op("Const");
        sum_axis_node.set_name(exp_sum_namescope + "/sum/reduction_indices");
        SetNodeAttr("dtype", DT_INT32, &sum_axis_node);
        Tensor axis_tensor(DT_INT32, TensorShape({}));
        axis_tensor.flat<int>()(0) = -1;
        SetNodeTensorAttr<int>("value", axis_tensor, &sum_axis_node);
        new_nodes->push_back(sum_axis_node);

        NodeDef sum_node;
        sum_node.set_name(exp_sum_namescope + "/sum");
        sum_node.set_op("Sum");
        SetNodeAttr("keep_dims", true, &sum_node);
        SetNodeAttr("T", DT_FLOAT, &sum_node);
        SetNodeAttr("Tidx", DT_INT32, &sum_node);
        AddNodeInput(cast_vsum_out_32.name(), &sum_node);
        // AddNodeInput(cast_vsum_in.name(), &sum_node);
        AddNodeInput(sum_axis_node.name(), &sum_node);
        new_nodes->push_back(sum_node);

        NodeDef cast_sum_out_16;
        cast_sum_out_16.set_name(exp_sum_namescope + "/cast_reduce_sum_out_16");
        cast_sum_out_16.set_op("Cast");
        SetNodeAttr("SrcT", DT_FLOAT, &cast_sum_out_16);
        SetNodeAttr("DstT", DT_BFLOAT16, &cast_sum_out_16);
        SetNodeAttr("Truncate", false, &cast_sum_out_16);
        AddNodeInput(sum_node.name(), &cast_sum_out_16);
        new_nodes->push_back(cast_sum_out_16);

        // softmax reciprocal
        const string reciprocal_namescope = softmax_name + "/reciprocal";

        NodeDef to_int_node;
        to_int_node.set_name(reciprocal_namescope + "/to_int");
        to_int_node.set_op("Bitcast");
        SetNodeAttr("T", DT_BFLOAT16, &to_int_node);
        SetNodeAttr("type", DT_INT16, &to_int_node);
        AddNodeInput(cast_sum_out_16.name(), &to_int_node);
        new_nodes->push_back(to_int_node);

        NodeDef complement_node;
        complement_node.set_op("Const");
        complement_node.set_name(reciprocal_namescope + "/complement");
        SetNodeAttr("dtype", DT_INT16, &complement_node);
        Tensor complement_tensor(DT_INT16, TensorShape({}));
        complement_tensor.flat<int16>()(0) = 0x7eb5;
        SetNodeTensorAttr<int16>("value", complement_tensor, &complement_node);
        new_nodes->push_back(complement_node);

        NodeDef sub_2_node;
        sub_2_node.set_name(reciprocal_namescope + "/sub_2");
        sub_2_node.set_op("Sub");
        SetNodeAttr("T", DT_INT16, &sub_2_node);
        AddNodeInput(complement_node.name(), &sub_2_node);
        AddNodeInput(to_int_node.name(), &sub_2_node);
        new_nodes->push_back(sub_2_node);

        NodeDef y0_node;
        y0_node.set_name(reciprocal_namescope + "/y0");
        y0_node.set_op("Bitcast");
        SetNodeAttr("T", DT_INT16, &y0_node);
        SetNodeAttr("type", DT_BFLOAT16, &y0_node);
        AddNodeInput(sub_2_node.name(), &y0_node);
        new_nodes->push_back(y0_node);

        NodeDef newton_k1;
        newton_k1.set_op("Const");
        newton_k1.set_name(reciprocal_namescope + "/mul_6/k1");
        SetNodeAttr("dtype", DT_BFLOAT16, &newton_k1);
        Tensor newton_k1_tensor(DT_BFLOAT16, TensorShape({}));
        newton_k1_tensor.flat<bfloat16>()(0) = (bfloat16)1.9395974;
        SetNodeTensorAttr<float>("value", newton_k1_tensor, &newton_k1);
        new_nodes->push_back(newton_k1);

        NodeDef mul_6_node;
        mul_6_node.set_name(reciprocal_namescope + "/mul_6");
        mul_6_node.set_op("Mul");
        SetNodeAttr("T", DT_BFLOAT16, &mul_6_node);
        AddNodeInput(y0_node.name(), &mul_6_node);
        AddNodeInput(newton_k1.name(), &mul_6_node);
        new_nodes->push_back(mul_6_node);

        NodeDef cast_11_node;
        cast_11_node.set_name(reciprocal_namescope + "/cast_11");
        cast_11_node.set_op("Cast");
        SetNodeAttr("DstT", DT_FLOAT, &cast_11_node);
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_11_node);
        SetNodeAttr("Truncate", false, &cast_11_node);
        AddNodeInput(cast_sum_out_16.name(), &cast_11_node);
        new_nodes->push_back(cast_11_node);

        NodeDef cast_12_node;
        cast_12_node.set_name(reciprocal_namescope + "/cast_12");
        cast_12_node.set_op("Cast");
        SetNodeAttr("DstT", DT_FLOAT, &cast_12_node);
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_12_node);
        SetNodeAttr("Truncate", false, &cast_12_node);
        AddNodeInput(y0_node.name(), &cast_12_node);
        new_nodes->push_back(cast_12_node);

        NodeDef mul_7_node;
        mul_7_node.set_name(reciprocal_namescope + "/mul_7");
        mul_7_node.set_op("Mul");
        SetNodeAttr("T", DT_FLOAT, &mul_7_node);
        AddNodeInput(cast_12_node.name(), &mul_7_node);
        AddNodeInput(cast_11_node.name(), &mul_7_node);
        new_nodes->push_back(mul_7_node);

        NodeDef newton_k2;
        newton_k2.set_op("Const");
        newton_k2.set_name(reciprocal_namescope + "/sub_3/k2");
        SetNodeAttr("dtype", DT_BFLOAT16, &newton_k2);
        Tensor newton_alpha_2_tensor(DT_BFLOAT16, TensorShape({}));
        newton_alpha_2_tensor.flat<bfloat16>()(0) = (bfloat16)1.436142;
        SetNodeTensorAttr<bfloat16>("value", newton_alpha_2_tensor, &newton_k2);
        new_nodes->push_back(newton_k2);

        NodeDef cast_13_node;
        cast_13_node.set_name(reciprocal_namescope + "/cast_13");
        cast_13_node.set_op("Cast");
        SetNodeAttr("DstT", DT_FLOAT, &cast_13_node);
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_13_node);
        SetNodeAttr("Truncate", false, &cast_13_node);
        AddNodeInput(newton_k2.name(), &cast_13_node);
        new_nodes->push_back(cast_13_node);

        NodeDef sub_3_node;
        sub_3_node.set_name(reciprocal_namescope + "/sub_3");
        sub_3_node.set_op("Sub");
        SetNodeAttr("T", DT_FLOAT, &sub_3_node);
        AddNodeInput(cast_13_node.name(), &sub_3_node);
        AddNodeInput(mul_7_node.name(), &sub_3_node);
        new_nodes->push_back(sub_3_node);

        NodeDef cast_14_node;
        cast_14_node.set_name(reciprocal_namescope + "/cast_14");
        cast_14_node.set_op("Cast");
        SetNodeAttr("SrcT", DT_FLOAT, &cast_14_node);
        SetNodeAttr("DstT", DT_BFLOAT16, &cast_14_node);
        SetNodeAttr("Truncate", false, &cast_14_node);
        AddNodeInput(sub_3_node.name(), &cast_14_node);
        new_nodes->push_back(cast_14_node);

        NodeDef y1_node;
        y1_node.set_name(reciprocal_namescope + "/y1");
        y1_node.set_op("Mul");
        SetNodeAttr("T", DT_BFLOAT16, &y1_node);
        AddNodeInput(mul_6_node.name(), &y1_node);
        AddNodeInput(cast_14_node.name(), &y1_node);
        new_nodes->push_back(y1_node);

        NodeDef cast_y1_node;
        cast_y1_node.set_name(reciprocal_namescope + "/cast_15");
        cast_y1_node.set_op("Cast");
        SetNodeAttr("DstT", DT_FLOAT, &cast_y1_node);
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_y1_node);
        SetNodeAttr("Truncate", false, &cast_y1_node);
        AddNodeInput(y1_node.name(), &cast_y1_node);
        new_nodes->push_back(cast_y1_node);

        NodeDef mul_9_node;
        mul_9_node.set_name(reciprocal_namescope + "/mul_9");
        mul_9_node.set_op("Mul");
        SetNodeAttr("T", DT_FLOAT, &mul_9_node);
        AddNodeInput(cast_y1_node.name(), &mul_9_node);
        AddNodeInput(cast_11_node.name(), &mul_9_node);
        new_nodes->push_back(mul_9_node);

        NodeDef newton_ones;
        newton_ones.set_op("Const");
        newton_ones.set_name(reciprocal_namescope + "/add/ones");
        SetNodeAttr("dtype", DT_BFLOAT16, &newton_ones);
        Tensor newton_ones_tensor(DT_BFLOAT16, TensorShape({}));
        newton_ones_tensor.flat<bfloat16>()(0) = (bfloat16)1.0;
        SetNodeTensorAttr<bfloat16>("value", newton_ones_tensor, &newton_ones);
        new_nodes->push_back(newton_ones);

        NodeDef cast_16_node;
        cast_16_node.set_name(reciprocal_namescope + "/cast_16");
        cast_16_node.set_op("Cast");
        SetNodeAttr("DstT", DT_FLOAT, &cast_16_node);
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_16_node);
        SetNodeAttr("Truncate", false, &cast_16_node);
        AddNodeInput(newton_ones.name(), &cast_16_node);
        new_nodes->push_back(cast_16_node);

        NodeDef sub_4_node;
        sub_4_node.set_name(reciprocal_namescope + "/sub_4");
        sub_4_node.set_op("Sub");
        SetNodeAttr("T", DT_FLOAT, &sub_4_node);
        AddNodeInput(cast_16_node.name(), &sub_4_node);
        AddNodeInput(mul_9_node.name(), &sub_4_node);
        new_nodes->push_back(sub_4_node);

        NodeDef cast_17_node;
        cast_17_node.set_name(reciprocal_namescope + "/cast_17");
        cast_17_node.set_op("Cast");
        SetNodeAttr("SrcT", DT_FLOAT, &cast_17_node);
        SetNodeAttr("DstT", DT_BFLOAT16, &cast_17_node);
        SetNodeAttr("Truncate", false, &cast_17_node);
        AddNodeInput(sub_4_node.name(), &cast_17_node);
        new_nodes->push_back(cast_17_node);

        NodeDef cast_18_node;
        cast_18_node.set_name(reciprocal_namescope + "/cast_18");
        cast_18_node.set_op("Cast");
        SetNodeAttr("DstT", DT_FLOAT, &cast_18_node);
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_18_node);
        SetNodeAttr("Truncate", false, &cast_18_node);
        AddNodeInput(cast_17_node.name(), &cast_18_node);
        new_nodes->push_back(cast_18_node);

        NodeDef mul_10_node;
        mul_10_node.set_name(reciprocal_namescope + "/mul_10");
        mul_10_node.set_op("Mul");
        SetNodeAttr("T", DT_FLOAT, &mul_10_node);
        AddNodeInput(cast_18_node.name(), &mul_10_node);
        AddNodeInput(cast_y1_node.name(), &mul_10_node);
        new_nodes->push_back(mul_10_node);

        NodeDef add_4_node;
        add_4_node.set_name(reciprocal_namescope + "/add_4");
        add_4_node.set_op("AddV2");
        SetNodeAttr("T", DT_FLOAT, &add_4_node);
        AddNodeInput(mul_10_node.name(), &add_4_node);
        AddNodeInput(cast_y1_node.name(), &add_4_node);
        new_nodes->push_back(add_4_node);

        NodeDef cast_19_node;
        cast_19_node.set_name(reciprocal_namescope + "/cast_19");
        cast_19_node.set_op("Cast");
        SetNodeAttr("SrcT", DT_FLOAT, &cast_19_node);
        SetNodeAttr("DstT", DT_BFLOAT16, &cast_19_node);
        SetNodeAttr("Truncate", false, &cast_19_node);
        AddNodeInput(add_4_node.name(), &cast_19_node);
        new_nodes->push_back(cast_19_node);

        NodeDef y2_node;
        y2_node.set_name(softmax_name + "/y2");
        y2_node.set_op("Mul");
        SetNodeAttr("T", DT_BFLOAT16, &y2_node);
        AddNodeInput(exp_x_node.name(), &y2_node);
        AddNodeInput(cast_19_node.name(), &y2_node);
        new_nodes->push_back(y2_node);

        NodeDef cast_y2_node;
        cast_y2_node.set_name(softmax_name + "/output_cast");
        cast_y2_node.set_op("Cast");
        SetNodeAttr("DstT", DT_FLOAT, &cast_y2_node);
        SetNodeAttr("SrcT", DT_BFLOAT16, &cast_y2_node);
        SetNodeAttr("Truncate", false, &cast_y2_node);
        AddNodeInput(y2_node.name(), &cast_y2_node);
        new_nodes->push_back(cast_y2_node);

        inputs_to_rename[softmax_name] = cast_y2_node.name();
        return Status::OK();
      },
      {true}, &processed_graph_def));
  TF_RETURN_IF_ERROR(RenameNodeInputs(processed_graph_def, inputs_to_rename,
                                      nodes_to_ignore, &current_graph_def));
  *output_graph_def = current_graph_def;
  return Status::OK();
}

}  // namespace decent_q
}  // namespace tensorflow
