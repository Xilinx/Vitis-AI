/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "resnet.hpp"

#include "xir/attrs/attrs.hpp"
#include "xir/graph/graph_imp.hpp"
#include "xir/graph/serialize_v2.hpp"
#include "xir/op/op_def.hpp"
#include "xir/op/op_imp.hpp"

namespace xir {

#define FILL_CONV_ATTR(attrs)                                                  \
  attrs->set_attr("kernel", vector<int>{Random(1, 9), Random(1, 9)});          \
  attrs->set_attr("stride", vector<int>{Random(1, 10), Random(1, 10)});        \
  attrs->set_attr("dilation", vector<int>{Random(1, 3), Random(1, 3)});        \
  attrs->set_attr("pad_mode", Random(0, 2));                                   \
  attrs->set_attr("pad", vector<int>{Random(0, 4), Random(0, 4), Random(0, 4), \
                                     Random(0, 4)});                           \
  attrs->set_attr("group", Random(1, 3));                                      \
  attrs->set_attr("shape_infer_mode", Random(0, 2));

#define FILL_POOL_ATTR(attrs)                                                  \
  attrs->set_attr("kernel", vector<int>{Random(1, 9), Random(1, 9)});          \
  attrs->set_attr("stride", vector<int>{Random(1, 10), Random(1, 10)});        \
  attrs->set_attr("dilation", vector<int>{Random(1, 3), Random(1, 3)});        \
  attrs->set_attr("pad_mode", Random(0, 2));                                   \
  attrs->set_attr("pad", vector<int>{Random(0, 4), Random(0, 4), Random(0, 4), \
                                     Random(0, 4)});                           \
  attrs->set_attr("shape_infer_mode", Random(0, 2));                           \
  attrs->set_attr("global", (bool)Random(0, 1));

#define FILL_CONST_ATTR(attrs)                                                 \
  attrs->set_attr("data", vector<char>(Random(128, 1024), Random(0, 128)));

const vector<int> Resnet::LayerType{
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_CONST, LAYER_TYPE_CONST,
    LAYER_TYPE_CONST, LAYER_TYPE_CONST,   LAYER_TYPE_DATA,  LAYER_TYPE_CONV,
    LAYER_TYPE_RELU,  LAYER_TYPE_MAXPOOL, LAYER_TYPE_CONV,  LAYER_TYPE_RELU,
    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,    LAYER_TYPE_CONV,  LAYER_TYPE_CONV,
    LAYER_TYPE_ELEW,  LAYER_TYPE_RELU,    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,
    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,    LAYER_TYPE_CONV,  LAYER_TYPE_ELEW,
    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,
    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,    LAYER_TYPE_ELEW,  LAYER_TYPE_RELU,
    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,
    LAYER_TYPE_CONV,  LAYER_TYPE_CONV,    LAYER_TYPE_ELEW,  LAYER_TYPE_RELU,
    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,
    LAYER_TYPE_CONV,  LAYER_TYPE_ELEW,    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,
    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,
    LAYER_TYPE_ELEW,  LAYER_TYPE_RELU,    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,
    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,    LAYER_TYPE_CONV,  LAYER_TYPE_ELEW,
    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,
    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,    LAYER_TYPE_CONV,  LAYER_TYPE_ELEW,
    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,
    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,    LAYER_TYPE_ELEW,  LAYER_TYPE_RELU,
    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,
    LAYER_TYPE_CONV,  LAYER_TYPE_ELEW,    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,
    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,
    LAYER_TYPE_ELEW,  LAYER_TYPE_RELU,    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,
    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,    LAYER_TYPE_CONV,  LAYER_TYPE_ELEW,
    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,
    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,    LAYER_TYPE_ELEW,  LAYER_TYPE_RELU,
    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,
    LAYER_TYPE_CONV,  LAYER_TYPE_CONV,    LAYER_TYPE_ELEW,  LAYER_TYPE_RELU,
    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,    LAYER_TYPE_CONV,  LAYER_TYPE_RELU,
    LAYER_TYPE_CONV,  LAYER_TYPE_ELEW,    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,
    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,    LAYER_TYPE_RELU,  LAYER_TYPE_CONV,
    LAYER_TYPE_ELEW,  LAYER_TYPE_RELU,
};

const vector<string> Resnet::LayerName{
    "res5c_branch2c_bias",
    "res5c_branch2c_weights",
    "res5c_branch2b_bias",
    "res5c_branch2b_weights",
    "res5c_branch2a_bias",
    "res5c_branch2a_weights",
    "res5b_branch2c_bias",
    "res5b_branch2c_weights",
    "res5b_branch2b_bias",
    "res5b_branch2b_weights",
    "res5b_branch2a_bias",
    "res5b_branch2a_weights",
    "res5a_branch2c_bias",
    "res5a_branch2c_weights",
    "res5a_branch2b_bias",
    "res5a_branch2b_weights",
    "res5a_branch2a_bias",
    "res5a_branch2a_weights",
    "res5a_branch1_bias",
    "res5a_branch1_weights",
    "res4f_branch2c_bias",
    "res4f_branch2c_weights",
    "res4f_branch2b_bias",
    "res4f_branch2b_weights",
    "res4f_branch2a_bias",
    "res4f_branch2a_weights",
    "res4e_branch2c_bias",
    "res4e_branch2c_weights",
    "res4e_branch2b_bias",
    "res4e_branch2b_weights",
    "res4e_branch2a_bias",
    "res4e_branch2a_weights",
    "res4d_branch2c_bias",
    "res4d_branch2c_weights",
    "res4d_branch2b_bias",
    "res4d_branch2b_weights",
    "res4d_branch2a_bias",
    "res4d_branch2a_weights",
    "res4c_branch2c_bias",
    "res4c_branch2c_weights",
    "res4c_branch2b_bias",
    "res4c_branch2b_weights",
    "res4c_branch2a_bias",
    "res4c_branch2a_weights",
    "res4b_branch2c_bias",
    "res4b_branch2c_weights",
    "res4b_branch2b_bias",
    "res4b_branch2b_weights",
    "res4b_branch2a_bias",
    "res4b_branch2a_weights",
    "res4a_branch2c_bias",
    "res4a_branch2c_weights",
    "res4a_branch2b_bias",
    "res4a_branch2b_weights",
    "res4a_branch2a_bias",
    "res4a_branch2a_weights",
    "res4a_branch1_bias",
    "res4a_branch1_weights",
    "res3d_branch2c_bias",
    "res3d_branch2c_weights",
    "res3d_branch2b_bias",
    "res3d_branch2b_weights",
    "res3d_branch2a_bias",
    "res3d_branch2a_weights",
    "res3c_branch2c_bias",
    "res3c_branch2c_weights",
    "res3c_branch2b_bias",
    "res3c_branch2b_weights",
    "res3c_branch2a_bias",
    "res3c_branch2a_weights",
    "res3b_branch2c_bias",
    "res3b_branch2c_weights",
    "res3b_branch2b_bias",
    "res3b_branch2b_weights",
    "res3b_branch2a_bias",
    "res3b_branch2a_weights",
    "res3a_branch2c_bias",
    "res3a_branch2c_weights",
    "res3a_branch2b_bias",
    "res3a_branch2b_weights",
    "res3a_branch2a_bias",
    "res3a_branch2a_weights",
    "res3a_branch1_bias",
    "res3a_branch1_weights",
    "res2c_branch2c_bias",
    "res2c_branch2c_weights",
    "res2c_branch2b_bias",
    "res2c_branch2b_weights",
    "res2c_branch2a_bias",
    "res2c_branch2a_weights",
    "res2b_branch2c_bias",
    "res2b_branch2c_weights",
    "res2b_branch2b_bias",
    "res2b_branch2b_weights",
    "res2b_branch2a_bias",
    "res2b_branch2a_weights",
    "res2a_branch2c_bias",
    "res2a_branch2c_weights",
    "res2a_branch2b_bias",
    "res2a_branch2b_weights",
    "res2a_branch2a_bias",
    "res2a_branch2a_weights",
    "res2a_branch1_bias",
    "res2a_branch1_weights",
    "conv1_bias",
    "conv1_weights",
    "conv1_input",
    "conv1",
    "conv1_relu",
    "pool1",
    "res2a_branch2a",
    "res2a_branch2a_relu",
    "res2a_branch2b",
    "res2a_branch2b_relu",
    "res2a_branch2c",
    "res2a_branch1",
    "res2a",
    "res2a_relu",
    "res2b_branch2a",
    "res2b_branch2a_relu",
    "res2b_branch2b",
    "res2b_branch2b_relu",
    "res2b_branch2c",
    "res2b",
    "res2b_relu",
    "res2c_branch2a",
    "res2c_branch2a_relu",
    "res2c_branch2b",
    "res2c_branch2b_relu",
    "res2c_branch2c",
    "res2c",
    "res2c_relu",
    "res3a_branch2a",
    "res3a_branch2a_relu",
    "res3a_branch2b",
    "res3a_branch2b_relu",
    "res3a_branch2c",
    "res3a_branch1",
    "res3a",
    "res3a_relu",
    "res3b_branch2a",
    "res3b_branch2a_relu",
    "res3b_branch2b",
    "res3b_branch2b_relu",
    "res3b_branch2c",
    "res3b",
    "res3b_relu",
    "res3c_branch2a",
    "res3c_branch2a_relu",
    "res3c_branch2b",
    "res3c_branch2b_relu",
    "res3c_branch2c",
    "res3c",
    "res3c_relu",
    "res3d_branch2a",
    "res3d_branch2a_relu",
    "res3d_branch2b",
    "res3d_branch2b_relu",
    "res3d_branch2c",
    "res3d",
    "res3d_relu",
    "res4a_branch2a",
    "res4a_branch2a_relu",
    "res4a_branch2b",
    "res4a_branch2b_relu",
    "res4a_branch2c",
    "res4a_branch1",
    "res4a",
    "res4a_relu",
    "res4b_branch2a",
    "res4b_branch2a_relu",
    "res4b_branch2b",
    "res4b_branch2b_relu",
    "res4b_branch2c",
    "res4b",
    "res4b_relu",
    "res4c_branch2a",
    "res4c_branch2a_relu",
    "res4c_branch2b",
    "res4c_branch2b_relu",
    "res4c_branch2c",
    "res4c",
    "res4c_relu",
    "res4d_branch2a",
    "res4d_branch2a_relu",
    "res4d_branch2b",
    "res4d_branch2b_relu",
    "res4d_branch2c",
    "res4d",
    "res4d_relu",
    "res4e_branch2a",
    "res4e_branch2a_relu",
    "res4e_branch2b",
    "res4e_branch2b_relu",
    "res4e_branch2c",
    "res4e",
    "res4e_relu",
    "res4f_branch2a",
    "res4f_branch2a_relu",
    "res4f_branch2b",
    "res4f_branch2b_relu",
    "res4f_branch2c",
    "res4f",
    "res4f_relu",
    "res5a_branch2a",
    "res5a_branch2a_relu",
    "res5a_branch2b",
    "res5a_branch2b_relu",
    "res5a_branch2c",
    "res5a_branch1",
    "res5a",
    "res5a_relu",
    "res5b_branch2a",
    "res5b_branch2a_relu",
    "res5b_branch2b",
    "res5b_branch2b_relu",
    "res5b_branch2c",
    "res5b",
    "res5b_relu",
    "res5c_branch2a",
    "res5c_branch2a_relu",
    "res5c_branch2b",
    "res5c_branch2b_relu",
    "res5c_branch2c",
    "res5c",
    "res5c_relu",
};

const vector<string> Resnet::LayerTypeName{
    "const", "data", "conv2d", "relu", "maxpool", "eltwise",
};

const vector<vector<string>> Resnet::LayerInputOpsName{
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {
        "conv1_weights",
        "conv1_bias",
        "conv1_input",
    },
    {
        "conv1",
    },
    {
        "conv1_relu",
    },
    {
        "res2a_branch2a_weights",
        "res2a_branch2a_bias",
        "pool1",
    },
    {
        "res2a_branch2a",
    },
    {
        "res2a_branch2b_weights",
        "res2a_branch2b_bias",
        "res2a_branch2a_relu",
    },
    {
        "res2a_branch2b",
    },
    {
        "res2a_branch2c_weights",
        "res2a_branch2c_bias",
        "res2a_branch2b_relu",
    },
    {
        "res2a_branch1_weights",
        "res2a_branch1_bias",
        "pool1",
    },
    {
        "res2a_branch1",
        "res2a_branch2c",
    },
    {
        "res2a",
    },
    {
        "res2b_branch2a_weights",
        "res2b_branch2a_bias",
        "res2a_relu",
    },
    {
        "res2b_branch2a",
    },
    {
        "res2b_branch2b_weights",
        "res2b_branch2b_bias",
        "res2b_branch2a_relu",
    },
    {
        "res2b_branch2b",
    },
    {
        "res2b_branch2c_weights",
        "res2b_branch2c_bias",
        "res2b_branch2b_relu",
    },
    {
        "res2a_relu",
        "res2b_branch2c",
    },
    {
        "res2b",
    },
    {
        "res2c_branch2a_weights",
        "res2c_branch2a_bias",
        "res2b_relu",
    },
    {
        "res2c_branch2a",
    },
    {
        "res2c_branch2b_weights",
        "res2c_branch2b_bias",
        "res2c_branch2a_relu",
    },
    {
        "res2c_branch2b",
    },
    {
        "res2c_branch2c_weights",
        "res2c_branch2c_bias",
        "res2c_branch2b_relu",
    },
    {
        "res2b_relu",
        "res2c_branch2c",
    },
    {
        "res2c",
    },
    {
        "res3a_branch2a_weights",
        "res3a_branch2a_bias",
        "res2c_relu",
    },
    {
        "res3a_branch2a",
    },
    {
        "res3a_branch2b_weights",
        "res3a_branch2b_bias",
        "res3a_branch2a_relu",
    },
    {
        "res3a_branch2b",
    },
    {
        "res3a_branch2c_weights",
        "res3a_branch2c_bias",
        "res3a_branch2b_relu",
    },
    {
        "res3a_branch1_weights",
        "res3a_branch1_bias",
        "res2c_relu",
    },
    {
        "res3a_branch1",
        "res3a_branch2c",
    },
    {
        "res3a",
    },
    {
        "res3b_branch2a_weights",
        "res3b_branch2a_bias",
        "res3a_relu",
    },
    {
        "res3b_branch2a",
    },
    {
        "res3b_branch2b_weights",
        "res3b_branch2b_bias",
        "res3b_branch2a_relu",
    },
    {
        "res3b_branch2b",
    },
    {
        "res3b_branch2c_weights",
        "res3b_branch2c_bias",
        "res3b_branch2b_relu",
    },
    {
        "res3a_relu",
        "res3b_branch2c",
    },
    {
        "res3b",
    },
    {
        "res3c_branch2a_weights",
        "res3c_branch2a_bias",
        "res3b_relu",
    },
    {
        "res3c_branch2a",
    },
    {
        "res3c_branch2b_weights",
        "res3c_branch2b_bias",
        "res3c_branch2a_relu",
    },
    {
        "res3c_branch2b",
    },
    {
        "res3c_branch2c_weights",
        "res3c_branch2c_bias",
        "res3c_branch2b_relu",
    },
    {
        "res3b_relu",
        "res3c_branch2c",
    },
    {
        "res3c",
    },
    {
        "res3d_branch2a_weights",
        "res3d_branch2a_bias",
        "res3c_relu",
    },
    {
        "res3d_branch2a",
    },
    {
        "res3d_branch2b_weights",
        "res3d_branch2b_bias",
        "res3d_branch2a_relu",
    },
    {
        "res3d_branch2b",
    },
    {
        "res3d_branch2c_weights",
        "res3d_branch2c_bias",
        "res3d_branch2b_relu",
    },
    {
        "res3c_relu",
        "res3d_branch2c",
    },
    {
        "res3d",
    },
    {
        "res4a_branch2a_weights",
        "res4a_branch2a_bias",
        "res3d_relu",
    },
    {
        "res4a_branch2a",
    },
    {
        "res4a_branch2b_weights",
        "res4a_branch2b_bias",
        "res4a_branch2a_relu",
    },
    {
        "res4a_branch2b",
    },
    {
        "res4a_branch2c_weights",
        "res4a_branch2c_bias",
        "res4a_branch2b_relu",
    },
    {
        "res4a_branch1_weights",
        "res4a_branch1_bias",
        "res3d_relu",
    },
    {
        "res4a_branch1",
        "res4a_branch2c",
    },
    {
        "res4a",
    },
    {
        "res4b_branch2a_weights",
        "res4b_branch2a_bias",
        "res4a_relu",
    },
    {
        "res4b_branch2a",
    },
    {
        "res4b_branch2b_weights",
        "res4b_branch2b_bias",
        "res4b_branch2a_relu",
    },
    {
        "res4b_branch2b",
    },
    {
        "res4b_branch2c_weights",
        "res4b_branch2c_bias",
        "res4b_branch2b_relu",
    },
    {
        "res4a_relu",
        "res4b_branch2c",
    },
    {
        "res4b",
    },
    {
        "res4c_branch2a_weights",
        "res4c_branch2a_bias",
        "res4b_relu",
    },
    {
        "res4c_branch2a",
    },
    {
        "res4c_branch2b_weights",
        "res4c_branch2b_bias",
        "res4c_branch2a_relu",
    },
    {
        "res4c_branch2b",
    },
    {
        "res4c_branch2c_weights",
        "res4c_branch2c_bias",
        "res4c_branch2b_relu",
    },
    {
        "res4b_relu",
        "res4c_branch2c",
    },
    {
        "res4c",
    },
    {
        "res4d_branch2a_weights",
        "res4d_branch2a_bias",
        "res4c_relu",
    },
    {
        "res4d_branch2a",
    },
    {
        "res4d_branch2b_weights",
        "res4d_branch2b_bias",
        "res4d_branch2a_relu",
    },
    {
        "res4d_branch2b",
    },
    {
        "res4d_branch2c_weights",
        "res4d_branch2c_bias",
        "res4d_branch2b_relu",
    },
    {
        "res4c_relu",
        "res4d_branch2c",
    },
    {
        "res4d",
    },
    {
        "res4e_branch2a_weights",
        "res4e_branch2a_bias",
        "res4d_relu",
    },
    {
        "res4e_branch2a",
    },
    {
        "res4e_branch2b_weights",
        "res4e_branch2b_bias",
        "res4e_branch2a_relu",
    },
    {
        "res4e_branch2b",
    },
    {
        "res4e_branch2c_weights",
        "res4e_branch2c_bias",
        "res4e_branch2b_relu",
    },
    {
        "res4d_relu",
        "res4e_branch2c",
    },
    {
        "res4e",
    },
    {
        "res4f_branch2a_weights",
        "res4f_branch2a_bias",
        "res4e_relu",
    },
    {
        "res4f_branch2a",
    },
    {
        "res4f_branch2b_weights",
        "res4f_branch2b_bias",
        "res4f_branch2a_relu",
    },
    {
        "res4f_branch2b",
    },
    {
        "res4f_branch2c_weights",
        "res4f_branch2c_bias",
        "res4f_branch2b_relu",
    },
    {
        "res4e_relu",
        "res4f_branch2c",
    },
    {
        "res4f",
    },
    {
        "res5a_branch2a_weights",
        "res5a_branch2a_bias",
        "res4f_relu",
    },
    {
        "res5a_branch2a",
    },
    {
        "res5a_branch2b_weights",
        "res5a_branch2b_bias",
        "res5a_branch2a_relu",
    },
    {
        "res5a_branch2b",
    },
    {
        "res5a_branch2c_weights",
        "res5a_branch2c_bias",
        "res5a_branch2b_relu",
    },
    {
        "res5a_branch1_weights",
        "res5a_branch1_bias",
        "res4f_relu",
    },
    {
        "res5a_branch1",
        "res5a_branch2c",
    },
    {
        "res5a",
    },
    {
        "res5b_branch2a_weights",
        "res5b_branch2a_bias",
        "res5a_relu",
    },
    {
        "res5b_branch2a",
    },
    {
        "res5b_branch2b_weights",
        "res5b_branch2b_bias",
        "res5b_branch2a_relu",
    },
    {
        "res5b_branch2b",
    },
    {
        "res5b_branch2c_weights",
        "res5b_branch2c_bias",
        "res5b_branch2b_relu",
    },
    {
        "res5a_relu",
        "res5b_branch2c",
    },
    {
        "res5b",
    },
    {
        "res5c_branch2a_weights",
        "res5c_branch2a_bias",
        "res5b_relu",
    },
    {
        "res5c_branch2a",
    },
    {
        "res5c_branch2b_weights",
        "res5c_branch2b_bias",
        "res5c_branch2a_relu",
    },
    {
        "res5c_branch2b",
    },
    {
        "res5c_branch2c_weights",
        "res5c_branch2c_bias",
        "res5c_branch2b_relu",
    },
    {
        "res5b_relu",
        "res5c_branch2c",
    },
    {
        "res5c",
    },
};

Resnet::Resnet() {
  for (auto i = 0; i < LAYER_NUM; i++) {
    if (LayerType[i] == LAYER_TYPE_CONV) {
      auto get_op_attrs = [this]() {
        auto attrs = Attrs::create();
        FILL_CONV_ATTR(attrs);
        return attrs;
      };
      op_attrs_.emplace_back(std::move(get_op_attrs));
    } else if (LayerType[i] == LAYER_TYPE_MAXPOOL) {
      auto get_op_attrs = [this]() {
        auto attrs = Attrs::create();
        FILL_POOL_ATTR(attrs);
        return attrs;
      };
      op_attrs_.emplace_back(std::move(get_op_attrs));
    } else {
      auto get_op_attrs = [this]() {
        auto attrs = Attrs::create();
        FILL_CONST_ATTR(attrs);
        return attrs;
      };
      op_attrs_.emplace_back(std::move(get_op_attrs));
    }
  }
}

unique_ptr<GraphImp> Resnet::Build() {
  auto g = make_unique<GraphImp>("resnet");

  for (auto i = 0; i < LAYER_NUM; i++) {
    const auto& op_name = LayerName[i];
    const auto& op_type = LayerTypeName[LayerType[i]];
    auto op_attrs = op_attrs_[i]();
    auto iom = get_iom(g, i);

    auto tensor = Tensor::create(
        op_name + "_tensor",
        vector<int>{Random(1, 7), Random(1, 7), Random(1, 7), Random(1, 7)},
        DataType{"INT8"});
    // tensor->malloc();
    auto op = g->add_op(op_name, op_type, std::move(op_attrs), iom);
    op->replace_output_tensor(std::move(tensor));
  }

  return g;
}

Resnet::InputOpsMap Resnet::get_iom(const unique_ptr<GraphImp>& g,
                                    int layer_id) {
  auto layer_type = LayerType[layer_id];

  switch (layer_type) {
    case LAYER_TYPE_CONST:
      return get_const_iom(g, layer_id);
    case LAYER_TYPE_DATA:
      return get_data_iom(g, layer_id);
    case LAYER_TYPE_CONV:
      return get_conv_iom(g, layer_id);
    case LAYER_TYPE_RELU:
      return get_relu_iom(g, layer_id);
    case LAYER_TYPE_MAXPOOL:
      return get_maxpool_iom(g, layer_id);
    case LAYER_TYPE_ELEW:
      return get_elew_iom(g, layer_id);
    default:
      LOG(ERROR) << "Not support type " << layer_type << endl;
      abort();
  }
}

Resnet::InputOpsMap Resnet::get_const_iom(const unique_ptr<GraphImp>& g,
                                          int layer_id) {
  return InputOpsMap{};
}

Resnet::InputOpsMap Resnet::get_data_iom(const unique_ptr<GraphImp>& g,
                                         int layer_id) {
  return InputOpsMap{};
}

Resnet::InputOpsMap Resnet::get_conv_iom(const unique_ptr<GraphImp>& g,
                                         int layer_id) {
  InputOpsMap iom;

  const auto& input_ops = LayerInputOpsName[layer_id];
  CHECK(input_ops.size() == 3)
      << ", " << LayerName[layer_id] << "(" << layer_id << ")";

  auto* op_w = g->get_op(input_ops[0]);
  CHECK(op_w != nullptr);
  auto* op_b = g->get_op(input_ops[1]);
  CHECK(op_b != nullptr);
  auto* op_i = g->get_op(input_ops[2]);
  CHECK(op_i != nullptr);

  iom["weights"] = vector<Op*>{op_w};
  iom["bias"] = vector<Op*>{op_b};
  iom["input"] = vector<Op*>{op_i};

  return iom;
}

Resnet::InputOpsMap Resnet::get_relu_iom(const unique_ptr<GraphImp>& g,
                                         int layer_id) {
  InputOpsMap iom;

  const auto& input_ops = LayerInputOpsName[layer_id];
  CHECK(input_ops.size() == 1);

  auto* op_i = g->get_op(input_ops[0]);
  CHECK(op_i != nullptr);

  iom["input"] = vector<Op*>{op_i};

  return iom;
}

Resnet::InputOpsMap Resnet::get_elew_iom(const unique_ptr<GraphImp>& g,
                                         int layer_id) {
  InputOpsMap iom;

  const auto& input_ops = LayerInputOpsName[layer_id];
  CHECK(input_ops.size() >= 1);

  vector<Op*> op_vec;
  for (const auto& e : input_ops) {
    auto* cur_op = g->get_op(e);
    CHECK(cur_op != nullptr);
    op_vec.push_back(cur_op);
  }

  iom["input"] = op_vec;

  return iom;
}

Resnet::InputOpsMap Resnet::get_maxpool_iom(const unique_ptr<GraphImp>& g,
                                            int layer_id) {
  InputOpsMap iom;

  const auto& input_ops = LayerInputOpsName[layer_id];
  CHECK(input_ops.size() == 1);

  auto* op_i = g->get_op(input_ops[0]);
  CHECK(op_i != nullptr);

  iom["input"] = vector<Op*>{op_i};

  return iom;
}

}  // namespace xir
