# Copyright 2019 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_cc_test",
    "tf_gen_op_wrappers_cc",
)

cc_library(
    name = "decent_q_transform_utils",
    srcs = [
        "transform_utils.cc",
    ],
    hdrs = [
        "transform_utils.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "decent_q_transform_utils_test",
    size = "small",
    srcs = ["transform_utils_test.cc"],
    deps = [
        ":decent_q_transform_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "decent_q_file_utils",
    srcs = [
        "file_utils.cc",
    ],
    hdrs = [
        "file_utils.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "decent_q_file_utils_test",
    size = "small",
    srcs = ["file_utils_test.cc"],
    deps = [
        ":decent_q_file_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "decent_q_quantize_utils",
    srcs = [
        "quantize_utils.cc",
        "known_patterns.cc",
    ],
    hdrs = [
        "quantize_utils.h",
        "known_patterns.h",
        "graph_quantizer.h",
        "cross_layers_equalization.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":decent_q_transform_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core/util/tensor_bundle",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/contrib/rnn:gru_ops_op_lib",
        "//tensorflow/contrib/rnn:lstm_ops_op_lib",
        "//tensorflow/core/kernels:quantization_utils",
    ],
)

cc_library(
    name = "decent_q_transforms_lib",
    srcs = [
        "cross_layers_equalization.cc",
        "remove_nodes.cc",
        "fold_constants.cc",
        "strip_unused_nodes.cc",
        "flatten_atrous.cc",
        "fold_batch_norms.cc",
        "graph_quantizer.cc",
        "deploy_quantized_graph.cc",
        "check_graph.cc",
        "separate_shared_constants.cc"
    ],
    hdrs = [
        "cross_layers_equalization.h",
        "remove_nodes.h",
        "fold_constants.h",
        "strip_unused_nodes.h",
        "flatten_atrous.h",
        "fold_batch_norms.h",
        "graph_quantizer.h",
        "deploy_quantized_graph.h",
        "check_graph.h",
        "separate_shared_constants.h"
    ],
    copts = tf_copts(),
    deps = [
        ":decent_q_transform_utils",
        ":decent_q_file_utils",
        ":decent_q_quantize_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core/util/tensor_bundle",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/contrib/rnn:gru_ops_op_lib",
        "//tensorflow/contrib/rnn:lstm_ops_op_lib",
        "//tensorflow/core/kernels:quantization_utils",
    ],
)

tf_gen_op_wrappers_cc(
    name = "fix_neuron_ops",
    op_lib_names = ["fix_neuron_ops"],
    pkg = "//tensorflow/contrib/decent_q",
)

tf_cc_test(
    name = "transforms_test",
    size = "medium",
    srcs = [
        "deploy_quantized_graph_test.cc",
        "flatten_atrous_test.cc",
        "fold_batch_norms_test.cc",
        "fold_constants_test.cc",
        "quantize_utils_test.cc",
        "remove_nodes_test.cc",
        "separate_shared_constants_test.cc",
        "sort_by_execution_order_test.cc",
        "strip_unused_nodes_test.cc",
    ],
    deps = [
        ":decent_q_transform_utils",
        ":decent_q_transform_lib",
        ":decent_q_transforms_lib",
        ":fix_neuron_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/contrib/decent_q:fix_neuron_ops_op_lib",
        "//tensorflow/core:bitwise_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:quantization_utils",
        "//tensorflow/core/kernels:quantized_ops",
        "//tensorflow/core/util/tensor_bundle",
    ],
)

cc_library(
    name = "decent_q_transform_lib",
    srcs = [
        "transform_graph.cc",
    ],
    hdrs = [
        "transform_graph.h"
    ],
    copts = tf_copts(),
    deps = [
        ":decent_q_transform_utils",
        ":decent_q_transforms_lib",
    ],
)

tf_cc_test(
    name = "decent_q_transform_graph_test",
    size = "medium",
    srcs = ["transform_graph_test.cc"],
    deps = [
        ":decent_q_transform_lib",
        ":decent_q_transform_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)


tf_cc_test(
    name = "graph_quantizer_test",
    size = "medium",
    srcs = ["graph_quantizer_test.cc"],
    deps = [
        ":decent_q_transform_lib",
        ":decent_q_transform_utils",
        ":decent_q_transforms_lib",
        ":fix_neuron_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/contrib/decent_q:fix_neuron_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)
# decent_q_transform python wrapper
py_library(
    name = "decent_q_transform_py",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:util",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
