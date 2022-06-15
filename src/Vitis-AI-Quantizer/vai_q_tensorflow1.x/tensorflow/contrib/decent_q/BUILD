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
    "tf_kernel_library",
    "tf_py_test",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_custom_op_py_library",
    "tf_gen_op_wrappers_cc",
)

tf_custom_op_py_library(
    name = "decent_q_py",
    srcs = [
    "__init__.py",
    "python/__init__.py",
    "python/decent_q.py",
    "python/input_fn.py",
    "python/quantize_graph.py",
    "python/utils.py",
    ] + glob(["python/ops/*.py"]),
    dso = [":python/ops/_fix_neuron_ops.so"],
    kernels = [
        ":fix_neuron_ops_kernels",
        ":fix_neuron_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":fix_neuron_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/contrib/decent_q/utils:decent_q_transform_py",
    ],
)

tf_py_test(
    name = "decent_q_test",
    size = "medium",
    srcs = ["python/decent_q_test.py"],
    additional_deps = [
        ":decent_q_py",
        ":fix_neuron_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "quantize_graph_test",
    size = "medium",
    srcs = ["python/quantize_graph_test.py"],
    additional_deps = [
        ":decent_q_py",
        ":fix_neuron_ops",
        "//tensorflow/core:protos_all_py",
    ],
)

tf_py_test(
    name = "py_utils_test",
    size = "small",
    srcs = ["python/utils_test.py"],
    additional_deps = [
        ":decent_q_py",
        "//tensorflow/core:protos_all_py",
    ],
    main = "python/utils_test.py",
)

tf_kernel_library(
    name = "fix_neuron_ops_kernels",
    srcs = [
        "kernels/fix_neuron_ops.cc",
        "kernels/fix_neuron_ops.h",
    ], 
    gpu_srcs = [
        "kernels/fix_neuron_ops_gpu.cu.cc",
        "kernels/fix_neuron_ops.h",
    ],
)

tf_custom_op_library(
    name = "python/ops/_fix_neuron_ops.so",
    srcs = [
        "kernels/fix_neuron_ops.cc",
        "kernels/fix_neuron_ops.h",
        "ops/fix_neuron_ops.cc",
    ],
    gpu_srcs = [
        "kernels/fix_neuron_ops_gpu.cu.cc",
        "kernels/fix_neuron_ops.h",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["fix_neuron_ops"],
)

tf_gen_op_wrapper_py(
    name = "fix_neuron_ops",
    deps = [":fix_neuron_ops_op_lib"],
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
