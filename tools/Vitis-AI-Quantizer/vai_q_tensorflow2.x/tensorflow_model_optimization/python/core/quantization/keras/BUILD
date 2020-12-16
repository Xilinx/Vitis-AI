package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
    "//third_party/tensorflow:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":quantize",
        # APIs are not exposed, but still needed for internal imports.
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations",
        "//tensorflow_model_optimization/python/core/quantization/keras/layers",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit",
    ],
)

py_library(
    name = "quant_ops",
    srcs = ["quant_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep1,
        # python:training tensorflow dep2,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_test(
    name = "quant_ops_test",
    size = "small",
    srcs = ["quant_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":quant_ops",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_library(
    name = "quantizers",
    srcs = [
        "quantizers.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quant_ops",
        # six dep1,
        # tensorflow dep1,
    ],
)

py_test(
    name = "quantizers_test",
    srcs = [
        "quantizers_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantizers",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:test_utils",
    ],
)

py_library(
    name = "quantize_config",
    srcs = [
        "quantize_config.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "quantize_registry",
    srcs = [
        "quantize_registry.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # six dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "quantize_layout_transform",
    srcs = [
        "quantize_layout_transform.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # tensorflow dep1,
        # python/keras tensorflow dep2,
    ],
)

py_library(
    name = "quantize_annotate",
    srcs = [
        "quantize_annotate.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_config",
        # tensorflow dep1,
    ],
)

py_test(
    name = "quantize_annotate_test",
    srcs = [
        "quantize_annotate_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_annotate",
        ":quantize_config",
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "quantize_aware_activation",
    srcs = [
        "quantize_aware_activation.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quant_ops",
        # tensorflow dep1,
    ],
)

py_test(
    name = "quantize_aware_activation_test",
    srcs = [
        "quantize_aware_activation_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_aware_activation",
        ":quantizers",
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "quantize_layer",
    srcs = [
        "quantize_layer.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantizers",
        # tensorflow dep1,
    ],
)

py_test(
    name = "quantize_layer_test",
    srcs = [
        "quantize_layer_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_layer",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "quantize_wrapper",
    srcs = [
        "quantize_wrapper.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_aware_activation",
        ":quantize_config",
        ":quantizers",
        # tensorflow dep1,
    ],
)

py_test(
    name = "quantize_wrapper_test",
    srcs = [
        "quantize_wrapper_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_aware_activation",
        ":quantize_wrapper",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
    ],
)

py_library(
    name = "quantize",
    srcs = [
        "quantize.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize_annotate",
        ":quantize_layer",
        ":quantize_wrapper",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_layout_transform",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras/layers:conv_batchnorm",
    ],
)

py_test(
    name = "quantize_test",
    srcs = ["quantize_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":quantize",
        ":quantize_layer",
        ":quantize_wrapper",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:test_utils",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
    ],
)

py_test(
    name = "quantize_integration_test",
    srcs = ["quantize_integration_test.py"],
    python_version = "PY3",
    deps = [
        ":quantize",
        ":utils",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/keras:test_utils",
    ],
)

py_test(
    name = "quantize_models_test",
    size = "large",
    srcs = ["quantize_models_test.py"],
    flaky = True,
    python_version = "PY3",
    shard_count = 10,
    deps = [
        ":quantize",
        ":utils",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/keras:test_utils",
    ],
)

py_test(
    name = "quantize_functional_test",
    size = "large",
    srcs = ["quantize_functional_test.py"],
    python_version = "PY3",
    # To match parallel runs of run_all_keras_modes.
    shard_count = 4,
    deps = [
        ":quantize",
        ":utils",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/keras:test_utils",
        "//tensorflow_model_optimization/python/core/keras/testing:test_utils_mnist",
    ],
)

py_library(
    name = "utils",
    srcs = [
        "utils.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        # tensorflow dep1,
    ],
)
