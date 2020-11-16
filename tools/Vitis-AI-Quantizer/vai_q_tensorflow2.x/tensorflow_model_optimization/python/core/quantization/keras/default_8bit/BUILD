package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "default_8bit",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [],
)

py_library(
    name = "default_8bit_quantizers",
    srcs = [
        "default_8bit_quantizers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
    ],
)

py_test(
    name = "default_8bit_quantizers_test",
    srcs = [
        "default_8bit_quantizers_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":default_8bit_quantizers",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "default_8bit_quantize_configs",
    srcs = [
        "default_8bit_quantize_configs.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_config",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
    ],
)

py_library(
    name = "default_8bit_quantize_registry",
    srcs = [
        "default_8bit_quantize_registry.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_config",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_configs",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantizers",
        "//tensorflow_model_optimization/python/core/quantization/keras/layers:conv_batchnorm",
    ],
)

py_test(
    name = "default_8bit_quantize_registry_test",
    srcs = [
        "default_8bit_quantize_registry_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":default_8bit_quantize_registry",
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)

py_library(
    name = "default_8bit_transforms",
    srcs = [
        "default_8bit_transforms.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_aware_activation",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_layer",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantizers",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_configs",
        "//tensorflow_model_optimization/python/core/quantization/keras/default_8bit:default_8bit_quantize_registry",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:transforms",
        "//tensorflow_model_optimization/python/core/quantization/keras/layers:conv_batchnorm",
    ],
)

py_test(
    name = "default_8bit_transforms_test",
    srcs = [
        "default_8bit_transforms_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":default_8bit_transforms",
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:model_transformer",
        "//tensorflow_model_optimization/python/core/quantization/keras/layers:conv_batchnorm_test_utils",
    ],
)

py_library(
    name = "default_8bit_quantize_layout_transform",
    srcs = [
        "default_8bit_quantize_layout_transform.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":default_8bit_transforms",
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize_layout_transform",
        "//tensorflow_model_optimization/python/core/quantization/keras/graph_transformations:model_transformer",
    ],
)

py_test(
    name = "quantize_numerical_test",
    srcs = ["quantize_numerical_test.py"],
    python_version = "PY3",
    deps = [
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:test_utils",
        "//tensorflow_model_optimization/python/core/quantization/keras:quantize",
        "//tensorflow_model_optimization/python/core/quantization/keras:utils",
    ],
)
