package(default_visibility = [
    "//tensorflow_model_optimization/python/core/internal/tensor_encoding:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "core",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":core_encoder",
        ":encoding_stage",
        ":gather_encoder",
        ":simple_encoder",
    ],
)

py_library(
    name = "core_encoder",
    srcs = ["core_encoder.py"],
    deps = [
        ":encoding_stage",
        # six dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils:py_utils",
    ],
)

py_test(
    name = "core_encoder_test",
    size = "small",
    srcs = ["core_encoder_test.py"],
    deps = [
        ":core_encoder",
        ":encoding_stage",
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)

py_library(
    name = "encoding_stage",
    srcs = ["encoding_stage.py"],
    deps = [
        # enum dep1,
        # six dep1,
        # tensorflow dep1,
    ],
)

py_test(
    name = "encoding_stage_test",
    size = "small",
    srcs = ["encoding_stage_test.py"],
    deps = [
        ":encoding_stage",
        # absl/testing:parameterized dep1,
        # mock dep1,
        # numpy dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)

py_library(
    name = "gather_encoder",
    srcs = ["gather_encoder.py"],
    deps = [
        ":core_encoder",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils:py_utils",
    ],
)

py_test(
    name = "gather_encoder_test",
    size = "small",
    srcs = ["gather_encoder_test.py"],
    deps = [
        ":core_encoder",
        ":encoding_stage",
        ":gather_encoder",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
        # python:framework_test_lib tensorflow dep2,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)

py_library(
    name = "simple_encoder",
    srcs = ["simple_encoder.py"],
    deps = [
        ":core_encoder",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils:py_utils",
    ],
)

py_test(
    name = "simple_encoder_test",
    size = "small",
    srcs = ["simple_encoder_test.py"],
    deps = [
        ":core_encoder",
        ":simple_encoder",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
        # python:framework_test_lib tensorflow dep2,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/testing:test_utils",
    ],
)
