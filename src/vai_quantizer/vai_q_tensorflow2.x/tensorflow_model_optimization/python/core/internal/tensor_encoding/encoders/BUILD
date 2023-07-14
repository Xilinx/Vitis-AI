package(default_visibility = [
    "//tensorflow_model_optimization/python/core/internal/tensor_encoding:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "encoders",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":common_encoders",
    ],
)

py_library(
    name = "common_encoders",
    srcs = ["common_encoders.py"],
    deps = [
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:core_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:gather_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:simple_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/stages:stages_impl",
    ],
)

py_test(
    name = "common_encoders_test",
    size = "small",
    srcs = ["common_encoders_test.py"],
    deps = [
        ":common_encoders",
        # absl/testing:parameterized dep1,
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:core_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:gather_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/core:simple_encoder",
        "//tensorflow_model_optimization/python/core/internal/tensor_encoding/utils:py_utils",
    ],
)
