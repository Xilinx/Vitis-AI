package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "api",
    srcs = [
        "__init__.py",
        "clustering/__init__.py",
        "clustering/keras/__init__.py",
        "quantization/__init__.py",
        "quantization/keras/__init__.py",
        "quantization/keras/quantizers/__init__.py",
        "sparsity/__init__.py",
        "sparsity/keras/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_model_optimization/python/core/clustering/keras",
        "//tensorflow_model_optimization/python/core/quantization/keras",
        "//tensorflow_model_optimization/python/core/sparsity/keras",
    ],
)
