package(default_visibility = [
    "//tensorflow_model_optimization/python/core/internal/tensor_encoding:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "utils",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":py_utils",
        ":tf_utils",
    ],
)

py_library(
    name = "py_utils",
    srcs = ["py_utils.py"],
    deps = [
        # enum dep1,
        # numpy dep1,
        # six dep1,
        # tensorflow dep1,
        # tree dep1,
    ],
)

py_test(
    name = "py_utils_test",
    size = "small",
    srcs = ["py_utils_test.py"],
    deps = [
        ":py_utils",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "tf_utils",
    srcs = ["tf_utils.py"],
    deps = [
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_test(
    name = "tf_utils_test",
    size = "medium",
    srcs = ["tf_utils_test.py"],
    deps = [
        ":tf_utils",
        # absl/testing:parameterized dep1,
        # numpy dep1,
        # scipy dep1,
        # tensorflow dep1,
    ],
)
