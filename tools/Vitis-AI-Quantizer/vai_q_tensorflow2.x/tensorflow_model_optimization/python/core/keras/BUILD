package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "keras",
    srcs = ["__init__.py"],
    deps = [
        ":compat",
    ],
)

# TODO(alanchiao): split into test_utils and utils.
py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep1,
        # tensorflow dep1,
    ],
)

py_library(
    name = "compat",
    srcs = ["compat.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep1,
    ],
)
