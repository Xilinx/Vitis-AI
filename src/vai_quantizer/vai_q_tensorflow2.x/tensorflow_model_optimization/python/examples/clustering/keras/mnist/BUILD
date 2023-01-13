package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
)

py_binary(
    name = "mnist_cnn",
    srcs = [
        "mnist_cnn.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        # absl:app dep1,
        # absl/flags dep1,
        # tensorflow dep1,
        # python/keras tensorflow dep2,
        # python/keras/optimizer_v2 tensorflow dep2,
        "//tensorflow_model_optimization/python/core/clustering/keras:cluster",
    ],
)
