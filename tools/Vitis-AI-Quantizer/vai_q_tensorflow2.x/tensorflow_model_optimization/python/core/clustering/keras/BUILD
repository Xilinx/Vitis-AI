package(default_visibility = [
    "//tensorflow_model_optimization:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cluster",
    ],
)

py_library(
    name = "cluster",
    srcs = ["cluster.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_config",
        ":cluster_wrapper",
        ":clustering_centroids",
        ":clustering_registry",
    ],
)

py_library(
    name = "cluster_config",
    srcs = ["cluster_config.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

py_library(
    name = "clustering_registry",
    srcs = ["clustering_registry.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":clusterable_layer",
    ],
)

py_library(
    name = "clusterable_layer",
    srcs = ["clusterable_layer.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

py_library(
    name = "clustering_centroids",
    srcs = ["clustering_centroids.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_config",
    ],
)

py_library(
    name = "cluster_wrapper",
    srcs = ["cluster_wrapper.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_config",
    ],
)

py_test(
    name = "cluster_test",
    size = "medium",
    srcs = ["cluster_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        # tensorflow dep1,
    ],
)

py_test(
    name = "clustering_centroids_test",
    size = "medium",
    srcs = ["clustering_centroids_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":clustering_centroids",
        # tensorflow dep1,
    ],
)

py_test(
    name = "cluster_wrapper_test",
    size = "medium",
    srcs = ["cluster_wrapper_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":cluster_wrapper",
        ":clusterable_layer",
        ":clustering_centroids",
        ":clustering_registry",
        # tensorflow dep1,
    ],
)

py_test(
    name = "clustering_registry_test",
    size = "medium",
    srcs = ["clustering_registry_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster_wrapper",
        ":clusterable_layer",
        ":clustering_registry",
        # tensorflow dep1,
    ],
)

py_test(
    name = "cluster_integration_test",
    size = "medium",
    srcs = ["cluster_integration_test.py"],
    python_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        # tensorflow dep1,
        "//tensorflow_model_optimization/python/core/keras:compat",
    ],
)
