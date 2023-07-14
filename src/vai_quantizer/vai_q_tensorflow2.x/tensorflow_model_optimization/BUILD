# Description:
#
#  TensorFlow Optimization is a repository for the training-time
#  portion of the Tensorflow Model Optimization Toolkit,
#  used to optimize machine learning models for deployment and execution.
#
#  https://github.com/tensorflow/model-optimization

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "tensorflow_model_optimization",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_model_optimization/python",
        "//tensorflow_model_optimization/python/core/api",
        "//tensorflow_model_optimization/python/core/clustering",
        "//tensorflow_model_optimization/python/core/keras:compat",
        "//tensorflow_model_optimization/python/core/quantization",
        "//tensorflow_model_optimization/python/core/sparsity",
    ],
)

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    python_version = "PY3",
    deps = [
        # tensorflow_docs/api_generator:generate_lib dep1,
        "//third_party/tensorflow_model_optimization",
    ],
)
