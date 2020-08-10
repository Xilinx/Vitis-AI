"""Build rules for tf.distritbute testing."""

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

def distribute_py_test(
        name,
        srcs = [],
        deps = [],
        tags = [],
        data = [],
        main = None,
        args = [],
        shard_count = 1,
        full_precision = False,
        **kwargs):
    """Generates py_test targets for CPU and GPU.

    Args:
        name: test target name to generate suffixed with `test`.
        srcs: source files for the tests.
        deps: additional dependencies for the test targets.
        tags: tags to be assigned to the different test targets.
        data: data files that need to be associated with the target files.
        main: optional main script.
        args: arguments to the tests.
        shard_count: number of shards to split the tests across.
        **kwargs: extra keyword arguments to the test.
    """

    _ignore = (full_precision)
    cuda_py_test(
        name = name,
        srcs = srcs,
        data = data,
        main = main,
        additional_deps = deps,
        shard_count = shard_count,
        tags = tags,
        args = args,
        **kwargs
    )
