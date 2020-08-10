# Provides the ruy_test macro for type-parametrized tests.

"""
ruy_test is a macro for building a test with multiple paths
corresponding to tuples of types for LHS, RHS, accumulator
and destination.
"""

def ruy_test(name, srcs, lhs_rhs_accum_dst, copts, tags = []):
    for (lhs, rhs, accum, dst) in lhs_rhs_accum_dst:
        native.cc_test(
            name = "%s_%s_%s_%s_%s" % (name, lhs, rhs, accum, dst),
            srcs = srcs,
            copts = copts + [
                "-DRUY_TEST_LHSSCALAR=%s" % lhs,
                "-DRUY_TEST_RHSSCALAR=%s" % rhs,
                "-DRUY_TEST_ACCUMSCALAR=%s" % accum,
                "-DRUY_TEST_DSTSCALAR=%s" % dst,
            ],
            deps = [
                "//tensorflow/lite/experimental/ruy:test_lib",
                "@com_google_googletest//:gtest_main",
            ],
            tags = tags,
        )

def ruy_benchmark(name, srcs, lhs_rhs_accum_dst, copts):
    tags = ["req_dep=@gemmlowp//:profiler"]
    for (lhs, rhs, accum, dst) in lhs_rhs_accum_dst:
        native.cc_binary(
            name = "%s_%s_%s_%s_%s" % (name, lhs, rhs, accum, dst),
            testonly = True,
            srcs = srcs,
            copts = copts + [
                "-DRUY_TEST_LHSSCALAR=%s" % lhs,
                "-DRUY_TEST_RHSSCALAR=%s" % rhs,
                "-DRUY_TEST_ACCUMSCALAR=%s" % accum,
                "-DRUY_TEST_DSTSCALAR=%s" % dst,
            ],
            deps = [
                "//tensorflow/lite/experimental/ruy:test_lib",
                "@gemmlowp//:profiler",  # Note also tagged as req_dep.
            ],
            tags = tags,
        )

def ruy_benchmark_opt_sets(name, opt_sets, srcs, lhs_rhs_accum_dst, copts):
    tags = ["req_dep=@gemmlowp//:profiler"]
    for opt_set in opt_sets:
        for (lhs, rhs, accum, dst) in lhs_rhs_accum_dst:
            native.cc_binary(
                name = "%s_%s_%s_%s_%s_%s" % (name, opt_set, lhs, rhs, accum, dst),
                testonly = True,
                srcs = srcs,
                copts = copts + [
                    "-DRUY_TEST_LHSSCALAR=%s" % lhs,
                    "-DRUY_TEST_RHSSCALAR=%s" % rhs,
                    "-DRUY_TEST_ACCUMSCALAR=%s" % accum,
                    "-DRUY_TEST_DSTSCALAR=%s" % dst,
                    "-DRUY_OPT_SET=0x%s" % opt_set,
                ],
                deps = [
                    "//tensorflow/lite/experimental/ruy:test_lib",
                    "@gemmlowp//:profiler",  # Note also tagged as req_dep.
                ],
                tags = tags,
            )
