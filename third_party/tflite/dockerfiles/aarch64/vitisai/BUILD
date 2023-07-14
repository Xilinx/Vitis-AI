load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite_combined")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "libvitisai_delegate.so",
    srcs = [
        "vitisai_delegate.cc",
        "vitisai_delegate.h",
    ],
    linkshared = True,
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/delegates/vitisai/pyxir:pyxir",
    ],
)
