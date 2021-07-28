load("//tensorflow/lite:special_rules.bzl", "tflite_extra_gles_deps")

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
#        "//tensorflow/lite/c:c_api_internal",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/delegates/vitisai/pyxir:pyxir",
    ],
)
