load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite_combined")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_import(
    name = "pyxir",
    hdrs = glob([
        "include/pyxir/**/*.hpp",
    ]),
    shared_library = "python/libpyxir.so",
)
