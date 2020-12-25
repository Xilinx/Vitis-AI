"""Generate Flatbuffer binary from json."""

load(
    "//tensorflow:tensorflow.bzl",
    "tf_binary_additional_srcs",
    "tf_cc_shared_object",
    "tf_cc_test",
)

def tflite_copts():
    """Defines compile time flags."""
    copts = [
        "-DFARMHASH_NO_CXX_STRING",
    ] + select({
        str(Label("//tensorflow:android_arm64")): [
            "-O3",
        ],
        str(Label("//tensorflow:android_arm")): [
            "-mfpu=neon",
            "-O3",
        ],
        str(Label("//tensorflow:ios_x86_64")): [
            "-msse4.1",
        ],
        str(Label("//tensorflow:windows")): [
            "/DTFL_COMPILE_LIBRARY",
            "/wd4018",  # -Wno-sign-compare
        ],
        "//conditions:default": [
            "-Wno-sign-compare",
        ],
    })

    return copts

EXPORTED_SYMBOLS = "//tensorflow/lite/java/src/main/native:exported_symbols.lds"
LINKER_SCRIPT = "//tensorflow/lite/java/src/main/native:version_script.lds"

def tflite_linkopts_unstripped():
    """Defines linker flags to reduce size of TFLite binary.

       These are useful when trying to investigate the relative size of the
       symbols in TFLite.

    Returns:
       a select object with proper linkopts
    """

    # In case you wonder why there's no --icf is because the gains were
    # negligible, and created potential compatibility problems.
    return select({
        "//tensorflow:android": [
            "-Wl,--no-export-dynamic",  # Only inc syms referenced by dynamic obj.
            "-Wl,--gc-sections",  # Eliminate unused code and data.
            "-Wl,--as-needed",  # Don't link unused libs.
        ],
        "//conditions:default": [],
    })

def tflite_jni_linkopts_unstripped():
    """Defines linker flags to reduce size of TFLite binary with JNI.

       These are useful when trying to investigate the relative size of the
       symbols in TFLite.

    Returns:
       a select object with proper linkopts
    """

    # In case you wonder why there's no --icf is because the gains were
    # negligible, and created potential compatibility problems.
    return select({
        "//tensorflow:android": [
            "-Wl,--gc-sections",  # Eliminate unused code and data.
            "-Wl,--as-needed",  # Don't link unused libs.
        ],
        "//conditions:default": [],
    })

def tflite_symbol_opts():
    """Defines linker flags whether to include symbols or not."""
    return select({
        "//tensorflow:android": [
            "-latomic",  # Required for some uses of ISO C++11 <atomic> in x86.
        ],
        "//conditions:default": [],
    }) + select({
        "//tensorflow:debug": [],
        "//conditions:default": [
            "-s",  # Omit symbol table, for all non debug builds
        ],
    })

def tflite_linkopts():
    """Defines linker flags to reduce size of TFLite binary."""
    return tflite_linkopts_unstripped() + tflite_symbol_opts()

def tflite_jni_linkopts():
    """Defines linker flags to reduce size of TFLite binary with JNI."""
    return tflite_jni_linkopts_unstripped() + tflite_symbol_opts()

def tflite_jni_binary(
        name,
        copts = tflite_copts(),
        linkopts = tflite_jni_linkopts(),
        linkscript = LINKER_SCRIPT,
        exported_symbols = EXPORTED_SYMBOLS,
        linkshared = 1,
        linkstatic = 1,
        testonly = 0,
        deps = [],
        tags = [],
        srcs = []):
    """Builds a jni binary for TFLite."""
    linkopts = linkopts + select({
        "//tensorflow:macos": [
            "-Wl,-exported_symbols_list,$(location {})".format(exported_symbols),
            "-Wl,-install_name,@rpath/" + name,
        ],
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-Wl,--version-script,$(location {})".format(linkscript),
            "-Wl,-soname," + name,
        ],
    })
    native.cc_binary(
        name = name,
        copts = copts,
        linkshared = linkshared,
        linkstatic = linkstatic,
        deps = deps + [linkscript, exported_symbols],
        srcs = srcs,
        tags = tags,
        linkopts = linkopts,
        testonly = testonly,
    )

def tflite_cc_shared_object(
        name,
        copts = tflite_copts(),
        linkopts = [],
        linkstatic = 1,
        deps = [],
        visibility = None):
    """Builds a shared object for TFLite."""
    tf_cc_shared_object(
        name = name,
        copts = copts,
        linkstatic = linkstatic,
        linkopts = linkopts + tflite_jni_linkopts(),
        framework_so = [],
        deps = deps,
        visibility = visibility,
    )

def tf_to_tflite(name, src, options, out):
    """Convert a frozen tensorflow graphdef to TF Lite's flatbuffer.

    Args:
      name: Name of rule.
      src: name of the input graphdef file.
      options: options passed to TOCO.
      out: name of the output flatbuffer file.
    """

    toco_cmdline = " ".join([
        "$(location //tensorflow/lite/toco:toco)",
        "--input_format=TENSORFLOW_GRAPHDEF",
        "--output_format=TFLITE",
        ("--input_file=$(location %s)" % src),
        ("--output_file=$(location %s)" % out),
    ] + options)
    native.genrule(
        name = name,
        srcs = [src],
        outs = [out],
        cmd = toco_cmdline,
        tools = ["//tensorflow/lite/toco:toco"] + tf_binary_additional_srcs(),
    )

def tflite_to_json(name, src, out):
    """Convert a TF Lite flatbuffer to JSON.

    Args:
      name: Name of rule.
      src: name of the input flatbuffer file.
      out: name of the output JSON file.
    """

    flatc = "@flatbuffers//:flatc"
    schema = "//tensorflow/lite/schema:schema.fbs"
    native.genrule(
        name = name,
        srcs = [schema, src],
        outs = [out],
        cmd = ("TMP=`mktemp`; cp $(location %s) $${TMP}.bin &&" +
               "$(location %s) --raw-binary --strict-json -t" +
               " -o /tmp $(location %s) -- $${TMP}.bin &&" +
               "cp $${TMP}.json $(location %s)") %
              (src, flatc, schema, out),
        tools = [flatc],
    )

def json_to_tflite(name, src, out):
    """Convert a JSON file to TF Lite's flatbuffer.

    Args:
      name: Name of rule.
      src: name of the input JSON file.
      out: name of the output flatbuffer file.
    """

    flatc = "@flatbuffers//:flatc"
    schema = "//tensorflow/lite/schema:schema_fbs"
    native.genrule(
        name = name,
        srcs = [schema, src],
        outs = [out],
        cmd = ("TMP=`mktemp`; cp $(location %s) $${TMP}.json &&" +
               "$(location %s) --raw-binary --unknown-json --allow-non-utf8 -b" +
               " -o /tmp $(location %s) $${TMP}.json &&" +
               "cp $${TMP}.bin $(location %s)") %
              (src, flatc, schema, out),
        tools = [flatc],
    )

# This is the master list of generated examples that will be made into tests. A
# function called make_XXX_tests() must also appear in generate_examples.py.
# Disable a test by adding it to the blacklists specified in
# generated_test_models_failing().
def generated_test_models():
    return [
        "abs",
        "add",
        "add_n",
        "arg_min_max",
        "avg_pool",
        "batch_to_space_nd",
        "cast",
        "ceil",
        "concat",
        "constant",
        "control_dep",
        "conv",
        "conv2d_transpose",
        "conv_with_shared_weights",
        "conv_to_depthwiseconv_with_shared_weights",
        "cos",
        "depthwiseconv",
        "depth_to_space",
        "div",
        "elu",
        "equal",
        "exp",
        "embedding_lookup",
        "expand_dims",
        "eye",
        "fill",
        "floor",
        "floor_div",
        "floor_mod",
        "fully_connected",
        "fused_batch_norm",
        "gather",
        "gather_nd",
        "gather_with_constant",
        "global_batch_norm",
        "greater",
        "greater_equal",
        "hardswish",
        "identity",
        "sum",
        "l2norm",
        "l2norm_shared_epsilon",
        "l2_pool",
        "leaky_relu",
        "less",
        "less_equal",
        "local_response_norm",
        "log_softmax",
        "log",
        "logical_and",
        "logical_or",
        "logical_xor",
        "lstm",
        "matrix_diag",
        "matrix_set_diag",
        "max_pool",
        "maximum",
        "mean",
        "minimum",
        "mirror_pad",
        "mul",
        "neg",
        "not_equal",
        "one_hot",
        "pack",
        "pad",
        "padv2",
        "placeholder_with_default",
        "prelu",
        "pow",
        "range",
        "rank",
        "reduce_any",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "relu",
        "relu1",
        "relu6",
        "reshape",
        "resize_bilinear",
        "resolve_constant_strided_slice",
        "reverse_sequence",
        "reverse_v2",
        "rfft2d",
        "round",
        "rsqrt",
        "shape",
        "sigmoid",
        "sin",
        "slice",
        "softmax",
        "space_to_batch_nd",
        "space_to_depth",
        "sparse_to_dense",
        "split",
        "splitv",
        "sqrt",
        "square",
        "squared_difference",
        "squeeze",
        "strided_slice",
        "strided_slice_1d_exhaustive",
        "strided_slice_np_style",
        "sub",
        "tile",
        "topk",
        "transpose",
        "transpose_conv",
        "uint8_hardswish",
        "unfused_gru",
        "unidirectional_sequence_lstm",
        "unidirectional_sequence_rnn",
        "unique",
        "unpack",
        "unroll_batch_matmul",
        "where",
        "zeros_like",
    ]

# List of models that fail generated tests for the conversion mode.
# If you have to disable a test, please add here with a link to the appropriate
# bug or issue.
def generated_test_models_failing(conversion_mode):
    if conversion_mode == "toco-flex":
        return [
            "lstm",  # TODO(b/117510976): Restore when lstm flex conversion works.
            "unidirectional_sequence_lstm",
            "unidirectional_sequence_rnn",
        ]
    elif conversion_mode == "forward-compat":
        return []
    return []

def generated_test_conversion_modes():
    """Returns a list of conversion modes."""

    return ["toco-flex", "forward-compat", ""]

def generated_test_models_all():
    """Generates a list of all tests with the different converters.

    Returns:
      List of tuples representing:
            (conversion mode, name of test, test tags, test args).
    """
    conversion_modes = generated_test_conversion_modes()
    tests = generated_test_models()
    options = []
    for conversion_mode in conversion_modes:
        failing_tests = generated_test_models_failing(conversion_mode)
        for test in tests:
            tags = []
            args = []
            if test in failing_tests:
                tags.append("notap")
                tags.append("manual")
            if conversion_mode:
                test += "_%s" % conversion_mode

            # Flex conversion shouldn't suffer from the same conversion bugs
            # listed for the default TFLite kernel backend.
            if conversion_mode == "toco-flex":
                args.append("--ignore_known_bugs=false")
            options.append((conversion_mode, test, tags, args))
    return options

def gen_zip_test(name, test_name, conversion_mode, **kwargs):
    """Generate a zipped-example test and its dependent zip files.

    Args:
      name: str. Resulting cc_test target name
      test_name: str. Test targets this model. Comes from the list above.
      conversion_mode: str. Which conversion mode to run with. Comes from the
        list above.
      **kwargs: tf_cc_test kwargs
    """
    toco = "//tensorflow/lite/toco:toco"
    flags = ""
    if conversion_mode == "toco-flex":
        flags += " --ignore_converter_errors --run_with_flex"
    elif conversion_mode == "forward-compat":
        flags += " --make_forward_compat_test"

    gen_zipped_test_file(
        name = "zip_%s" % test_name,
        file = "%s.zip" % test_name,
        toco = toco,
        flags = flags + " --save_graphdefs",
    )
    tf_cc_test(name, **kwargs)

def gen_zipped_test_file(name, file, toco, flags):
    """Generate a zip file of tests by using :generate_examples.

    Args:
      name: str. Name of output. We will produce "`file`.files" as a target.
      file: str. The name of one of the generated_examples targets, e.g. "transpose"
      toco: str. Pathname of toco binary to run
      flags: str. Any additional flags to include
    """
    native.genrule(
        name = file + ".files",
        cmd = (("$(locations :generate_examples) --toco $(locations {0}) " +
                " --zip_to_output {1} {2} $(@D)").format(toco, file, flags)),
        outs = [file],
        tools = [
            ":generate_examples",
            toco,
        ],
    )

    native.filegroup(
        name = name,
        srcs = [file],
    )

def gen_selected_ops(name, model):
    """Generate the library that includes only used ops.

    Args:
      name: Name of the generated library.
      model: TFLite model to interpret.
    """
    out = name + "_registration.cc"
    tool = "//tensorflow/lite/tools:generate_op_registrations"
    tflite_path = "//tensorflow/lite"
    native.genrule(
        name = name,
        srcs = [model],
        outs = [out],
        cmd = ("$(location %s) --input_model=$(location %s) --output_registration=$(location %s) --tflite_path=%s") %
              (tool, model, out, tflite_path[2:]),
        tools = [tool],
    )

def flex_dep(target_op_sets):
    if "SELECT_TF_OPS" in target_op_sets:
        return ["//tensorflow/lite/delegates/flex:delegate"]
    else:
        return []

def gen_model_coverage_test(src, model_name, data, failure_type, tags):
    """Generates Python test targets for testing TFLite models.

    Args:
      src: Main source file.
      model_name: Name of the model to test (must be also listed in the 'data'
        dependencies)
      data: List of BUILD targets linking the data.
      failure_type: List of failure types (none, toco, crash, inference)
        expected for the corresponding combinations of op sets
        ("TFLITE_BUILTINS", "TFLITE_BUILTINS,SELECT_TF_OPS", "SELECT_TF_OPS").
      tags: List of strings of additional tags.
    """
    i = 0
    for target_op_sets in ["TFLITE_BUILTINS", "TFLITE_BUILTINS,SELECT_TF_OPS", "SELECT_TF_OPS"]:
        args = []
        if failure_type[i] != "none":
            args.append("--failure_type=%s" % failure_type[i])
        i = i + 1
        native.py_test(
            name = "model_coverage_test_%s_%s" % (model_name, target_op_sets.lower().replace(",", "_")),
            srcs = [src],
            main = src,
            size = "large",
            args = [
                "--model_name=%s" % model_name,
                "--target_ops=%s" % target_op_sets,
            ] + args,
            data = data,
            srcs_version = "PY2AND3",
            python_version = "PY2",
            tags = [
                "no_oss",
                "no_windows",
            ] + tags,
            deps = [
                "//tensorflow/lite/testing/model_coverage:model_coverage_lib",
                "//tensorflow/lite/python:lite",
                "//tensorflow/python:client_testlib",
            ] + flex_dep(target_op_sets),
        )
