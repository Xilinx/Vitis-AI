# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""VAI_Q_TENSORFLOW: Xilinx's Quantize Tool For Tensorflow

This script is designed to quantize a frozen floating point model into fixed point graph and
deploy to Xilinx hardware. This script takes a frozen GraphDef proto, input_nodes,
output_nodes and some configure as input, and then quantize the weights/biases
and activations to given bit_width. In order to improve the pricison of fixed point graph,
the quantizer needs to run some iterations of inference of the whole model to
calibration the activations, thus needing an input function to load data.

## Steps for quantizatin:

# 1) freeze the float graph:
In most situations, training a model with TensorFlow will give you a folder
containing a GraphDef file (usually ending with the .pb or .pbtxt extension)
and a set of checkpoint files. What you need for mobile or embedded deployment
is a single GraphDef file that's been 'frozen', or had its variables converted
into inline constants so everything's in one file. To handle the conversion,
Tensorflow provided freeze_graph.py, which is automatically installed with vai_q_tensorflow.

An example of command-line usage is:

freeze_graph \
    --input_graph=/tmp/inception_v3_inf_graph.pb \
    --input_checkpoint=/tmp/checkpoints/model.ckpt-1000 \
    --input_binary=true \
    --output_graph=/tmp/frozen_graph.pb \
    --output_node_names=InceptionV3/Predictions/Reshape_1

Note[1]: type `freeze_graph --help` for more options
Note[2]: The input and output node names will vary depending on the model,
but you can inspect and estimate them with vai_q_tensorflow.

An example of command-line usage is:

vai_q_tensorflow inspect --input_frozen_graph=/tmp/inception_v3_inf_graph.pb

# 2) quantize the frozen graph:
This step takes a frozen graph as input, together with the graph's input/output
information and a input_fn to do quantization, and outputs the quantized graph.

An example of command-line usage is:

vai_q_tensorflow quantize \
    --input_frozen_graph=/tmp/frozen_graph.pb \
    --input_nodes=input \
    --input_shapes=?,224,224,3 \
    --output_nodes=resnet_v1_50/predictions/Reshape_1 \
    --input_fn=inception_v1_input_fn.calib_input \
    --gpu=0

The input_fn is a python function to provide real input data for the graph.
In this case, the inception_v1_input_fn.py looks like:
"`inception_v1_input_fn.py

def calib_input(iter_num):
  image = load_image(iter_num)
  preprocessed_image = do_preprocess(image)
  return {"input": preprocessed_images}
"

Note[1]: type `vai_q_tensorflow --help` for more options

## The output of this scripts are:
1) quantize_eval_model.pb: a standard tf_model which can be used to evaluate the quantized model.
2) deploy_model.pb: an extended tf_model, which cannot be imported by standard tensorflow.
Users can use Xilinx's compilers to compile the models using this file and deploy it to DPU.

"""
import os
import shutil
import argparse
import importlib
import sys
import time
import tempfile
from copy import deepcopy
from progressbar import ProgressBar

import tensorflow as tf
from tensorflow.core.framework import types_pb2, graph_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client.session import Session
from tensorflow.python import Graph

from .utils import *
from .quantize_graph import *

sys.path.append("..")
from vai_q_tensorflow.gen_files.version import __version__, __git_version__

#  try:
#    import xnnc
#    from pathlib import Path
#  except ImportError:
#    print(
#        "[INFO] Not found xnnc package. Disable the support for the generation of XIR model."
#    )

FLAGS = None



def _parse_input_frozen_graph(input_frozen_graph):
  """Parse input_frozen_graph configurations"""
  if input_frozen_graph == '':
    raise ValueError(INVALID_INPUT + NOT_FOUND_MESSAGE + " No --input_frozen_graph assigned.")
  if not tf.io.gfile.exists(input_frozen_graph):
    raise ValueError(INVALID_INPUT + NOT_FOUND_MESSAGE + "Input frozen graph file '" + input_frozen_graph +
                     "' does not exist!")
  graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(input_frozen_graph, "rb") as f:
    graph_def.ParseFromString(f.read())
  return graph_def


def _parse_input_graph(input_graph):
  """Parse input_graph configurations"""
  if input_frozen_graph == '':
    raise ValueError("No --input_graph assigned.")
  if not tf.gfile.Exists(input_graph):
    raise ValueError("Input graph file '" + input_graph + "' does not exist!")
  graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(input_graph, "rb") as f:
    graph_def.ParseFromString(f.read())
  return graph_def


def _parse_input_meta_graph(input_meta_graph):
  """Parse input_meta_graph configurations"""
  if not tf.gfile.Exists(input_meta_graph):
    raise ValueError("Input meta graph file '" + input_meta_graph +
                     "' does not exist.")
  meta_graph_def = tf.MetaGraphDef()
  with tf.io.gfile.GFile(input_meta_graph, "rb") as f:
    meta_graph_def.ParseFromString(f.read())
  return meta_graph_def


def _parse_input_nodes(input_graph_def, input_nodes_str):
  """Parse input_nodes configurations"""
  input_nodes = []
  if input_nodes_str:
    input_nodes = input_nodes_str.split(",")
    check_node_names(input_graph_def, input_nodes)
  else:
    raise ValueError("No --input_nodes assigned.")
  return input_nodes


def _parse_output_nodes(input_graph_def, output_nodes_str):
  """Parse output_nodes configurations"""
  output_nodes = []
  if output_nodes_str:
    output_nodes = output_nodes_str.split(",")
    check_node_names(input_graph_def, output_nodes)
  else:
    raise ValueError("No --output_nodes assigned.")
  return output_nodes


def _parse_nodes_bit(input_graph_def, nodes_bit_str):
  """Parse nodes_bit configurations"""
  nodes_bit = []
  if nodes_bit_str:
    nodes_bit = nodes_bit_str.split(",")
    for param in nodes_bit:
      node_name = [param.strip().split(":")[0]]
      check_node_names(input_graph_def, node_name)
      bit = int(param.strip().split(":")[-1])
      if bit < 1:
        raise ValueError(INVALID_BITWITH + INVALID_PARAM_MESSAGE + " Error mehtod number, method must be \
                >=1 but got ", bit)
  return nodes_bit


def _parse_nodes_method(input_graph_def, nodes_method_str):
  """Parse nodes_method configurations"""
  nodes_method = []
  if nodes_method_str:
    nodes_method = nodes_method_str.split(",")
    for param in nodes_method:
      node_name = [param.strip().split(":")[0]]
      check_node_names(input_graph_def, node_name)
      method = int(param.strip().split(":")[-1])
      if method not in [0, 1, 2]:
        raise ValueError(INVALID_METHOD + INVALID_PARAM_MESSAGE + "Error mehtod number, method must be one \
                of [0, 1, 2] but got {}".format(method))
  return nodes_method


def _parse_ignore_nodes(input_graph_def, ignore_nodes_str):
  """Parse ignore configurations"""
  ignore_nodes = []
  if ignore_nodes_str:
    ignore_nodes = ignore_nodes_str.split(",")
  if ignore_nodes:
    check_node_names(input_graph_def, ignore_nodes)
  return ignore_nodes


def _parse_input_shapes(input_nodes, input_shapes_str):
  """Parse quant input_shapes configurations"""
  input_shapes = []
  if not input_shapes_str:
    raise ValueError("No --input_shapes assigned.")
  elif len(input_shapes_str.split(":")) != len(input_nodes):
    raise ValueError(LENGHT_MISSMATCH + INVALID_PARAM_MESSAGE + "input_shapes should be the same length as input_nodes")

  for input_shape in input_shapes_str.split(":"):
    shape_str = input_shape.split(",")
    if len(shape_str) != 4:
      raise ValueError(LENGHT_MISSMATCH + INVALID_PARAM_MESSAGE +
          ' Input_shapes should be 4-dimension int shape list (support unknown batch_size), e.g. ?,224,224,3 or 1,299,299,3'
      )
    st = 1 if shape_str[0] == '?' else 0
    shape = []
    if shape_str[0] == '?':
      shape_str[0] = -1
    try:
      shape[0:3] = [int(s) for s in shape_str]
    except Exception as e:
      raise ValueError(LENGHT_MISSMATCH + INVALID_PARAM_MESSAGE +
          'Input_shapes should be 4-dimension int shape list (support unknown batch_size), e.g. ?,224,224,3 or 1,299,299,3'
      )
    input_shapes.append(shape)
  return input_shapes


def _parse_input_fn(input_fn_str):
  """Parse input_fn configurations"""
  input_fn = None
  if input_fn_str == "":
    raise ValueError('No input_fn assigned.')
  else:
    try:
      sys.path.append('./')
      module = __import__(input_fn_str.rsplit('.', 1)[0], fromlist=True)
      input_fn = getattr(module, input_fn_str.rsplit('.', 1)[1])
    except Exception as e:
      raise ValueError(INVALID_INPUT_FN + FAIL_IMPORT_MESSAGE + 'Fail to import input_fn, error: ', e)
  return input_fn


def _parse_session_config(gpu_memory_fraction):
  """Parse session configurations"""
  s_config = tf.compat.v1.ConfigProto()
  s_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
  # Disable graph optimizer and rewiter to make sure every quantize node works correctly
  # the next two operation have been moved into `quantize_frozen` and `dump` function
  # s_config.graph_options.optimizer_options.opt_level = -1
  # s_config.graph_options.rewrite_options.disable_meta_optimizer = True
  return s_config


def check_float_graph(input_graph_def, input_fn, q_config, s_config):
  """Check if float graph and input_fn is validate before quantization"""
  graph = tf.Graph()
  with graph.as_default():
    tf.graph_util.import_graph_def(input_graph_def, name='')
    print("INFO: Checking Float Graph...")
    input_tensors = [
        op.outputs[0] for op in graph.get_operations()
        if op.type == 'Placeholder'
    ]
    output_tensors = [
        graph.get_tensor_by_name(name + ':0')
        for name in gen_quantized_node_names(graph, q_config.output_nodes)
    ]

    with Session(graph=graph, config=s_config) as sess:
      inputs = input_fn(0)
      feed_dict = gen_feed_dict(input_tensors, inputs)
      sess.run(output_tensors, feed_dict)
  print("INFO: Float Graph Check Done.")

IGNORE_OP_TYPES = ["Enter", "Merge", "LoopCond", "Switch", "Exit", "Less", \
        "LogicalAnd", "LogicalOr", "LogicalNot", "Assert"]
def get_shape_info(input_graph_def, input_fn, s_config, ignore_node_names):
  graph = tf.Graph()
  with graph.as_default():
    tf.graph_util.import_graph_def(input_graph_def, name='')
    input_tensors = [
        op.outputs[0] for op in graph.get_operations()
        if op.type == 'Placeholder'
    ]

    output_tensors = []
    output_names = []
    for op in graph.get_operations():
      if len(op.outputs) > 0 and op.type not in IGNORE_OP_TYPES \
           and op.name not in ignore_node_names:
        output_tensors.append(op.outputs[0])
        output_names.append(op.name)


    output_tensor_val = []
    with Session(graph=graph, config=s_config) as sess:
      inputs = input_fn(0)
      ## just use one image
      for k,v in inputs.items():
        inputs[k] = v[0:1]
      feed_dict = gen_feed_dict(input_tensors, inputs)
      for t in output_tensors:
        try:
          output_tensor_val.append(sess.run(t, feed_dict))
        except Exception as e:
          output_tensor_val.append(None)
      #################################
    shape_info = {}
    for name, tensor in zip(output_names, output_tensor_val):
      if tensor is not None:
        shape_info[name] = tensor.shape
    pass
  return shape_info

def calibrate_frozen(input_graph_def, input_fn, q_config, s_config,
        add_shapes=False, fold_constant=True):
  """Transform float graph to quantized graph and do calibration"""

  temp_path = os.path.join(q_config.output_dir, "temp")
  if not os.path.exists(temp_path):
    os.makedirs(temp_path)

  if fold_constant:
    from tensorflow.tools.graph_transforms import TransformGraph
    input_names = q_config.input_nodes
    output_names = q_config.output_nodes
    transforms = ["fold_constants(ignore_errors=true)"]
    input_graph_def = TransformGraph(input_graph_def, input_names,
                                           output_names, transforms)
    # save_pb_file(input_graph_def,
    #              os.path.join(q_config.output_dir, "decent_debug/fold_constants.pb"))

  # Calibration
  calib_graph_def = CreateQuantizeCalibrationGraphDef(input_graph_def,
                                                      q_config)
  graph = tf.Graph()
  with graph.as_default():
    tf.graph_util.import_graph_def(calib_graph_def, name='')
    print("INFO: Calibrating for {} iterations...".format(q_config.calib_iter))
    input_tensors = [
        op.outputs[0] for op in graph.get_operations()
        if op.type == 'Placeholder'
    ]
    output_tensors = [
        graph.get_tensor_by_name(name + ':0')
        for name in gen_quantized_node_names(graph, q_config.output_nodes)
    ]
    with Session(graph=graph, config=s_config) as sess:
      calib_graph_def = sess.graph.as_graph_def(add_shapes=add_shapes)
      progress = ProgressBar()
      for i in progress(range(0, q_config.calib_iter)):
        inputs = input_fn(i)
        feed_dict = gen_feed_dict(input_tensors, inputs)
        sess.run(output_tensors, feed_dict)
  print("INFO: Calibration Done.")

  # Quantized Evaluation
  quantize_eval_graph_def = CreateQuantizeEvaluationGraphDef(
      calib_graph_def, q_config)
  shutil.rmtree(temp_path)
  return quantize_eval_graph_def


def deploy_frozen(quantize_eval_graph_def, q_config):
  """Deploy quantized graph to DPU"""
  print("INFO: Generating Deploy Model...")
  q_config.output_nodes = get_quantized_nodes(quantize_eval_graph_def,
                                              q_config.output_nodes)
  deploy_graph_def = CreateQuantizeDeployGraphDef(quantize_eval_graph_def,
                                                  q_config)
  save_pb_file(deploy_graph_def,
               os.path.join(q_config.output_dir, "deploy_model.pb"))
  print("INFO: Deploy Model Generated.")
  return deploy_graph_def


def convert_datatype(input_graph_def, q_config, convert_datatype):
  """Convert input graph to target data type
  Args:
    input_graph_def: A `GraphDef` object, frozen model to be quantized.
    q_config: A `QuantizeConfig` object, the configuration for Quantize.
    convert_datatype: int value. the value of flags.convert_datatype.
  """
  # Summarize Quantize Results
  datatype = ["", "fp16", "double", "bfloat16", "fp32"]
  if isinstance(convert_datatype, int):
    save_file_name = "converted_model_{}.pb".format(datatype[convert_datatype])
  elif isinstance(convert_datatype, str) and convert_datatype in datatype[1:]:
    save_file_name = "converted_model_{}.pb".format(datatype[convert_datatype])
  else:
    raise ValueError(INVALID_TARGET_DTYPE  + INVALID_PARAM_MESSAGE +
            "Please provide correct convert_datatype one of the str type value({}) or "
            " one of int type value({}).".format(datatype[1:], list(range(len(datatype[1:])))))



  # fold constant
  from tensorflow.tools.graph_transforms import TransformGraph
  input_names = q_config.input_nodes
  output_names = q_config.output_nodes
  transforms = ["fold_constants(ignore_errors=true)"]
  input_graph_def = TransformGraph(input_graph_def, input_names,
                                         output_names, transforms)

  # fold batch norm
  q_config.fold_bn_only = 1
  input_graph_def = CreateOptimizedGraphDef(input_graph_def, q_config)

  # convert datatype
  if convert_datatype == 1:
      dtype = types_pb2.DT_HALF
  elif convert_datatype == 2:
      dtype = types_pb2.DT_DOUBLE
  elif convert_datatype == 3:
      dtype = types_pb2.DT_BFLOAT16
  else:
      dtype = types_pb2.DT_FLOAT

  target_graph_def = graph_pb2.GraphDef()
  target_graph_def.versions.CopyFrom(input_graph_def.versions)
  for node in input_graph_def.node:
    if "FusedBatchNorm" in node.op:
      raise ValueError(UNSUPPORTED_OP + UNSUPPORTED_OP_MESSAGE + 'Error: exist unsupported pattern in PB file, node'
              ' name:{} OP type: {}.'.format(node.name, node.op))
    new_node = target_graph_def.node.add()
    new_node.op = node.op
    new_node.name = node.name
    new_node.input.extend(node.input)

    attrs = list(node.attr.keys())
    for attr in attrs:
      if node.attr[attr].type == types_pb2.DT_FLOAT:
        # modify node dtype
        node.attr[attr].type = dtype

      if attr == "value":
        tensor = node.attr[attr].tensor
        if tensor.dtype == types_pb2.DT_FLOAT:
          # if float_val exists
          if tensor.float_val:
            float_val = tf.make_ndarray(node.attr[attr].tensor)
            new_node.attr[attr].tensor.CopyFrom(tf.make_tensor_proto(float_val, dtype=dtype))
            continue
          # if tensor content exists
          if tensor.tensor_content:
            tensor_shape = [x.size for x in tensor.tensor_shape.dim]
            tensor_weights = tf.make_ndarray(tensor)
            # reshape tensor
            tensor_weights = np.reshape(tensor_weights, tensor_shape)
            tensor_proto = tf.compat.v1.make_tensor_proto(tensor_weights, dtype=dtype)
            new_node.attr[attr].tensor.CopyFrom(tensor_proto)
            continue
      new_node.attr[attr].CopyFrom(node.attr[attr])
  save_pb_file(target_graph_def,
               os.path.join(q_config.output_dir, save_file_name))

  print("********************* Quantization Summary *********************\
      \nINFO: Output: \
      \n  converted datatype model file: {} ".format(
      os.path.join(q_config.output_dir, save_file_name)))

def quantize_frozen(input_graph_def,
                    input_fn,
                    q_config=QuantizeConfig(),
                    s_config=tf.compat.v1.ConfigProto(),
                    skip_check=0,
                    dump_as_xir=False,
                    fuse_op_config=None,
                    add_shapes=False,
                    keep_float_weight=True,
                    output_format="pb",
                    custom_op_set=set(),
                    fold_constant=True):
  """Quantize calibrate and then deploy to DPU.

  Args:
    input_graph_def: A `GraphDef` object, frozen model to be quantized.
    input_fn: A `function` object, the function that provides input data for the placeholder nodes.
      if set function object, the function should take a `int` value as input indicating the calibration
      step number, and should return a dict`(placeholder_node_name, numpy.Array)` object for each call, which
      will be fed into the model's placeholder nodes.
    skip_check: If set 1, the check for float model will be skipped, useful when only part of
      the input model is quantized.
    s_config: A `ConfigProto` object, the configuration for Session.
    dump_as_xir: Specify whether dump forzen model as an xmodel.

  Returns:
    quantize_eval_graph_def: A `GraphDef` object, the quantized model for evaluation on gpu or cpu.
    deploy_graph_def: A `GraphDef` object, the quantized model for dpu deployment.
  """

  s_config.graph_options.optimizer_options.opt_level = -1
  s_config.graph_options.rewrite_options.disable_meta_optimizer = True
  if not skip_check:
    check_float_graph(input_graph_def, input_fn, q_config, s_config)

  quantize_eval_graph_def = calibrate_frozen(input_graph_def, input_fn,
                                             q_config, s_config,
                                             add_shapes=add_shapes,
                                             fold_constant=fold_constant)

  if not keep_float_weight:
    max_dump_batches = 1
    dump_float = False
    name2qdq_tensor = dump(quantize_eval_graph_def, input_fn,
         q_config.output_dir, max_dump_batches, dump_float, s_config=s_config,
         get_weight_only=True)
    for node in quantize_eval_graph_def.node:
      w_quant_name = node.name + "/wquant"
      if w_quant_name in name2qdq_tensor:
        weight = name2qdq_tensor[w_quant_name]
        dtype = node.attr["dtype"].type
        tensor_proto = tf.AttrValue(tensor=tf.make_tensor_proto(weight,
            dtype=dtype, shape=weight.shape))
        node.attr["value"].CopyFrom(tensor_proto)

  # deploy_graph_def = deploy_frozen(quantize_eval_graph_def, q_config)
  # print("INFO: skip create deploy_model.pb, not support create \
  #         deploy_model.pb in the future")

  plugin_nodes = {} ##{plugin_name:[node name]}
  plugin_output_nodes = {}##{plugin_name:[node name]}
  deploy_model_describe = ""
  if fuse_op_config:
    deploy_graph_def = deepcopy(quantize_eval_graph_def)

    namescope_map = get_fuse_config(fuse_op_config)
    namescope_map = check_namescope_map(namescope_map,
            deploy_graph_def)

    plugin_nodes, plugin_output_nodes = get_plugin_output(input_graph_def,
            namescope_map)
    exclude_nodes = [node for node in deploy_graph_def.node if node.op
            == "FixNeuron"]

    deploy_graph_def = fuse_ops(deploy_graph_def, namescope_map, exclude_nodes=exclude_nodes)

  ignore_node_names = []
  target_node_names = []
  for pn, nodes_lst in plugin_output_nodes.items():
    target_node_names.extend(nodes_lst)
  for node in quantize_eval_graph_def.node:
    if not (node.op in custom_op_set or node.name in target_node_names):
      ignore_node_names.append(node.name)

  shape_info = get_shape_info(quantize_eval_graph_def, input_fn, s_config,
          ignore_node_names)

  quantize_eval_graph_def = set_shape_info(quantize_eval_graph_def,
          shape_info, plugin_output_nodes)
  if output_format == "pb":
    save_pb_file(quantize_eval_graph_def,
                 os.path.join(q_config.output_dir, "quantize_eval_model.pb"))

    if fuse_op_config:
      deploy_graph_def = set_shape_info(deploy_graph_def,
              shape_info, plugin_output_nodes)
      deploy_graph_path = os.path.join(q_config.output_dir, "deploy_model.pb")
      deploy_model_describe = "\n  deploy_model: {} \nplease use this " \
              " deploy_model.pb to deploy model".format(deploy_graph_path)
      save_pb_file(deploy_graph_def,
                   os.path.join(q_config.output_dir, "deploy_model.pb"))

    # Summarize Quantize Results
    print("********************* Quantization Summary *********************\
        \nINFO: Output: \
        \n  quantize_eval_model: {} ".format(
        os.path.join(q_config.output_dir, "quantize_eval_model.pb"))
        + deploy_model_describe)
  elif output_format == "onnx":
    try:
      import tf2onnx
      input_names = [name + ":0" for name in q_config.input_nodes]
      output_names = [name + ":0" for name in q_config.output_nodes]
      output_path = os.path.join(q_config.output_dir,
                    "quantize_eval_model.onnx")
      model_proto, external_tensor_storage = tf2onnx.convert.from_graph_def(quantize_eval_graph_def,
                name="", input_names=input_names, output_names=output_names,
                opset=13, custom_ops=None, custom_op_handlers=None, custom_rewriter=None,
                inputs_as_nchw=None, extra_opset=None,
                shape_override=None, target=None, large_model=False,
                output_path=output_path)
      print("********************* Quantization Summary *********************\
          \nINFO: Output: \
          \n  quantize_eval_model: {} ".format(
          os.path.join(q_config.output_dir, "quantize_eval_model.onnx")))
    except ImportError:
      print(
          "[INFO] Not found tf2onnx package. Disable the support for exporting onnx model."
      )
  else:
    raise ValueError("Please provide correct output format ['pb', 'onnx'], but got ", output_format)

  #  if dump_as_xir:
  #    in_shapes = None
  #    if q_config.input_shapes is not None and len(q_config.input_shapes) > 0:
  #      in_shapes = q_config.input_shapes
  #    fname = Path(q_config.output_dir)
  #    fname = fname / "quantize_eval_model.xmodel"
  #    convert tf frozen model into xir model and dump the result
  #    xnnc.xir.from_tensorflow(
  #        graph_def=quantize_eval_graph_def,
  #        fname=fname,
  #        layout=xnnc.Layout.NHWC,
  #        in_shapes=in_shapes,
  #    )
  #
  #    print("******************* Serialize to XIR Model *********************\
  #          \nINFO: Output: \
  #          \n  xir_model: {}".format(fname.absolute()))

  return


def quantize_train(input_meta_graph_def, q_config):

  temp_path = os.path.join(q_config.output_dir, "temp")
  if not os.path.exists(temp_path):
    os.makedirs(temp_path)

  float_graph_def = None
  if input_meta_graph_def:
    float_graph_def = input_meta_graph_def.graph_def
  else:
    raise ValueError(
        "You need to provide a `MetaGraphDef` for quantize train.")

  quantize_train_graph_def = CreateQuantizeTrainingGraphDef(
      float_graph_def, q_config)

  input_meta_graph_def.graph_def.Clear()
  input_meta_graph_def.graph_def.CopyFrom(quantize_train_graph_def)

  quantize_train_path = os.path.join(q_config.output_dir, "quantize_train")
  if not os.path.exists(quantize_train_path):
    os.makedirs(quantize_train_path)
  quantize_train_meta_path = os.path.join(quantize_train_path,
                                          "quantize_train.ckpt.meta")
  save_pb_file(input_meta_graph_def, quantize_train_meta_path)
  print("INFO: Quantize train graph are generated in: {}".format(
      quantize_train_meta_path))
  return


def quantize_evaluate(input_meta_graph_def, q_config):

  temp_path = os.path.join(q_config.output_dir, "temp")
  if not os.path.exists(temp_path):
    os.makedirs(temp_path)

  quantize_train_graph_def = None
  if input_meta_graph_def:
    quantize_train_graph_def = input_meta_graph_def.graph_def
  else:
    raise ValueError(
        "You need to provide a `MetaGraphDef` for quantize train.")

  quantize_eval_graph_def = CreateQuantizeEvaluationGraphDef(
      quantize_train_graph_def, q_config)

  input_meta_graph_def.graph_def.Clear()
  input_meta_graph_def.graph_def.CopyFrom(quantize_eval_graph_def)

  quantize_eval_path = os.path.join(q_config.output_dir, "quantize_eval")
  if not os.path.exists(quantize_eval_path):
    os.makedirs(quantize_eval_path)
  quantize_eval_meta_path = os.path.join(quantize_eval_path,
                                         "quantize_eval.ckpt.meta")
  save_pb_file(input_meta_graph_def, quantize_eval_meta_path)
  print("INFO: Quantize eval graph are generated in: {}".format(
      quantize_eval_meta_path))
  return


def deploy_checkpoint(input_meta_graph_def, input_checkpoint, q_config):

  if not tf.train.checkpoint_exists(input_checkpoint):
    raise ValueError("Input checkpoint '" + input_checkpoint +
                     "' does not exits.")

  if tf.gfile.IsDirectory(input_checkpoint):
    input_checkpoint = tf.train.latest_checkpoint(
        input_checkpoint)

  if not os.path.exists(q_config.output_dir):
    os.makedirs(q_config.output_dir)

  quantize_eval_graph_def = None
  if input_meta_graph_def:
    quantize_eval_graph_def = input_meta_graph_def.graph_def
  else:
    raise ValueError("You need to provide a `MetaGraphDef` for deploy.")

  q_config.output_nodes = get_quantized_nodes(quantize_eval_graph_def,
                                              q_config.output_nodes)
  saver = tf.train.import_meta_graph(input_meta_graph_def, clear_devices=True)
  with Session() as sess:
    saver.restore(sess, input_checkpoint)
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, quantize_eval_graph_def, q_config.output_nodes)

  if not os.path.exists(os.path.join(q_config.output_dir, "deploy")):
    os.makedirs(os.path.join(q_config.output_dir, "deploy"))
  quantize_deploy_graph_def = CreateQuantizeDeployGraphDef(
      frozen_graph_def, q_config)
  save_pb_file(quantize_deploy_graph_def,
               os.path.join(q_config.output_dir, "deploy/deploy_model.pb"))

  print("INFO: Quantize deploy graph are generated in: {}".format(
      os.path.join(q_config.output_dir, "deploy")))
  return


def inspect(input_graph_def, input_frozen_graph):
  """Inspect the float graph and parse quantizable patterns,
  then generate possible vai_q_tensorflow command"""
  CheckGraphDef(input_graph_def, input_frozen_graph)


def dump(input_graph_def,
         input_fn,
         output_dir,
         max_dump_batches,
         dump_float,
         s_config=tf.compat.v1.ConfigProto(),
         dump_input_tensors='',
         get_weight_only=False,
         fuse_op_config=None,
         custom_op_set=None):
  """Dump weights and activation data"""
  s_config.graph_options.optimizer_options.opt_level = -1
  s_config.graph_options.rewrite_options.disable_meta_optimizer = True
  w_q_map = dict()
  a_q_map = dict()
  for node in input_graph_def.node:
    if node.op == "FixNeuron":
      if node.name.endswith("wquant"):
        w_q_map[node.name] = int(node.attr['quantize_pos'].i)
      else:
        a_q_map[node.name] = int(node.attr['quantize_pos'].i)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  graph = tf.Graph()
  with graph.as_default():
    if dump_input_tensors:
      # TODO: Support multi input tensors
      image = tf.placeholder(tf.float32,
                             shape=(None, None, None, 3),
                             name="image_tensor")
      tf.graph_util.import_graph_def(input_graph_def,
                          name='',
                          input_map={dump_input_tensors: image})
    else:
      tf.graph_util.import_graph_def(input_graph_def, name='')

    ignore_node_names = []
    if fuse_op_config:
      namescope_map = get_fuse_config(fuse_op_config)
      namescope_map = check_namescope_map(namescope_map,
              input_graph_def)

      plugin_nodes, plugin_output_nodes = get_plugin_output(input_graph_def,
              namescope_map)

      output_nodes_path = os.path.join(output_dir, "output_nodes.txt")
      with open(output_nodes_path, "w") as f:
        for ns, node_names in plugin_output_nodes.items():
          f.write("namescope_map [{} : {}] \n".format(ns, " ".join(node_names)))
      for pn, nodes_lst in plugin_nodes.items():
        for node_name in nodes_lst:
          if node_name not in plugin_output_nodes[pn]:
            ignore_node_names.append(node_name)

    # Get fetches
    w_fetch_tensors = []
    w_fetch_names = []
    a_fetch_tensors = []
    a_fetch_names = []
    for op in graph.get_operations():
      if op.type == "FixNeuron":
        if op.name.endswith("wquant"):
          w_fetch_tensors.append(op.outputs[0])
          w_fetch_names.append(op.name)
        else:
          a_fetch_tensors.append(op.outputs[0])
          a_fetch_names.append(op.name)
      elif dump_float:
        try:
          if op.type not in IGNORE_OP_TYPES and \
                  op.name not in ignore_node_names:
            a_fetch_tensors.append(op.outputs[0])
            a_fetch_names.append(op.name)
          if custom_op_set and op.type in custom_op_set:
            for i in range(1, len(op.inputs)):
              w_tensor = op.inputs[i]
              w_op = w_tensor.op
              if w_op.type == "Const":
                w_fetch_tensors.append(w_tensor)
                w_fetch_names.append(w_op.name)
        except KeyError:
          continue

    # Dump weights/biases
    print("INFO: Start Dumping for {} batches".format(max_dump_batches))
    with Session(config=s_config) as sess:
      print("INFO: Dumping weights/biases...")
      w_fetch_results = sess.run(w_fetch_tensors)
      if get_weight_only:
        name2qdq_tensor = dict(zip(w_fetch_names, w_fetch_results))
        return name2qdq_tensor

      dump_folder = os.path.join(output_dir, "dump_results_weights")
      if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)

      index = 0
      for name, res in zip(w_fetch_names, w_fetch_results):
        index = index + 1
        filename = os.path.join(dump_folder, name.replace("/", "_"))
        print("INFO: Dumping ({}/{}): {}".format(index, len(w_fetch_names),
                                                 name))
        res = res.flatten()

        if name in w_q_map:
          res = res * 2**w_q_map[name]
          res.astype(np.int8).tofile(filename + ".bin")
          np.savetxt(filename + ".txt",
                     res.astype(np.int8),
                     fmt="%s",
                     delimiter=",")

    # Build feed_dict
    input_tensors = [
        op.outputs[0] for op in graph.get_operations()
        if op.type == 'Placeholder'
    ]

    # Run inference and dump activations
    print("INFO: Start Dumping for {} batches".format(max_dump_batches))
    with Session(config=s_config) as sess:
      for i in range(max_dump_batches):
        dump_folder = os.path.join(output_dir, "dump_results_" + str(i))
        if not os.path.exists(dump_folder):
          os.makedirs(dump_folder)

        print("INFO: Dumping for batch: {}/{} ...".format(
            i + 1, max_dump_batches))
        inputs = input_fn(iter=i)
        feed_dict = gen_feed_dict(input_tensors, inputs)
        a_fetch_results = sess.run(a_fetch_tensors, feed_dict)

        index = 0
        for name, res in zip(a_fetch_names, a_fetch_results):
          index = index + 1
          filename = os.path.join(dump_folder, name.replace("/", "_"))
          print("INFO: Dumping ({}/{}): {}".format(index, len(a_fetch_names),
                                                   name))
          res = res.flatten()

          if dump_float:
            res.tofile(filename + "_float.bin")
            np.savetxt(filename + "_float.txt", res, fmt="%s", delimiter=",")

          if name in a_q_map:
            res = res * 2**a_q_map[name]
            res.astype(np.int8).tofile(filename + ".bin")
            np.savetxt(filename + ".txt",
                       res.astype(np.int8),
                       fmt="%s",
                       delimiter=",")
  print("INFO: Dump results are saved in {}.".format(output_dir))
  return

def update_old_model(input_graph_def, save_path):
  """Update model pb for compatible"""
  for n in input_graph_def.node:
    if n.op == "FixNeuron":
      n.attr['T'].type = tf.float32.as_datatype_enum
  save_pb_file(input_graph_def, save_path)
  print("INFO: Old version quantized model pb file has been updated to new version and save inplace")


def main(unused_args, flags):
  os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu
  if not os.getenv("TF_CPP_MIN_LOG_LEVEL"):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
  if not os.getenv("TF_CPP_MIN_VLOG_LEVEL"):
    os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"


  custom_op_set = set()
  if flags.custom_op_so:
    custom_op_so = flags.custom_op_so.split(",")
    for so_pair in custom_op_so:
      op_type, so_path = so_pair.split(":")
      op_type = op_type.strip()
      so_path = so_path.strip()
      custom_op_set.add(op_type)
      _custom_ops = tf.load_op_library(
          tf.resource_loader.get_path_to_datafile(so_path))

  if flags.command == "quantize":
    # Parse flags

    if flags.mode == "frozen":
      input_graph_def = _parse_input_frozen_graph(flags.input_frozen_graph)
      input_nodes = _parse_input_nodes(input_graph_def, flags.input_nodes)
      output_nodes = _parse_output_nodes(input_graph_def, flags.output_nodes)
      input_shapes = _parse_input_shapes(input_nodes, flags.input_shapes)
      ignore_nodes = _parse_ignore_nodes(input_graph_def, flags.ignore_nodes)
      nodes_bit = _parse_nodes_bit(input_graph_def, flags.nodes_bit)
      nodes_method = _parse_nodes_method(input_graph_def, flags.nodes_method)
      q_config = QuantizeConfig(input_nodes=input_nodes,
                                output_nodes=output_nodes,
                                input_shapes=input_shapes,
                                ignore_nodes=ignore_nodes,
                                weight_bit=flags.weight_bit,
                                activation_bit=flags.activation_bit,
                                nodes_bit=nodes_bit,
                                nodes_method=nodes_method,
                                method=flags.method,
                                calib_iter=flags.calib_iter,
                                output_dir=flags.output_dir,
                                align_concat=flags.align_concat,
                                align_pool=flags.align_pool,
                                adjust_shift_bias=flags.adjust_shift_bias,
                                adjust_shift_cut=flags.adjust_shift_cut,
                                simulate_dpu=flags.simulate_dpu,
                                scale_all_avgpool=flags.scale_all_avgpool,
                                do_cle=flags.do_cle,
                                replace_relu6=flags.replace_relu6,
                                replace_sigmoid=flags.replace_sigmoid,
                                replace_softmax=flags.replace_softmax)
      if flags.convert_datatype:
        convert_datatype(input_graph_def, q_config, flags.convert_datatype)
        return
      input_fn = _parse_input_fn(flags.input_fn)
      s_config = _parse_session_config(flags.gpu_memory_fraction)

      add_shapes = (flags.add_shapes != 0)
      keep_float_weight = (flags.keep_float_weight != 0)
      fold_constant = (flags.fold_constant != 0)
      quantize_frozen(input_graph_def, input_fn, q_config, s_config,
                      flags.skip_check, flags.dump_as_xir,
                      flags.fuse_op_config, add_shapes, keep_float_weight,
                      flags.output_format, custom_op_set,
                      fold_constant)

    elif flags.mode == "train":
      input_meta_graph_def = _parse_input_meta_graph(flags.input_meta_graph)
      input_graph_def = input_meta_graph_def.graph_def
      input_nodes = _parse_input_nodes(input_graph_def, flags.input_nodes)
      output_nodes = _parse_output_nodes(input_graph_def, flags.output_nodes)
      input_shapes = _parse_input_shapes(input_nodes, flags.input_shapes)
      ignore_nodes = _parse_ignore_nodes(input_graph_def, flags.ignore_nodes)
      nodes_bit = _parse_nodes_bit(input_graph_def, flags.nodes_bit)
      nodes_method = _parse_nodes_method(input_graph_def, flags.nodes_method)
      q_config = QuantizeConfig(input_nodes=input_nodes,
                                output_nodes=output_nodes,
                                input_shapes=input_shapes,
                                ignore_nodes=ignore_nodes,
                                weight_bit=flags.weight_bit,
                                activation_bit=flags.activation_bit,
                                nodes_bit=nodes_bit,
                                nodes_method=nodes_method,
                                method=flags.method,
                                calib_iter=flags.calib_iter,
                                output_dir=flags.output_dir,
                                align_concat=flags.align_concat,
                                align_pool=flags.align_pool,
                                adjust_shift_bias=flags.adjust_shift_bias,
                                adjust_shift_cut=flags.adjust_shift_cut,
                                simulate_dpu=flags.simulate_dpu,
                                scale_all_avgpool=flags.scale_all_avgpool)

      quantize_train(input_meta_graph_def, q_config)

    elif flags.mode == "eval":
      input_meta_graph_def = _parse_input_meta_graph(flags.input_meta_graph)
      input_graph_def = input_meta_graph_def.graph_def
      input_nodes = _parse_input_nodes(input_graph_def, flags.input_nodes)
      output_nodes = _parse_output_nodes(input_graph_def, flags.output_nodes)
      input_shapes = _parse_input_shapes(input_nodes, flags.input_shapes)
      ignore_nodes = _parse_ignore_nodes(input_graph_def, flags.ignore_nodes)
      nodes_bit = _parse_nodes_bit(input_graph_def, flags.nodes_bit)
      nodes_method = _parse_nodes_method(input_graph_def, flags.nodes_method)
      q_config = QuantizeConfig(input_nodes=input_nodes,
                                output_nodes=output_nodes,
                                input_shapes=input_shapes,
                                ignore_nodes=ignore_nodes,
                                weight_bit=flags.weight_bit,
                                activation_bit=flags.activation_bit,
                                nodes_bit=nodes_bit,
                                nodes_method=nodes_method,
                                method=flags.method,
                                calib_iter=flags.calib_iter,
                                output_dir=flags.output_dir,
                                align_concat=flags.align_concat,
                                align_pool=flags.align_pool,
                                adjust_shift_bias=flags.adjust_shift_bias,
                                adjust_shift_cut=flags.adjust_shift_cut,
                                simulate_dpu=flags.simulate_dpu,
                                scale_all_avgpool=flags.scale_all_avgpool)
      quantize_evaluate(input_meta_graph_def, q_config)

    else:
      print("Unknown mode for quantize: " + flags.mode)
      return -1

  elif flags.command == "deploy":
    input_meta_graph_def = _parse_input_meta_graph(flags.input_meta_graph)
    input_graph_def = input_meta_graph_def.graph_def
    input_nodes = _parse_input_nodes(input_graph_def, flags.input_nodes)
    output_nodes = _parse_output_nodes(input_graph_def, flags.output_nodes)
    input_shapes = _parse_input_shapes(input_nodes, flags.input_shapes)
    ignore_nodes = _parse_ignore_nodes(input_graph_def, flags.ignore_nodes)
    nodes_bit = _parse_nodes_bit(input_graph_def, flags.nodes_bit)
    nodes_method = _parse_nodes_method(input_graph_def, flags.nodes_method)
    q_config = QuantizeConfig(input_nodes=input_nodes,
                              output_nodes=output_nodes,
                              input_shapes=input_shapes,
                              ignore_nodes=ignore_nodes,
                              weight_bit=flags.weight_bit,
                              activation_bit=flags.activation_bit,
                              nodes_bit=nodes_bit,
                              nodes_method=nodes_method,
                              method=flags.method,
                              calib_iter=flags.calib_iter,
                              output_dir=flags.output_dir)
    deploy_checkpoint(input_meta_graph_def, flags.input_checkpoint, q_config)

  elif flags.command == "inspect":
    input_graph_def = _parse_input_frozen_graph(flags.input_frozen_graph)
    inspect(input_graph_def, flags.input_frozen_graph)

  elif flags.command == "dump":
    os.environ["ARCH_TYPE"] = flags.arch_type
    input_graph_def = _parse_input_frozen_graph(flags.input_frozen_graph)
    input_fn = _parse_input_fn(flags.input_fn)
    s_config = _parse_session_config(flags.gpu_memory_fraction)
    dump(input_graph_def, input_fn, flags.output_dir, flags.max_dump_batches,
         flags.dump_float, s_config, flags.dump_input_tensors,
         flags.fuse_op_config, custom_op_set)
  elif flags.command == "update":
    input_graph_def = _parse_input_frozen_graph(flags.input_frozen_graph)
    update_old_model(input_graph_def, flags.input_frozen_graph)


  else:
    print("Unknown Command: " + flags.command)
    return -1


def version_string():
  version = "Vai_q_tensorflow " + __version__
  version += " build for Tensorflow " + pywrap_tensorflow.__version__
  version += "\ngit version " + __git_version__
  return version


def usage_string():
  usage = """
    usage: vai_q_tensorflow command [Options]

    examples:
      show help       : vai_q_tensorflow --help
      quantize a model: vai_q_tensorflow quantize --input_frozen_graph frozen_graph.pb --input_nodes xxx --output_nodes yyy --input_shapes zzz --input_fn module.calib_input
      inspect a model : vai_q_tensorflow inspect --input_frozen_graph frozen_graph.pb
      dump quantized model : vai_q_tensorflow dump --input_frozen_graph quantize_results/quantize_eval_model.pb --input_fn module.dump_input
      update old quantized model: vai_q_tensorflow update --input_frozen_graph quantize_results/quantize_eval_model.pb
  """
  return usage


def run_main():
  parser = argparse.ArgumentParser(
      description="Xilinx's Quantization Tools" + version_string(),
      usage=usage_string(),
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--version", action="version", version=version_string())
  parser.register("type", "bool", lambda v: v.lower() == "true")

  parser.add_argument("command",
                      type=str,
                      default="",
                      help="Specify a command for vai_q_tensorflow",
                      choices=['quantize', 'inspect', 'dump', 'deploy', 'update'])

  ####################
  #  Main Arguments  #
  ####################
  parser.add_argument("--mode",
                      type=str,
                      default="frozen",
                      help="Mode for quantization.",
                      choices=['frozen', 'train', 'eval'])
  parser.add_argument("--input_frozen_graph",
                      type=str,
                      default="",
                      help="Path to input frozen graph(.pb)")
  parser.add_argument("--input_meta_graph",
                      type=str,
                      default="",
                      help="Path to input meta graph(.meta)")
  parser.add_argument("--input_checkpoint",
                      type=str,
                      default="",
                      help="Path to checkpoint files")
  parser.add_argument("--output_dir",
                      type=str,
                      default="./quantize_results",
                      help="The directory to save the quantization results")

  ######################################
  #  Quantize Configuration Arguments  #
  ######################################
  parser.add_argument("--weight_bit",
                      type=int,
                      default=8,
                      help="The target bit width for weights/biases")
  parser.add_argument("--activation_bit",
                      type=int,
                      default=8,
                      help="The target bit width for activation")
  parser.add_argument(
      "--nodes_bit",
      type=str,
      default="",
      help="Specify bit width of nodes, nodes name and bit width \
      form a pair of parameter joined by a colon, and parameters \
      are comma separated. When specify conv op name only vai_q_tensorflow \
      will quantize weights of conv op using specified bit width . \
      e.g 'conv1/Relu:16,conv1/weights:8,conv1:16'  If using python api then \
      should be used like this, nodes_bit=['input:16','Conv2D:16', 'add:16']")
  parser.add_argument(
      "--nodes_method",
      type=str,
      default="",
      help="Specify method of nodes, nodes name and method \
      form a pair of parameter joined by a colon, and parameter pairs \
      are comma separated. When specify conv op name only vai_q_tensorflow \
      will quantize weights of conv op using specified method. \
      e.g 'conv1/Relu:1,depthwise_conv1/weights:2,conv1:1',  If using python api then \
      should be used like this, nodes_method=['input:0','Conv2D:1', 'add:2']")
  parser.add_argument("--method",
                      type=int,
                      default=1,
                      choices=[0, 1, 2],
                      help=" The method for quantization, options are: \
      0: non-overflow method, make sure no values are saturated during quantization, \
        may get bad results incase of outliers. \
      1: min-diffs method, allow saturation for large values during quantization to get smaller quantization errors. \
        This method is slower than method 0 but has higher endurance to outliers. \
      2: min-diffs method with strategy for depthwise, allow saturation for large values during quantization to get smaller quantization errors. \
        And apply special strategy for depthwise weights, but do implement method 1 to normal weights and activation. \
        This method is slower than method 0 but has higher endurance to outliers."
                      )
  parser.add_argument(
      "--calib_iter",
      type=int,
      default=100,
      help=
      "The iterations of calibration, total number of images for calibration = calib_iter * batch_size"
  )
  parser.add_argument(
      "--input_nodes",
      type=str,
      default="",
      help=
      "The name list of input nodes of the subgraph to be quantized, comma separated. \
      Used together with output_nodes.  When generating the model for deploy, only the subgraph between input_nodes and \
      output_nodes will be included. Please set it to the begining of the main body fo the model to quantize, \
      such as the nodes after data preprocessing and augmentation.")
  parser.add_argument(
      "--input_shapes",
      type=str,
      default="",
      help=
      "The shape list of input_nodes, The shape must be a 4-dimension shape for each node, comma separated, e.g. 1,224,224,3;\
      Unknown size for batchsize is supported, e.g. ?,224,224,3; \
      In case of multiple input_nodes, please assign the shape list of each node,\
      separated by `:`. e.g. ?,224,224,3:?,300,300,1")
  parser.add_argument(
      "--output_nodes",
      type=str,
      default="",
      help=
      "The name list of output nodes of the subgraph to be quantized, comma separated. \
      Used together with input_nodes.  When generating the model for deploy, only the subgraph between input_nodes and \
      output_nodes will be included. Please set it to the end of the main body of the model to quantize, \
      such as the nodes before postprocessing.")
  parser.add_argument(
      "--ignore_nodes",
      type=str,
      default="",
      help=
      "The name list of nodes to be ignored during quantization, comma separated. The ignored nodes \
      will be left unquantized during quantization even if it is quantizable.  This argument has no effect for non-quantizable nodes. \
      e.g 'conv1/Relu,depthwise_conv1/weights,conv1',  If using python api then \
      should be used like this, ignore_nodes=['conv1/Relu','depthwise_conv1/weights','conv1']")
  parser.add_argument("--skip_check",
                      type=int,
                      default=0,
                      choices=[0, 1],
                      help="Set to 1 to skip the check for float model.")
  parser.add_argument("--fold_constant",
                      type=int,
                      default=1,
                      choices=[0, 1],
                      help="Set to 1 to do fold_constant for graph_def.")
  parser.add_argument(
      "--align_concat",
      type=int,
      default=0,
      choices=[0, 1, 2],
      help=
      "The strategy for alignment of the input quantize positions for concat nodes. Set to 0 to align \
      all concat nodes, 1 to align the output concat nodes, 2 to disable alignment"
  )
  parser.add_argument(
      "--align_pool",
      type=int,
      default=0,
      choices=[0, 1, 2],
      help=
      "The strategy for alignment of the input quantize positions for maxpool/avgpool nodes. Set to 0 to align \
      all maxpool/avgpool nodes, 1 to align the output maxpool/avgpool nodes, 2 to disable alignment"
  )
  parser.add_argument(
      "--adjust_shift_bias",
      type=int,
      default=1,
      choices=[0, 1, 2],
      help=
      "The strategy for shift bias check and adjustment for DPU compiler. Set to 0 to disable shift bias\
          check and adjustment, 1 to enable with static constraints, 2 to enable with dynamic constraints."
  )
  parser.add_argument(
      "--adjust_shift_cut",
      type=int,
      default=1,
      choices=[0, 1],
      help=
      "The strategy for shift cut check and adjustment for DPU compiler. Set to 0 to disable shift cut\
          check and adjustment, 1 to enable with static constraints.")
  parser.add_argument(
      "--simulate_dpu",
      type=int,
      default=1,
      choices=[0, 1],
      help=
      "Set to 1 to enable simulation of DPU. The behavior of DPU for some operations are different from tensorflow. \
      For example, the dividing in LeakyRelu and AvgPooling are replaced by bit-shifting, so there maybe slight difference \
      between DPU outputs and CPU/GPU outputs. This quantizer will simulate the behavior for these operations if this flag is set to 1"
  )
  parser.add_argument(
      "--scale_all_avgpool",
      type=int,
      default=1,
      choices=[0, 1],
      help=
      "Set to 1 to enable scale output of AvgPooling op to simulate DPU. Only kernel_size <= 256 will be scaled. \
      This operation do not affect the special case such as kernel_size=3,5,6,7,14"
  )
  parser.add_argument(
      "--do_cle",
      type=int,
      default=0,
      choices=[0, 1],
      help=
      "Set to 1 to enable implement cross layer equalization to adjust the weights distribution . \
      Set to 0 will skip cross layer equalization operation "
  )
  parser.add_argument(
      "--replace_relu6",
      type=int,
      default=1,
      choices=[0, 1],
      help=
      "Set to 1 to enable replace relu6 with relu. \
      Set to 0 will skip replacement."
  )
  parser.add_argument(
      "--replace_sigmoid",
      type=int,
      default=0,
      choices=[0, 1],
      help=
      "Set to 1 to enable replace sigmoid with hard-sigmoid. \
      Set to 0 will skip replacement."
  )
  parser.add_argument(
      "--replace_softmax",
      type=int,
      default=0,
      choices=[0, 1],
      help=
      "Set to 1 to enable replace softmax with dpu version softmax. \
      Set to 0 will skip replacement."
  )
  parser.add_argument(
      "--custom_op_so",
      type=str,
      default="",
      help=
      "[experimental function]Pass the op type and path to vitis tensorflow 1.15 quantize tool.  The op type and path \
      are connected by a colon to form a complete pair of custom op parameters.  If there are multiple \
      custom *.so that need to be loaded, separate them with a comma.  For example \
      ParamRelu:_param_relu_ops.so,TimeTwo:_time_two_ops.so")
  parser.add_argument(
      "--add_shapes",
      type=int,
      default=1,
      choices=[0, 1],
      help=
      "Set to 1 to enable save shape information into graph_def. \
      Set to 0 will not save."
  )
  parser.add_argument(
      "--convert_datatype",
      type=int,
      default=0,
      choices=[0, 1, 2, 3, 4],
      help=
      "Set to 1 will do fold bn and convert to data type float point 16. \
      Set to 2 will do fold bn and convert to data type double. \
      Set to 3 will do fold bn and convert to data type bfloat16. \
      Set to 4 will do fold bn and convert to data type float. \
      Set to 0 will skip convert data type."
  )
  parser.add_argument(
      "--keep_float_weight",
      type=int,
      default=1,
      choices=[0, 1],
      help=
      "Set to 1 to keep original float weight. \
      Set to 0 will set weight as the value of corresponding wquant output."
  )
  parser.add_argument(
      "--output_format",
      type=str,
      default="pb",
      choices=["pb", "onnx"],
      help=
      "indicates what format to save the quantized model, 'pb' for saving tensorflow " \
      " frozen pb, 'onnx' for saving onnx model.")

  ############################################
  #  Input Function Configuration Arguments  #
  ############################################
  parser.add_argument(
      "--input_fn",
      type=str,
      default="",
      help=
      "The python importable function that provides the input data. The format is \
      `module_name.input_fn_name`, e.g. 'my_input_fn.input_fn'. The input_fn should take a `int` object as input \
      indicating the calibration step, and should return a dict`(placeholder_node_name : numpy.Array)` object \
      for each call, which will be fed into the model's placeholder nodes.")

  parser.add_argument(
      "--fuse_op_config",
      type=str,
      default="",
      help=
      "[experimental function] The json file that indicate how to fuse ops into one.")

  ##################################
  #  Dump Configuration Arguments  #
  ##################################
  parser.add_argument("--max_dump_batches",
                      type=int,
                      default=1,
                      help="The maximum batches to be dumped")
  parser.add_argument(
      "--dump_float",
      type=int,
      default=0,
      choices=[0, 1],
      help=
      "Set to 1 to dump the float weights/biases and activation tensors together with the quantized tensors."
  )
  parser.add_argument(
      "--dump_input_tensors",
      type=str,
      default="",
      help=
      "Specify input tensor name of Graph when graph entrance is not a placeholder. "
      "We will add a placeholder according to the dump_input_tensor, so that input_fn can feed data."
  )
  parser.add_argument(
      "--dump_as_xir",
      default=False,
      action="store_true",
      help=
      "Specify whether dump quantized tensorflow frozen model as xir model.",
  )
  parser.add_argument(
      "--arch_type",
      type=str,
      default="DEFAULT",
      choices=["DEFAULT",  'DPUCADF8H'],
      help=
      "Specify the arch type for fix neuron. 'DEFAULT' means quantization range of both wquant and aquant \
      is [-128, 127]. 'DPUCADF8H' means wquant quantization range is [-128, 127] while aquant is [-127, 127]",
  )

  #####################################
  #  Session Configuration Arguments  #
  #####################################
  parser.add_argument(
      "--gpu",
      type=str,
      default="0",
      help="The gpu id used for quantization, comma separated.")
  parser.add_argument(
      "--gpu_memory_fraction",
      type=float,
      default=0.5,
      help="The gpu memory fraction used for quantization, between 0-1.")

  FLAGS, unparsed = parser.parse_known_args()
  if unparsed:
    raise ValueError("Unknown arguments: ", unparsed)

  my_main = lambda unused_args: main(unused_args, FLAGS)
  tf.compat.v1.app.run(main=my_main, argv=[sys.argv[0]] + unparsed)


if __name__ == '__main__':
  run_main()
