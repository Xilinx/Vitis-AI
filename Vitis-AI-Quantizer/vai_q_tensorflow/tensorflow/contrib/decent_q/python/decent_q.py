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

def calib_input(iter):
  image = load_image(iter)
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
import sys
import time
import tempfile
from progressbar import ProgressBar

from tensorflow.python import pywrap_tensorflow
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client.session import Session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import app
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import saver as saver_lib

from tensorflow.contrib.decent_q.python.utils import *
from tensorflow.contrib.decent_q.python.quantize_graph import *

FLAGS = None


def _parse_input_frozen_graph(input_frozen_graph):
  """Parse input_frozen_graph configurations"""
  if input_frozen_graph == '':
    raise ValueError("No --input_frozen_graph assigned.")
  if not gfile.Exists(input_frozen_graph):
    raise ValueError("Input frozen graph file '" + input_frozen_graph +
                     "' does not exist!")
  graph_def = graph_pb2.GraphDef()
  with gfile.GFile(input_frozen_graph, "rb") as f:
    graph_def.ParseFromString(f.read())
  return graph_def


def _parse_input_graph(input_graph):
  """Parse input_graph configurations"""
  if input_frozen_graph == '':
    raise ValueError("No --input_graph assigned.")
  if not gfile.Exists(input_graph):
    raise ValueError("Input graph file '" + input_graph + "' does not exist!")
  graph_def = graph_pb2.GraphDef()
  with gfile.GFile(input_graph, "rb") as f:
    graph_def.ParseFromString(f.read())
  return graph_def


def _parse_input_meta_graph(input_meta_graph):
  """Parse input_meta_graph configurations"""
  if not gfile.Exists(input_meta_graph):
    raise ValueError("Input meta graph file '" + input_meta_graph +
                     "' does not exist.")
  meta_graph_def = MetaGraphDef()
  with gfile.GFile(input_meta_graph, "rb") as f:
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


def _parse_nodes_bit(input_graph_def, nodes_bit_str):
  """Parse nodes_bit configurations"""
  nodes_bit = []
  if nodes_bit_str:
    nodes_bit = nodes_bit_str.split(",")
  return nodes_bit


def _parse_output_nodes(input_graph_def, output_nodes_str):
  """Parse output_nodes configurations"""
  output_nodes = []
  if output_nodes_str:
    output_nodes = output_nodes_str.split(",")
    check_node_names(input_graph_def, output_nodes)
  else:
    raise ValueError("No --output_nodes assigned.")
  return output_nodes


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
    raise ValueError("input_shapes should be the same length as input_nodes")

  for input_shape in input_shapes_str.split(":"):
    shape_str = input_shape.split(",")
    if len(shape_str) != 4:
      raise ValueError(
          'Input_shapes should be 4-dimension int shape list (support unknown batch_size), e.g. ?,224,224,3 or 1,299,299,3'
      )
    st = 1 if shape_str[0] == '?' else 0
    shape = []
    if shape_str[0] == '?':
      shape_str[0] = -1
    try:
      shape[0:3] = [int(s) for s in shape_str]
    except Exception as e:
      raise ValueError(
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
      raise ValueError('Fail to import input_fn, error: ', e)
  return input_fn


def _parse_session_config(gpu, gpu_memory_fraction):
  """Parse session configurations"""
  s_config = config_pb2.ConfigProto()
  s_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
  # Disable graph optimizer and rewiter to make sure every quantize node works correctly
  s_config.graph_options.optimizer_options.opt_level = -1
  s_config.graph_options.rewrite_options.disable_meta_optimizer = True
  return s_config


def check_float_graph(input_graph_def, input_fn, q_config, s_config):
  """Check if float graph and input_fn is validate before quantization"""
  graph = ops.Graph()
  with graph.as_default():
    importer.import_graph_def(input_graph_def, name='')
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
      inputs = input_fn(iter=0)
      feed_dict = gen_feed_dict(input_tensors, inputs)
      sess.run(output_tensors, feed_dict)
  print("INFO: Float Graph Check Done.")


def calibrate_frozen(input_graph_def, input_fn, q_config, s_config):
  """Transform float graph to quantized graph and do calibration"""

  temp_path = os.path.join(q_config.output_dir, "temp")
  if not os.path.exists(temp_path):
    os.makedirs(temp_path)

  # Calibration
  calib_graph_def = CreateQuantizeCalibrationGraphDef(input_graph_def,
                                                      q_config)
  graph = ops.Graph()
  with graph.as_default():
    importer.import_graph_def(calib_graph_def, name='')
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
      progress = ProgressBar()
      for i in progress(range(0, q_config.calib_iter)):
        inputs = input_fn(iter=i)
        feed_dict = gen_feed_dict(input_tensors, inputs)
        sess.run(output_tensors, feed_dict)
  print("INFO: Calibration Done.")

  # Quantized Evaluation
  quantize_eval_graph_def = CreateQuantizeEvaluationGraphDef(
      calib_graph_def, q_config)
  save_pb_file(quantize_eval_graph_def,
               os.path.join(q_config.output_dir, "quantize_eval_model.pb"))
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


def quantize_frozen(input_graph_def,
                    input_fn,
                    q_config=QuantizeConfig(),
                    s_config=config_pb2.ConfigProto(),
                    skip_check=0):
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

  Returns:
    quantize_eval_graph_def: A `GraphDef` object, the quantized model for evaluation on gpu or cpu.
    deploy_graph_def: A `GraphDef` object, the quantized model for dpu deployment.
  """

  if not skip_check:
    check_float_graph(input_graph_def, input_fn, q_config, s_config)

  quantize_eval_graph_def = calibrate_frozen(input_graph_def, input_fn,
                                             q_config, s_config)
  deploy_graph_def = deploy_frozen(quantize_eval_graph_def, q_config)

  # Summarize Quantize Results
  print("********************* Quantization Summary *********************\
      \nINFO: Output: \
      \n  quantize_eval_model: {} \
      \n  deploy_model: {}".format(
      os.path.join(q_config.output_dir, "quantize_eval_model.pb"),
      os.path.join(q_config.output_dir, "deploy_model.pb")))
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

  if not checkpoint_management.checkpoint_exists(input_checkpoint):
    raise ValueError("Input checkpoint '" + input_checkpoint +
                     "' does not exits.")

  if gfile.IsDirectory(input_checkpoint):
    input_checkpoint = checkpoint_management.latest_checkpoint(
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
  saver = saver_lib.import_meta_graph(input_meta_graph_def, clear_devices=True)
  with Session() as sess:
    saver.restore(sess, input_checkpoint)
    frozen_graph_def = graph_util.convert_variables_to_constants(
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
         s_config,
         dump_input_tensors=''):
  """Dump weights and activation data"""
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

  graph = ops.Graph()
  with graph.as_default():
    if dump_input_tensors:
      # TODO: Support multi input tensors
      image = array_ops.placeholder(dtypes.float32,
                             shape=(None, None, None, 3),
                             name="image_tensor")
      importer.import_graph_def(input_graph_def,
                          name='',
                          input_map={dump_input_tensors: image})
    else:
      importer.import_graph_def(input_graph_def, name='')

    # Get fetches
    w_fetch_tensors = []
    w_fetch_names = []
    a_fetch_tensors = []
    a_fetch_names = []
    for op in graph.get_operations():
      if dump_float:
        try:
          a_fetch_tensors.append(op.outputs[0])
          a_fetch_names.append(op.name)
        except KeyError:
          continue
      elif op.type == "FixNeuron":
        if op.name.endswith("wquant"):
          w_fetch_tensors.append(op.outputs[0])
          w_fetch_names.append(op.name)
        else:
          a_fetch_tensors.append(op.outputs[0])
          a_fetch_names.append(op.name)

    # Dump weights/biases
    print("INFO: Start Dumping for {} batches".format(max_dump_batches))
    with Session(config=s_config) as sess:
      dump_folder = os.path.join(output_dir, "dump_results_weights")
      if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)

      print("INFO: Dumping weights/biases...")
      w_fetch_results = sess.run(w_fetch_tensors)

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


def main(unused_args, flags):
  os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu
  if not os.getenv("TF_CPP_MIN_LOG_LEVEL"):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
  if not os.getenv("TF_CPP_MIN_VLOG_LEVEL"):
    os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"

  if flags.command == "quantize":
    # Parse flags

    if flags.mode == "frozen":
      input_graph_def = _parse_input_frozen_graph(flags.input_frozen_graph)
      input_nodes = _parse_input_nodes(input_graph_def, flags.input_nodes)
      output_nodes = _parse_output_nodes(input_graph_def, flags.output_nodes)
      input_shapes = _parse_input_shapes(input_nodes, flags.input_shapes)
      ignore_nodes = _parse_ignore_nodes(input_graph_def, flags.ignore_nodes)
      nodes_bit = _parse_nodes_bit(input_graph_def, flags.nodes_bit)
      q_config = QuantizeConfig(input_nodes=input_nodes,
                                output_nodes=output_nodes,
                                input_shapes=input_shapes,
                                ignore_nodes=ignore_nodes,
                                weight_bit=flags.weight_bit,
                                activation_bit=flags.activation_bit,
                                nodes_bit=nodes_bit,
                                method=flags.method,
                                calib_iter=flags.calib_iter,
                                output_dir=flags.output_dir,
                                align_concat=flags.align_concat,
                                adjust_shift_bias=flags.adjust_shift_bias,
                                adjust_shift_cut=flags.adjust_shift_cut,
                                simulate_dpu=flags.simulate_dpu)
      input_fn = _parse_input_fn(flags.input_fn)
      s_config = _parse_session_config(flags.gpu, flags.gpu_memory_fraction)

      quantize_frozen(input_graph_def, input_fn, q_config, s_config,
                      flags.skip_check)

    elif flags.mode == "train":
      input_meta_graph_def = _parse_input_meta_graph(flags.input_meta_graph)
      input_graph_def = input_meta_graph_def.graph_def
      input_nodes = _parse_input_nodes(input_graph_def, flags.input_nodes)
      output_nodes = _parse_output_nodes(input_graph_def, flags.output_nodes)
      input_shapes = _parse_input_shapes(input_nodes, flags.input_shapes)
      ignore_nodes = _parse_ignore_nodes(input_graph_def, flags.ignore_nodes)
      nodes_bit = _parse_nodes_bit(input_graph_def, flags.nodes_bit)
      q_config = QuantizeConfig(input_nodes=input_nodes,
                                output_nodes=output_nodes,
                                input_shapes=input_shapes,
                                ignore_nodes=ignore_nodes,
                                weight_bit=flags.weight_bit,
                                activation_bit=flags.activation_bit,
                                nodes_bit=nodes_bit,
                                method=flags.method,
                                calib_iter=flags.calib_iter,
                                output_dir=flags.output_dir,
                                align_concat=flags.align_concat,
                                adjust_shift_bias=flags.adjust_shift_bias,
                                adjust_shift_cut=flags.adjust_shift_cut,
                                simulate_dpu=flags.simulate_dpu)

      quantize_train(input_meta_graph_def, q_config)

    elif flags.mode == "eval":
      input_meta_graph_def = _parse_input_meta_graph(flags.input_meta_graph)
      input_graph_def = input_meta_graph_def.graph_def
      input_nodes = _parse_input_nodes(input_graph_def, flags.input_nodes)
      output_nodes = _parse_output_nodes(input_graph_def, flags.output_nodes)
      input_shapes = _parse_input_shapes(input_nodes, flags.input_shapes)
      ignore_nodes = _parse_ignore_nodes(input_graph_def, flags.ignore_nodes)
      nodes_bit = _parse_nodes_bit(input_graph_def, flags.nodes_bit)
      q_config = QuantizeConfig(input_nodes=input_nodes,
                                output_nodes=output_nodes,
                                input_shapes=input_shapes,
                                ignore_nodes=ignore_nodes,
                                weight_bit=flags.weight_bit,
                                activation_bit=flags.activation_bit,
                                nodes_bit=nodes_bit,
                                method=flags.method,
                                calib_iter=flags.calib_iter,
                                output_dir=flags.output_dir,
                                align_concat=flags.align_concat,
                                adjust_shift_bias=flags.adjust_shift_bias,
                                adjust_shift_cut=flags.adjust_shift_cut,
                                simulate_dpu=flags.simulate_dpu)
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
    q_config = QuantizeConfig(input_nodes=input_nodes,
                              output_nodes=output_nodes,
                              input_shapes=input_shapes,
                              ignore_nodes=ignore_nodes,
                              weight_bit=flags.weight_bit,
                              activation_bit=flags.activation_bit,
                              nodes_bit=nodes_bit,
                              method=flags.method,
                              calib_iter=flags.calib_iter,
                              output_dir=flags.output_dir)
    deploy_checkpoint(input_meta_graph_def, flags.input_checkpoint, q_config)

  elif flags.command == "inspect":
    input_graph_def = _parse_input_frozen_graph(flags.input_frozen_graph)
    inspect(input_graph_def, flags.input_frozen_graph)

  elif flags.command == "dump":
    input_graph_def = _parse_input_frozen_graph(flags.input_frozen_graph)
    input_fn = _parse_input_fn(flags.input_fn)
    s_config = _parse_session_config(flags.gpu, flags.gpu_memory_fraction)
    dump(input_graph_def, input_fn, flags.output_dir, flags.max_dump_batches,
         flags.dump_float, s_config, flags.dump_input_tensors)

  else:
    print("Unknown Command: " + flags.command)
    return -1


def version_string():
  version_number = "v1.2.0"
  version = "Vai_q_tensorflow " + version_number
  version += " build for Tensorflow " + pywrap_tensorflow.__version__
  return version


def usage_string():
  usage = """
    usage: vai_q_tensorflow command [Options]

    examples:
      show help       : vai_q_tensorflow --help
      quantize a model: vai_q_tensorflow quantize --input_frozen_graph frozen_graph.pb --input_nodes xxx --output_nodes yyy --input_shapes zzz --input_fn module.calib_input
      inspect a model : vai_q_tensorflow inspect --input_frozen_graph frozen_graph.pb
      dump quantized model : vai_q_tensorflow dump --input_frozen_graph quantize_results/quantize_eval_model.pb --input_fn module.dump_input
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
                      choices=['quantize', 'inspect', 'dump', 'deploy'])

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
      are comma separated. As node name may be different, node \
      name should be from quanized graph rather than from float \
      graph. e.g 'conv1/Relu:16,conv1/weights:8'")
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
        And apply special strategy for depthwise weights. This method is slower than method 0 but has higher endurance to outliers."
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
      will be left unquantized during quantization even if it is quantizable. This argument has no effect for non-quantizable nodes."
  )
  parser.add_argument("--skip_check",
                      type=int,
                      default=0,
                      choices=[0, 1],
                      help="Set to 1 to skip the check for float model.")
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
  app.run(main=my_main, argv=[sys.argv[0]] + unparsed)


if __name__ == '__main__':
  run_main()
