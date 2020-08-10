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
"""quantize graph python api"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time

# pylint: disable=unused-import,wildcard-import, line-too-long
from tensorflow.python.framework.ops import get_default_graph
from tensorflow.python.framework.ops import Operation
from tensorflow.python.client.session import Session
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.framework import graph_util

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.pywrap_tensorflow import DecentQCheckGraphWithStringInputs
from tensorflow.python.pywrap_tensorflow import DecentQConvertConstantsToVariablesWithStringInputs
from tensorflow.python.pywrap_tensorflow import DecentQCreateOptimizedGraphWithStringInputs
from tensorflow.python.pywrap_tensorflow import DecentQCreateQuantizeCalibrationGraphWithStringInputs
from tensorflow.python.pywrap_tensorflow import DecentQCreateQuantizeTrainingGraphWithStringInputs
from tensorflow.python.pywrap_tensorflow import DecentQCreateQuantizeEvaluationGraphWithStringInputs
from tensorflow.python.pywrap_tensorflow import DecentQConvertFoldedBatchnormsWithStringInputs
from tensorflow.python.pywrap_tensorflow import DecentQCreateQuantizeDeployGraphWithStringInputs
from tensorflow.contrib.decent_q.python.utils import *


class QuantizeConfig:
  """Quantize configuration"""
  def __init__(self,
               input_nodes=[],
               output_nodes=[],
               input_shapes=[],
               ignore_nodes=[],
               weight_bit=8,
               activation_bit=8,
               nodes_bit=[],
               method=1,
               calib_iter=100,
               output_dir="./quantize_results",
               align_concat=0,
               adjust_shift_bias=0,
               adjust_shift_cut=0,
               simulate_dpu=0):
    if not (isinstance(input_nodes, list)):
      raise TypeError('input_nodes should be list(str)')
    if not (isinstance(input_shapes, list)):
      raise TypeError('input_shapes should be list(list(int))')
    if not (isinstance(output_nodes, list)):
      raise TypeError('output_nodes should be list(str)')
    if not (isinstance(ignore_nodes, list)):
      raise TypeError('ignore_nodes should be list(str)')

    self.input_nodes = input_nodes
    self.output_nodes = output_nodes
    self.input_shapes = input_shapes
    self.ignore_nodes = ignore_nodes
    self.weight_bit = weight_bit
    self.activation_bit = activation_bit
    self.nodes_bit = nodes_bit
    self.method = method
    self.calib_iter = calib_iter
    self.output_dir = output_dir
    self.align_concat = align_concat
    self.adjust_shift_bias = adjust_shift_bias
    self.adjust_shift_cut = adjust_shift_cut
    self.simulate_dpu = simulate_dpu

  def to_string(self):
    config_string = ''
    for name in self.input_nodes:
      config_string += 'input_nodes,' + name + ','
    for name in self.output_nodes:
      config_string += 'output_nodes,' + name + ','
    for shape in self.input_shapes:
      if not (isinstance(shape, list)):
        raise TypeError('input_shapes should be list(list(int))')
      shape = [str(s) for s in shape]
      config_string += 'input_shapes,' + '*'.join(shape) + ','
    for name in self.ignore_nodes:
      config_string += 'ignore_nodes,' + name + ','
    for node_bit in self.nodes_bit:
      config_string += 'nodes_bit,' + node_bit + ','
    config_string += 'weight_bit,' + str(self.weight_bit) + ','
    config_string += 'activation_bit,' + str(self.activation_bit) + ','
    config_string += 'method,' + str(self.method) + ','
    config_string += 'calib_iter,' + str(self.calib_iter) + ','
    config_string += 'output_dir,' + str(self.output_dir) + ','
    config_string += 'align_concat,' + str(self.align_concat) + ','
    config_string += 'adjust_shift_bias,' + str(self.adjust_shift_bias) + ','
    config_string += 'adjust_shift_cut,' + str(self.adjust_shift_cut) + ','
    config_string += 'simulate_dpu,' + str(self.simulate_dpu) + ','
    return compat.as_bytes(config_string)


def CheckGraphDef(input_graph_def, graph_path):
  """Python wrapper for the decent_q check graph tool.

  Args:
    input_graph_def: GraphDef object containing a model to be checked.
    graph_path: string object of the graph path

  Returns:
    None
  """

  input_graph_def_string = input_graph_def.SerializeToString()
  graph_path_string = compat.as_bytes(graph_path)
  with errors.raise_exception_on_not_ok_status() as status:
    DecentQCheckGraphWithStringInputs(input_graph_def_string,
                                      graph_path_string, status)
  return


def ConvertConstantsToVariables(input_graph_def, config):
  """Python wrapper for the decent_q convert constants to variables tool.

  Args:
    input_graph_def: GraphDef object containing a model to be transformed.
    config: QuantizeConfig object

  Returns:
    Transformed GraphDef with variables.
  """

  input_graph_def_string = input_graph_def.SerializeToString()
  with errors.raise_exception_on_not_ok_status() as status:
    output_graph_def_string = DecentQConvertConstantsToVariablesWithStringInputs(
        input_graph_def_string, config.to_string(), status)
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.ParseFromString(output_graph_def_string)
  return output_graph_def


def CreateOptimizedGraphDef(input_graph_def, config):
  """Python wrapper for the decent_q create optimized graph_def tool.

  Args:
    input_graph_def: GraphDef object containing a model to be transformed.
    config: QuantizeConfig object

  Returns:
    Optimized GraphDef for quantization.
  """

  input_graph_def_string = input_graph_def.SerializeToString()
  with errors.raise_exception_on_not_ok_status() as status:
    output_graph_def_string = DecentQCreateOptimizedGraphWithStringInputs(
        input_graph_def_string, config.to_string(), status)
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.ParseFromString(output_graph_def_string)
  return output_graph_def


def CreateQuantizeCalibrationGraphDef(input_graph_def, config):
  """Python wrapper for the decent_q create calibration graph_def tool.

  Args:
    input_graph_def: GraphDef object containing a model to be transformed.
    config: QuantizeConfig object

  Returns:
    Transformed GraphDef for quantize calibration.
  """

  input_graph_def_string = input_graph_def.SerializeToString()
  with errors.raise_exception_on_not_ok_status() as status:
    output_graph_def_string = DecentQCreateQuantizeCalibrationGraphWithStringInputs(
        input_graph_def_string, config.to_string(), status)
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.ParseFromString(output_graph_def_string)
  return output_graph_def


def CreateQuantizeTrainingGraphDef(input_graph_def, config):
  """Python wrapper for the decent_q create training graph_def tool.

  Args:
    input_graph_def: GraphDef object containing a model to be transformed.
    config: QuantizeConfig object

  Returns:
    Transformed GraphDef for quantize training.
  """

  input_graph_def_string = input_graph_def.SerializeToString()
  with errors.raise_exception_on_not_ok_status() as status:
    output_graph_def_string = DecentQCreateQuantizeTrainingGraphWithStringInputs(
        input_graph_def_string, config.to_string(), status)
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.ParseFromString(output_graph_def_string)
  return output_graph_def


def CreateQuantizeEvaluationGraphDef(input_graph_def, config):
  """Python wrapper for the decent_q create evaluation graph_def tool.

  Args:
    input_graph_def: GraphDef object containing a model to be transformed.
    config: QuantizeConfig object

  Returns:
    Transformed GraphDef for quantize evaluation.
  """

  input_graph_def_string = input_graph_def.SerializeToString()
  with errors.raise_exception_on_not_ok_status() as status:
    output_graph_def_string = DecentQCreateQuantizeEvaluationGraphWithStringInputs(
        input_graph_def_string, config.to_string(), status)
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.ParseFromString(output_graph_def_string)
  return output_graph_def


def ConvertFoldedBatchnorms(input_graph_def, config):
  """Python wrapper for the decent_q create deploy graph_def tool.

  Args:
    input_graph_def: GraphDef object containing a model to be transformed.
    config: QuantizeConfig object

  Returns:
    Transformed GraphDef for converting folded batchnorms.
  """

  input_graph_def_string = input_graph_def.SerializeToString()
  with errors.raise_exception_on_not_ok_status() as status:
    output_graph_def_string = DecentQConvertFoldedBatchnormsWithStringInputs(
        input_graph_def_string, config.to_string(), status)
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.ParseFromString(output_graph_def_string)
  return output_graph_def


def CreateQuantizeDeployGraphDef(input_graph_def, config):
  """Python wrapper for the decent_q create deploy graph_def tool.

  Args:
    input_graph_def: GraphDef object containing a model to be transformed.
    config: QuantizeConfig object

  Returns:
    Transformed GraphDef for quantize deploy.
  """

  input_graph_def_string = input_graph_def.SerializeToString()
  with errors.raise_exception_on_not_ok_status() as status:
    output_graph_def_string = DecentQCreateQuantizeDeployGraphWithStringInputs(
        input_graph_def_string, config.to_string(), status)
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.ParseFromString(output_graph_def_string)
  return output_graph_def


def RerouteTensor(t0, t1, can_modify=None):
  """Reroute the end of the tensor t0 to the ends of the tensor t1.

  Args:
    t0: a Tensor.
    t1: a Tensor.
    can_modify: iterable of operations which can be modified. Any operation
      outside within_ops will be left untouched by this function.

  Returns:
    The number of individual modifications made by the function.
  """
  nb_update_inputs = 0
  consumers = t1.consumers()
  if can_modify is not None:
    consumers = [c for c in consumers if c in can_modify]
  consumers_indices = {}
  for c in consumers:
    consumers_indices[c] = [i for i, t in enumerate(c.inputs) if t is t1]
  for c in consumers:
    for i in consumers_indices[c]:
      c._update_input(i, t0)  # pylint: disable=protected-access
      nb_update_inputs += 1
  return nb_update_inputs


def AppendNode(node_def, consumers, existed=False):
  """Append new node to the default graph and connect it to the consumers"""
  g = get_default_graph()
  if existed:
    t_output = g.get_tensor_by_name(node_def.name + ":0")
  else:
    t_inputs = [g.get_tensor_by_name(input + ":0") for input in node_def.input]
    new_op = Operation(node_def, g, inputs=t_inputs)
    t_output = new_op.outputs[0]
  # print("Append node: ", node_def.name, "-->", consumers)
  for consumer, index in consumers.items():
    t_consumer = consumer.inputs[index]
    tensor_modified_count = RerouteTensor(t_output,
                                          t_consumer,
                                          can_modify=consumers)
  return


def GetNodeConsumers(node_def, graph_def):
  """Get the consumers of node in a given graph_def, return a dict of (name, index)"""
  g = get_default_graph()
  cur_op_names = [op.name for op in g.get_operations()]

  consumers = dict()
  for node in graph_def.node:
    if node.name in cur_op_names and node_def.name in node.input:
      consumers[g.get_operation_by_name(node.name)] = [
          i for i, t in enumerate(node.input) if t == node_def.name
      ][0]
  return consumers


def MergeNodesFromGraphDef(graph_def):
  """Merge the new nodes from a quantize graph def into the default graph inplace."""
  g = get_default_graph()
  cur_op_names = [op.name for op in g.get_operations()]

  graph_def_nodes = [node.name for node in graph_def.node]
  existed_nodes = []
  for node in graph_def_nodes:
    if graph_def_nodes.count(node) > 1 and \
       node not in existed_nodes:
      existed_nodes.append(node)
  new_nodes = []
  new_node_names = []
  for node in graph_def.node:
    if node.name in new_node_names:
      continue
    if node.name not in cur_op_names or \
       node.name in existed_nodes:
      new_nodes.append(node)
      new_node_names.append(node.name)
  for node in new_nodes:
    consumers = GetNodeConsumers(node, graph_def)
    existed = True if node.name in existed_nodes else False
    AppendNode(node, consumers, existed)
  return


def GenerateGraphTransform(func):
  """Decorator to convert graph_def transforming to graph_transforming."""
  def inner_decorator(*args, **kwargs):
    input_graph_def = get_default_graph().as_graph_def()

    # Do GraphDef Transforming
    output_graph_def = func(input_graph_def, kwargs['config'])

    # Apply modifications to default graph
    MergeNodesFromGraphDef(output_graph_def)
    return

  return inner_decorator


def CreateQuantizeTrainingGraph(graph=None, config=None):
  """Python wrapper for the decent_q create training graph tool.

  Args:
    graph: the graph to be quantized, default graph will be used if set None.
    config: the QuantizeConfig

  Returns:
    Transformed Graph(as default) for quantize training.
  """
  if config is None:
    raise ValueError("Please set the QuantizeConfig.")
  elif not isinstance(config, QuantizeConfig):
    raise ValueError("Config shoulb be a QuantizeConfig object.")

  # Create the output_dir and temp folder
  temp_folder = os.path.join(config.output_dir, "temp")
  if not os.path.exists(temp_folder):
    try:
      os.makedirs(temp_folder)
    except Exception as e:
      print(e)

  if graph is None:
    graph = get_default_graph()
  input_graph_def = graph.as_graph_def()

  # Do GraphDef Transforming
  print("INFO: Creating the quantize train graph...")
  quantize_train_graph_def = CreateQuantizeTrainingGraphDef(
      input_graph_def, config)

  # Apply modifications to default graph
  MergeNodesFromGraphDef(quantize_train_graph_def)

  # Save the model
  quantize_train_path = os.path.join(config.output_dir,
                                     "quantize_train_graph.pb")
  save_pb_file(quantize_train_graph_def, quantize_train_path)
  print("INFO: Quantize train graph is generated in: {}".format(
      quantize_train_path))
  return


def CreateQuantizeEvaluationGraph(graph=None, config=None):
  """Python wrapper for the decent_q create evaluation graph tool.

  Args:
    graph: the graph to be quantized, default graph will be used if set None.
    config: the QuantizeConfig

  Returns:
    Transformed Graph(as default) for quantize evaluation.
  """
  if config is None:
    raise ValueError("Please set the QuantizeConfig.")
  elif not isinstance(config, QuantizeConfig):
    raise ValueError("Config shoulb be a QuantizeConfig object.")

  # Create the output_dir
  if not os.path.exists(config.output_dir):
    try:
      os.makedirs(config.output_dir)
    except Exception as e:
      print(e)

  if graph is None:
    graph = get_default_graph()
  input_graph_def = graph.as_graph_def()

  # Do GraphDef Transforming
  print("INFO: Creating the quantize evaluation graph...")
  quantize_train_graph_def = CreateQuantizeTrainingGraphDef(
      input_graph_def, config)
  quantize_eval_graph_def = CreateQuantizeEvaluationGraphDef(
      quantize_train_graph_def, config)

  # Apply modifications to default graph
  MergeNodesFromGraphDef(quantize_eval_graph_def)

  # Save the model
  quantize_eval_path = os.path.join(
      config.output_dir, "quantize_eval_graph_{}.pb".format(
          time.strftime("%Y%m%d%H%M%S", time.localtime())))
  save_pb_file(quantize_eval_graph_def, quantize_eval_path)
  print("INFO: Quantize eval graph is generated in: {}".format(
      quantize_eval_path))
  return


def CreateQuantizeDeployGraph(graph=None, checkpoint='', config=None):
  """Python wrapper for the decent_q create deploy graph tool.

  Args:
    graph: the graph to be quantized, default graph will be used if set None.
    checkpoint: the checkpoint path
    config: the QuantizeConfig

  Returns:
    Transformed Graph(as default) for quantize deploy.
  """
  if config is None:
    raise ValueError("Please set the QuantizeConfig.")
  elif not isinstance(config, QuantizeConfig):
    raise ValueError("Config shoulb be a QuantizeConfig object.")

  # Create the output_dir
  if not os.path.exists(config.output_dir):
    try:
      os.makedirs(config.output_dir)
    except Exception as e:
      print(e)

  if graph is None:
    graph = get_default_graph()
  quantize_eval_graph_def = graph.as_graph_def()

  if os.path.isdir(checkpoint):
    checkpoint = checkpoint_management.latest_checkpoint(checkpoint)
  else:
    pass
  print("INFO: Creating quantize eval model from: {}".format(checkpoint))
  step_in_ckpt = checkpoint.rsplit("-")[-1]

  # Freeze the checkpoint into the graph
  config.output_nodes = get_quantized_nodes(quantize_eval_graph_def,
                                            config.output_nodes)
  saver = saver_lib.Saver()
  with Session() as sess:
    saver.restore(sess, checkpoint)
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess, quantize_eval_graph_def, config.output_nodes)

  # Convert folded batchnorms
  frozen_quantize_eval_graph_def = ConvertFoldedBatchnorms(
      frozen_graph_def, config)
  frozen_quantize_eval_path = os.path.join(
      config.output_dir, "quantize_eval_model_{}_{}.pb".format(
          step_in_ckpt, time.strftime("%Y%m%d%H%M%S", time.localtime())))
  save_pb_file(frozen_quantize_eval_graph_def, frozen_quantize_eval_path)
  print("INFO: Quantize eval model is generated in: {}".format(
      frozen_quantize_eval_path))

  # Deploy
  quantize_deploy_graph_def = CreateQuantizeDeployGraphDef(
      frozen_quantize_eval_graph_def, config)

  # Save the model
  deploy_path = os.path.join(
      config.output_dir, "deploy_model_{}_{}.pb".format(
          step_in_ckpt, time.strftime("%Y%m%d%H%M%S", time.localtime())))
  save_pb_file(quantize_deploy_graph_def, deploy_path)
  print("INFO: Deploy model is generated in: {}".format(deploy_path))
  return
