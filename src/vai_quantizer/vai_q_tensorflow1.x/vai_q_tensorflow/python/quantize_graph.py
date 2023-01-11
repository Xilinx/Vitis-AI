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

import os, time, copy

import sys
sys.path.append("..")

import tensorflow as tf
from tensorflow.python.client.session import Session

from vai_q_tensorflow.gen_files.vai_wrapper import DecentQCheckGraphWithStringInputs
from vai_q_tensorflow.gen_files.vai_wrapper import DecentQConvertConstantsToVariablesWithStringInputs
from vai_q_tensorflow.gen_files.vai_wrapper import DecentQCreateOptimizedGraphWithStringInputs
from vai_q_tensorflow.gen_files.vai_wrapper import DecentQCreateQuantizeCalibrationGraphWithStringInputs
from vai_q_tensorflow.gen_files.vai_wrapper import DecentQCreateQuantizeTrainingGraphWithStringInputs
from vai_q_tensorflow.gen_files.vai_wrapper import DecentQCreateQuantizeEvaluationGraphWithStringInputs
from vai_q_tensorflow.gen_files.vai_wrapper import DecentQConvertFoldedBatchnormsWithStringInputs
from vai_q_tensorflow.gen_files.vai_wrapper import DecentQCreateQuantizeDeployGraphWithStringInputs
from vai_q_tensorflow.python.utils import *

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
               nodes_method=[],
               method=1,
               calib_iter=100,
               output_dir="./quantize_results",
               align_concat=0,
               align_pool=0,
               adjust_shift_bias=1,
               adjust_shift_cut=1,
               simulate_dpu=1,
               scale_all_avgpool=1,
               do_cle=0,
               replace_relu6=1,
               replace_sigmoid=0,
               fold_bn_only=0,
               replace_softmax=0):
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
    self.nodes_method = nodes_method
    self.method = method
    self.calib_iter = calib_iter
    self.output_dir = output_dir
    self.align_concat = align_concat
    self.align_pool = align_pool
    self.adjust_shift_bias = adjust_shift_bias
    self.adjust_shift_cut = adjust_shift_cut
    self.simulate_dpu = simulate_dpu
    self.scale_all_avgpool = scale_all_avgpool
    self.do_cle = do_cle
    self.replace_relu6 = replace_relu6
    self.replace_sigmoid = replace_sigmoid
    self.fold_bn_only = fold_bn_only
    self.replace_softmax = replace_softmax

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
    for node_method in self.nodes_method:
      config_string += 'nodes_method,' + node_method + ','
    config_string += 'weight_bit,' + str(self.weight_bit) + ','
    config_string += 'activation_bit,' + str(self.activation_bit) + ','
    config_string += 'method,' + str(self.method) + ','
    config_string += 'calib_iter,' + str(self.calib_iter) + ','
    config_string += 'output_dir,' + str(self.output_dir) + ','
    config_string += 'align_concat,' + str(self.align_concat) + ','
    config_string += 'align_pool,' + str(self.align_pool) + ','
    config_string += 'adjust_shift_bias,' + str(self.adjust_shift_bias) + ','
    config_string += 'adjust_shift_cut,' + str(self.adjust_shift_cut) + ','
    config_string += 'simulate_dpu,' + str(self.simulate_dpu) + ','
    config_string += 'scale_all_avgpool,' + str(self.scale_all_avgpool) + ','
    config_string += 'do_cle,' + str(self.do_cle) + ','
    config_string += 'replace_relu6,' + str(self.replace_relu6) + ','
    config_string += 'replace_sigmoid,' + str(self.replace_sigmoid) + ','
    config_string += 'fold_bn_only,' + str(self.fold_bn_only) + ','
    config_string += 'replace_softmax,' + str(self.replace_softmax) + ','
    return tf.compat.as_bytes(config_string)


def CheckGraphDef(input_graph_def, graph_path):
  """Python wrapper for the decent_q check graph tool.

  Args:
    input_graph_def: GraphDef object containing a model to be checked.
    graph_path: string object of the graph path

  Returns:
    None
  """

  input_graph_def_string = input_graph_def.SerializeToString()
  graph_path_string = tf.compat.as_bytes(graph_path)
  with tf.compat.v1.errors.raise_exception_on_not_ok_status() as status:
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
  with tf.compat.v1.errors.raise_exception_on_not_ok_status() as status:
    output_graph_def_string = DecentQConvertConstantsToVariablesWithStringInputs(
      input_graph_def_string, config.to_string(), status)
  output_graph_def = tf.compat.v1.GraphDef()
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

  input_graph_def = add_shapes_to_graph_def(input_graph_def)
  input_graph_def_string = input_graph_def.SerializeToString()
  with tf.compat.v1.errors.raise_exception_on_not_ok_status() as status:
    output_graph_def_string = DecentQCreateOptimizedGraphWithStringInputs(
      input_graph_def_string, config.to_string(), status)
  output_graph_def = tf.compat.v1.GraphDef()
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

  input_graph_def = add_shapes_to_graph_def(input_graph_def)
  input_graph_def_string = input_graph_def.SerializeToString()
  with tf.compat.v1.errors.raise_exception_on_not_ok_status() as status:
    output_graph_def_string = DecentQCreateQuantizeCalibrationGraphWithStringInputs(
      input_graph_def_string, config.to_string(), status)
  output_graph_def = tf.compat.v1.GraphDef()
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

  input_graph_def = add_shapes_to_graph_def(input_graph_def)
  input_graph_def_string = input_graph_def.SerializeToString()
  with tf.compat.v1.errors.raise_exception_on_not_ok_status() as status:
    output_graph_def_string = DecentQCreateQuantizeTrainingGraphWithStringInputs(
      input_graph_def_string, config.to_string(), status)
  output_graph_def = tf.compat.v1.GraphDef()
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
  with tf.compat.v1.errors.raise_exception_on_not_ok_status() as status:
    output_graph_def_string = DecentQCreateQuantizeEvaluationGraphWithStringInputs(
      input_graph_def_string, config.to_string(), status)
  output_graph_def = tf.compat.v1.GraphDef()
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
  with tf.compat.v1.errors.raise_exception_on_not_ok_status() as status:
    output_graph_def_string = DecentQConvertFoldedBatchnormsWithStringInputs(
      input_graph_def_string, config.to_string(), status)
  output_graph_def = tf.compat.v1.GraphDef()
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
  with tf.compat.v1.errors.raise_exception_on_not_ok_status() as status:
    output_graph_def_string = DecentQCreateQuantizeDeployGraphWithStringInputs(
      input_graph_def_string, config.to_string(), status)
  output_graph_def = tf.compat.v1.GraphDef()
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


def AppendNode(node_def, consumers, name_to_op, existed=False):
  """Append new node to the default graph and connect it to the consumers"""
  g = tf.get_default_graph()
  if existed:
    t_output = g.get_tensor_by_name(node_def.name + ":0")
  else:
    t_inputs = [g.get_tensor_by_name(input + ":0") for input in node_def.input]
    new_op = tf.Operation(node_def, g, inputs=t_inputs)
    t_output = new_op.outputs[0]
    name_to_op[new_op.name] = new_op
  #  print("Append node: ", node_def.name, "-->", consumers)
  for consumer_name, index in consumers.items():
    if consumer_name not in name_to_op:
      continue
    consumer_op = name_to_op[consumer_name]
    t_consumer = consumer_op.inputs[index]
    tensor_modified_count = RerouteTensor(t_output,
                                          t_consumer,
                                          can_modify=[consumer_op])
  return


def GetNodeConsumers(node_def, graph_def):
  """Get the consumers of node in a given graph_def, return a dict of (name, index)"""
  g = tf.get_default_graph()
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
  g = tf.get_default_graph()
  name_to_op = dict()
  for op in g.get_operations():
    name_to_op[op.name] = op

  graph_def_nodes = [node.name for node in graph_def.node]
  existed_nodes = []
  for node in graph_def_nodes:
    if graph_def_nodes.count(node) > 1 and node not in existed_nodes:
      existed_nodes.append(node)

  new_nodes = []
  new_node_names = []
  for node in graph_def.node:
    if node.name in new_node_names:
      continue
    if node.name not in name_to_op or node.name in existed_nodes:
      new_nodes.append(node)
      new_node_names.append(node.name)

  node_to_consumers = dict()
  for node in graph_def.node:
    if node.name in new_node_names:
      node_to_consumers[node.name] = dict()
  g = tf.get_default_graph()
  for node in graph_def.node:
    for index, inp in enumerate(node.input):
      if inp in new_node_names:
        node_to_consumers[inp][node.name] = index

  for node in new_nodes:
    consumers = node_to_consumers[node.name]
    existed = True if node.name in existed_nodes else False
    AppendNode(node, consumers, name_to_op, existed)
  return


def GenerateGraphTransform(func):
  """Decorator to convert graph_def transforming to graph_transforming."""
  def inner_decorator(*args, **kwargs):
    input_graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

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
    graph = tf.get_default_graph()
  input_graph_def = graph.as_graph_def(add_shapes=True)

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
    graph = tf.get_default_graph()
  input_graph_def = graph.as_graph_def(add_shapes=True)

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
  save_pb_file(graph.as_graph_def(add_shapes=True), quantize_eval_path)
  print(
    "INFO: Quantize eval graph is generated in: {}".format(quantize_eval_path))
  return


def _get_removed_nodes(name_to_node, target_node_name):
  """
  Get nodes before and include the specified node by recursive calls
  Args:
    name_to_node: the graph_def to be replaced
    target_node_name: the specified node name
  Returns:
    list of the node names which is before target_node_name
  """
  remove_list = []
  remove_list.append(target_node_name)
  input_nodes = name_to_node[target_node_name].input
  if not input_nodes:
    return remove_list
  # when input is one of the input node's multi-output eg:['IteratorGetNext:1']
  input_node = input_nodes[0].split(":")[0]
  remove_list.extend(_get_removed_nodes(name_to_node, input_node))
  return remove_list


def SetInputNodesAsPlaceholder(graph_def, target_node_name, shape):
  """
  Replace original input node as placeholder. Freezing graph will include some
  nodes of tf.data. eg "IteratorGetNext". This function remove these nodes and
  replace them with placeholder node.
  Args:
    input_graph_def: the graph_def to be replaced
    target_node_name: the specified node to be replaced
  Returns:
    Replaced GraphDef
  """
  shape[0] = None if shape[0] == -1 else shape[0]
  name_to_node = {}
  for node in graph_def.node:
    name_to_node[node.name] = node
  if name_to_node[target_node_name].op == "Placeholder":
    return graph_def
  remove_nodes = _get_removed_nodes(name_to_node, target_node_name)
  placeholder = tf.placeholder(tf.float32, shape=shape, name=target_node_name)
  placeholder_def = placeholder.op.node_def
  placeholder_def.name = target_node_name
  replaced_graph_def = tf.compat.v1.GraphDef()
  replaced_graph_def.node.extend([placeholder_def])
  for node in graph_def.node:
    if node.name in remove_nodes:
      print("INFO: remove node:", node.name)
      continue
    if node.input and node.input[0] == target_node_name:
      print("INFO: re-map input node to ", node.name)
      new_node = copy.deepcopy(node)
      new_node.input[0] = placeholder_def.name
      replaced_graph_def.node.extend([new_node])
    else:
      replaced_graph_def.node.extend([copy.deepcopy(node)])
  return replaced_graph_def


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
    graph = tf.get_default_graph()
  quantize_eval_graph_def = graph.as_graph_def(add_shapes=True)

  if os.path.isdir(checkpoint):
    checkpoint = tf.train.latest_checkpoint(checkpoint)
  else:
    pass
  print("INFO: Creating quantize eval model from: {}".format(checkpoint))
  step_in_ckpt = checkpoint.rsplit("-")[-1]

  # Freeze the checkpoint into the graph
  config.output_nodes = get_quantized_nodes(quantize_eval_graph_def,
                                            config.output_nodes)
  saver = tf.train.Saver()
  with Session() as sess:
    saver.restore(sess, checkpoint)
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, quantize_eval_graph_def, config.output_nodes)

  # Convert folded batchnorms
  frozen_quantize_eval_graph_def = ConvertFoldedBatchnorms(
    frozen_graph_def, config)

  # Deploy
  # quantize_deploy_graph_def = CreateQuantizeDeployGraphDef(
  #   frozen_quantize_eval_graph_def, config)

  # Save the model
  # for quantize finetune model, replace input node with placeholder
  # replaced_graph_def = frozen_quantize_eval_graph_def
  for target_node_name, shape in zip(config.input_nodes, config.input_shapes):
    frozen_quantize_eval_graph_def = SetInputNodesAsPlaceholder(
      frozen_quantize_eval_graph_def, target_node_name, shape)

  frozen_quantize_eval_path = os.path.join(
    config.output_dir, "quantize_eval_model_{}_{}.pb".format(
      step_in_ckpt, time.strftime("%Y%m%d%H%M%S", time.localtime())))
  frozen_quantize_eval_graph_def = tf.compat.v1.graph_util.extract_sub_graph(frozen_quantize_eval_graph_def,
          config.output_nodes)
  save_pb_file(frozen_quantize_eval_graph_def, frozen_quantize_eval_path)
  print("INFO: Quantize eval model is generated in: {}".format(
    frozen_quantize_eval_path))

  # deploy_path = os.path.join(
  #   config.output_dir, "deploy_model_{}_{}.pb".format(
  #     step_in_ckpt, time.strftime("%Y%m%d%H%M%S", time.localtime())))
  # save_pb_file(quantize_deploy_graph_def, deploy_path)
  # print("INFO: Deploy model is generated in: {}".format(deploy_path))
  return
