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

import os
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary
import numpy as np

FLAGS = None


def check_node_names(graph_def, node_names):
  """Check if graph_def has node names"""
  if not isinstance(node_names, list):
    raise TypeError('node_names should be list(str)')

  node_list = []
  for node in graph_def.node:
    node_list.append(node.name)
  for node_name in node_names:
    if not node_name in node_list:
      raise NameError("Node '{}' not found in graph.".format(node_name))


def gen_quantized_node_names(graph, node_names):
  """Generate quantized node name from normal node names"""
  node_list = []
  for node in graph.get_operations():
    node_list.append(node.name)
  quantized_node_names = []
  for node_name in node_names:
    if node_name + "/wquant" in node_list:
      quantized_node_names.append(node_name + "/wquant")
    elif node_name + "/aquant" in node_list:
      quantized_node_names.append(node_name + "/aquant")
    else:
      quantized_node_names.append(node_name)
  return quantized_node_names


def get_node_dtypes(input_graph_def, target_nodes):
  """Get target node dtypes form input_graph_def"""
  node_dtypes = []
  for target_node in target_nodes:
    for node in input_graph_def.node:
      if node.name == target_node:
        if 'dtype' in node.attr:
          node_dtypes.append(dtypes.as_dtype(node.attr['dtype'].type).name)
        elif 'T' in node.attr:
          node_dtypes.append(dtypes.as_dtype(node.attr['T'].type).name)
        elif 'type' in node.attr:
          node_dtypes.append(dtypes.as_dtype(node.attr['type'].type).name)
        else:
          raise ValueError("Fail to get data_type of node: {}".format(node))
  return node_dtypes


def get_node_shapes(input_graph_def, target_nodes):
  """Get shapes of target nodes from input_graph_def, shapes may be partial"""
  node_shapes = []
  for target in target_nodes:
    for node in input_graph_def.node:
      if node.name == target:
        if not 'shape' in node.attr:
          print("Warning: Fail to get output shape of node: {}".format(node))
        node_shapes.append(
            tensor_shape.as_shape(node.attr['shape'].shape).as_list())
  return node_shapes


def get_quantized_nodes(input_graph_def, target_nodes):
  """Get the quantized fix_neuron nodes for target nodes. Some nodes will be followed
  by fix_neuron node(named xxx/aquant) during quantization, if exists return the fix_neuron_node, 
  otherwise return itself"""
  node_list = [
      node.name.replace('/aquant', '')
      for node in input_graph_def.node
      if node.op == "FixNeuron"
  ]
  quantized_nodes = []
  for node in target_nodes:
    quantized_nodes.append(node + "/aquant" if node in node_list else node)
  return quantized_nodes


def save_pb_file(graph_def, filename):
  """Save graph_def to pb_file"""
  with gfile.GFile(filename, mode='wb') as f:
    f.write(graph_def.SerializeToString())


def show_pb_in_tensorboard(graph_def, port=6006):
  """Show pb_file in tensorboard"""
  _ = importer.import_graph_def(graph_def, name="")
  summary_write = summary.FileWriter("./logdir/", graph)
  os.system('tensorboard --logdir ./logdir/ --port 6006')


def get_edge_tensors(graph, input_nodes, output_nodes):
  """Get input and output tensors of a graph"""
  input_tensors = [
      graph.get_tensor_by_name(name + ":0") for name in input_nodes
  ]
  output_tensors = [
      graph.get_tensor_by_name(name + ":0")
      for name in gen_quantized_node_names(graph, output_nodes)
  ]
  return input_tensors, output_tensors


def gen_feed_dict(input_tensors, inputs):
  """Generate feed dict"""
  feed_dict = dict()
  if not isinstance(inputs, dict):
    raise ValueError(
        "Expect dict(input_node_name, numpy.Array) for input_fn, but got: ",
        type(inputs))
  if len(inputs) != len(input_tensors):
    raise ValueError(
        "len(inputs) != len(input_nodes), please check your input_fn.")
  for input_tensor in input_tensors:
    name = input_tensor.op.name
    if name not in inputs:
      raise ValueError(
          "key {} not found, please check your input_fn.".format(name))
    feed_dict[input_tensor] = inputs[name]
  return feed_dict
