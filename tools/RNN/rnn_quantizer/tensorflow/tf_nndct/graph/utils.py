#
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
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2

from tf_nndct.graph import ops
from tf_nndct.utils import generic_utils
from tf_nndct.utils import viz

def maybe_export_graph(path, graph):
  if not os.environ.get('VAI_TF_PARSER_DEBUG', ''):
    return

  dir_name = os.path.dirname(path)
  generic_utils.mkdir_if_not_exist(dir_name)

  if isinstance(graph, tf.Graph):
    graph = graph.as_graph_def()
    write_binary_proto(path, graph)
  elif isinstance(graph, graph_pb2.GraphDef):
    write_binary_proto(path, graph)
  elif isinstance(graph, ops.Graph):
    viz.export_to_netron(path, graph)
  else:
    pass

def write_proto(path, message, as_text=False):
  dir_name = os.path.dirname(path)
  generic_utils.mkdir_if_not_exist(dir_name)
  if dir_name:
    os.makedirs(dir_name, exist_ok=True)
  if as_text:
    with open(path, "w") as f:
      f.write(text_format.MessageToString(message))
  else:
    with open(path, "wb") as f:
      f.write(message.SerializeToString())

def write_text_proto(path, message):
  write_proto(path, message, as_text=True)

def write_binary_proto(path, message):
  write_proto(path, message, as_text=False)

def op_param_by_name(op, name):
  for param in op.ParamName:
    if param.value == name:
      return param
  return None

def topological_sort(graph):
  # Kahn's algorithm.
  # See https://en.wikipedia.org/wiki/Topological_sorting
  num_ready_inputs = {}
  for node in graph.nodes:
    num_ready_inputs[node.name] = 0

  ready_nodes = []
  for node in graph.inputs:
    ready_nodes.append(node)

  reordered_nodes = []
  while ready_nodes:
    node = ready_nodes.pop(0)
    reordered_nodes.append(node)
    for node_name in node.out_nodes:
      out_node = graph.node(node_name)
      num_ready_inputs[node_name] = num_ready_inputs[node_name] + 1
      if num_ready_inputs[node_name] == out_node.num_inputs:
        ready_nodes.append(out_node)

  if len(reordered_nodes) != graph.node_size:
    all_nodes = {node.name for node in graph.nodes}
    ready_nodes = {node.name for node in reordered_nodes}
    not_ready_nodes = all_nodes - ready_nodes
    detailed_message = ['node_name: num_ready_inputs vs. num_inputs']
    for node_name in not_ready_nodes:
      node = graph.node(node_name)
      detailed_message.append('{}: {} vs. {}'.format(
          node_name, num_ready_inputs[node_name], node.num_inputs))
    raise RuntimeError(('Couldn\'t sort the graph in topological order as '
                        'there is at least one cycle in the graph. Not ready '
                        'nodes: \n{}'.format('\n'.join(detailed_message))))

  topo_graph = ops.Graph()
  for node in reordered_nodes:
    topo_graph.add_node(node)
  return topo_graph

def stringfy_to_write(x):
  if isinstance(x, str):
    x = "'{}'".format(x)
  return '{}'.format(x)
