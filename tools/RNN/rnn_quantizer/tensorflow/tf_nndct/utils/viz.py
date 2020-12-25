

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

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile

def export_to_netron(filepath, graph):
  """Export the nndct `graph` to a serialized file specified by `filepath`.
  Here we use GraphDef as netron's input.
  See https://github.com/lutzroeder/netron
  """
  graph_def = _to_graph_def(graph)
  with gfile.GFile(filepath, "wb") as f:
    f.write(graph_def.SerializeToString())

def _to_graph_def(graph):
  """Convert nndct graph to tensorflow's GraphDef."""
  graph_def = graph_pb2.GraphDef()
  # TODO(yuwang): Add attrs to node.
  for node in graph.nodes:
    node_def = graph_def.node.add()
    node_def.name = node.name
    node_def.op = node.op.type
    for in_node in node.in_nodes:
      node_def.input.extend([in_node])
  return graph_def

def export_to_graphviz(graph):
  pass
