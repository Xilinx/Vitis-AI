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

from pytorch_nndct.utils import module_util as mod_util

class TopoNode(object):
  """ A wrapper of graph node with addtional information."""

  def __init__(self,
               name,
               in_quant_part,
               spec=None,
               module=None,
               inputs=None,
               op=None):
    self.name = name
    self.in_quant_part = in_quant_part
    self.spec = spec
    self.module = module
    self.inputs = inputs
    self.op = op

  def __str__(self):
    return '{}({}) <- {}'.format(
        self.name,
        self.module._get_name() if self.module else None,
        ', '.join([str(inp) for inp in self.inputs]))

class ModelTopology(object):

  def __init__(self):
    self.nodes = []
    self._node_by_name = {}

    self.inputs = []
    self.outputs = []

  def add_node(self, node):
    self._node_by_name[node.name] = len(self.nodes)
    self.nodes.append(node)

  def node(self, name):
    return self.nodes[self._node_by_name[name]]

  def __str__(self):
    strs = []
    for node in self.nodes:
      strs.append(str(node))
    return '\n'.join(strs)

def topo_node_name(node):
  module_name = mod_util.module_name_from_node(node)
  node_name = node if isinstance(node, str) else node.name
  # Use node name for non-module node so that
  # we can have a complete topology.
  return module_name if module_name else node_name

def build_model_topo(graph, node_to_spec):
  model_topo = ModelTopology()
  for node in graph.nodes:
    name = topo_node_name(node)
    inputs = []
    for input_name in node.in_nodes:
      inputs.append(topo_node_name(input_name))
    spec = node_to_spec.get(node.name, None)
    model_topo.add_node(
        TopoNode(
            name,
            node.in_quant_part,
            spec,
            module=None,
            inputs=inputs,
            op=node.op))
  return model_topo
