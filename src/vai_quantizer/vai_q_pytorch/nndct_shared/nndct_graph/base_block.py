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

from .graph_node_list import (APPEND_INTERVAL, FORWARD_DIR, BACKWARD_DIR, GraphNodeList, GraphNodeListIterator, POSITION_LOWER_BOUND, POSITION_UPPER_BOUND)


class Block(object):
  """ Graph object of NNDCT, contain list of NndctNodes.
    That will be used for topology or export to XGraph"""

  def __init__(self, graph, node, input_node, return_node):
    self._graph = graph
    self._node = node
    self._output = return_node
    self._input = input_node
    self._input.next_node = self._output
    self._input.prev_node = self._output
    self._output.next_node = self._input
    self._output.prev_node = self._input
    self._output.owning_block = self
    self._input.owning_block = self
    self._output.topo_position = POSITION_UPPER_BOUND
    self._input.topo_position = POSITION_LOWER_BOUND
    self._graph.add_block(self)

  
  def __deepcopy__(self, memo):
    raise NotImplementedError("Deep copy is prohibited, use `clone_from` instead.")  
  
  def clone_from(self, src_block, local_map, converted_nodes):
    for node in src_block.nodes:
      if node.name in converted_nodes:
        continue
      self.append_node(self.owning_graph.create_node_from(node, local_map, converted_nodes))
      
    
  @property
  def input_node(self):
    return self._input
  
  @input_node.setter
  def input_node(self, node):
    self._input = node

  @property
  def return_node(self):
    return self._output
  
  @property
  def nodes(self):
    # Implement iterable object
    return GraphNodeList(self._input, FORWARD_DIR)

  @property
  def reverse_nodes(self):
    # Implement iterable object
    return GraphNodeList(self._output, BACKWARD_DIR)

  @property
  def owning_graph(self):
    return self._graph

  @owning_graph.setter
  def owning_graph(self, graph):
    self._graph = graph
    if self._graph:
      self._graph.add_block(self)

  @property
  def owning_node(self):
    return self._node
  
  def append_node(self, node):
    assert node.owning_graph is self.owning_graph and (not node.in_node_list()) 
    node.insert_before(self._output)

  def reindex_topo(self):
    cur_pos = POSITION_LOWER_BOUND
    for node in self.nodes:
      cur_pos += APPEND_INTERVAL
      node.topo_position = cur_pos
