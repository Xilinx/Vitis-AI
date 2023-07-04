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
"""Classes and functions used to build a graph."""
import copy
import enum
import numpy as np
import types

from tensorflow.python.util import nest

from nndct_shared.nndct_graph import base_graph
from nndct_shared.nndct_graph import base_operator
from nndct_shared.nndct_graph import base_tensor
from nndct_shared.nndct_graph import base_node

import nndct_shared
AutoName = nndct_shared.nndct_graph.base_operator.AutoName
Attr = nndct_shared.nndct_graph.base_operator.NndctIrAttr
Node = nndct_shared.nndct_graph.base_node.Node
OpTypes = nndct_shared.base.NNDCT_OP
OccurenceType = nndct_shared.nndct_graph.base_operator.OccurenceType

class Graph(base_graph.Graph):

  def __init__(self, name=None):
    super(Graph, self).__init__(name)

    self.data_format = None
    self.input_signature = None
    self.structured_output_tensors = None

  def add_node(self, node):
    if node.name in self._nodes_by_name:
      raise ValueError('Node {} has been added.'.format(node.name))
    node.graph = self
    self._nodes_by_name[node.name] = node
    for tensor in node.out_tensors:
      self._add_tensor(tensor)

  def remove_node(self, node):
    if node.num_inputs > 1:
      raise ValueError(
          'Node has multiple inputs is not allowed to remove: {}'.format(node))

    parents = self.parents(node)
    children = self.children(node)

    parent = parents[0] if len(parents) else None
    if parent:
      assert parent.num_outputs < 2 and node.num_outputs < 2, (
          'Not allowd to remove node {}'.format(node))

    for child in children:
      inputs_to_remove = []
      for index, tensor in enumerate(child.in_tensors):
        if tensor.is_produced_by(node):
          if parent:
            child.consume(parent.out_tensors[0], index)
          else:
            inputs_to_remove.append(tensor)

      for tensor in inputs_to_remove:
        child.remove_input(tensor)

    del self._nodes_by_name[node.name]

  def clear_nodes(self):
    self._nodes_by_name = {}
    self._tensors = {}

  def _add_tensor(self, tensor):
    if not tensor.name:
      raise ValueError('Unnamed tensor.')
    self._tensors[tensor.name] = tensor

  def add_tensor(self, tensor):
    raise ValueError(
        'Add a tensor to a graph directly is not allowed.'
        'When a node is added to a graph, its output tensors are also added to'
        'the graph')

  def tensor(self, name):
    return self._tensors[name]

  def clone(self):
    nodes = {}
    tensors = {}
    for node in self.nodes:
      cloned_node = node.clone()
      for tensor in node.out_tensors:
        tensors[tensor.name] = cloned_node.produce(tensor.name)
      nodes[node.name] = cloned_node

    for node in self.nodes:
      for tensor in node.in_tensors:
        nodes[node.name].consume(tensors[tensor.name])

    graph = self.__class__(self.name)
    for node in nodes.values():
      graph.add_node(node)

    graph.data_format = self.data_format
    graph.input_signature = copy.deepcopy(self.input_signature)
    output_tensors = [
        tensors[tensor.name]
        for tensor in nest.flatten(self.structured_output_tensors)
    ]
    graph.structured_output_tensors = nest.pack_sequence_as(
        self.structured_output_tensors, output_tensors)
    return graph

  @property
  def nodes(self):
    for node in self._nodes_by_name.values():
      yield node

  @property
  def node_size(self):
    return len(self._nodes_by_name)

  @property
  def end_tensors(self):
    end_tensors = {}
    for tensor in nest.flatten(self.structured_output_tensors):
      if tensor.name in end_tensors:
        continue
      end_tensors[tensor.name] = tensor
    return end_tensors.values()

  def __str__(self):
    lines = [
        'Graph {', f'name: {self.name}',
        f'input_signature: {self.input_signature}',
        f'structured_output_tensors: {self.structured_output_tensors}'
    ]
    for node in self._nodes_by_name.values():
      lines.append('node {}'.format(str(node)))
    lines.append('}')
    return '\n'.join(lines)

class Node(base_node.Node):

  def __init__(self, name, op=None, dtype=None):
    super(Node, self).__init__(name, op, dtype)

    self._graph = None

    # i/o tensor names
    self.input_names = []
    self.output_names = []

    self.layer_name = None
    self.inbound_nodes = []
    self.in_quant_part = True

  def consume(self, tensor, index=None):
    if index is None:
      self._in_tensors.append(tensor)
    elif index >= self.num_inputs:
      raise IndexError('index out of range')
    else:
      self._in_tensors[index] = tensor

  def is_consuming(self, tensor):
    for t in self._in_tensors:
      if t.name == tensor.name:
        return True
    return False

  def produce(self, name):
    tensor = Tensor(name, producer=self)
    self._out_tensors.append(tensor)
    return tensor

  def remove_input_at(self, index):
    del self._in_tensors[index]

  def remove_input(self, tensor):
    index_to_remove = None
    for i, t in enumerate(self._in_tensors):
      if t.name == tensor.name:
        index_to_remove = i
        break
    self.remove_input_at(index_to_remove)

  def remove_output_at(self, index):
    del self._out_tensors[index]

  def remove_output(self, tensor):
    index_to_remove = None
    for i, t in enumerate(self._out_tensors):
      if t.name == tensor.name:
        index_to_remove = i
        break
    self.remove_output_at(index_to_remove)

  def description(self):
    node_des = {}
    node_des['name'] = self._name
    node_des['op'] = self._op.description()
    node_des['in_nodes'] = [n for n in self.in_nodes]
    node_des['out_nodes'] = [n for n in self.out_nodes]
    node_des['in_tensors'] = [t.name for t in self.in_tensors]
    node_des['out_tensors'] = [t.name for t in self.out_tensors]
    return node_des

  def clone(self):
    node = self.__class__(self.name, self.op.clone(), self.dtype)
    # Note: in_tensors and out_tensors will not be cloned here.
    # They should be set by graph.clone().
    node.input_names = self.input_names[:]
    node.output_names = self.output_names[:]
    node.layer_name = self.layer_name
    node.inbound_nodes = self.inbound_nodes[:]
    node.in_quant_part = self.in_quant_part
    return node

  @property
  def num_inputs(self):
    return len(self._in_tensors)

  @property
  def num_outputs(self):
    return len(self._out_tensors)

  @property
  def in_nodes(self):
    return list({tensor.producer.name for tensor in self._in_tensors})

  @property
  def out_nodes(self):
    out_nodes = set()
    for tensor in self._out_tensors:
      for node in self._graph.nodes:
        if node.is_consuming(tensor):
          out_nodes.add(node.name)
    return list(out_nodes)

  @property
  def graph(self):
    return self._graph

  @graph.setter
  def graph(self, graph):
    self._graph = graph

class Operation(base_operator.Operation):

  def __init__(self, op_type):
    super(Operation, self).__init__(op_type)

  def get_param(self, name):
    if name not in self._params:
      raise KeyError('Parameter "{}" not found'.format(name))
    return self._params[name]

  def set_param(self, name, value):
    # String param name is allowed, this because custom LSTM may use any
    # weight names, so we can't pre-define ParamName before we see them.
    # In this case, we use the original weight name as key to save params.
    if not isinstance(name, (str, enum.Enum)):
      raise ValueError("Invalid param name: {}".format(type(name)))
    self._params[name] = value

  def clone(self):
    # Some trival ops doesn't have their own class definition.
    # They just use base_operator.Operation with a string 'op_type' to
    # indicate their operations.
    if type(self) == Operation:
      op = self.__class__(self.type)
    else:
      op = self.__class__()

    for key in self.configs:
      op.set_config(key, copy.deepcopy(self.get_config(key)))

    for key in self.attrs:
      attr = self.get_attr(key)
      cloned_attr = attr.clone() if isinstance(attr,
                                               Tensor) else copy.deepcopy(attr)
      op.set_attr(key, cloned_attr)

    for key in self.params:
      op.set_param(key, self.get_param(key).clone())
    return op

  @property
  def params(self):
    # Make params read-only
    return types.MappingProxyType(self._params)

class Tensor(base_tensor.Tensor):

  def __init__(self,
               name=None,
               shape=None,
               dtype=None,
               data=None,
               producer=None):
    super(Tensor, self).__init__(name, shape, dtype, data=data, node=producer)

    self._producer = producer

  @classmethod
  def from_numpy(cls, name, data):
    tensor = cls(name)
    tensor.from_ndarray(data)
    return tensor

  def is_produced_by(self, node):
    if not self._producer:
      return False
    return self._producer.name == node.name

  def transpose(self, axes):
    if self._data is None:
      shape = [self._shape[i] for i in axes]
    else:
      data = self._data.transpose(axes)
      data = np.ascontiguousarray(data)
      shape = data.shape
    self._data = data
    self._shape = shape
    return self

  def clone(self):
    tensor = self.__class__(self.name)
    tensor.clone_from(self)
    return tensor

  @property
  def producer(self):
    return self._producer
