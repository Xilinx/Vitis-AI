

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

import copy
import json
import weakref
from abc import ABC, abstractproperty
from typing import Optional
from collections import OrderedDict
from .base_operator import Operation
from .base_tensor import Tensor
from .graph_node_list import (APPEND_INTERVAL, MID_POSITION,
                              POSITION_LOWER_BOUND, POSITION_UPPER_BOUND)
from .operator_definition import CustomOp


class Use(object):
  def __init__(self, user, offset):
    self._user = user
    self._offset = offset

  @property
  def user(self):
    return self._user

  @property
  def offset(self):
    return self._offset

  @user.setter
  def user(self, user):
    self._user = user

  @offset.setter
  def offset(self, offset):
    self._offset = offset




class NodeBase(ABC):
  @abstractproperty
  def op_type(self):
    """ op type
    Returns:
    str: node op type
    """


class Node(NodeBase):
  """A node contains an op and its input and output tensor.
  """

  def __init__(self, name: str,
               op: Optional[str] = None,
               dtype: Optional[str] = None,
               in_quant_part: Optional[bool] = False):
    super().__init__()
    self._name = name
    self._op = op
    self._dtype = dtype
    self._idx = -1
    self._scope_name = ""
    self._source_range = ""


    self._in_tensors = []
    self._out_tensors = []
    self._in_nodes = []
    self._out_nodes = []
    self._blocks = []
    self._is_quantizable = in_quant_part
    self._is_merged = False
    self._transpose_in_order = None
    self._transpose_out_order = None

    self._topo_position = 0
    self._block = None
    self._graph = None
    self._neighbor_nodes = [None, None]
    self._target_device = None

  def __repr__(self):
    return f"Node(name={self.name}, id={self.idx}, op_type={self.op.type}, quant_state={self.in_quant_part})"

  def __str__(self):
    return json.dumps(self.description(), indent=2, separators=(',', ': '))

  def __deepcopy__(self, memo):
    raise NotImplementedError("Deep copy is prohibited, use `clone_from` instead.")

  def clone_from(self, src_node, local_map):
    tmp_attrs = src_node.op._attrs
    tmp_params = src_node.op._params
    tmp_configs = src_node.op._configs
    src_node.op._params = copy.copy(tmp_params)
    src_node.op._attrs = copy.copy(tmp_attrs)
    src_node.op._configs = copy.copy(tmp_configs)
    self.op = copy.copy(src_node.op)
    self.op._export_attr_and_param()
    src_node.op._attrs = tmp_attrs
    src_node.op._params = tmp_params
    src_node.op._configs = tmp_configs
    self.op.clone_from(src_node.op, local_map)



  @property
  def scope_name(self):
    return self._scope_name

  @scope_name.setter
  def scope_name(self, name):
    self._scope_name = name

  def description(self):
    node_des = {}
    node_des['name'] = self._name
    node_des['scope_name'] = self._scope_name
    node_des['idx'] = self._idx
    node_des['dtype'] = self._dtype
    node_des['enable_quant'] = self._is_quantizable
    node_des['in_nodes'] = [i for i in self.in_nodes]
    node_des['out_nodes'] = [o for o in self.out_nodes]
    node_des['in_tensors'] = [it.description() for it in self.in_tensors]
    node_des['out_tensors'] = [ot.description() for ot in self.out_tensors]
    node_des['op'] = self._op.description()
    if self._blocks:
      for i, block in enumerate(self._blocks):
        node_des[f'block_{i}'] = []
        for n in sorted(block.nodes, key=lambda n: n.idx):
          node_des[f'block_{i}'].append(n.description())
    return node_des

  def clean_connections(self):
    self._in_nodes = []
    self._out_nodes = []

  def add_in_node(self, node_name: str):
    if node_name not in self._in_nodes:
      self._in_nodes.append(node_name)

  def add_out_node(self, node_name: str):
    if node_name not in self._out_nodes:
      self._out_nodes.append(node_name)

  @property
  def in_tensors(self):
    return self._in_tensors

  @property
  def out_tensors(self):
    return self._out_tensors


  @property
  def in_nodes(self):
    nodes = []
    for tensor in self.in_tensors:
      if tensor.node is not None:
        nodes.append(tensor.node.name)

    return nodes

  @property
  def out_nodes(self):
    nodes = []
    for out in self.out_tensors:
      for use in out.uses:
        nodes.append(use.user.name)
    return nodes


  def node_attr(self, key):
    return self._op.get_attr(key)

  def set_node_attr(self, key, value):
    if all([val is None for val in self._op._attr_value_mem[key]]):
      self._op.set_attr(key, value)
    else:
      self._op.update_attr(key, value)


  def node_config(self, key):
    return self._op.get_config(key)

  def set_node_config(self, key, value):
    self._op.set_config(key, value)

  def has_bound_params(self):
    return self._op.has_native_params()

  @property
  def op_type(self):
    return self.op.type

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, value):
    self._name = value

  @property
  def idx(self):
    return self._idx

  @idx.setter
  def idx(self, index):
    self._idx = index
    self.owning_graph.update_node_idx(self, index)

  @property
  def op(self):
    return self._op

  @op.setter
  def op(self, op):
    self._op = op

  @property
  def dtype(self):
    return self._dtype

  # @property
  # def alias(self):
  #   return self._alias

  @property
  def in_quant_part(self) -> bool:
    return self._is_quantizable

  @in_quant_part.setter
  def in_quant_part(self, quant_state: bool) -> None:
    self._is_quantizable = quant_state

  @property
  def module(self):
    return self._module()

  @module.setter
  def module(self, module):
    self._module = weakref.ref(module)

  @property
  def blocks(self):
    return self._blocks

  def add_block(self, block):
    self._blocks.append(block)

  def has_custom_op(self):
    return isinstance(self.op, CustomOp)

  def get_attr_val(self, attr_name):
    attr = self.node_attr(attr_name)
    return attr.data if isinstance(attr, Tensor) else attr

  @property
  def merged(self):
    return self._is_merged

  @merged.setter
  def merged(self, flag):
    self._is_merged = flag


  @property
  def transpose_in_order(self):
    return self._transpose_in_order

  @transpose_in_order.setter
  def transpose_in_order(self, order):
    self._transpose_in_order = order

  @property
  def transpose_out_order(self):
    return self._transpose_out_order

  @transpose_out_order.setter
  def transpose_out_order(self, order):
    self._transpose_out_order = order


  def set_node_attr_tensor_value(self, old_tensor, new_tensor):
    for attr_name, attr_value in self.op.attrs.items():
      if attr_value.value is old_tensor:
        self.set_node_attr(attr_name, new_tensor)

  def destroy(self):
    if len(self.blocks) > 0:
      raise RuntimeError("Can't destroy if or loop node.")

    while len(self.out_tensors) > 0:
      self.remove_output(len(self.out_tensors) - 1)

    self.remove_all_inputs()

    if self.in_node_list():
      self.remove_from_list()

    self.owning_graph.free_node(self)


  def remove_output(self, i):

    assert i < len(self.out_tensors)
    assert len(self.out_tensors[i].uses) == 0

    output = self.out_tensors.pop(i)
    self.owning_graph.remove_tensor(output)
    for output_offset in range(i, len(self.out_tensors)):
      self.out_tensors[output_offset].offset -= 1

  def replace_input_at(self, i, new_tensor):
    old_tensor = self.in_tensors[i]
    if old_tensor is new_tensor:
        return
    self.in_tensors[i] = new_tensor
    uses = [u for u in old_tensor.uses]
    attr_uses = [attr_u for attr_u in old_tensor.attr_uses]
    for u in uses:
      if u.user is self:
        new_tensor.uses.append(u)
        old_tensor.uses.remove(u)

    for attr_u in attr_uses:
      if attr_u.user is self.op:
        old_tensor.replace_attr_with_new_tensor_v2(attr_u, new_tensor)



  def remove_input(self, i):
    self.drop_input(i)
    for j in range(i + 1, len(self._in_tensors)):
      it = self.find_use_for_input(j)
      it.offset -= 1

    self._in_tensors.pop(i)


  def remove_all_inputs(self):
    for i in range(len(self.in_tensors)):
      self.drop_input(i)

    self.in_tensors.clear()

  def drop_input(self, i):
    assert i < len(self.in_tensors)
    input_value = self.in_tensors[i]
    use_it = self.find_use_for_input(i)
    input_value.uses.remove(use_it)
    self.in_tensors[i] = None
    return input_value

  def find_use_for_input(self, i):
    use_it = None
    for use in self.in_tensors[i].uses:
      if use.offset == i and use.user is self:
        use_it = use
    assert use_it is not None
    return use_it


  @property
  def owning_block(self):
    return self._block

  @owning_block.setter
  def owning_block(self, block):
    self._block = block

  @property
  def owning_graph(self):
    return self._graph

  @owning_graph.setter
  def owning_graph(self, graph):
    self._graph = graph
    if self._graph:
      self._graph.add_node(self)

  @property
  def topo_position(self):
    return self._topo_position

  @topo_position.setter
  def topo_position(self, pos):
    self._topo_position = pos


  def insert_before(self, node):
    assert node.in_node_list()
    self.insert_after(node.prev_node)


  def insert_after(self, node):
    assert not self.in_node_list() and node.in_node_list()
    assert node.owning_block is not None
    self._block = node.owning_block
    next_node = node.next_node
    node.next_node = self
    self.prev_node = node
    self.next_node = next_node
    next_node.prev_node = self
    self.update_topo_position()


  def update_topo_position(self):
    is_first_node = self.prev_node is self.owning_block.input_node
    is_last_node = self.next_node is self.owning_block.return_node
    prev_pos = self.prev_node.topo_position
    next_pos = self.next_node.topo_position

    if is_last_node:
      if is_first_node:
        self.topo_position = MID_POSITION
        return

      if prev_pos >= (POSITION_UPPER_BOUND - APPEND_INTERVAL):
        self.owning_block.reindex_topo()
        return

      self.topo_position = prev_pos + APPEND_INTERVAL

    elif is_first_node:
      if next_pos <= (POSITION_LOWER_BOUND + APPEND_INTERVAL):
        self.owning_block.reindex_topo()
        return

      self.topo_position = next_pos - APPEND_INTERVAL

    else:
      pos_between = prev_pos + (next_pos - prev_pos) / 2
      if pos_between == prev_pos:
        self.owning_block.reindex_topo()
        return

      self.topo_position = pos_between

  @property
  def next_node(self):
    return self._neighbor_nodes[1]

  @next_node.setter
  def next_node(self, node):
    self._neighbor_nodes[1] = node

  @property
  def prev_node(self):
    return self._neighbor_nodes[0]

  @prev_node.setter
  def prev_node(self, node):
    self._neighbor_nodes[0] = node

  def in_node_list(self):
    if self.next_node is None:
      assert self.prev_node is None

    return self.next_node is not None



  def remove_from_list(self):
    assert self.in_node_list()
    if self.owning_block.input_node is self:
        self.owning_block.input_node = self.next_node
    self.owning_block = None
    next_node = self.next_node
    prev_node = self.prev_node
    prev_node.next_node = next_node
    next_node.prev_node = prev_node
    self.next_node = None
    self.prev_node = None


  def add_in_tensor(self, tensor):
    tensor.uses.append(Use(self, len(self.in_tensors)))
    self._in_tensors.append(tensor)
    self.owning_graph.add_tensor(tensor)

  def add_out_tensor(self, tensor):
    tensor.offset = len(self.out_tensors)
    self._out_tensors.append(tensor)
    tensor.node = self
    self.owning_graph.add_tensor(tensor)

  @property
  def target_device(self):
    return self._target_device

  @target_device.setter
  def target_device(self, device):
    self._target_device = device


  @property
  def scope_name(self):
    return self._scope_name

  @scope_name.setter
  def scope_name(self, scope_name):
    self._scope_name = scope_name

  @property
  def source_range(self):
    return self._source_range

  @source_range.setter
  def source_range(self, source_range):
    self._source_range = source_range
