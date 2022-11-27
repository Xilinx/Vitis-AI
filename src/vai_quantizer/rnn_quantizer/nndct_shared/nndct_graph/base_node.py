

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
from nndct_shared.base import NNDCT_OP
from .base_operator import Operation
from .base_tensor import Tensor
from .operator_definition import CustomOp
from typing import Optional
import weakref
from abc import ABC, abstractproperty


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
               idx: Optional[int] = -1,
               in_quant_part: Optional[bool] = False):
    super().__init__()
    self._name = name
    self._op = op
    self._dtype = dtype
    self._idx = idx
    self._scope_name = ''
    self._alias = set()

    self._in_tensors = []
    self._out_tensors = []
    self._in_nodes = []
    self._out_nodes = []
    self._blocks = []
    self._is_quantizable = in_quant_part
    self._is_merged = False
    self._transpose_order = None

  def __repr__(self):
    return f"Node(name={self.name}, id={self.idx}, op_type={self.op.type}, quant_state={self.in_quant_part})"

  def __str__(self):
    return json.dumps(self.description(), indent=4, separators=(',', ': '))

  def __deepcopy__(self, memo):
    # Create a new class
    new = object.__new__(type(self))
    memo[id(self)] = new   # add the new class to the memo
    # Insert a deepcopy of all instance attributes
    new.__dict__.update(copy.deepcopy(self.__dict__, memo))
    # Manually update the weakref to be correct
    for ot in new.out_tensors:
      ot.node = new
    return new

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
    node_des['in_nodes'] = [i for i in self._in_nodes]
    node_des['out_nodes'] = [o for o in self._out_nodes]
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
    return self._in_nodes

  @property
  def out_nodes(self):
    return self._out_nodes

  def add_alias(self, name):
    if name and not name == self._name:
      self._alias.add(name)

  def set_dtype(self, dtype):
    self._dtype = dtype

  def set_optype(self, optype):
    self._op.set_optype(optype)

  def set_idx(self, idx):
    self._idx = idx

  def _load_from_description(self, node_des):
    for k, v in node_des.items():
      if k == 'bottom':
        for i in v:
          self.add_input(i)
      elif k == 'top':
        self._out_nodes.add(v)
      elif k == 'op_type':
        self._op.set_optype(v)
      elif k == 'alias':
        for a in v:
          self.add_alias(a)
      elif k == 'configs':
        self._op.set_configs(v)
      elif k == 'attrs':
        self._op.set_attrs(v)
      elif k in ['idx', 'op_params', 'dtype', 'name']:
        pass
      else:
        raise KeyError("got unexpected key {}: {}".format(k, v))

  def as_description(self, helper={}):

    def __named_params():
      return [{
          k: [helper['params'][k].shape, helper['params'][k].dtype]
      } for k in self.op.params]

    node_des = {
        'name': self._name,
        'bottom': helper['inputs'],
        'top': self._name,
        'idx': self._idx,
        'op_params': __named_params(),
        'op_type': self._op.type,
        'dtype': self._dtype,
        'alias': list(self._alias),
    }
    node_des['configs'] = {
        k: copy.deepcopy(v) for k, v in self._op.configs.items()
    }

    node_des['attrs'] = {k: copy.deepcopy(v) for k, v in self._op.attrs.items()}
    return node_des

  def deepcopy(self, helper={}):
    return self.__class__.load(node_des=self.as_description(helper))

  @classmethod
  def load(cls, node_des):
    node = cls(
        node_des['name'],
        dtype=node_des['dtype'],
        idx=node_des['idx'],
        op=Operation(
            node_des['op_type'],
            params=[list(item.keys())[0] for item in node_des['op_params']]))
    node._load_from_description(node_des)
    return node

  def node_attr(self, key):
    return self._op.get_attr(key)

  def set_node_attr(self, key, value):
    self._op.set_attr(key, value)

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

  @property
  def op(self):
    return self._op

  @op.setter
  def op(self, op):
    self._op = op

  @property
  def dtype(self):
    return self._dtype

  @property
  def alias(self):
    return self._alias

  @property
  def in_quant_part(self) -> bool:
    return self._is_quantizable

  @in_quant_part.setter
  def in_quant_part(self, quant_state: bool) -> None:
    self._is_quantizable = quant_state

  # @property
  # def in_quant_part(self) -> bool:
  #   return self._is_quantizable

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
  def transpose_order(self):
    return self._transpose_order
  
  @transpose_order.setter
  def transpose_order(self, order):
    self._transpose_order = order
    
  
    
