from collections import OrderedDict, ChainMap
import json
import torch
import weakref
from .utils import *

_NODE_NAME_SEPERATOR = '/'

class TorchGraph(object):
  _graph_id = 0
  def __init__(self, graph_name):
    self._name = graph_name
    self._nodes = []
    self._blob_values = {}
    self._param_values = {}
    self._param_alias = {}  #key: value_name, value: real param name
    self._ret_values = OrderedDict()
    # self._node_idx = 0

  @classmethod
  def new_graph(cls, graph_name):
    name = graph_name if graph_name else f"NndctGraph{TorchGraph._graph_id}" 
    TorchGraph._graph_id += 1
    return cls(name)
  
  def __str__(self):
    strs = ["{}/{}:".format(self.__class__.__name__, self._name)]
    for n in sorted(self._nodes, key=lambda n: n.idx):
      strs.append("node {}".format(n))
    return "\n".join(strs)

  def _infer_dtype(self, node):
    if node.outputs[0].dtype == "unknown":
      dtype = list(node.flatten_inputs)[0].dtype
    else:
      dtype = node.outputs[0].dtype
    if dtype == "unknown":
      raise RuntimeError(
          f"Can't infer dtype for node(idx={node.idx}, kind={node.kind})")
    return dtype

  def add_node(self, node):
    node.idx = len(self._nodes)
    node.dtype = self._infer_dtype(node)
    self._nodes.append(node)
    for output in node.outputs:
      self.add_blob_value(output)

  def add_param_alias(self, name, host_name):
    if isinstance(host_name, list):
      self._param_alias[name] = host_name
    else:
      if host_name not in self._param_alias:
        if host_name not in self._param_values:
          raise RuntimeError(f"{host_name} is not a Graph parameter")
        self._param_alias[name] = host_name
      else:
        self._param_alias[name] = self._param_alias[host_name]

  def add_ret_value(self, value):
    self._ret_values[value.name] = value

  def add_blob_value(self, value):
    self._blob_values[value.name] = value

  def add_param_value(self, value):
    self._param_values[value.name] = value

  def param_names(self):
    for key in ChainMap(self._param_values, self._param_alias).keys():
      yield key

  def blobs_name(self):
    for key in self._blob_values.keys():
      yield key

  def get_blob_value_by_name(self, name: str):
    return self._blob_values[name]

  def get_param_value_by_name(self, name: str):
    name = self.get_host_param_name(name)
    if isinstance(name, list):
      return [self._param_values[n] for n in name]
    else:
      return self._param_values[name]

  def get_host_param_name(self, name: str):
    return self._param_alias.get(name, name)

  def ret_values(self):
    return self._ret_values

  @property
  def nodes(self):
    return self._nodes
  @property
  def name(self):
    return self._name
  
class TorchValue(object):

  def __init__(self, value: torch.Value):

    def _get_value(node):
      sel = node.kindOf("value")
      return getattr(node, sel)("value")

    self._name = unique_name(value)
    self._scope_name = value.node().scopeName()
    self._node = None
    self._shape = list(
        value.type().sizes()) if value.isCompleteTensor() else None
    self._is_none = False
    self._is_plain_value = False
    self._type = str(value.type())
    self._data = None
    if value.node().mustBeNone():
      self._is_none = True
    elif value.node().kind().split("::")[-1] == "Constant":
      self._data = _get_value(value.node())
      if self._type != "Tensor":
        self._is_plain_value = True

    self._init_dtype(value)

  def _init_dtype(self, value):
    known_types = ['int', 'long', 'float', 'double', 'bool']
    if self.is_tensor() and value.isCompleteTensor():
      dtype = value.type().scalarType().lower()
    elif self.is_none():
      dtype = None
    else:
      dtype = self._type.lower()

    if dtype is None:
      self._dtype = None
    else:
      self._dtype = '.'.join(['torch', dtype
                             ]) if dtype in known_types else 'unknown'

  def is_tensor(self):
    return self._type == "Tensor"

  def is_plain_value(self):
    return self._is_plain_value

  def is_none(self):
    return self._is_none

  @property
  def name(self):
    return self._name

  @property
  def node(self):
    return self._node() if self._node else self._node

  @node.setter
  def node(self, value):
    self._node = weakref.ref(value) if value else value

  @property
  def dtype(self):
    return self._dtype

  @property
  def scope_name(self):
    return self._scope_name

  @property
  def shape(self):
    return self._shape

  @property
  def data(self):
    return self._data

class TorchNode(object):

  def __init__(self, node=None):
    self._idx = None
    self._kind = node.kind().split("::")[-1] if node else None
    self._inputs = []
    self._outputs = []
    self._in_nodes = []
    self._out_nodes = []
    self._dtype = "unknown"

  def __str__(self):
    return json.dumps(self.description(), indent=4, separators=(',', ': '))

  def description(self):
    node_des = {}
    node_des['name'] = self.name
    node_des['index'] = self.idx
    node_des['kind'] = self.kind
    node_des['dtype'] = self.dtype
    node_des['in_nodes'] = [i.idx for i in self._in_nodes]
    node_des['out_nodes'] = [o.idx for o in self._out_nodes]

    node_des['in_value'] = []
    for ip in self._inputs:
      if isinstance(ip, list):
        node_des['in_value'].append([it.name for it in ip])
      else:
        node_des['in_value'].append(ip.name)

    node_des['out_value'] = [ot.name for ot in self._outputs]
    return node_des

  def add_input(self, value):
    self._inputs.append(value)

  def add_output(self, value):
    self._outputs.append(value)
    value.node = self if value.node else None

  def add_out_node(self, node):
    self._out_nodes.append(node)

  def add_in_node(self, node):
    self._in_nodes.append(node)

  def clean_connection(self):
    self._out_nodes.clear()
    self._in_nodes.clear()

  @property
  def inputs(self):
    return self._inputs

  @property
  def flatten_inputs(self):
    for ip in self._inputs:
      if isinstance(ip, list):
        for i in ip:
          yield i
      else:
        yield ip

  @property
  def outputs(self):
    return self._outputs

  @outputs.setter
  def outputs(self, outputs):
    self._outputs = outputs

  @property
  def idx(self):
    return self._idx

  @idx.setter
  def idx(self, idx):
    self._idx = idx

  @property
  def kind(self):
    return self._kind

  @kind.setter
  def kind(self, kind):
    self._kind = kind

  @property
  def in_nodes(self):
    return self._in_nodes

  @property
  def out_nodes(self):
    return self._out_nodes

  @property
  def dtype(self):
    return self._dtype

  @dtype.setter
  def dtype(self, value):
    self._dtype = value

  @property
  def name(self):
    if hasattr(self, "_name"):
      return getattr(self, "_name")
    else:
      if self.outputs[0].scope_name:
        return _NODE_NAME_SEPERATOR.join(
            [self.outputs[0].scope_name, self.outputs[0].name])
      else:
        return self.outputs[0].name

  @name.setter
  def name(self, name):
    setattr(self, "_name", name)
