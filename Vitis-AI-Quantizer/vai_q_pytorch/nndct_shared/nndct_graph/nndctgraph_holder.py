import copy
from itertools import chain
from functools import reduce

from nndct_shared.base import NNDCT_KEYS, NNDCT_OP, NNDCT_PARAM
from nndct_shared.utils.log import NndctDebugger
from nndct_shared import utils as nndct_utils

from .base_graph import Graph
from .base_node import Node
from .base_operator import Operation

from .utils import *

def cast_to_basegraph(graph_or_file, prefix='', keep_all_graph_info=False):
  if isinstance(graph_or_file, str):
    return Graph.load(des_or_file=graph_or_file)
  elif keep_all_graph_info:
    # assert isinstance(graph_or_file,Graph),"keep_all_graph_info only works for Graph"
    return graph_or_file
  elif any(p.data is not None for name, p in graph_or_file.params.items()):
    # any(len(node.op.computes)>0 for node in graph_or_file.nodes):
    return graph_or_file.deepcopy(
        graph_cls=Graph, with_params=False, prefix=prefix)
  else:
    return graph_or_file

class NndctGraphHolder(NndctDebugger):

  @nndct_utils.nndct_pre_processing
  def __init__(self, *args, **kwargs):
    self.__Nndctgraph = None
    self.__NameScope, self.__VarScope = '', ''

    #for quantization
    self.QUANTIZABLE_OPS = [
        NNDCT_OP.AVG_POOL, NNDCT_OP.ADAPTIVEAVGPOOL2D, NNDCT_OP.CONVTRANSPOSE2D,
        NNDCT_OP.BATCH_NORM, NNDCT_OP.BIAS_ADD, NNDCT_OP.BASIC_LSTM, NNDCT_OP.BASIC_GRU,
        NNDCT_OP.CONV2D, NNDCT_OP.CONCAT, NNDCT_OP.DEPTHWISE_CONV2D,
        NNDCT_OP.DENSE, NNDCT_OP.ADD, NNDCT_OP.MULTIPLY, NNDCT_OP.DEVIDE,
        NNDCT_OP.LEAKY_RELU, NNDCT_OP.MAX_POOL, NNDCT_OP.MAXIMUM, NNDCT_OP.MEAN,
        NNDCT_OP.MINIMUM, NNDCT_OP.RESIZE, NNDCT_OP.SIGMOID, NNDCT_OP.TANH,
        NNDCT_OP.SUB, NNDCT_OP.RSUB
    ]
    self.LSTM_QUANTIZABLE_OPS = [
        #NNDCT_OP.PLACEHOLDER,
        NNDCT_OP.CONV2D, NNDCT_OP.DENSE,
        NNDCT_OP.ADD, NNDCT_OP.MULTIPLY,
        NNDCT_OP.SIGMOID, NNDCT_OP.TANH,
        NNDCT_OP.SUB, NNDCT_OP.RSUB
    ]
    self.HARD_QUANT_OPS = [NNDCT_OP.BATCH_NORM, NNDCT_OP.BASIC_LSTM, NNDCT_OP.BASIC_GRU]
    self.QUANTIZE_IGNORE_OPS = [
        NNDCT_OP.ZEROS, NNDCT_OP.INT, NNDCT_OP.TENSOR, NNDCT_OP.EMPTY,
        NNDCT_OP.EMBEDDING, NNDCT_OP.RELU
    ]
    self.QUANTIZE_IGNORE_OPS_FOR_CONSTANT_TENSOR = [
        NNDCT_OP.TRANSPOSE, NNDCT_OP.CAST, NNDCT_OP.EXPAND, NNDCT_OP.EXP,
        NNDCT_OP.REPEAT, NNDCT_OP.INPLACE_COPY, NNDCT_OP.FLOOR, NNDCT_OP.SELECT,
        NNDCT_OP.STRIDED_SLICE, NNDCT_OP.RESHAPE, NNDCT_OP.SHAPE
    ]
    self.QUANTIZABLE_DTYPES = ['float32', 'float64']
 
    # other utils
    
    if self.graph_or_file is not None:
      self.load_Nndctgraph(self.graph_or_file)

  def load_Nndctgraph(self, graph_or_file):
    #TODO:need data file? for loading parameters in nndctgraph?
    self.__Nndctgraph = cast_to_basegraph(
        graph_or_file, keep_all_graph_info=self.keep_all_graph_info)

  def get_model_type(self):
    return self.model_type or 'Nndct'


 
  def get_Nndctnode(self, node_name=None, params=None, idx=None, inputs=None):
    #TODO: use parameters to find node, use normal parameters
    def __from_node_name():
      for node in self.__Nndctgraph.nodes:
        if node_name == node.name:
          return node
        elif node_name in node.alias:
          #TODO: is taht safe?
          return node
      raise KeyError(
          "{} do not exist in Graph, please check!".format(node_name))

    def __from_params():
      for node in self.__Nndctgraph.nodes:
        valid_params = node.op.params
        if all(p in valid_params for p in params):
          return node
      raise KeyError(
          "node with params {} do not exist in Graph, please check!".format(
              params))

    def __from_inputs():
      mapped_inputs = [
          nndct_utils.node_from_output(i, self.get_model_type()) for i in inputs
      ]
      for node in self.__Nndctgraph.nodes:
        if all(i in node.inputs for i in mapped_inputs):
          return node
      raise KeyError(
          "node with inputs {}(map to {}) do not exist in Graph, please check!"
          .format(inputs, mapped_inputs))

    def _from_idx():
      for node in self.__Nndctgraph.nodes:
        if node.idx == idx:
          return node
      raise KeyError(
          "node with idx {} do not exist in Graph, please check!".format(idx))

    if node_name is not None:
      return __from_node_name()
    elif idx is not None:
      return _from_idx()
    elif params and len(params) > 0:
      return __from_params()
    elif inputs and len(inputs) > 0:
      return __from_inputs()
    else:
      raise Exception(
          "One of node name,params and inputs should be given to locate Xnode in the graph"
      )

  def is_node_quantizable(self, node_or_name, lstm):
    node = self._find_node(node_or_name)
    if should_pass_by_elemwise_node(node, self.__Nndctgraph):
      return False
    if not lstm:
      return node.op.type in self.QUANTIZABLE_OPS
    else:
      return node.op.type in self.LSTM_QUANTIZABLE_OPS

  def load_quant_info(self, quant_groups):
    self.__QuantGroups = quant_groups

  def quant_node_params(self, node_or_name):
    node = self._find_node(node_or_name)
    if node.op.type == NNDCT_OP.BATCH_NORM:
      return {
          k: v
          for k, v in node.op.params.items()
          if k in [NNDCT_PARAM.GAMMA, NNDCT_PARAM.BETA]
      }
    else:
      return node.op.params

  def quant_start_node(self, node_or_name):
    node = self._find_node(node_or_name)
    return self.__Nndctgraph.node(self.__QuantGroups[node.name][0])

  def quant_is_start_node(self, node_or_name):
    node = self._find_node(node_or_name)
    return node == self.__Nndctgraph.node(self.__QuantGroups[node.name][0])

  def quant_end_node(self, node_or_name):
    node = self._find_node(node_or_name)
    return self.__Nndctgraph.node(self.__QuantGroups[node.name][-1])

  def quant_inputs(self, node_or_name, inputs, params=None, validate=True):
    node = self._find_node(node_or_name)
    valid_inputs = []
    for i in inputs:
      try:
        input_node = self.get_Nndctnode(
            node_name=nndct_utils.node_from_output(i, self.get_model_type()))
      except KeyError:
        valid_inputs.append(None)
        continue
      if input_node.op.type == NNDCT_OP.INPUT:
        valid_input = input_node.name
        valid_inputs.append(valid_input)
      else:
        valid_inputs.append(None)
    return valid_inputs

  def quant_output(self, node_or_name, params=None):
    node = self._find_node(node_or_name)
    end_node = self.quant_end_node(node)
    quant_members = self.__QuantGroups[node.name]
    follow_members = quant_members[quant_members.index(node.name) + 1:]
    return end_node.name, True

  def quant_params(self, node_or_name):
    node = self._find_node(node_or_name)
    return list(
        chain.from_iterable(
            [self.node(n).op.params for n in self.__QuantGroups[node.name]]))




  def _find_node(self, node_or_name):
    if isinstance(node_or_name, str):
      return self.get_Nndctnode(node_name=node_or_name)
    else:
      return node_or_name

  @property
  def quant_groups(self):
    return self.__QuantGroups



  @property
  def Nndctgraph(self):
    return self.__Nndctgraph

  @property
  def name_scp(self):
    return self.__NameScope

  @property
  def var_scp(self):
    return self.__VarScope

  @property
  def default_commanders(self):
    return {}

  @property
  def default_kwargs(self):
    return {
        'graph_or_file': None,
        'keep_all_graph_info': False,
        'model_type': None
    }
