import copy

from nndct_shared import utils as nndct_utils
from nndct_shared.base import NNDCT_DEBUG_LVL, NNDCT_OP
from nndct_shared.nndct_graph import Tensor
from nndct_shared.utils import NndctDebugLogger, NndctOption
from nndct_shared.utils.log import NndctDebugger

from .commander import OptimizeCommander
from .op_evaluator import Evaluator


class NndctOptimizer(NndctDebugger):

  @nndct_utils.nndct_pre_processing
  def __init__(self, use_quant, *args, **kwargs):
    self.set_debug_lv(NNDCT_DEBUG_LVL.BUILD_GRAPH)
    self.use_quant = use_quant

  def fuse_Nndctnode(self, graph, node):
    assert len(graph.children(node.name)) == 1
    child = graph.children(node.name)[0]
    graph.remove_node(child)

  def optimize(self, graph, commands=[]):
    param_info = {}
    for command in commands:
      graph, param_info = self.optimizer[command](self, graph, param_info)

    graph.remove_node_by_types([NNDCT_OP.DROPOUT])
    return graph

  @property
  def default_commanders(self):
    return {OptimizeCommander: 'optimizer'}

  @property
  def default_kwargs(self):
    return {'sort_strategy': 'bfs', 'model_type': 'Nndct'}


class GraphOptimizer(object):
  """Optimze graph for device computation
 
  """

  def __init__(self, nndct_graph):
    self._dev_graph = copy.deepcopy(nndct_graph)
    self._evalute_func_map = {
        NNDCT_OP.SHAPE: Evaluator.shape,
        NNDCT_OP.CAST: Evaluator.cast,
        NNDCT_OP.INT: Evaluator.int,
        NNDCT_OP.SCALAR_MUL: Evaluator.mul,
        NNDCT_OP.TENSOR: Evaluator.tensor,
        NNDCT_OP.FLOOR: Evaluator.floor
        
    }
    # self._redundant_ops = [NNDCT_OP.CONTIGUOUS]

  def get_frozen_graph(self):
    self._infer_tensor_layout()
    self._strip_redundant_ops()
    self._constant_folding()
    if NndctOption.nndct_parse_debug.value >= 3:
      NndctDebugLogger.write(f"\nfrozen dev graph:\n{self._dev_graph}")
    return self._dev_graph

  def _strip_redundant_ops(self):
    # remove unsupported op in xmodel
    redundant_op_types = [NNDCT_OP.CONTIGUOUS]
    self._dev_graph.remove_node_by_types(redundant_op_types)
    
    # remove redundant permute op
    permute_nodes = [node for node in self._dev_graph.nodes if node.op.type == NNDCT_OP.PERMUTE]
    for permute in permute_nodes:
      if permute.node_attr(permute.op.AttrName.ORDER) == [0, 1, 2, 3]:
        self._dev_graph.remove_node(permute)
          
  def _constant_folding(self):
    folding_nodes = set()
    for node in self._dev_graph.nodes:
      if hasattr(node.op, "AttrName"):
        for attr_name in node.op.attrs.keys():
          attr_val = node.node_attr(attr_name)
          if isinstance(attr_val, list):
            for i, val in enumerate(attr_val):
              attr_val[i] = self._materialize(node, val, folding_nodes)
          else:
            attr_val = self._materialize(node, attr_val, folding_nodes)
          if node.op.attrs[attr_name].type == list:
            attr_val = [attr_val]
          node.set_node_attr(attr_name, attr_val)
         
    for node_name in folding_nodes:
      self._dev_graph.remove_node_forcely(self._dev_graph.node(node_name))
      
    self._dev_graph.reconnect_nodes()

  @staticmethod
  def _infer_op_value_immediately(op_type):
    return op_type in [NNDCT_OP.SHAPE]

  def _materialize(self, cur_node, value, visited):
    eval_list = []

    def dfs(node):
      visited.add(node.name)
      if self._infer_op_value_immediately(node.op.type):
        eval_list.append(node)
        return

      for tensor in node.in_tensors:
        if tensor.node.name not in visited:
          dfs(tensor.node)

      eval_list.append(node)

    if not isinstance(value, Tensor):
      return value
    else:
      dfs(value.node)
      for node in eval_list:
        if node.out_tensors[0].data is None:
          self._evalute_func_map[node.op.type](node)
      
      cur_node.in_tensors.remove(value)
      return eval_list[-1].out_tensors[0].data
    
  def _infer_tensor_layout(self):
    # TODO: Don't support NHWC in pytorch inference
    for node in self._dev_graph.nodes:
      
      if node.op.type == NNDCT_OP.PERMUTE and node.in_tensors[0].ndim == 4:
        if node.in_tensors[0].layout is None:
          if node.node_attr(node.op.AttrName.ORDER) == [0, 1, 2, 3]:
            node.out_tensors[0].layout = Tensor.Layout.NHWC
        else:
          node.out_tensors[0].layout = node.in_tensors[0].layout
      elif node.out_tensors and node.in_tensors:
        node.out_tensors[0].layout = node.in_tensors[0].layout
      else:
        continue
