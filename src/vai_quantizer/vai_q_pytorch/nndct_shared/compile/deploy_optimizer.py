

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
from collections import defaultdict, deque, namedtuple
from typing import List
from nndct_shared.quantization import BaseQuantizer
from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph import Graph, Node, Tensor, GraphSearcher
from nndct_shared.nndct_graph import operator_definition as base_op
from nndct_shared.utils import (GLOBAL_MAP, NNDCT_KEYS, QError, QWarning,
                               NndctOption, NndctScreenLogger)
from nndct_shared.utils import PatternType, permute_data
from .fuse_trans_matmul import TransMatmulActvHandler
from .fuse_trans_reduction_op import TransReductionOpActvHandler
from nndct_shared.optimization.fuse_pad import PadFuseHandler
from nndct_shared.optimization.merge_permute_in_linear import PermuteMergeHandler
from nndct_shared.optimization.merge_reshape import ReshapeMergeHandler
# from nndct_shared.nndct_graph.operator_definition import Reshape
from .attr_transform import *
from .op_evaluator import Evaluator
import numpy as np
DeployGraphInfo = namedtuple("DeployGraphInfo", ["dev_graph", "quant_info"])


def get_xmodel_and_dump_infos(quantizer: BaseQuantizer, deploy_graphs_list: List[List[Graph]]):
  if len(deploy_graphs_list) == 1:
    graph_quant_info = get_deploy_graph_infos(quantizer, deploy_graphs_list[0])
    return graph_quant_info, graph_quant_info
  elif len(deploy_graphs_list) == 2:
    xmodel_quant_info = get_deploy_graph_infos(quantizer, deploy_graphs_list[0])
    dump_quant_info = get_deploy_graph_infos(quantizer, deploy_graphs_list[1])
    return xmodel_quant_info, dump_quant_info
  else:
    raise RuntimeError("Length of graphs list to deploy should be 1 or 2")


def get_deploy_graph_infos(quantizer: BaseQuantizer, deploy_graphs: List[Graph]) -> List[DeployGraphInfo]:
  graph_quant_info_list = []
  quant_groups = copy.deepcopy(quantizer.configer.quant_groups)
  quant_config = {"param": {}, "output": {}, "input": {}}
  if not NndctOption.nndct_quant_off.value:
    quant_config["param"].update(quantizer.quant_config["param"])
    quant_config["input"].update(quantizer.quant_config["input"])
    for blob_name, quant_info in quantizer.quant_config["output"].items():
      if any([blob_name in dev_graph for dev_graph in deploy_graphs]):
        quant_config["output"][blob_name] = copy.deepcopy(quant_info)
      else:
        
        def find_possible_quant_pn(node):
          if len(node.in_nodes) == 1 and len(node.out_tensors) == 1: 
            pn = quantizer.Nndctgraph.parents(node)[0]
            if any([pn.name in dev_graph for dev_graph in deploy_graphs]):
              if len(pn.out_tensors) == 1:
                return pn
          
            else:
              return find_possible_quant_pn(pn) 
        node = quantizer.Nndctgraph.node(blob_name)
        pn = find_possible_quant_pn(node)
        if pn is not None and pn.name not in quant_config["output"]:
          quant_config['output'][pn.name] = copy.deepcopy(quant_info)
        
  for dev_graph in deploy_graphs:
    graph_quant_info_list.append(DeployGraphInfo(dev_graph=dev_graph, quant_info=quant_config))
  
  return graph_quant_info_list
    
        
class DevGraphOptimizer(object):
  """Optimze graph for device computation
 
  """

  def __init__(self, nndct_graph):
    self._dev_graph = Graph(graph_name=nndct_graph.name)
    self._dev_graph.clone_from(nndct_graph)
    self._evalute_func_map = {
        NNDCT_OP.SHAPE: Evaluator.shape,
        NNDCT_OP.CAST: Evaluator.cast,
        NNDCT_OP.INT: Evaluator.int,
        NNDCT_OP.SCALAR_MUL: Evaluator.mul,
        NNDCT_OP.TENSOR: Evaluator.tensor,
        NNDCT_OP.FLOOR: Evaluator.floor,
        NNDCT_OP.DIV: Evaluator.elemwise_div,
        NNDCT_OP.FLOOR_DIV: Evaluator.floor_div, 
        NNDCT_OP.ADD: Evaluator.add,
        NNDCT_OP.SCALAR_ADD: Evaluator.add,
        NNDCT_OP.SCALAR_REMAINDER: Evaluator.remainder
        
    }

 
   
  def strip_redundant_ops(self):
    # remove unsupported op in xmodel
    redundant_op_types = [NNDCT_OP.CONTIGUOUS]
    self._dev_graph.remove_node_by_types(redundant_op_types)
    

  def update_op_attrs(self):
    for node in self._dev_graph.nodes:
      if node.op.type == NNDCT_OP.STRIDED_SLICE:
        input_dims = node.in_tensors[0].ndim
        begin = [0] * input_dims
        last = [NNDCT_CONSTANT.INT_MAX] * input_dims
        strides = [1] * input_dims
        dims = node.node_attr(node.op.AttrName.DIMS)
        start = node.node_attr(node.op.AttrName.BEGIN)
        step = node.node_attr(node.op.AttrName.STRIDES)
        end = node.node_attr(node.op.AttrName.END)
        for i, pos in enumerate(dims):
          begin[pos] = start[i]
          if isinstance(end[i], Tensor) or (isinstance(end[i], int) and end[i] < last[pos]):
            last[pos] = end[i]

          strides[pos] = step[i]

        begin_mask = 0
        for dim, pos in enumerate(begin):
          if pos == 0:
            begin_mask |= 1 << dim
        
        end_mask = 0
        for dim, pos in enumerate(end):
          if isinstance(pos, int) and pos >= NNDCT_CONSTANT.INT_MAX:
            end_mask |= 1 << dim

        node.set_node_attr(node.op.AttrName.BEGIN, begin)
        node.set_node_attr(node.op.AttrName.BEGIN_MASK, begin_mask)
        node.set_node_attr(node.op.AttrName.END, last)
        node.set_node_attr(node.op.AttrName.END_MASK, end_mask)
        node.set_node_attr(node.op.AttrName.STRIDES, strides)
      elif node.op.type in [NNDCT_OP.SQUEEZE, NNDCT_OP.SUM, NNDCT_OP.MAX, NNDCT_OP.MEAN]:
        new_dims = []
        input_dims = node.in_tensors[0].ndim
        if node.node_attr(node.op.AttrName.DIMS) == [None]:
          new_dims = [i for i in range(len(node.in_tensors[0].shape))]
        else:
          for dim in node.node_attr(node.op.AttrName.DIMS):
            if isinstance(dim, int) and dim < 0:
              new_dims.append(input_dims + dim)
            else:
              new_dims.append(dim)
        node.set_node_attr(node.op.AttrName.DIMS, new_dims)
        
      elif node.op.type == NNDCT_OP.TRANSPOSE:
        input_dims = node.in_tensors[0].ndim
        new_order = list(range(input_dims)) 
        transpose_order = node.node_attr(node.op.AttrName.ORDER)
        tmp = new_order[transpose_order[0]]
        new_order[transpose_order[0]] = new_order[transpose_order[1]]
        new_order[transpose_order[1]] = tmp
        node.set_node_attr(node.op.AttrName.ORDER, new_order)
        node.op.type = NNDCT_OP.PERMUTE
      elif node.op.type in [NNDCT_OP.CONCAT, NNDCT_OP.SHAPE, NNDCT_OP.SOFTMAX]:
        input_dims = node.in_tensors[0].ndim
        dim = node.node_attr(node.op.AttrName.AXIS)
        if isinstance(dim, int) and dim < 0:
          dim = input_dims + dim
          node.set_node_attr(node.op.AttrName.AXIS, dim)   

      elif node.op.type == NNDCT_OP.FLATTEN:
        input_dims = node.in_tensors[0].ndim
        start_dim = node.node_attr(node.op.AttrName.START_DIM)
        end_dim = node.node_attr(node.op.AttrName.END_DIM)
        if isinstance(start_dim, int) and start_dim < 0:
          start_dim = start_dim + input_dims
        node.set_node_attr(node.op.AttrName.START_DIM, start_dim)
        if isinstance(end_dim, int) and end_dim < 0:
          end_dim = end_dim + input_dims
        node.set_node_attr(node.op.AttrName.END_DIM, end_dim)

    
            

  def constant_folding(self):
    folding_nodes = set()
    for node in self._dev_graph.nodes:    
      if node.in_quant_part is False:
          continue
      # if hasattr(node.op, "AttrName") and node.op.type not in [NNDCT_OP.ADD, NNDCT_OP.SUB, NNDCT_OP.MULTIPLY, NNDCT_OP.DIV]:
      if hasattr(node.op, "AttrName"):
        for attr_name in node.op.attrs.keys():
          if attr_name.name in ["INPUT", "OTHER"]:
            continue
          attr_val = node.node_attr(attr_name)
          if isinstance(attr_val, list):
            for i, val in enumerate(attr_val):
              attr_val[i] = self._materialize(node, val, folding_nodes)
          else:
            attr_val = self._materialize(node, attr_val, folding_nodes)
          if node.op.attrs[attr_name].type == list:
            attr_val = [attr_val]
          node.set_node_attr(attr_name, attr_val)  
    
    if folding_nodes:   
      for node_name in folding_nodes:
        node = self._dev_graph.node(node_name)
        for out in node.out_tensors:
          while out.uses:
            out.uses[0].user.remove_input(out.uses[0].offset)

        self._dev_graph.node(node_name).destroy()
      
      self._dev_graph.reconnect_nodes()

  @staticmethod
  def _infer_op_value_immediately(op_type):
    return op_type in [NNDCT_OP.SHAPE, NNDCT_OP.CONST]

  def _eval_node_value(self, node):
    if node.out_tensors[0].data is None:
      self._evalute_func_map[node.op.type](node)

  def _materialize(self, cur_node, value, folding_nodes):
    if not isinstance(value, Tensor) or value.node is None:
      return value
    
    evaluator_s = []
    tran_q = deque()
    value_node = value.node
    tran_q.append(value_node)
    while len(tran_q) > 0:
      node = tran_q.popleft()
      evaluator_s.append(node)
      if node.op.type not in [NNDCT_OP.SHAPE, NNDCT_OP.CONST] and node.name not in folding_nodes:
        for pn in node.owning_graph.parents(node):
          tran_q.append(pn)

    while len(evaluator_s) > 0:
      node = evaluator_s.pop()
      self._eval_node_value(node)
      folding_nodes.add(node.name)
    

    return value_node.out_tensors[0].data            



    
  """     
  def _materialize(self, cur_node, value, folding_nodes):
    visited = set()

    def dfs(node):
      visited.add(node.name)
      if self._infer_op_value_immediately(node.op.type):
        folding_nodes.add(node.name)
        self._eval_node_value(node)
        return True
      elif hasattr(node, "const_folding") and node.const_folding is True:
        folding_nodes.add(node.name)
        self._eval_node_value(node)
        return True
      elif node.op.type not in self._evalute_func_map:
        return False

      find_evaluable_op = False
      for tensor in node.in_tensors:
        if tensor.node and tensor.node.name not in visited:  # and tensor.data is None:
          find_evaluable_op = dfs(tensor.node)
          if find_evaluable_op is False:
            break
      
      if find_evaluable_op:
        folding_nodes.add(node.name)
        self._eval_node_value(node)
        
      return find_evaluable_op

    if not isinstance(value, Tensor):
      return value
    else:
      # if hasattr(value.node, "const_folding") and value.node.const_folding is True:
      #   folding_nodes.add(value.node.name)
      is_evaluable = dfs(value.node)
      if is_evaluable:
        data = value.node.out_tensors[0].data          
        input_idx = cur_node.in_tensors.index(value)
        cur_node.remove_input(input_idx)

        if not cur_node.in_tensors and cur_node.op.type not in [NNDCT_OP.ZEROS, NNDCT_OP.QUANT_STUB]:
          cur_node.const_folding = True
        
        return data
      else:
        return value
  """
  def fuse_transpose_matmul(self):
    fuse_trans_handler = TransMatmulActvHandler()
    graph_searcher = GraphSearcher(self._dev_graph)
    node_sets = graph_searcher.find_nodes_from_type(
      [PatternType(pattern=[NNDCT_OP.PERMUTE, NNDCT_OP.MATMUL],
                   action=fuse_trans_handler)])
    removed_nodes = set()
    for _, node_list in node_sets.items():
      for nodeset in node_list:
        trans_node = nodeset[0]
        matmul_node = nodeset[1]
        if matmul_node.is_trans_b and trans_node not in removed_nodes:
          removed_nodes.add(trans_node)

    for trans_node in removed_nodes:
      self._dev_graph.remove_node(trans_node)
    



  def layout_tranform(self):
    """layout_transform TORCH(NCHW) -> XIR(NHWC)"""
 
    
    custom2xir =GLOBAL_MAP.get_ele(NNDCT_KEYS.CUSTOM_TO_XIR_LIST) 
    if custom2xir is None:
      custom2xir = []
    
    def _find_swim_order(ndim):
      return {
        2: [0, 1],
        3: [0, 2, 1],
        4: [0, 2, 3, 1],
        5: [0, 3, 4, 2, 1]
      }[ndim]
    
    def _find_sink_order(ndim):
      return {
        2: [0, 1],
        3: [0, 2, 1],
        4: [0, 3, 1, 2],
        5: [0, 4, 3, 1, 2]
      }[ndim]
    
    def _is_dim_transparent(node):
      return node.in_tensors[0].ndim and node.out_tensors[0].ndim and node.in_tensors[0].ndim == node.out_tensors[0].ndim
    
    def _not_change_memory_reshape(node, transpose_order):
      if node.op.type in [NNDCT_OP.RESHAPE]:
        rand_in_tensor = np.random.random(node.in_tensors[0].shape)
        reshape_transpose_tensor = rand_in_tensor.reshape(node.out_tensors[0].shape).transpose(transpose_order).flatten()
        transpose_reshape_tensor = rand_in_tensor.transpose(transpose_order).reshape(node.out_tensors[0].shape).flatten()
        if any(reshape_transpose_tensor != transpose_reshape_tensor):
          return False
        else:
          return True
      else:
        return True
    
    def _is_shape_transparent(node):
      return node.in_tensors[0].shape and node.out_tensors[0].shape and node.in_tensors[0].shape == node.out_tensors[0].shape
    
    def _have_special_layout(node):
      return node.out_tensors[0].ndim and  node.out_tensors[0].ndim >=3
    
    def _is_custom_op(node):
      return isinstance(node.op, base_op.CustomOp) and node.op.type not in custom2xir
    
    def _is_permute_op(node):
      return isinstance(node.op, base_op.Permute)

    def _is_terminate_op(node):
      return node.op.type == NNDCT_OP.RETURN

    def _is_same_scope(node_1, node_2):
      return node_1.owning_graph == node_2.owning_graph and node_1.owning_block == node_2.owning_block
    
    implicit_ops = [NNDCT_OP.CONV2D, 
                    NNDCT_OP.DEPTHWISE_CONV2D, 
                    NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D,
                    NNDCT_OP.CONVTRANSPOSE2D,
                    NNDCT_OP.MAX_POOL,
                    NNDCT_OP.AVG_POOL,
                    NNDCT_OP.ADAPTIVEAVGPOOL2D,
                    NNDCT_OP.INTERPOLATE,
                    NNDCT_OP.UP_SAMPLING,
                    NNDCT_OP.RESIZE,
                    NNDCT_OP.BATCH_NORM,
                    NNDCT_OP.MAX_POOL1D,
                    NNDCT_OP.CONV1D,
                    NNDCT_OP.CONV3D,
                    NNDCT_OP.DEPTHWISE_CONV3D,
                    NNDCT_OP.DEPTHWISE_CONVTRANSPOSE3D,
                    NNDCT_OP.CONVTRANSPOSE3D,
                    NNDCT_OP.PIXEL_SHUFFLE,
                    NNDCT_OP.PIXEL_UNSHUFFLE,
                    NNDCT_OP.RESIZE_3D,
                    NNDCT_OP.RESIZE_NEAREST_3D,
                    NNDCT_OP.REORG,
                    NNDCT_OP.CORRELATION1D_ELEMWISE,
                    NNDCT_OP.CORRELATION2D_ELEMWISE,
                    NNDCT_OP.COST_VOLUME]
    
    special_ops_fn = {
      NNDCT_OP.RESHAPE: shape_attr_transform_fn,
      NNDCT_OP.CONCAT: axis_attr_transform_fn,
      NNDCT_OP.STRIDED_SLICE: slice_attr_transform_fn,
      NNDCT_OP.SUM: reduce_op_attr_transform_fn,
      NNDCT_OP.MAX: reduce_op_attr_transform_fn,
      NNDCT_OP.MEAN: reduce_op_attr_transform_fn,
      NNDCT_OP.SHAPE: axis_attr_transform_fn,
      NNDCT_OP.SOFTMAX: axis_attr_transform_fn,
      NNDCT_OP.ZEROS: shape_attr_transform_fn,
      NNDCT_OP.ARGMAX_DIM: axis_attr_transform_fn,

    }          
    
    
    # collect insert point for transpose
    insert_pos = []
    for node in self._dev_graph.nodes:
      if node.op.type in implicit_ops:
        insert_pos.append(node)

    swim_transpose = defaultdict(list)
    swim_in_transpose = defaultdict(list)
    sink_transpose = defaultdict(list)

    for node in insert_pos:
      tranpose_out_order = tuple(_find_swim_order(node.out_tensors[0].ndim))
      swim_transpose[tranpose_out_order].append(node)
      tranpose_in_order = tuple(_find_swim_order(node.in_tensors[0].ndim))
      swim_in_transpose[node] = tranpose_in_order
      tranpose_out_order = tuple(_find_sink_order(node.out_tensors[0].ndim))
      sink_transpose[tranpose_out_order].append(node)

      
      
    nodes_need_to_remove = []
    transpose_insert_between_swim = defaultdict(list)
    visited = []
    # swim_transpose_order, nodes = next(iter(swim_transpose.items()))
    for swim_transpose_order, nodes in swim_transpose.items():
      for insert_node in nodes:
        q = deque()
        q.append(insert_node)
        visited.append(insert_node)
        insert_node.transpose_out_order = swim_transpose_order
        insert_node.transpose_in_order = swim_in_transpose[insert_node]
        while len(q) > 0:
          node = q.popleft()
          for pn in self._dev_graph.parents(node):
            if pn not in visited:

              if not _have_special_layout(pn) or pn.op.type in implicit_ops:
                continue
      
              elif pn.op.type in [NNDCT_OP.INPUT, NNDCT_OP.QUANT_STUB, NNDCT_OP.CONST, NNDCT_OP.ZEROS]  or _is_dim_transparent(pn) and (not _is_permute_op(pn)) and  (not _is_custom_op(pn)) and _not_change_memory_reshape(pn, node.transpose_in_order):
                pn.transpose_out_order = node.transpose_in_order
                pn.transpose_in_order = pn.transpose_out_order
                if pn.op.type in special_ops_fn:
                  special_ops_fn[pn.op.type](pn, pn.transpose_out_order)
                q.append(pn) 
                visited.append(pn)
                   
              else:
                # pn.transpose_out_order = [0, 2, 3, 1]
                transpose_insert_between_swim[swim_transpose_order].append((pn, node))
    
    index = 0
    for transpose_order, node_pairs in transpose_insert_between_swim.items():
      for pn, cn in node_pairs:
        if not _is_same_scope(pn, cn):
          continue
        node_name = "_".join([pn.name, "swim_transpose", f"{index}"])
        op = base_op.Permute(NNDCT_OP.PERMUTE)
        new_node = Node(node_name, op=op, dtype=pn.dtype, in_quant_part=pn.in_quant_part)
        new_node.set_node_attr(new_node.op.AttrName.ORDER, list(transpose_order))
        self._dev_graph.insert_node_between_nodes(new_node, pn, cn)
        nodes_need_to_remove.append(new_node)
        index += 1
                   
    if transpose_insert_between_swim:
      self._dev_graph.reconnect_nodes()
      
    # debug
    # print("#####swim######")
    # for node in self._dev_graph.nodes:
    #   print(node.op.type, node.name, node.transpose_out_order)
      
    transpose_insert_between_sink = defaultdict(list)
    visited = []
    for node in self._dev_graph.nodes:
      if node.transpose_out_order:
        nodes = sink_transpose[tuple(_find_sink_order(len(node.transpose_out_order)))]
        if node not in nodes:
          nodes.append(node)

    for sink_transpose_order, nodes in sink_transpose.items():
      for insert_node in nodes:
        if insert_node not in visited:
          q = deque()
          q.append(insert_node)
          visited.append(insert_node)
          while len(q) > 0:
            node = q.popleft()
            for cn in self._dev_graph.children(node):
              if cn not in visited:
                if cn.op.type in implicit_ops or _is_terminate_op(cn):
                  continue
                elif cn.op.type == NNDCT_OP.SHAPE:
                  visited.append(cn)
                  if node.transpose_out_order:
                    special_ops_fn[cn.op.type](cn, node.transpose_out_order)
                    continue
                elif cn.transpose_out_order:
                  q.append(cn)
                  visited.append(cn)
                elif _is_dim_transparent(cn) and (not _is_permute_op(cn)) and (not _is_custom_op(cn)) and _not_change_memory_reshape(cn, node.transpose_out_order):
                  cn.transpose_in_order = node.transpose_out_order
                  cn.transpose_out_order = cn.transpose_in_order
                  q.append(cn)
                  visited.append(cn)
                  if cn.op.type in special_ops_fn:
                    special_ops_fn[cn.op.type](cn, cn.transpose_out_order)
                else:
                  transpose_insert_between_sink[sink_transpose_order].append((node, cn))
                
    index = 0
    for transpose_order, node_pairs in transpose_insert_between_sink.items():
      for pn, cn in node_pairs:
        if not _is_same_scope(pn, cn):
          continue
        node_name = "_".join([pn.name, "sink_transpose", f"{index}"])
        op = base_op.Permute(NNDCT_OP.PERMUTE)
        new_node = Node(node_name, op=op, dtype=pn.dtype, in_quant_part=cn.in_quant_part)
        new_node.set_node_attr(new_node.op.AttrName.ORDER, list(transpose_order))
        self._dev_graph.insert_node_between_nodes(new_node, pn, cn)
       
        nodes_need_to_remove.append(new_node)
        index += 1
        
    if transpose_insert_between_sink:  
      self._dev_graph.reconnect_nodes()
      
    # debug
    # print("#####sink######")
    # for node in self._dev_graph.nodes:
    #   print(node.op.type, node.name, node.transpose_out_order)
    neighbor_broadcast = {}
    for node in self._dev_graph.nodes:
      if len(node.in_nodes) <= 1 or len(node.out_nodes) == 0 or node in implicit_ops:
        continue
      if all([node.transpose_out_order is None for node in self._dev_graph.parents(node)]) or all([node.transpose_out_order is not None for node in self._dev_graph.parents(node)]):
        continue
      #if node.out_tensors[0].dtype != "float32":
      #  continue
      transpose_order = None
      for pn in self._dev_graph.parents(node):
        transpose_order = pn.transpose_out_order
        if transpose_order is not None:
          break
        
      neighbor_broadcast[node] = transpose_order

    have_neighbors = False
    for node, transpose_order in neighbor_broadcast.items():
      index = 0
      for pn in self._dev_graph.parents(node):
        if pn.transpose_out_order is None and pn.out_tensors[0].ndim and node.out_tensors[0].ndim and pn.out_tensors[0].ndim == node.out_tensors[0].ndim and _is_same_scope(pn, node):
          # pn.transpose_out_order = node.transpose_out_order
          node_name = "_".join([node.name, "neighbor_transpose", f"{index}"])
          op = base_op.Permute(NNDCT_OP.PERMUTE)
          new_node = Node(node_name, op=op, dtype=node.dtype, in_quant_part=pn.in_quant_part)
          new_node.set_node_attr(new_node.op.AttrName.ORDER, list(transpose_order))
          self._dev_graph.insert_node_between_nodes(new_node, pn, node)
         
          index += 1
        
          nodes_need_to_remove.append(new_node)
          have_neighbors = True
          
    if have_neighbors:
      self._dev_graph.reconnect_nodes()
    
    # Debug
    # print("####neightbor######")
    # for node in self._dev_graph.nodes:
    #   print(node.op.type, node.name, node.transpose_out_order)    
    # remove consecutive transpose

    def merge_transpose_group(transpose_group):
      order = []
      reserved_trans = None
      for trans in transpose_group:
        if trans not in nodes_need_to_remove:
          reserved_trans = trans
          
        if not order:
          order = trans.node_attr(trans.op.AttrName.ORDER)
        else:
          new_order = len(order) * [None]
          tmp_order = trans.node_attr(trans.op.AttrName.ORDER)
          for i in range(len(order)):
            t_i = tmp_order[i]
            new_order[i] = order[t_i]
          order = new_order 
      
      if reserved_trans is None:
        reserved_trans = transpose_group[-1]
      
      reserved_trans.set_node_attr(reserved_trans.op.AttrName.ORDER, order)
      reserverd_nodes.append(reserved_trans)
          
      transpose_group.clear()

    def merge_father_and_child_recursion(node, visited, transpose_group, reserverd_nodes):
      visited.append(node)
      if _is_permute_op(node):
        if node.out_nodes and all([_is_permute_op(cn) for cn in self._dev_graph.children(node)]):
          if len(node.out_nodes) > 1:
            transpose_group.append(node)
            merge_transpose_group(transpose_group)
        else:
          transpose_group.append(node)
          merge_transpose_group(transpose_group)

      for cn in self._dev_graph.children(node):
        if cn not in visited:
          merge_father_and_child(cn, visited, transpose_group, reserverd_nodes)

    def merge_father_and_child_iteration(node_in, visited_in, transpose_group_in, reserverd_nodes_in):
      task_stack = [[node_in, visited_in, transpose_group_in, reserverd_nodes_in]]
      while len(task_stack) > 0:
          node, visited, transpose_group, reserverd_nodes = task_stack.pop()
          visited.append(node)
          if _is_permute_op(node):
            if node.out_nodes and all([_is_permute_op(cn) for cn in self._dev_graph.children(node)]):
              if len(node.out_nodes)>1:
                transpose_group.append(node)
                merge_transpose_group(transpose_group)
              else:
                transpose_group.append(node)
            else:
              transpose_group.append(node)
              merge_transpose_group(transpose_group)

          task_list = []
          for cn in self._dev_graph.children(node):
            if cn not in visited:
              task_list.append([cn, visited, transpose_group, reserverd_nodes])
          task_list.reverse()
          for task in task_list:
              task_stack.append(task)

    def merge_father_and_child(node, visited, transpose_group, reserverd_nodes):
      pass
 
    if NndctOption.nndct_traversal_graph_mode.value == 1:
        merge_father_and_child = merge_father_and_child_recursion
    else:
        merge_father_and_child = merge_father_and_child_iteration     
        
    def merge_brothers(reserverd_nodes):
      remove_nodes = []
      for node in self._dev_graph.nodes:
        if len(node.out_nodes) > 1 and all([_is_permute_op(cn) for cn in self._dev_graph.children(node)]):
          need_merge = True
          order = None
          for trans_node in self._dev_graph.children(node):
            if order is not None:
              if order != trans_node.node_attr(trans_node.op.AttrName.ORDER):
                need_merge = False
                break
            else:
              order = trans_node.node_attr(trans_node.op.AttrName.ORDER)
          
          if need_merge:
            reserverd_node = None
            for trans_node in self._dev_graph.children(node):
              if trans_node not in nodes_need_to_remove:
                reserverd_node = trans_node
                
            if reserverd_node is None:
              reserverd_node = self._dev_graph.children(node)[0]
            
            for trans_node in self._dev_graph.children(node):
              if trans_node is not reserverd_node and trans_node in reserverd_nodes:
                remove_nodes.append(trans_node)
                
                out_tensor = trans_node.out_tensors[0]
                out_tensor.replace_uses_with(reserverd_node.out_tensors[0])
           
                  
      for node in remove_nodes:
        node.destroy()
         
      if remove_nodes:
        self._dev_graph.reconnect_nodes()         
    
    source_nodes = []
    for node in self._dev_graph.nodes:
      if not node.in_tensors:
        source_nodes.append(node)
    
    transpose_group = []
    reserverd_nodes = []
    visited = []
    for source in source_nodes:
      merge_father_and_child(source, visited, transpose_group, reserverd_nodes)
      
    nodes_need_to_remove = [node for node in nodes_need_to_remove if node not in reserverd_nodes]
    
    for node in reserverd_nodes:
      order = node.node_attr(node.op.AttrName.ORDER)
      keep_order = True
      if any([index != dim for index, dim in enumerate(order)]):
        keep_order = False
      if keep_order:
        nodes_need_to_remove.append(node)
        
    for node in nodes_need_to_remove:
      self._dev_graph.remove_node(node)
       
    merge_brothers(reserverd_nodes)
    # debug
    # print("#####finalize######")       
    # for node in self._dev_graph.nodes:
    #   print(node.op.type, node.name, node.transpose_out_order)

    def delete_transpose_of_correlation(self):
      nodes_need_to_delete_for_special_ops = []
      nodes_need_to_insert_aster_special_ops = []
      nodes_need_to_merge_for_special_ops = []
      for node in self._dev_graph.nodes:
        if node.op.type == NNDCT_OP.MEAN and not node.node_attr(node.op.AttrName.KEEP_DIMS) and self._dev_graph.parents(node):
          pn = self._dev_graph.parents(node)[0]
          if pn.in_tensors and _is_permute_op(pn) and self._dev_graph.parents(pn):
            gpn = self._dev_graph.parents(pn)[0]
            if gpn.op.type in [NNDCT_OP.CORRELATION1D_ELEMWISE, NNDCT_OP.CORRELATION2D_ELEMWISE] and node.out_tensors[0].ndim and gpn.out_tensors[0].ndim == 5 and node.out_tensors[0].ndim == 4:

              nodes_need_to_delete_for_special_ops.append(pn)
              
              node.transpose_in_order = tuple(_find_swim_order(5))
              node.transpose_out_order = tuple(_find_swim_order(4))
              special_ops_fn[node.op.type](node, node.transpose_in_order)

              nodes_need_to_insert_aster_special_ops.append(node)
      index = 0
      for node in nodes_need_to_insert_aster_special_ops:
        cn = self._dev_graph.children(node)[0]
        node_name = "_".join([node.name, "sink_transpose", f"{index}"])
        op = base_op.Permute(NNDCT_OP.PERMUTE)
        new_node = Node(node_name, op=op, dtype=node.dtype, in_quant_part=node.in_quant_part)
        new_node.set_node_attr(new_node.op.AttrName.ORDER, tuple(_find_sink_order(4)))
        self._dev_graph.insert_node_between_nodes(new_node, node, cn)
        nodes_need_to_merge_for_special_ops.append(new_node)
        index += 1

      for node in nodes_need_to_delete_for_special_ops:
        self._dev_graph.remove_node(node)

      source_nodes = []
      for node in self._dev_graph.nodes:
        if not node.in_tensors:
          source_nodes.append(node)

      transpose_group = []
      reserverd_nodes = []
      visited = []
      for source in nodes_need_to_merge_for_special_ops:
        merge_father_and_child(source, visited, transpose_group, reserverd_nodes)

      nodes_need_to_merge_for_special_ops = [node for node in nodes_need_to_merge_for_special_ops if node not in reserverd_nodes]
    
      for node in reserverd_nodes:
        order = node.node_attr(node.op.AttrName.ORDER)
        keep_order = True
        if any([index != dim for index, dim in enumerate(order)]):
          keep_order = False
        if keep_order:
          nodes_need_to_merge_for_special_ops.append(node)
          
      for node in nodes_need_to_merge_for_special_ops:
        self._dev_graph.remove_node(node)
        
      merge_brothers(reserverd_nodes)

    delete_transpose_of_correlation(self)

  def fuse_redundant_transpose(self):
    def fuse_transpose_reduction_op(self):
      fuse_trans_handler = TransReductionOpActvHandler()
      graph_searcher = GraphSearcher(self._dev_graph)
      node_sets = graph_searcher.find_nodes_from_type(
        [PatternType(pattern=[NNDCT_OP.PERMUTE, NNDCT_OP.MEAN],
                    action=fuse_trans_handler)])
      removed_nodes = set()
      for _, node_list in node_sets.items():
        for nodeset in node_list:
          trans_node = nodeset[0]
          reduction_op_node = nodeset[1]
          if hasattr(reduction_op_node, "fuse_trans_reduction_op") and reduction_op_node.fuse_trans_reduction_op and trans_node not in removed_nodes:
            removed_nodes.add(trans_node)
            reduction_op_node.set_node_attr(reduction_op_node.op.AttrName.DIMS, reduction_op_node.new_dims)

      for trans_node in removed_nodes:
        self._dev_graph.remove_node(trans_node)
    
    fuse_transpose_reduction_op(self)
      
  def partition_by_quant_part(self) -> List[List[Graph]]:

    if not any([node.op.type == NNDCT_OP.QUANT_STUB for node in self._dev_graph.nodes]):
      return [[self._dev_graph]]
    
    id2nodes = defaultdict(set)
        
    def collect_node_set(node, set_id, visited=None):
    
      if visited is None:
        visited = []
      
      if node.op.type == NNDCT_OP.RETURN:
        return

      if not hasattr(node, "set_id"):
        node.set_id = set_id
      
      id2nodes[set_id].add(node)
      visited.append(node)
      
      for cn in self._dev_graph.children(node):
        if cn not in visited and cn.in_quant_part:
          collect_node_set(cn, set_id, visited)  
          
    def get_set_id_from_nodeset(nodeset):
      return min([node.set_id for node in nodeset])
    
    def partition_check(quant_graphs, node_graph_id):
      for node_name, graph_id in node_graph_id.items():
        if len(graph_id) > 1:
          NndctScreenLogger().error2user(QError.GRAPH_PARTITION, f"The subgraph{graph_id} hold {node_name} at the same time.")
      for node in self._dev_graph.nodes:
        if node.op.type == NNDCT_OP.RETURN:
          continue
        if node.in_quant_part and all([node not in graph for graph in quant_graphs]):
          raise RuntimeError(f"Please check graph partition: the quant node '{node.name}' should be in quant graph.")
        elif not node.in_quant_part and any([node in graph for graph in quant_graphs]):
          raise RuntimeError(f"Please check graph partition: the non-quant node '{node.name}' included in quant graph.")   
    
    set_id = 0           
    for node in self._dev_graph.nodes:
      visited = []
      if node.op.type == NNDCT_OP.QUANT_STUB or (not node.in_nodes and node.in_quant_part):
        collect_node_set(node, set_id, visited)
        set_id += 1
    
    # for _, nodeset in id2nodes.items():
    #   sorted_node = self._dev_graph.top_sort_nodeset(nodeset)
    #   print(f"id: {_}")
    #   for node in sorted_node:
    #     print(node.name, node.op.type)

    merged_id2nodes = defaultdict(set)
    for _, nodeset in id2nodes.items():
      id = get_set_id_from_nodeset(nodeset)
      merged_id2nodes[id].update(nodeset)
    
    quant_dev_graph = []
    node_graph_id = defaultdict(list)
    for graph_id, nodes in merged_id2nodes.items():
      for node in nodes:
        node_graph_id[node.name].append(graph_id)
      subgraph = Graph.create_subgraph_from_nodeset(self._dev_graph, nodes, f"{self._dev_graph.name}_{graph_id}")
      quant_dev_graph.append(subgraph)

    partition_check(quant_dev_graph, node_graph_id)
    if NndctOption.nndct_dump_no_quant_part.value: 
      return [quant_dev_graph, [self._dev_graph]]
    else:
      return [quant_dev_graph]


  @property
  def dev_graph(self):
    return self._dev_graph


  def fuse_pad(self):
    fuse_pad_handler = PadFuseHandler()
    graph_searcher = GraphSearcher(self._dev_graph)
    nodesets = graph_searcher.find_nodes_from_type(
        [PatternType(pattern=[NNDCT_OP.PAD],
                     action=fuse_pad_handler), 
        ])
    removed_pad = set()
    for id, node_list in nodesets.items():
      for nodeset in node_list:
        pad_node = nodeset[0]
        if pad_node.merged and pad_node.name not in removed_pad:
          removed_pad.add(pad_node)
          self._dev_graph.remove_node(pad_node)

  def convert_reshapelike_to_reshape(self):
    def _mem_layout_changed(node):
      order = node.node_attr(node.op.AttrName.ORDER)
      input_shape = node.in_tensors[0].shape
      output_shape = node.out_tensors[0].shape
      q = deque()
      for i, dim_shape in enumerate(input_shape):
        if dim_shape > 1:
          q.append(i)

      for dim_shape, dim in zip(output_shape, order):
        if dim_shape > 1:
          if dim != q.popleft():
            return True
      
      return False

    for node in self._dev_graph.nodes:
      if node.op.type in [NNDCT_OP.FLATTEN, NNDCT_OP.SQUEEZE, NNDCT_OP.UNSQUEEZE] or node.op.type in [NNDCT_OP.PERMUTE, NNDCT_OP.TRANSPOSE] and (not _mem_layout_changed(node)):
        output_shape = node.out_tensors[0].shape
        new_reshape_op = base_op.Reshape(NNDCT_OP.RESHAPE)
        new_reshape_op.set_attr(new_reshape_op.AttrName.SHAPE, output_shape)
        node.op = new_reshape_op
  
  def convert_shape_tensor_to_const(self):
    for node in self._dev_graph.nodes:
      if node.op.type == NNDCT_OP.SHAPE_AS_TENSOR:
        input_shape = node.in_tensors[0].shape
        new_const_op = base_op.Constant(NNDCT_OP.CONST)
        new_const_op.set_attr(new_const_op.AttrName.DATA, list(input_shape))
        node.op = new_const_op
        node.remove_all_inputs()


  def merge_permute_to_linear(self):
    merge_permute_handler = PermuteMergeHandler()
    graph_searcher = GraphSearcher(self._dev_graph)
    nodesets = graph_searcher.find_nodes_from_type(
        [PatternType(pattern=[NNDCT_OP.PERMUTE, NNDCT_OP.FLATTEN, NNDCT_OP.DENSE],
                     action=merge_permute_handler), 
         PatternType(pattern=[NNDCT_OP.PERMUTE, NNDCT_OP.RESHAPE, NNDCT_OP.DENSE], 
                     action=merge_permute_handler),
         PatternType(pattern=[NNDCT_OP.PERMUTE, NNDCT_OP.SQUEEZE, NNDCT_OP.DENSE], 
                     action=merge_permute_handler),
        ])

    removed_permute = set()
    for id, node_list in nodesets.items():
      for nodeset in node_list:
        permute_node = nodeset[0]
        if permute_node.merged and permute_node.name not in removed_permute:
          removed_permute.add(permute_node)
          self._dev_graph.remove_node(permute_node)

  def merge_consecutive_reshape(self):
    while True:
      merge_reshape_handler = ReshapeMergeHandler()
      graph_searcher = GraphSearcher(self._dev_graph)
      nodesets = graph_searcher.find_nodes_from_type(
          [PatternType(pattern=[NNDCT_OP.RESHAPE, NNDCT_OP.RESHAPE],
                      action=merge_reshape_handler)])
    
      nodes_list = []
      for _, nodes in nodesets.items():
        nodes_list.extend(nodes)
      if all([not nodeset for nodeset in nodes_list]):
        break
     
      removed_reshape = set()
      for node in self._dev_graph.nodes:
        if node.op.type == NNDCT_OP.RESHAPE and len(node.out_tensors[0].uses) == 0:
          removed_reshape.add(node)
      
      if not removed_reshape:
        break
      
      for node in removed_reshape:
        node.destroy()

  def broadcast_const_for_binary_op(self):
    for node in self._dev_graph.nodes:
      if node.op.type in [NNDCT_OP.ADD, NNDCT_OP.SUB, NNDCT_OP.RSUB, NNDCT_OP.MULTIPLY, NNDCT_OP.DIV]:
        input, other = node.node_attr(node.op.AttrName.INPUT), node.node_attr(node.op.AttrName.OTHER)
        if input.data.shape == other.data.shape:
          continue
       
        if other.is_param_tensor():
          dtype = input.data.dtype
          operand2 = np.array(np.ones(input.shape, dtype=dtype) * other.data)
          other.from_ndarray(operand2)
        elif other.node.op.type == NNDCT_OP.CONST:
          dtype = input.data.dtype
          operand2 = np.array(np.ones(input.shape, dtype=dtype) * other.data)
          other.from_ndarray(operand2)
          other.node.transpose_out_order = None
          if not other.is_param_tensor():
            other.node.set_node_attr(other.node.op.AttrName.DATA, operand2)


  def convert_rsub_to_sub(self):
    for node in self._dev_graph.nodes:
      if node.op.type == NNDCT_OP.RSUB:
        new_sub_op = base_op.BinaryOp(NNDCT_OP.SUB)
        input, other = node.node_attr(node.op.AttrName.INPUT), node.node_attr(node.op.AttrName.OTHER)
        new_sub_op.set_attr(new_sub_op.AttrName.INPUT, other)
        new_sub_op.set_attr(new_sub_op.AttrName.OTHER, input)
        node.op = new_sub_op

  def update_node_data(self):
    for node in self._dev_graph.nodes:
      for tensor in node.out_tensors:
        if isinstance(tensor.data, np.ndarray):
          output_data = permute_data(tensor.data, node.transpose_out_order)
          tensor.from_ndarray(output_data) 
    
    permute_nodes = self._dev_graph.find_nodes_by_types([NNDCT_OP.PERMUTE])
    for node in permute_nodes:
      in_data = node.in_tensors[0].data
      if in_data is not None:
        data = permute_data(in_data, node.node_attr(node.op.AttrName.ORDER))
        node.out_tensors[0].from_ndarray(data)
      else:
        NndctScreenLogger().warning2user(QWarning.DATA_DUMP, f"{node.__repr__()} has no data.")
 
  def convert_adaptive_pool_to_pool(self):
    for node in self._dev_graph.nodes:
      if node.op.type == NNDCT_OP.ADAPTIVEAVGPOOL2D:
        input_size = node.in_tensors[0].shape[1:3]
        output_size = node.out_tensors[0].shape[1:3]
        mod = [input_size[i] % output_size[i] for i in range(0, len(input_size))]
        if mod != [0] * len(mod):
          continue

        stride_h = int(input_size[0] / output_size[0])
        stride_w = int(input_size[1] / output_size[1])
        kernel_h = input_size[0] - (output_size[0] - 1) * stride_h
        kernel_w = input_size[1] - (output_size[1] - 1) * stride_w
        op = base_op.AvgPool(NNDCT_OP.AVG_POOL)
        op.set_attr(op.AttrName.KERNEL, [kernel_w, kernel_h])
        op.set_attr(op.AttrName.STRIDE, [stride_w, stride_h])
        op.set_attr(op.AttrName.COUNT_INCLUDE_PAD, True)
        op.set_attr(op.AttrName.PAD_MODE, 0)
        op.set_attr(op.AttrName.PAD, [0, 0, 0, 0])
        if list(output_size) == [1, 1]:
          op.set_attr(op.AttrName.GLOBAL, True)
          
        node.op = op




        
    
  

    
