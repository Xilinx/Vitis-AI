

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

from collections import ChainMap, OrderedDict, defaultdict
from copy import copy

from nndct_shared.nndct_graph import GraphSearcher
from nndct_shared.utils import NndctDebugLogger, NndctOption, PatternType
from pytorch_nndct.utils.jit_utils import *

from .opt_pass import OptPass
from .torch_graph import *


class TorchGraphHandler(object):

  def __init__(self):
    self._transparent_ops = ["ListUnpack", "TupleUnpack"]
    
  def build_torch_graph(self, graph_name, module, input_args, train=False):
    self._module = module
    fw_graph, params = self._trace_graph_from_model(input_args, train)

    self._node_kinds = {node.kind().split(":")[-1] for node in fw_graph.nodes()}
    if NndctOption.nndct_parse_debug.value >= 1:
      NndctDebugLogger.write(f"jit graph:\n{fw_graph}")
      NndctDebugLogger.write(f"\nparsing nodes types:\n{self._node_kinds}\n")

    raw_graph, raw_params = self._build_raw_graph(graph_name, fw_graph, params)
    self._infer_layout(raw_graph)
    if NndctOption.nndct_parse_debug.value >= 2:
      NndctDebugLogger.write(f"\ntorch raw graph:\n{raw_graph}")   
    opt_graph = self._opt_raw_graph(raw_graph)
    if NndctOption.nndct_parse_debug.value >= 2:
      NndctDebugLogger.write(f"\ntorch opt graph:\n{raw_graph}")
      
    return opt_graph, raw_params

  @property
  def graph_nodes(self):
    for node in [self._graph.param_node()] + list(self._graph.nodes()):
      yield node

  def _trace_graph_from_model(self, input_args, train):
    graph, output = trace_and_get_graph_from_model(self._module, input_args,
                                                    train)
    graph = optimize_graph(graph)
    params = rename_graph_param_name(self._module, graph)
    return graph, params
      
  def _opt_raw_graph(self, raw_graph):
    graph_searcher = GraphSearcher(raw_graph)
    
   
    # classification
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["t", "addmm"])])
    OptPass.merge_param_transpose_with_addmm(raw_graph, node_sets)
    
    # shufflenet
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["ListUnpack"]), 
                                                        PatternType(pattern=["TupleUnpack"])])
    OptPass.unpack_ListUnpack_op(raw_graph, node_sets)
        
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["slice"])])
    
    OptPass.slice_to_strided_slice(raw_graph, node_sets)
      
    # yolo_v3
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["select", "copy_"])])
    OptPass.select_to_slice_inplace_copy(raw_graph, node_sets)
    
    # FADnet
    while True:
      node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["select", "strided_slice"])])
      if not OptPass.merge_select_to_strided_slice(raw_graph, node_sets):
        break
    # FADnet
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["strided_slice", "strided_slice"])])
    OptPass.merge_consecutive_strided_slice(raw_graph, node_sets)
    
    # 3d pointpillar
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["strided_slice", "index_put_"])])
    OptPass.stride_slice_to_index_inplace_put(raw_graph, node_sets)
    
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["strided_slice", "copy_"])])
    OptPass.create_stride_slice_inplace_copy(raw_graph, node_sets)
    
    # nd(>2) linear (JIRA 2646)
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["matmul", "add"]),
                                                     PatternType(pattern=["matmul", "add_"])])
    OptPass.merge_matmul_with_add(raw_graph, node_sets)                                                

      
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["ListConstruct"]),
                                                        PatternType(pattern=["TupleConstruct"])])
    OptPass.pack_ListConstruct_op(raw_graph, node_sets)
    
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["embedding_bag"])])
    OptPass.strip_reduantant_tensors_in_embedding_bag(raw_graph, node_sets)
    
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["empty", "zero_"])])
    OptPass.merge_empty_with_zero(raw_graph, node_sets)
    
    # delete node should be done after merge stride_slice
    # delete reduantant view FADnet. 
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["view"])])
    OptPass.remove_reduantant_view(raw_graph, node_sets)

    return raw_graph

  def _build_raw_graph(self, graph_name, fw_graph, params):
    raw_graph = TorchGraph.new_graph(graph_name)
    extra_inputs_of_node = defaultdict(list)
    for fw_value in fw_graph.param_node().outputs():
      if unique_name(fw_value) not in params:
        input_node = TorchNode(fw_graph.param_node())
        value = TorchValue(fw_value)
        value.node = input_node
        input_node.add_output(value)
        raw_graph.add_node(input_node)
      else:
        value = TorchValue(fw_value)
        raw_graph.add_param_value(value)

    extra_count = 0
    for fw_node in fw_graph.nodes():
      if fw_node.kind().split("::")[-1] != "Constant":
        for attr_name in fw_node.attributeNames():
          value = get_attr_value(fw_node, attr_name)
          torch_value = TorchValue(value, name=f"extra_{extra_count}")
          extra_inputs_of_node[fw_node].append(torch_value)
          raw_graph.add_blob_value(torch_value)
          extra_count += 1
          
        if get_node_type(fw_node) == 'prim::PythonOp':
          for value in fw_node.scalar_args():
            torch_value = TorchValue(value, name=f"extra_{extra_count}")
            extra_inputs_of_node[fw_node].append(torch_value)
            raw_graph.add_blob_value(torch_value)
            extra_count += 1
          # ip = fw_graph.insertConstant(value)
          # fw_node.addInput(ip)
      else:
        const_value = TorchValue(list(fw_node.outputs())[0])
        if const_value.is_plain_value() or const_value.is_none():
          raw_graph.add_blob_value(const_value)
        else:
          const_node = TorchNode(fw_node)
          const_value.node = const_node
          const_node.add_output(const_value)
          raw_graph.add_node(const_node)
        
    for fw_node in fw_graph.nodes():
      self.add_torch_node(raw_graph, fw_node, extra_inputs_of_node[fw_node])

    for ip in fw_graph.return_node().inputs():
      ret_value = raw_graph.get_blob_value_by_name(unique_name(ip))
      if ret_value.node.kind in ["TupleConstruct"]:
        for ip in ret_value.node.inputs:
          raw_graph.add_ret_value(ip)
        raw_graph.remove_node(ret_value.node)
      else:
        raw_graph.add_ret_value(ret_value)
    self._connect_nodes(raw_graph)
    
    raw_params = {param_name: raw_graph.get_param_value_by_name(param_name) for param_name in raw_graph.param_names()}
    
    return raw_graph, raw_params

  def reconnect_nodes(self, raw_graph):
    for idx, node in enumerate(raw_graph.nodes):
      node.idx = idx
      node.clean_connection()
    self._connect_nodes(raw_graph)

  @staticmethod
  def _connect_nodes(raw_graph):
    for nodeA in raw_graph.nodes:
      for ip in nodeA.flatten_inputs:
        for nodeB in raw_graph.nodes:
          if nodeB is not nodeA and ip in nodeB.outputs:
            nodeB.add_out_node(nodeA)
            nodeA.add_in_node(ip.node)

  def add_torch_node(self, raw_graph, fw_node, extra_inputs_of_node):
    
    schema = get_node_schema(fw_node)
    # assert schema
    node = TorchNode(fw_node)
    node.schema = schema
    for ip in fw_node.inputs():
      ip_value = raw_graph.get_blob_value_by_name(unique_name(ip)) \
      if unique_name(ip) in raw_graph.blobs_name() else \
      raw_graph.get_param_value_by_name(unique_name(ip))
      
      node.add_input(ip_value)
    for op in fw_node.outputs():
      value = TorchValue(op)
      value.node = node
      node.add_output(value)
    
    for extra_input in extra_inputs_of_node:
      node.add_input(extra_input)

    if node.inputs:
      raw_graph.add_node(node)
      
  def _infer_layout(self, raw_graph):
    
    def layout_transform(node):
      if node.kind == "transpose":
        layout = node.outputs[0].layout   
        dim0 = node.inputs[1].data
        dim1 = node.inputs[2].data
        layout[dim0], layout[dim1] = layout[dim1], layout[dim0]
        return layout
      elif node.kind == "permute":
        dims = node.inputs[1].data
        layout = [None] * 4
        dim_map = {i: dim for i, dim in enumerate(dims)}
        for i, dim in enumerate(node.outputs[0].layout):
          layout[dim_map[i]] = dim
        return layout
             
  
    def set_layout_between_anchor(node, visited_nodes):
      for pn in raw_graph.parents(node):
        if pn in visited_nodes:
          continue
        elif pn in source_nodes:
          assert node.inputs[0].layout == nchw
        else:
          visited_nodes.add(pn)
          if pn.outputs[0].ndim == 4 and pn.kind in layout_transformation_op:
            pn.inputs[0].layout = layout_transform(pn)
          else:
            for inp in pn.inputs:
              if inp.layout is None:
                inp.layout = pn.outputs[0].layout
              
          set_layout_between_anchor(pn, visited_nodes)   
    
    layout_anchor_op = ["_convolution",
                          "max_pool2d",
                          "avg_pool2d",
                          "adaptive_avg_pool2d",
                          "max_pool2d_with_indices",
                          "upsample_bilinear2d",
                          "upsample_nearest2d",
                          "replication_pad2d",
                          ]
    layout_transformation_op = ["transpose", "permute"]
    nchw = ['N', 'C', 'H', 'W']
    source_nodes = []
    for node in raw_graph.nodes:
      if node.kind in layout_anchor_op:
        source_nodes.append(node)
        node.outputs[0].layout = copy(nchw)
        node.inputs[0].layout = copy(nchw)
        
    visited_nodes = set()
    for node in source_nodes:
      set_layout_between_anchor(node, visited_nodes)
