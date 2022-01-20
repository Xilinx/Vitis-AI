
from collections import defaultdict
from typing import Dict, List
from nndct_shared.nndct_graph import GraphSearcher
from nndct_shared.utils import (NndctDebugLogger, NndctOption,
                                PatternType)

from .opt_pass import OptPass
from .torch_graph import *
from pytorch_nndct.utils.jit_utils import *

class TorchScriptModuleHandler(object):
  def __init__(self):
    self._extra_node_input_args = defaultdict(list)
    
  def build_torch_graph(self, graph_name, script_module, *args):
    def rename_graph_inputs(graph):
      for i, inp in enumerate(list(graph.inputs())[1:]):
        set_unique_name(inp, 'input_' + str(i))
    script_graph = script_module.graph.copy()
    script_graph = optimize_graph(script_graph, is_jit_graph=True)
    rename_graph_inputs(script_graph)
    # print(script_graph) 
    post_process_script_graph(script_graph)
    params = self._create_params_value(script_graph, script_module)
    # print("new:\n", script_graph)
    raw_graph, raw_params = self._build_raw_graph(graph_name if graph_name else script_module.__class__.__name__, script_graph, params=params)
    self._optimize_raw_graph(raw_graph)
    # print(raw_graph)

    return raw_graph, raw_params
  
  def _optimize_raw_graph(self, graph):
    for subgraph in graph.subgraphs():
      self._optimize_raw_graph(subgraph)
    self._execute_optimize(graph)
        
  def _build_raw_graph(self, graph_name, fw_graph, params=None, blobs=None):
    
    raw_graph = TorchGraph.new_graph(graph_name)
    if params:
      for param in params:
        raw_graph.add_param_value(param)
    if blobs:
      for blob in blobs:
        raw_graph.add_blob_value(blob)
        
    self._create_attrs_value(fw_graph, raw_graph)
    self._create_inputs_value(fw_graph, raw_graph)
  
    for fw_name, fw_node in get_fw_op_nodes(fw_graph):
      if list(fw_node.blocks()):
        self._add_node(fw_node, raw_graph)
        blobs = []
        for blob_name in raw_graph.blobs_name():
          blobs.append(raw_graph.get_blob_value_by_name(blob_name))
                    
        block_node = list(raw_graph.nodes)[-1]
        for i, fw_block in enumerate(fw_node.blocks()):
          raw_block, _ = self._build_raw_graph(f"{fw_name}_block_{i}", fw_block, params, blobs)
          block_node.add_block(raw_block)
          
      elif node_type(fw_node) == "prim::ListConstruct" and should_construct_dynamic_list(fw_node):
        list_val = TorchValue(list(fw_node.outputs())[0])
        list_node = TorchNode(fw_node)
        list_val.node = list_node
        list_node.add_output(list_val)
        raw_graph.add_node(list_node)
      else:
        self._add_node(fw_node, raw_graph)
    
    self._create_ret_value(fw_graph, raw_graph)   
    raw_graph.connect_nodes()
    raw_params = {param_name: raw_graph.get_param_value_by_name(param_name) for param_name in raw_graph.param_names()}
    return raw_graph, raw_params
  
  def _create_ret_value(self, graph, raw_graph):
     for ip in get_fw_graph_ret_value(graph):
      ret_value = raw_graph.get_blob_value_by_name(unique_name(ip))
      raw_graph.add_ret_value(ret_value)
      """
      if ret_value.node and ret_value.node.kind in ["TupleConstruct"]:
        for ip in ret_value.node.inputs:
          raw_graph.add_ret_value(ip)
        raw_graph.remove_node(ret_value.node)
      else:
        raw_graph.add_ret_value(ret_value)
      """
    
  def _create_inputs_value(self, graph, raw_graph):
     for ip in get_fw_graph_inputs(graph):
      input_node = TorchNode(ip.node())
      value = TorchValue(ip)
      value.node = input_node
      input_node.add_output(value)
      raw_graph.add_node(input_node)
      
  def _create_attrs_value(self, graph, raw_graph):
    for fw_name, fw_node in get_fw_op_nodes(graph):
      if node_type(fw_node) != "prim::Constant":
        extra_count = 0
        for attr_name in fw_node.attributeNames():
          value = get_attr_value(fw_node, attr_name)
          torch_value = TorchValue(value, name=f"{fw_name}_extra_{extra_count}")
          self._extra_node_input_args[fw_node].append(torch_value)
          raw_graph.add_blob_value(torch_value)
          # self._visited_values[torch_value.name] = torch_value
          extra_count += 1
      else:
        const_value = TorchValue(list(fw_node.outputs())[0])
        # self._visited_values[const_value.name] = const_value
        if const_value.is_plain_value() or const_value.is_none():
          raw_graph.add_blob_value(const_value)
          
        else:
          const_node = TorchNode(fw_node)
          const_value.node = const_node
          const_node.add_output(const_value)
          raw_graph.add_node(const_node)
    
  def _create_params_value(self, graph, script_module):
    params: List[TorchValue] = []
    getattr_nodes = graph.findAllNodes("prim::GetAttr", recurse=True)
    visited: Dict[str, torch.Value] = {}
    state_dict = script_module.state_dict()
    for node in getattr_nodes:
      if get_node_output_name(node) in visited:
        continue

      for getattrs in get_attr_chains(node):
        full_attr = getattr_full_name(getattrs)  # self.conv.weight -> conv.weight
        if full_attr in state_dict and full_attr not in visited:
          # print(f"set {unique_name(getattrs[-1].output())} => {full_attr}")
          set_unique_name(getattrs[-1].output(), full_attr)
          torch_tensor = state_dict[full_attr]
          value = TorchValue(getattrs[-1].output())
          value.dtype = {torch.float: 'torch.float', 
                          torch.float64: 'torch.double'}.get(torch_tensor.dtype, None)
          assert value.dtype
          value.shape = list(torch_tensor.size())
          visited[full_attr] = getattrs[-1].output()
          # raw_graph.add_param_value(value)
          # self._visited_values[value.name] = value
          params.append(value)
        elif full_attr in visited:
          re_use_param = visited[full_attr]
          getattrs[-1].output().replaceAllUsesWith(re_use_param) 
    return params
  
  def _add_node(self, fw_node, raw_graph):
   
    node = TorchNode(fw_node)
    for ip in fw_node.inputs():
      ip_value = raw_graph.get_blob_value_by_name(unique_name(ip)) \
      if unique_name(ip) in raw_graph.blobs_name() else \
      raw_graph.get_param_value_by_name(unique_name(ip))
      
      node.add_input(ip_value)
    for op in fw_node.outputs():
      value = TorchValue(op)
      value.node = node
      node.add_output(value)
    
    for extra_input in self._extra_node_input_args[fw_node]:
      node.add_input(extra_input)

    if node.inputs:
      raw_graph.add_node(node)
      
        
  def _execute_optimize(self, raw_graph):
    graph_searcher = GraphSearcher(raw_graph)
    
    # classification
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["t", "addmm"])])
    OptPass.merge_param_transpose_with_addmm(raw_graph, node_sets)
    
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["TupleConstruct", "TupleUnpack"])])
    OptPass.penetrate_pack_unpack(raw_graph, node_sets)
    
    # shufflenet
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["ListUnpack"]), 
                                                        PatternType(pattern=["TupleUnpack"])])
    OptPass.unpack_ListUnpack_op(raw_graph, node_sets)
    
    # node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["slice"])])
    
    # OptPass.slice_to_strided_slice(raw_graph, node_sets)
      
    # yolo_v3
    # node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["select", "copy_"])])
    # OptPass.select_to_slice_inplace_copy(raw_graph, node_sets)
    
    # 3d pointpillar
    # node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["strided_slice", "index_put_"])])
    # OptPass.stride_slice_to_index_inplace_put(raw_graph, node_sets)
    
    # node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["strided_slice", "copy_"])])
    # OptPass.create_stride_slice_inplace_copy(raw_graph, node_sets)
    
    # nd(>2) linear (JIRA 2646)
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["matmul", "add"]),
                                                     PatternType(pattern=["matmul", "add_"])])
    OptPass.merge_matmul_with_add(raw_graph, node_sets)                                                

      
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["ListConstruct"]),
                                                        PatternType(pattern=["TupleConstruct"])])
    OptPass.pack_ListConstruct_op(raw_graph, node_sets)
    
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["embedding_bag"])])
    OptPass.strip_reduantant_tensors_in_embedding_bag(raw_graph, node_sets)
    
    return raw_graph

  # def reconnect_nodes(self, raw_graph):
  #   for idx, node in enumerate(raw_graph.nodes):
  #     node.idx = idx
  #     node.clean_connection()
  #   self._connect_nodes(raw_graph)

  # @staticmethod
  # def _connect_nodes(raw_graph):
  #   for nodeA in raw_graph.nodes:
  #     for ip in nodeA.flatten_inputs:
  #       for nodeB in raw_graph.nodes:
  #         if nodeB is not nodeA and ip in nodeB.outputs:
  #           nodeB.add_out_node(nodeA)
  #           nodeA.add_in_node(ip.node)
