

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
from nndct_shared.utils import NndctDebugLogger, NndctOption, PatternType, NndctScreenLogger
from pytorch_nndct.utils.jit_utils import *

from .opt_pass import OptPass
from .torch_graph import *

class TorchGraphHandler(object):

  def __init__(self):
    self._transparent_ops = ["ListUnpack", "TupleUnpack"]
    
  def build_torch_graph(self, graph_name, module, input_args, train=False):
    self._module = module
    NndctScreenLogger().info("Start to trace model...")
    fw_graph, params = self._trace_graph_from_model(input_args, train)
    NndctScreenLogger().info("Finish tracing.")

    self._node_kinds = {node.kind().split(":")[-1] for node in fw_graph.nodes()}
    if NndctOption.nndct_parse_debug.value >= 1:
      NndctDebugLogger.write(f"jit graph:\n{fw_graph}")
      NndctDebugLogger.write(f"\nparsing nodes types:\n{self._node_kinds}\n")

    raw_graph, raw_params = self._build_raw_graph(graph_name, fw_graph, params)
    if NndctOption.nndct_parse_debug.value >= 2:
      NndctDebugLogger.write(f"\ntorch raw graph:\n{raw_graph}")   
    opt_graph = self._opt_raw_graph(raw_graph)
    if NndctOption.nndct_parse_debug.value >= 2:
      NndctDebugLogger.write(f"\ntorch opt graph:\n{raw_graph}")
    
    if NndctOption.nndct_parse_debug.value >= 3:
      self._check_stub_topology(opt_graph)
    
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
    
    # torch 1.8.x will generate redundant type_as op before element-wise add.
    if get_torch_version() >= 180 and get_torch_version() < 190:
      node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["type_as", "add"])])
      OptPass.merge_internal_type_as(raw_graph, node_sets)
      
    # classification
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["t", "addmm"]), PatternType(pattern=["t", "matmul"])])
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

    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["add"]), PatternType(pattern=["sub"]), PatternType(pattern=["mul"]), PatternType(pattern=["div"])])
    OptPass.transform_const_scalar_to_const_tensor(raw_graph, node_sets)
    
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
          if attr_name == "inplace":
            continue
          value = get_attr_value(fw_node, attr_name)
          torch_value = TorchValue(value, name=f"extra_{extra_count}")
          extra_inputs_of_node[fw_node].append(torch_value)
          raw_graph.add_blob_value(torch_value)
          extra_count += 1
          
        if node_type(fw_node) == 'prim::PythonOp':
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

  
  def _check_stub_topology(self, raw_graph):
    def _path_str(path):
      return "->\n".join([f"({node.name}, {node.kind})" for node in path])
      
    if not any([node.kind in ["QuantStubF", "DeQuantStubF"] for node in raw_graph.nodes]):
      return
  
    sources = []
    ends = []
    for node in raw_graph.nodes:
      if not node.in_nodes or node.kind in ["tensor", "zeros", "new_zeros"]:
        sources.append(node)
      elif not node.out_nodes:
        ends.append(node)
        
    first_quant_stubs = []
    stack = []
    lack_dequant_stub_paths = []
    for source in sources:
      visited = []
      stack.append(source)
      while stack:
        node = stack.pop()
        visited.append(node)
        if node.kind == "QuantStubF" and node not in first_quant_stubs:
          first_quant_stubs.append(node)
          continue
          
        for cn in node.out_nodes:
          if cn not in visited:
            stack.append(cn)

    stack = []
    for quant_stub in first_quant_stubs:
      path_to = {}
      path_list = []
      visited = []
      stack.append(quant_stub)
      while stack:
        node = stack.pop()
        visited.append(node)
        if node.kind == "DeQuantStubF":
          continue
        elif not node.out_nodes:
          pn = node
          while pn:
            path_list.append(pn)
            pn = path_to[pn] if pn in path_to else None
          path_list.reverse()
          lack_dequant_stub_paths.append(path_list)
        
        for cn in node.out_nodes:
          if cn not in visited:
            stack.append(cn)
            path_to[cn] = node
    
    if lack_dequant_stub_paths:
      print("####lack of dequant op:")
      for path in lack_dequant_stub_paths:
        print(f"\n{_path_str(path)}")
          
    
    last_dequant_stubs = []
    stack = []
    lack_quant_stub_paths = []
    for end in ends:
      visited = []
      # path_to = {}
      # path_list = []
      stack.append(end)
      while stack:
        node = stack.pop()
        visited.append(node)
        if node.kind == "DeQuantStubF" and node not in last_dequant_stubs:
          last_dequant_stubs.append(node)
          continue
      
        for pn in node.in_nodes:
          if pn not in visited:
            stack.append(pn)

    stack = []
    for dequant_stub in last_dequant_stubs:
      path_to = {}
      path_list = []
      visited = []
      stack.append(dequant_stub)
      while stack:
        node = stack.pop()
        visited.append(node)
        if node.kind == "QuantStubF":
          continue
        elif not node.in_nodes:
          cn = node
          while cn:
            path_list.append(cn)
            cn = path_to[cn] if cn in path_to else None
          
          lack_quant_stub_paths.append(path_list)
        
        for pn in node.in_nodes:
          if pn not in visited:
            stack.append(pn)
            path_to[pn] = node
  
    if lack_quant_stub_paths:
      print("####lack of quant stub ops:")
      for path in lack_quant_stub_paths:
        print(f"\n{_path_str(path)}")
    
    # quant_dequant_pairs = []
    # for quant_stub in first_quant_stubs:
    #   for dequant_stub in last_dequant_stubs:
    #     if quant_stub.idx < dequant_stub.idx:
    #       quant_dequant_pairs.append((quant_stub, dequant_stub))
          
    path_stack = []
    adj_stack = []
    path_list = []
    all_paths = defaultdict(list)
   
    for quant_stub in first_quant_stubs:
      path_stack.append(quant_stub)
      adj_stack.append(quant_stub.out_nodes)
      while path_stack:
        adj_nodes = adj_stack.pop()
        if adj_nodes:
          for adj in adj_nodes:
            if adj not in path_stack:
              path_stack.append(adj)
              adj_stack.append([node for node in adj_nodes if node is not adj])
              adj_stack.append(adj.out_nodes)
              break
            
        else:
          path_stack.pop()
          continue

        if path_stack[-1] in last_dequant_stubs:
          path_list = path_stack.copy()
          all_paths[(quant_stub, path_stack[-1])].append(path_list)
          path_stack.pop()
          adj_stack.pop()

    # path_stub_pair_num = defaultdict(list)
    for _, paths in all_paths.items():
      for path in paths:
        quant_dequant_stack = []
        pair_num = 0
        for node in path:
            if quant_dequant_stack and quant_dequant_stack[-1].kind == "QuantStubF" and node.kind in ["DeQuantStubF", "size"]:
              quant_dequant_stack.pop()
              pair_num += 1
            elif node.kind in ["QuantStubF", "DeQuantStubF"]:
              quant_dequant_stack.append(node)
              
        # path_stub_pair_num[stub_pair].append(pair_num)
      
        if quant_dequant_stack:
          # stubs are not paired
          print(f"####quant/dequant stubs unpaired path\n: {_path_str(path)}")
      
      
      # nums = []
      # if len(set(path_stub_pair_num[stub_pair])) != 1:
      #   print(f"#### num pair of quant/dequant on each path")
      #   for pair_num, path in zip(path_stub_pair_num[stub_pair], all_paths[stub_pair]):
      #     if pair_num not in nums:
      #       nums.append(pair_num)
      #       print(f"{stub_pair[0].name} to {stub_pair[1].name}(pair_num:{pair_num})\n:{_path_str(path)}")
          
          
    
      
            
  

