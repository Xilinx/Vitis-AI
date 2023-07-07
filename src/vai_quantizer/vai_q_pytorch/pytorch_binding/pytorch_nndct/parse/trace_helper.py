

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

import sys
from collections import ChainMap, OrderedDict, defaultdict
from copy import copy
from typing import Dict
from nndct_shared.nndct_graph import GraphSearcher
from nndct_shared.utils import NndctDebugLogger, NndctOption, PatternType, NndctScreenLogger, GLOBAL_MAP, NNDCT_KEYS
from nndct_shared.utils import QError
from pytorch_nndct.utils.jit_utils import *
from pytorch_nndct.utils.module_util import get_module_name
from .opt_pass import OptPass
from .torch_graph import *

class TorchGraphHandler(object):

  def __init__(self):
    self._cur_graph = None
    self._cur_block = None
    self._extra_inputs_of_node = defaultdict(list)
    self._params = None
    self._block_idx = -1
    # self._is_control_flow_graph = False

  @staticmethod
  def _check_control_flow(fw_graph):
    op_kind = {node.kind() for node in fw_graph.nodes()}
    return any([op in op_kind for op in ["prim::If", "prim::Loop"]])

  def _get_fw_graph_from_module(self, module, input_args, train):
    # 1. Only support trace jit script for torch vesion 1.12 and above
    # 2. If traced from model contain control flow("if"/"loop"), then fall back to trace jit script
    if get_torch_version() >= 1120 and isinstance(module, torch.jit.ScriptModule):
      NndctScreenLogger().check2user(QError.TRACED_NOT_SUPPORT, "The model produced by 'torch.jit.script' is not supported in quantizer", type(module) == torch.jit._trace.TopLevelTracedModule)
      NndctScreenLogger().info(f"The input model {get_module_name(module)} is ScriptModule.")
      fw_graph = self._get_graph_from_script(module, input_args)
      _is_control_flow_graph = True
      #_is_control_flow_graph = self._check_control_flow(fw_graph)
    else:
      NndctScreenLogger().info(f"The input model {get_module_name(module)} is torch.nn.Module.")
      fw_graph = self._trace_graph_from_model(module, input_args, train)
      _is_control_flow_graph = self._check_control_flow(fw_graph)
      if _is_control_flow_graph or NndctOption.nndct_jit_trace.value is True:
        NndctScreenLogger().check2user(QError.TORCH_VERSION, f"The quantizer only support network with control flow for torch version > 1.11.", get_torch_version() >= 1120)
        NndctScreenLogger().info(f"Find the control flow operation in {get_module_name(module)} and retry jit trace to keep it.")
        traced_module = torch.jit.trace(module.eval(), input_args)
        fw_graph = self._get_graph_from_script(traced_module, input_args)

    return fw_graph, _is_control_flow_graph



  def build_torch_graph(self, graph_name, module, input_args, train=False):
    # self._module = module
    NndctScreenLogger().info("Start to trace and freeze model...")
    if NndctOption.nndct_parse_debug.value != 0:
      NndctDebugLogger.write("####Parser Debug Info:\n")
    try:
      fw_graph, is_control_flow_graph = self._get_fw_graph_from_module(module, input_args, train)
    except Exception as e:
      NndctScreenLogger().error2user(QError.PYTORCH_TRACE, f"Failed to get graph from model and input args. The PyTorch internal failed reason is:\n{str(e)}")
      sys.exit(1)
    NndctScreenLogger().info("Finish tracing.")
     
    node_kinds = {node.kind().split(":")[-1] for node in fw_graph.nodes()}
    if NndctOption.nndct_parse_debug.value >= 1:
      NndctDebugLogger.write(f"jit graph:\n{fw_graph}")
      NndctDebugLogger.write(f"\nparsing nodes types:\n{node_kinds}\n")
    
    raw_graph = self._create_raw_graph(graph_name, fw_graph)
    if NndctOption.nndct_parse_debug.value >= 2:
      NndctDebugLogger.write(f"\ntorch raw graph:\n{raw_graph}")   
    self._opt_raw_graph(raw_graph, is_control_flow_graph)
    if NndctOption.nndct_parse_debug.value >= 2:
      NndctDebugLogger.write(f"\ntorch opt graph:\n{raw_graph}")
    
    if NndctOption.nndct_parse_debug.value >= 3:
      self._check_stub_topology(raw_graph)

    return raw_graph


  def _trace_graph_from_model(self, module, input_args, train):
    assert isinstance(module, torch.nn.Module)
    graph, _ = trace_and_get_graph_from_model(module, input_args,
                                                    train)
    graph = optimize_graph(graph)
    params = rename_graph_param_name(module, graph)
    self._params = params
    return graph
  

  def _get_graph_from_script(self, module, input):
    assert isinstance(module, torch.jit.ScriptModule)
    script_graph = module.forward.graph
    torch._C._jit_pass_onnx_function_substitution(script_graph)
    frozen_module = freeze_graph_wo_opt(module.eval(), preserved_attrs=['training'])
    script_graph = frozen_module.graph
    script_graph = optimize_graph(script_graph, is_jit_graph=True)
    rename_graph_inputs(script_graph)
    if NndctOption.nndct_parse_debug.value >= 1:
      NndctDebugLogger.write(f"origin jit graph:\n{script_graph}")
    post_process_script_graph(script_graph)

    self._params = self._get_param_names(script_graph)
    GLOBAL_MAP.set_map(NNDCT_KEYS.TORCH_SCRIPT_MODEL, frozen_module)
      
    return script_graph

  def _get_param_names(self, graph):
    params = [] 
    for _, node in get_fw_op_nodes(graph):
      for block in node_blocks(node):
        block_param = self._get_param_names(block)
        params.extend(block_param)
      if node_type(node) == "prim::Constant":
        output_name = get_node_output_name(node)
        if "self" in output_name and get_node_output_type(node) == "TensorType":
          full_attr = ".".join(output_name.split(".")[1:])
          set_unique_name(node.output(), full_attr)
          params.append(full_attr) 
    return params

  
  def _opt_raw_graph(self, raw_graph, is_control_flow_graph):
    if is_control_flow_graph is True:
      self._opt_raw_block(raw_graph.top_block)
    else:
      self._optimize(raw_graph.top_block)
      #self._optimize_for_jit(raw_graph.top_block)
      #self._optimize(raw_graph.top_block)
  
  def _opt_raw_block(self, raw_block):
    for node in raw_block.nodes:
      if not node.blocks:
        continue
      for block in node.blocks:
        self._opt_raw_block(block)

    self._optimize_for_jit(raw_block)
   
  

  def _optimize_for_jit(self, raw_block):
    graph_searcher = GraphSearcher(raw_block)
    
    # torch 1.8.x will generate redundant type_as op before element-wise add.
    if get_torch_version() >= 180 and get_torch_version() < 190:
      node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["type_as", "add"])])
      OptPass.merge_internal_type_as(raw_block, node_sets)
      
    # classification
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["t", "addmm"]), PatternType(pattern=["t", "matmul"])])
    OptPass.merge_param_transpose_with_addmm(raw_block, node_sets)
    
    
    # nd(>2) linear (JIRA 2646)
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["matmul", "add"]),
                                                     PatternType(pattern=["matmul", "add_"])])
    OptPass.merge_matmul_with_add(raw_block, node_sets)                                                
    
    # LSTM
    while True:
      node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["TupleConstruct", "TupleConstruct"])])
      if not node_sets:
        break
      OptPass.merge_consecutive_tuple(raw_block, node_sets)
    
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["ListConstruct"]),
                                                          PatternType(pattern=["TupleConstruct"])])
    OptPass.pack_ListConstruct_op(raw_block, node_sets)
      
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["embedding_bag"])])
    OptPass.strip_reduantant_tensors_in_embedding_bag(raw_block, node_sets)
    
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["empty", "zero_"])])
    OptPass.merge_empty_with_zero(raw_block, node_sets)
    
    # delete node should be done after merge stride_slice
    # delete reduantant view FADnet. 
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["view"])])
    OptPass.remove_reduantant_view(raw_block, node_sets)

  

  def _optimize(self, raw_block):
    
    graph_searcher = GraphSearcher(raw_block)
    
    # torch 1.8.x will generate redundant type_as op before element-wise add.
    if get_torch_version() >= 180 and get_torch_version() < 190:
      node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["type_as", "add"])])
      OptPass.merge_internal_type_as(raw_block, node_sets)
      
    # classification
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["t", "addmm"]), PatternType(pattern=["t", "matmul"])])
    OptPass.merge_param_transpose_with_addmm(raw_block, node_sets)
    
    # shufflenet
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["ListUnpack"]), 
                                                        PatternType(pattern=["TupleUnpack"])])
    OptPass.unpack_ListUnpack_op(raw_block, node_sets)
        
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["slice"])])
    
    OptPass.slice_to_strided_slice(raw_block, node_sets)
      
    # yolo_v3
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["select", "copy_"])])
    OptPass.select_to_slice_inplace_copy(raw_block, node_sets)
    
    # FADnet
    while True:
      node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["select", "strided_slice"])])
      if not OptPass.merge_select_to_strided_slice(raw_block, node_sets):
        break
    # FADnet
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["strided_slice", "strided_slice"])])
    OptPass.merge_consecutive_strided_slice(raw_block, node_sets)
    
    # 3d pointpillar
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["strided_slice", "index_put_"])])
    OptPass.stride_slice_to_index_inplace_put(raw_block, node_sets)
    
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["strided_slice", "copy_"])])
    OptPass.create_stride_slice_inplace_copy(raw_block, node_sets)
    
    # nd(>2) linear (JIRA 2646)
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["matmul", "add"]),
                                                     PatternType(pattern=["matmul", "add_"])])
    OptPass.merge_matmul_with_add(raw_block, node_sets)                                                
    
    # LSTM
    while True:
      node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["TupleConstruct", "TupleConstruct"])])
      if not node_sets:
        break
      OptPass.merge_consecutive_tuple(raw_block, node_sets)
    
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["ListConstruct"]),
                                                          PatternType(pattern=["TupleConstruct"])])
    OptPass.pack_ListConstruct_op(raw_block, node_sets)
      
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["embedding_bag"])])
    OptPass.strip_reduantant_tensors_in_embedding_bag(raw_block, node_sets)
    
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["empty", "zero_"])])
    OptPass.merge_empty_with_zero(raw_block, node_sets)
    
    # delete node should be done after merge stride_slice
    # delete reduantant view FADnet. 
    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["view"])])
    OptPass.remove_reduantant_view(raw_block, node_sets)

    node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["add"]), PatternType(pattern=["sub"]), PatternType(pattern=["mul"]), PatternType(pattern=["div"]), PatternType(pattern=["rsub"])])
    OptPass.transform_const_scalar_to_const_tensor(raw_block, node_sets)

    # clamp
    # node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["clamp"])])
    # OptPass.transform_const_scalar_to_const_tensor_clamp(raw_block, node_sets)

    # node_sets = graph_searcher.find_nodes_from_type([PatternType(pattern=["ScalarImplicit"])])
    # OptPass.remove_tensor2scalar(raw_block, node_sets)
    
  
  def _create_attrs_value(self, graph):
    assert self._cur_graph
    assert self._cur_block
    extra_count = 0
    exclude_attributes = ["inplace", "module"]
    for name, fw_node in get_fw_op_nodes(graph):
      if node_type(fw_node) != "prim::Constant":
        for attr_name in fw_node.attributeNames():
          if attr_name in exclude_attributes:
            continue
          value = get_attr_value(fw_node, attr_name)
          torch_value = TorchValue(value, name=f"extra_{extra_count}")
          self._extra_inputs_of_node[fw_node].append(torch_value)
          self._cur_graph.add_blob_value(torch_value)
          extra_count += 1
          
        if node_type(fw_node) == 'prim::PythonOp':
          for value in fw_node.scalar_args():
            torch_value = TorchValue(value, name=f"extra_{extra_count}")
            self._extra_inputs_of_node[fw_node].append(torch_value)
            self._cur_graph.add_blob_value(torch_value)
            extra_count += 1
          # ip = fw_graph.insertConstant(value)
          # fw_node.addInput(ip)
      elif name not in self._params:
        const_value = TorchValue(list(fw_node.outputs())[0])
        if const_value.is_plain_value() or const_value.is_none():
          self._cur_graph.add_blob_value(const_value)
        else:
          const_node = TorchNode(fw_node)
          const_node.owning_graph = self._cur_graph
          # const_value.node = const_node
          const_node.add_output(const_value)
          self._cur_block.append_node(const_node)

  def _create_inputs_value(self, graph):
    for ip in get_fw_graph_inputs(graph):
      if "self" == unique_name(ip).split(".")[0]:
        continue
      if unique_name(ip) not in self._params:
        input_node = TorchNode(ip.node())
        input_node.owning_graph = self._cur_graph
        value = TorchValue(ip)
        input_node.add_output(value)
        self._cur_block.append_node(input_node)
      else:
        value = TorchValue(ip)
        self._cur_graph.add_param_value(value)  
    
    for name, fw_node in get_fw_op_nodes(graph):
      if node_type(fw_node) == "prim::Constant":
        if name in self._params:
          value = TorchValue(fw_node.output())
          self._cur_graph.add_param_value(value)  
   
    getattr_nodes = graph.findAllNodes("prim::GetAttr", recurse=True)
    if getattr_nodes:
      node_visited = set()
      state_dict = self._module.state_dict()
      for node in getattr_nodes:
        if get_node_output_name(node) in node_visited or get_node_output_name(node) in self._params:
          continue
        # print("node_name:",get_node_output_name(node))
        for getattrs in get_attr_chains(node):
          node_visited.update(map(get_node_output_name, getattrs))
          full_attr = getattr_full_name(getattrs)  # self.conv.weight -> conv.weight
          # print("attr_name:", full_attr)
          if full_attr in state_dict and full_attr not in self._params:
            # print(f"set {unique_name(getattrs[-1].output())} => {full_attr}")
            set_unique_name(getattrs[-1].output(), full_attr)
            torch_tensor = state_dict[full_attr]
            value = TorchValue(getattrs[-1].output())
            value.dtype = {torch.float: 'torch.float', 
                            torch.float64: 'torch.double'}.get(torch_tensor.dtype, torch_tensor.dtype)
            assert value.dtype
            value.shape = list(torch_tensor.size())
            self._cur_graph.add_param_value(value) 
            self._params.append(value.name)
            


  def _build_block_graph(self, fw_block):
    block_idx = self._block_idx
    
    self._create_attrs_value(fw_block)
    self._create_inputs_value(fw_block)  

    for _, fw_node in get_fw_op_nodes(fw_block):
      if node_type(fw_node) == "prim::Constant" and self._is_param_const_node(fw_node):
        continue
      raw_node = self.add_torch_node(fw_node)
      if raw_node is None:
        continue
      # raw_node.owning_graph = self._cur_graph
      # self._cur_block.append_node(raw_node)
      for sub_fw_block in fw_node.blocks():
        saved_block = self._cur_block
        self._cur_block = TorchBlock(self._cur_graph, raw_node)
        self._block_idx += 1
        self._build_block_graph(sub_fw_block)
        raw_node.add_block(self._cur_block)
        self._cur_block = saved_block

    fw_block_return = get_fw_graph_ret_node(fw_block)
    return_node = TorchNode(fw_block_return)
    return_node.owning_graph = self._cur_graph
    return_node.name = "_".join(["return", str(block_idx)])
    for ip in fw_block_return.inputs():
      ip_value = self._cur_graph.get_blob_value_by_name(unique_name(ip)) \
      if unique_name(ip) in self._cur_graph.blobs_name() else \
      self._cur_graph.get_param_value_by_name(unique_name(ip))
      
      return_node.add_input(ip_value)
    
    self._cur_block.append_node(return_node)
    
  def _create_raw_graph(self, graph_name, fw_graph):
    raw_graph = TorchGraph(graph_name)
    self._cur_graph = raw_graph
    top_block = TorchBlock(raw_graph, None)
    raw_graph.set_top_block(top_block)
    self._cur_block = top_block
    self._block_idx += 1
    self._build_block_graph(fw_graph)
    raw_graph.connect_nodes()
    return raw_graph
  

  def _is_param_const_node(self, fw_node):
    if get_node_output_name(fw_node) in self._params:
      return True
    else:
      return False
 
    
 

  def add_torch_node(self, fw_node):
    
    schema = get_node_schema(fw_node)
    # assert schema
    node = TorchNode(fw_node)
   
    node.schema = schema
    for ip in fw_node.inputs():
      ip_value = self._cur_graph.get_blob_value_by_name(unique_name(ip)) \
      if unique_name(ip) in self._cur_graph.blobs_name() else \
      self._cur_graph.get_param_value_by_name(unique_name(ip))
      
      node.add_input(ip_value)
      
    for extra_input in self._extra_inputs_of_node[fw_node]:
      node.add_input(extra_input)
    
    
    if node.inputs or should_construct_dynamic_list(fw_node):
      node.owning_graph = self._cur_graph      
      self._cur_block.append_node(node)
      for op in fw_node.outputs():
        value = TorchValue(op)
        node.add_output(value)
    
      return node
    else:
      return None
    

  
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
          
          
    
      
            
  

