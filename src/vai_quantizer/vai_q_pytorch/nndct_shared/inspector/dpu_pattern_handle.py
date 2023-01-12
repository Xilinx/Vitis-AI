import os
import pwd
import random
import string

from nndct_shared.base import NNDCT_OP
from nndct_shared.compile import CompilerFactory
from nndct_shared.compile.xir_helper import XIRHelper
from nndct_shared.nndct_graph import GraphBase
from nndct_shared.nndct_graph import operator_definition as base_op
from nndct_shared.utils import (DataXopError, DeviceInfo, DeviceType,
                                NndctScreenLogger, QError, QNote, QWarning,
                                create_work_dir, NndctOption)

from .utils import QuantConfig, convert_quant_config_to_dict


class PatternGraph(GraphBase):
  def __init__(self, graph_name):
    self._name = graph_name
    self._nodes = []
    self._nodes_by_name = {}
  
  @classmethod
  def create_pattern_graph_from_nodeset(cls, name, nodeset):
    pattern = cls(name)
    for node in nodeset:
      pattern.add_node(node)
    return pattern
    
  def parents(self, node):
    if isinstance(node, str):
      node = self.node(node)
    return [self.node(node_name) for node_name in node.in_nodes]

  def children(self, node):
    if isinstance(node, str):
      node = self.node(node)
    return [self.node(node_name) for node_name in node.out_nodes]

  def add_node(self, node):
    if node not in self._nodes:
      self._nodes.append(node)
      self._nodes_by_name[node.name] = node

  def node(self, name):
    return self._nodes_by_name.get(name, None)

  @property
  def nodes(self):
    return self._nodes

  @property
  def op_types(self):
    return {node.op.type for node in self.nodes}


  @property
  def name(self):
    return self._name

  def get_topological_graph_nodes_list(self):
    return [node for node in self.nodes]


class DPUPatternHandle(object):
  def __init__(self, device_allocator):
    self._pn_op_map = {}
    self._default_fp =  device_allocator.get_quant_config().get_fake_fp()
    self._bw = device_allocator.get_quant_config().get_bw()
    self._device_alloc = device_allocator

  def replace_pn_op_of_nodes_with_input_type(self, nodes):
    owning_graph = nodes[0].owning_graph
    for node in nodes:
      for i, pn in enumerate(owning_graph.parents(node)):
        if len(pn.in_nodes) == 0 and pn.op.type in self._device_alloc._supported_nndct_type:
          continue
        self._pn_op_map[pn.name] = pn.op
        op = base_op.UnaryOp(NNDCT_OP.INPUT)
        op.set_attr(op.AttrName.INPUT, f"placeholder_{i}")
        pn.op = op

  def recover_pn_op(self, nodes):
    owning_graph = nodes[0].owning_graph
    for node in nodes:
      for pn in owning_graph.parents(node):
        if pn.name in self._pn_op_map:
          pn.op = self._pn_op_map[pn.name]
    
  def update_quant_config(self, pattern_quant_config):
    for key in pattern_quant_config.get_param_keys():
      if key not in self._device_alloc.get_quant_config().get_param_keys():
        self._device_alloc.insert_param_quant_fp(key, pattern_quant_config.get_param_bw_fp(key)[-1])
    
    for key in pattern_quant_config.get_input_keys():
      if key not in self._device_alloc.get_quant_config().get_input_keys():
        self._device_alloc.insert_input_quant_fp(key, pattern_quant_config.get_input_bw_fp(key)[-1])

    for key in pattern_quant_config.get_output_keys():
      if key not in self._device_alloc.get_quant_config().get_output_keys():
        self._device_alloc.insert_output_quant_fp(key, pattern_quant_config.get_output_bw_fp(key)[-1])

  def _init_quant_config(self, nodeset, quant_node):
    quant_config = QuantConfig(self._bw, self._default_fp)
    for node in nodeset:
      for _, param_tensor in node.op.params.items():
        quant_config.insert_param_quant_fp(param_tensor.name)

    for node in quant_node:
      quant_config.insert_output_quant_fp(node.name)

    return quant_config
  
  @staticmethod
  def build_pattern(nodeset, pattern):
    owning_graph = nodeset[0].owning_graph
    for n in nodeset:
      for pn in owning_graph.parents(n):
        if pn.op.type == NNDCT_OP.INPUT:
          pattern.add_node(pn)
   
    for node in nodeset:
      pattern.add_node(node)

  def assign_target_device(self, nodeset, compiled_xpattern):
    for node in nodeset:
      xops = XIRHelper.find_xops_from_nndct_node(node, compiled_xpattern)
      if len(xops) > 0:
        if all([XIRHelper.get_xop_device_type(xop) == XIRHelper.get_xop_device_type(xops[0]) for xop in xops]):
          subgraph_type = XIRHelper.get_xop_device_type(xops[0])
          target_device = DeviceInfo(DeviceType.DPU) if subgraph_type == "DPU" else DeviceInfo(DeviceType.CPU)
        else:
          target_device = DeviceInfo(DeviceType.CPU)
        # node.target_device = target_device
        self._device_alloc.set_node_device(node.name, target_device)
      else:
        device_type = self._device_alloc.get_node_device(nodeset[0].name).get_device_type()
        NndctScreenLogger().warning2user(QWarning.INSPECTOR_PATTERN, f"{node.op.type}: {node.name} may be fused by compiler and will be assigned to {device_type}.")
        target_device = DeviceInfo(device_type) 
        # node.target_device = target_device
        self._device_alloc.set_node_device(node.name, target_device)

  def assign_dpu_device(self, nodeset):
    for node in nodeset:
      target_device = DeviceInfo(DeviceType.DPU) 
      self._device_alloc.set_node_device(node.name, target_device)

  def assign_partition_msg(self, partition_msg, nodeset):
    for node in nodeset:
      self._device_alloc.set_node_partition_msg(node.name, partition_msg)

  def process_nndct_pattern(self, nodeset):
    expanded_heads, _, expanded_nodeset = self.expand_nodeset(nodeset)
    in_node = [nodeset[0]]
    # self.replace_pn_op_of_nodes_with_input_type(expanded_heads)
    self.replace_pn_op_of_nodes_with_input_type(in_node)
    random_suffix = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    print(f"Find pattern:")
    for node in nodeset:
      print(node.name)
    patter_name_str = "_".join([node.op.type for node in nodeset])
    pattern = PatternGraph(f"{patter_name_str}_{random_suffix}")
    # self.build_pattern(expanded_nodeset, pattern)
    self.build_pattern(nodeset, pattern)
    quant_config = self.init_quant_config(nodeset, expanded_heads)
    xir_convert = CompilerFactory.get_compiler("xmodel")
    xmodel_file_name = os.path.join("/tmp", pattern.name)
    fake_quant_config = convert_quant_config_to_dict(quant_config)
    xgraph = xir_convert.do_compile(pattern,  output_file_name=xmodel_file_name, quant_config_info=fake_quant_config)
    xcompiler = CompilerFactory.get_compiler("xcompiler")
    compiled_xpattern = xcompiler.compile_and_reload(xmodel_file_name, self._device_alloc.target)
    self.assign_target_device(nodeset, compiled_xpattern)
    if self._device_alloc.is_dpu_node(nodeset[0]):
      self.update_quant_config(quant_config)
    # self.recover_pn_op(expanded_heads)
    self.recover_pn_op(in_node)

  def process_pattern(self, nodeset, quant_node):
    expanded_heads = self.expanded_heads(nodeset)
    quant_node = expanded_heads + quant_node
    in_nodes = self._find_input_nodes_for_nodeset(nodeset)
    self.replace_pn_op_of_nodes_with_input_type(in_nodes)
    random_suffix = ''.join(random.sample(string.ascii_letters + string.digits, 16))
    backup_shapes = {}
    if not any([node.op.type in [NNDCT_OP.RESHAPE, NNDCT_OP.CONCAT] for node in nodeset]):
      for node in expanded_heads:
        shape = node.out_tensors[0].shape
        if shape and shape[0] > 1:
          backup_shapes[node] = list(shape)
          shape[0] = 1
          node.out_tensors[0].shape = shape
          NndctScreenLogger().warning2user(QWarning.INSPECTOR_PATTERN, f"The First dimension of pattern data node {node.name}'s  shape is {backup_shapes[node][0]} > 1 which will be set to 1 temporarily for pattern matching.")
    
    patter_name_str = "_".join([node.op.type for node in nodeset])
    pattern = PatternGraph.create_pattern_graph_from_nodeset(f"{patter_name_str}_{random_suffix}", expanded_heads + nodeset)
    quant_config = self._init_quant_config(nodeset, quant_node)
    xir_convert = CompilerFactory.get_compiler("xmodel")
    user_name = pwd.getpwuid(os.getuid()).pw_name
    xmodel_dir = os.path.join("/tmp", user_name)
    create_work_dir(xmodel_dir)
    xmodel_file_name = os.path.join(xmodel_dir, pattern.name)
    fake_quant_config = convert_quant_config_to_dict(quant_config)
    convert_xgraph_failed_msg = ""
    output_file = xmodel_file_name if NndctOption.nndct_inspect_debug.value else ""
    try:
      xgraph = xir_convert.do_compile(pattern, output_file_name=output_file, quant_config_info=fake_quant_config)
    except Exception as e:
      convert_xgraph_failed_msg = f"Convert nndct graph to XIR failed.({str(e)})"
      self.assign_partition_msg(convert_xgraph_failed_msg, nodeset)
    
    if not convert_xgraph_failed_msg:
      xcompiler = CompilerFactory.get_compiler("xcompiler")
      #compiled_xpattern = xcompiler.compile_and_reload(xmodel_file_name, self._device_alloc.target)
      compiled_xpattern = xcompiler.compile_xgraph(xmodel_file_name, xgraph, self._device_alloc.target, self._device_alloc.fingerprint)
      
      if XIRHelper.is_valid_compiled_pattern(compiled_xpattern) and XIRHelper.is_dpu_pattern(compiled_xpattern):
          self.assign_dpu_device(nodeset)
          self.update_quant_config(quant_config)
      else:
        pattern_partition_msg = XIRHelper.get_pattern_partition_msg(compiled_xpattern)
        if not pattern_partition_msg:
          pattern_partition_msg = f"Try to assign {pattern.name} to DPU failed."      
      # pattern_partition_msg = XIRHelper.get_pattern_partition_msg(compiled_xpattern)
        self.assign_partition_msg(pattern_partition_msg, nodeset)
    
    if backup_shapes:
      for node, shape in backup_shapes.items():
        node.out_tensors[0].shape = shape

    self.recover_pn_op(in_nodes)

  def expanded_heads(self, nodeset):
    expanded_heads = []
    graph = nodeset[0].owning_graph
    input_nodes = self._find_input_nodes_for_nodeset(nodeset)
    for node in input_nodes:
      for pn in graph.parents(node):
        expanded_heads.append(pn)
    return expanded_heads

  def expand_nodeset(self, nodeset):
    expanded_nodeset = []
    expanded_heads = []
    expanded_tails = []
    graph = nodeset[0].owning_graph
    for pn in graph.parents(nodeset[0]):
      expanded_heads.append(pn)

    for cn in graph.children(nodeset[-1]):
      if len(cn.in_nodes) == 1: 
        expanded_tails.append(cn)
    
    expanded_nodeset = expanded_heads + nodeset + expanded_tails
    return expanded_heads, expanded_tails, expanded_nodeset
  
  def _find_input_nodes_for_nodeset(self, nodeset):
    input_node = []
    graph = nodeset[0].owning_graph
    for node in nodeset:
      if all([graph.node(in_node) not in nodeset for in_node in node.in_nodes]):
        input_node.append(node)
    return input_node



class OpWithActHandle(DPUPatternHandle):
  def __init__(self, device_allocator):
    super().__init__(device_allocator)

  def init_quant_config(self, nodeset, heads):
    quant_config = QuantConfig(self._bw, self._default_fp)
    # quant_config = {"param": {}, "output": {}, "input": {}}
    for node in nodeset:
      for _, param_tensor in node.op.params.items():
        quant_config.insert_param_quant_fp(param_tensor.name)
    
    for node in heads:
      quant_config.insert_output_quant_fp(node.name)

    quant_config.insert_output_quant_fp(nodeset[-1].name)
    # quant_config["output"][nodeset[-1].name] = self.fake_quant_info

    return quant_config

  def __call__(self, *args, **kwargs):
    # import ipdb
    # ipdb.set_trace()
    _, nodeset = args
    node_0 = nodeset[0]
    node_1 = nodeset[1]

    if self._device_alloc.has_assigned(node_0.name) or len(node_0.out_nodes) != 1:
      return 

    self.process_nndct_pattern(nodeset)


class SingleOpHandle(DPUPatternHandle):
  def __init__(self, device_allocator):
    super().__init__(device_allocator)

 
  def init_quant_config(self, nodeset, heads):
    quant_config = QuantConfig(self._bw, self._default_fp)
    # quant_config = {"param": {}, "output": {}, "input": {}}
    for node in nodeset:
      for _, param_tensor in node.op.params.items():
        quant_config.insert_param_quant_fp(param_tensor.name)
      quant_config.insert_output_quant_fp(node.name)

    for node in heads:
      quant_config.insert_output_quant_fp(node.name)

    return quant_config

  def __call__(self, *args, **kwargs):
    # import ipdb
    # ipdb.set_trace()
    _, nodeset = args
    node = nodeset[0]
    if self._device_alloc.has_assigned(node.name):
      return 

    self.process_nndct_pattern(nodeset)


class CombOpHandle(SingleOpHandle):
  def __init__(self, device_allocator, fake_quant_fp=None):
    super().__init__(device_allocator)
    self._default_fp = fake_quant_fp if fake_quant_fp is not None else self._default_fp


  def __call__(self, *args, **kwargs):
    # import ipdb
    # ipdb.set_trace()
    _, nodeset = args
    # node = nodeset[0]
    if any([self._device_alloc.has_assigned(node.name) for node in nodeset]) or any([len(node.out_nodes) > 1 for node in nodeset[:-1]]):
      return 

    self.process_nndct_pattern(nodeset)
