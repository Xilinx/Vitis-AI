from nndct_shared.base import NNDCT_OP
from nndct_shared.utils import DeviceInfo, DeviceType, NndctScreenLogger, QError
from .dpu_pattern_transform import build_patterns_from_dpu_templates, drop_fix_in_pattern
from .dpu_pattern_handle import DPUPatternHandle
from .dpu_pattern_match import SubgraphMatcher, Match
from .utils import QuantConfig, build_xir_nndct_op_map, log_debug_info


class DPUAllocator(object):
  def __init__(self, target_or_fingerprint) -> None:
      super().__init__()
      if target_or_fingerprint.startswith("0x"):
        self._target = None
        self._fingerprint = target_or_fingerprint
      else:
        self._target = target_or_fingerprint
        self._fingerprint = None
        import target_factory
        is_registered_target = target_factory.is_registered_target(target_or_fingerprint)
        NndctScreenLogger().check2user(QError.UNREGISTERED_TARGET, f"{target_or_fingerprint} is invalid target name.Valid names are: AMD_AIE2_1x4_Overlay=>0x800020500fffffe,AMD_AIE2_4x4_Overlay=>0x80000000020c244,AMD_AIE2_4x5_Overlay=>0x80000000020c23f,AMD_AIE2_Nx4_Overlay=>0x800020500ffffff,DPUCADF8H_ISA0=>0x700000000000000,DPUCAHX8H_ISA2=>0x20200000010002a,DPUCAHX8H_ISA2_DWC=>0x20200000010002b,DPUCAHX8H_ISA2_ELP2=>0x20200000000002e,DPUCAHX8L_ISA0=>0x30000000000001d,DPUCAHX8L_ISA0_SP=>0x30000000000101d,DPUCV2DX8G_ISA0_C16M4B1=>0x900000044000414,DPUCVDX8G_ISA3_C32B1=>0x603000b56011811,DPUCVDX8G_ISA3_C32B3=>0x603000b56011831,DPUCVDX8G_ISA3_C32B3_PSMNET=>0x603000b16026831,DPUCVDX8G_ISA3_C32B6=>0x603000b56011861,DPUCVDX8G_ISA3_C64B1=>0x603000b56011812,DPUCVDX8G_ISA3_C64B3=>0x603000b56011832,DPUCVDX8G_ISA3_C64B5=>0x603000b56011852,DPUCVDX8H_ISA1_F2W2_8PE=>0x501000000140fee,DPUCVDX8H_ISA1_F2W4_4PE=>0x5010000001e082f,DPUCVDX8H_ISA1_F2W4_6PE_aieDWC=>0x501000000160c2f,DPUCVDX8H_ISA1_F2W4_6PE_aieMISC=>0x5010000001e082e,DPUCZDI4G_ISA0_B4096_DEMO_SSD=>0x400002003220206,DPUCZDI4G_ISA0_B8192D8_DEMO_SSD=>0x400002003220207,DPUCZDX8G_ISA1_B1024=>0x101000056010402,DPUCZDX8G_ISA1_B1152=>0x101000056010203,DPUCZDX8G_ISA1_B1600=>0x101000056010404,DPUCZDX8G_ISA1_B2304=>0x101000056010405,DPUCZDX8G_ISA1_B3136=>0x101000056010406,DPUCZDX8G_ISA1_B4096=>0x101000056010407,DPUCZDX8G_ISA1_B512=>0x101000056010200,DPUCZDX8G_ISA1_B800=>0x101000056010201,V70_2x3_Overlay=>0x80000000020c242", is_registered_target)

      self._quant_config = QuantConfig(bw=8, fake_fp=7)
      self._node_device_map = {}
      self._supported_nndct_type, _ = build_xir_nndct_op_map()
      self._node_partition_msg = {}
      self._patterns = build_patterns_from_dpu_templates()
      self._supported_pattern_type = set()
      for pattern in self._patterns:
        for node in pattern.nodes:
          self._supported_pattern_type.update(pattern.get_node_types(node))
          
      self._matched_pattern = []

  def process(self, graph):
    self._assign_pattern_device(graph)
    self._post_process(graph)
  
  def _assign_pattern_device(self, graph):
    def node_match(node_1, node_2):
      return node_1 and node_1.op.type in node_2.get_types() and node_1.name not in self._node_device_map and node_1.in_quant_part is True

    pattern_handler = DPUPatternHandle(self)
    matcher = SubgraphMatcher(graph)
    seen_patterns = []
    for pattern in self._patterns:
      pattern_without_fix = drop_fix_in_pattern(pattern)
      duplicated = False
      for seen_pattern in seen_patterns:
        if pattern_without_fix == seen_pattern:
          duplicated = True
          duplicated_pattern = seen_pattern
          break
      if duplicated is True:
        log_debug_info(f"{pattern_without_fix.name} is duplicated with {duplicated_pattern.name} and will be ignored.")
        continue
      seen_patterns.append(pattern_without_fix)
      matched_subgraphs = matcher.findPatternMatches(pattern_without_fix, node_match)
      for subgraph in matched_subgraphs:
        if pattern.name not in self._matched_pattern:
          self._matched_pattern.append(pattern.name)
        match = Match.create_match(pattern, subgraph)
        subgraph_str = ""
        for node in match.nodeset:
          subgraph_str += f"node name:{node.name}, op type:{node.op.type}, output shape: {node.out_tensors[0].shape}\n"
        NndctScreenLogger().info(f"""Find subgraph for {pattern.name}:
{subgraph_str}
""")
        pattern_handler.process_pattern(match.nodeset, match.quant_node)
        


  def get_node_device(self, node_name):
    return self._node_device_map[node_name]

  def get_node_device_type(self, node_name):
    return self._node_device_map[node_name].get_device_type()

  def get_node_device_msg(self, node_name):
    return self._node_device_map[node_name].get_filter_message()
    
  def get_quant_config(self):
    return self._quant_config

  def set_node_device(self, node_name, device_info):
    self._node_device_map[node_name] = device_info

  def set_node_partition_msg(self, node_name, msg):
    self._node_partition_msg[node_name] = msg
  
  def get_node_partition_msg(self, node_name):
    return self._node_partition_msg.get(node_name, "")

  def is_dpu_node(self, node):
    return node.name in self._node_device_map and self._node_device_map[node.name].get_device_type() == DeviceType.DPU
  
  def _post_process(self, graph):
    for node in graph.nodes:
      if node.name not in self._node_device_map:
        if node.op.type in [NNDCT_OP.INPUT, NNDCT_OP.QUANT_STUB, NNDCT_OP.RETURN]:
          device_info = DeviceInfo(DeviceType.USER)
          self.set_node_device(node.name, device_info) 
        elif not node.in_nodes:
          if node.op.type in self._supported_nndct_type:
            if any([self.is_dpu_node(cn)for cn in graph.children(node)]):
              self.set_node_device(node.name, DeviceInfo(DeviceType.DPU))      
            else:
              device_info = DeviceInfo(DeviceType.CPU)
              device_info.set_filter_message("All the children nodes are assigned to CPU.")
              self.set_node_device(node.name, device_info)
          else:
            device_info = DeviceInfo(DeviceType.CPU)
            device_info.set_filter_message(f"{node.op.type} can't be converted to XIR.")
            self.set_node_device(node.name, device_info)

        elif node.op.type not in self._supported_nndct_type:
          device_info = DeviceInfo(DeviceType.CPU)
          device_info.set_filter_message(f"{node.op.type} can't be converted to XIR.")
          self.set_node_device(node.name, device_info)
        elif node.op.type not in self._supported_pattern_type:
          device_info = DeviceInfo(DeviceType.CPU)
          device_info.set_filter_message(f"{node.op.type} can't be assigned to DPU.")
          self.set_node_device(node.name, device_info)    
        else:
          device_info = DeviceInfo(DeviceType.CPU)
          msg = self.get_node_partition_msg(node.name)
          if not msg:
            msg = f"{node.op.type} can't be assigned to DPU."
          device_info.set_filter_message(msg)
          self.set_node_device(node.name, device_info)

    sorted_nodes = graph.top_sort_nodeset(list(graph.nodes))
    for node in sorted_nodes:
      if node.op.type in [NNDCT_OP.RESHAPE, NNDCT_OP.FLATTEN, NNDCT_OP.SQUEEZE, NNDCT_OP.UNSQUEEZE]:
        pn = graph.parents(node)[0]
        if self.is_dpu_node(pn):
          if (not (all([self.is_dpu_node(cn) for cn in graph.children(node)]))) and pn.out_tensors[0].shape[0] != node.out_tensors[0].shape[0]:
              msg = "First dimension is changed."
              device_info = DeviceInfo(DeviceType.CPU)
              device_info.set_filter_message(msg)
              self.set_node_device(node.name, device_info)
          else:
            self.set_node_device(node.name, DeviceInfo(DeviceType.DPU))
            if node.name not in self._quant_config.get_output_keys():
              self._quant_config.insert_output_quant_fp(node.name)
        else:
          msg = "The input of reshape is not on DPU."
          device_info = DeviceInfo(DeviceType.CPU)
          device_info.set_filter_message(msg)
          self.set_node_device(node.name, device_info)

      elif node.op.type == NNDCT_OP.CONCAT:
        if self.is_dpu_node(node):
          output_tensor_size = node.out_tensors[0].ndim
          axis = node.node_attr(node.op.AttrName.AXIS)
          if all([not self.is_dpu_node(pn) for pn in graph.parents(node)]):
            msg = "All input of concat are on CPU."
            device_info = DeviceInfo(DeviceType.CPU)
            device_info.set_filter_message(msg)
            self.set_node_device(node.name, device_info)
            
            if all([not self.is_dpu_node(cn) for cn in graph.children(node)]):
              self._quant_config.remove_output_fp(node.name)
            for pn in graph.parents(node):
              if  len(pn.out_nodes) == 1:
                self._quant_config.remove_output_fp(pn.name)

          if any([not self.is_dpu_node(pn) for pn in graph.parents(node)]):
            if output_tensor_size != 4 or axis != 3:
              msg = f"One of the input is from non-DPU device. And output is not 4d. And axis == {axis} is not supported."
              device_info = DeviceInfo(DeviceType.CPU)
              device_info.set_filter_message(msg)
              self.set_node_device(node.name, device_info)
              if all([not self.is_dpu_node(cn) for cn in graph.children(node)]):
                self._quant_config.remove_output_fp(node.name)
              for pn in graph.parents(node):
                if (not self.is_dpu_node(pn)) and len(pn.out_nodes) == 1:
                  self._quant_config.remove_output_fp(pn.name) 
  
  @property
  def target(self):
    return self._target

  @property
  def fingerprint(self):
    return self._fingerprint

  def insert_output_quant_fp(self, key, fp):
    self._quant_config.insert_output_quant_fp(key, fp)
  
  def insert_input_quant_fp(self, key, fp):
    self._quant_config.insert_input_quant_fp(key, fp)

  def insert_param_quant_fp(self, key, fp):
    self._quant_config.insert_param_quant_fp(key, fp)

  def get_bw(self):
    return self._quant_config.get_bw()
  

  def has_assigned(self, node_name):
    return True if node_name in self._node_device_map else False





