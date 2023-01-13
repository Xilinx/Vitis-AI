
import copy
from collections import OrderedDict
from nndct_shared.nndct_graph import NndctGraphHolder
from nndct_shared.inspector.device_allocator import DPUAllocator
from nndct_shared.inspector.utils import convert_quant_config_to_dict, log_debug_info
from nndct_shared.utils import DeviceType, NNDCT_OP

 
class TargetQuantInfoMgr(NndctGraphHolder):

  def __init__(self, target, graph, model_type):
    super().__init__(graph, model_type)
    self._quant_groups = None
    self._quant_info = None
    self._device_allocator = DPUAllocator(target)

  def setup_quant_info(self, dev_graph):
    self._device_allocator.process(dev_graph)
    log_debug_info("\nNon DPU node assignment:")
    for node in dev_graph.nodes:
      if self._device_allocator.get_node_device_type(node.name) != DeviceType.DPU:
        log_debug_info(f"{node.name}, {node.op.type}, {self._device_allocator.get_node_device_type(node.name)}, {self._device_allocator.get_node_device_msg(node.name)}")
    quant_config = self._device_allocator.get_quant_config()
    self._quant_info = convert_quant_config_to_dict(quant_config, init=True)

    keys = list(self._quant_info["output"].keys())
    # update key value
    for key in keys:
      key_node = dev_graph.node(key)
      if len(key_node.out_tensors) > 1 and self._device_allocator.get_node_device_type(key_node.name) != DeviceType.DPU:
        value = self._quant_info["output"][key][0]
        new_value = []
        for tensor in dev_graph.node(key).out_tensors:
          if any([self._device_allocator.get_node_device_type(u.user.name) == DeviceType.DPU for u in tensor.uses]):
            new_value.append(copy.deepcopy(value))
          else:
            new_value.append(None)
        self._quant_info["output"][key] = new_value 

    # pop key 
    # import ipdb
    # ipdb.set_trace()
    keys = list(self._quant_info["output"].keys())
    for key in keys:
      if dev_graph.node(key).op.type == NNDCT_OP.CONCAT:
        for tensor in dev_graph.node(key).in_tensors:
          if len(tensor.uses) == 1 and tensor.node is not None and tensor.node.name in self._quant_info["output"].keys():
              if len(tensor.node.out_tensors) > 1:
                idx = tensor.node.out_tensors.index(tensor)
                self._quant_info["output"][tensor.node.name][idx] = None
              else:
                self._quant_info["output"].pop(tensor.node.name)         
    
    keys = list(self._quant_info["output"].keys())
    for key in keys:
      if dev_graph.node(key).op.type in [NNDCT_OP.RESHAPE, NNDCT_OP.PIXEL_SHUFFLE, NNDCT_OP.PIXEL_UNSHUFFLE] and self._device_allocator.get_node_device_type(key) == DeviceType.DPU: 
        pn = dev_graph.parents(key)[0]
        if pn.name in self._quant_info["output"].keys():
          self._quant_info["output"].pop(pn.name)

    keys = list(self._quant_info["output"].keys())
    for key in keys:
      if dev_graph.node(key).op.type == NNDCT_OP.RESHAPE and self._device_allocator.get_node_device_type(key) == DeviceType.CPU:
          if all([self._device_allocator.get_node_device_type(u.user.name) != DeviceType.DPU for u in dev_graph.node(key).in_tensors[0].uses]):
            pn = dev_graph.parents(key)[0]
            if pn.name in self._quant_info["output"].keys():
              self._quant_info["output"].pop(pn.name)
            
          if all([self._device_allocator.get_node_device_type(u.user.name) != DeviceType.DPU for u in dev_graph.node(key).out_tensors[0].uses]):
            self._quant_info["output"].pop(key)

    # replace key
    keys = list(self._quant_info["output"].keys())
    for key in keys:
      if self.Nndctgraph.node(key) is None:
        value = self._quant_info["output"][key][0]
        self._quant_info["output"].pop(key)
        pn = dev_graph.parents(key)[0]
        if pn.name not in keys:
          self._quant_info["output"][pn.name] = [value]

    self._sort_quant_info()
  
  def _sort_quant_info(self):
    sorted_quant_info = {'param': {}, 'output': {}, 'input': {}}
    for node in self.Nndctgraph.nodes:
      for param_type, tensor in node.op.params.items():
        if tensor.name in self._quant_info["param"]:
          sorted_quant_info["param"][tensor.name] = self._quant_info["param"][tensor.name]
      if node.name in self._quant_info["output"]:
        sorted_quant_info["output"][node.name] = self._quant_info["output"][node.name]
      
      if node.name in self._quant_info["input"]:
        sorted_quant_info["input"][node.name] = self._quant_info["input"][node.name]
    
    self._quant_info = sorted_quant_info





  def setup_quant_group(self):
    self._QuantGroups = {}
    assert self._quant_info is not None
    sorted_nodes = self.Nndctgraph.top_sort_nodeset(list(self.Nndctgraph.nodes))
    for node in sorted_nodes:
      if not node.in_quant_part:
        continue
      node_group = self.find_node_group(node)
      self._QuantGroups[node.name] = node_group
      
  def find_node_group(self, node):
    def dfs(n, visited, group):
      visited.add(n.name)
      for cn in n.owning_graph.children(n):
        if cn.name not in visited and cn.name not in group and self._device_allocator.is_dpu_node(cn):
          group.append(cn.name)
          if cn.name not in self._quant_info["output"]:
            dfs(cn, visited, group)

    for k, v in self._QuantGroups.items():
      if node.name in v:
        return v
  
    group = []
    group.append(node.name)
    if not(node.name in self._quant_info["output"]) and self._device_allocator.is_dpu_node(node):
      visited = set()
      dfs(node, visited, group)
    return group

  def is_node_quantizable(self, node, lstm):
    if self._device_allocator.is_dpu_node(node):
      return True
    elif node.op.type in [NNDCT_OP.INPUT, NNDCT_OP.QUANT_STUB]:
      if any([self._device_allocator.is_dpu_node(self.Nndctgraph.node(node_name)) for node_name in node.out_nodes]):
        return True
    return False

  def quant_output(self, node_or_name):
    node = self._find_node(node_or_name)
    if not node.in_quant_part:
      return node
  
    if self.is_node_quantizable(node, False):
      idx = -1
      end_node = self.Nndctgraph.node(self._QuantGroups[node.name][idx])
      return end_node
    else:
     
      def traverse_up_to_find_nearest_quant_output(n):
        if n.name in self.quant_info["output"]:
          return n
        else:
          assert len(node.in_nodes) == 1
          pn = n.owning_graph.parents(n)[0]
          return traverse_up_to_find_nearest_quant_output(pn)
      return traverse_up_to_find_nearest_quant_output(node)

          


   

  def node_output_quantizable(self, node):
    if self.is_node_quantizable(node, False):
      return self.quant_output(node) is node
    return False
      

  @property
  def quant_info(self):
    return self._quant_info
  
  @property
  def quant_groups(self):
    return self._QuantGroups


  @property
  def quant_algo(self):
    return None
 
