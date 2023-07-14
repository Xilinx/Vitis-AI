# Copyright 2022 Xilinx Inc.
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

import os
import copy
# import pandas as pd
import torch
from collections import defaultdict
from tabulate import tabulate
from nndct_shared.compile import DevGraphOptimizer
from nndct_shared.base import NNDCT_OP
from nndct_shared.nndct_graph import Graph
from nndct_shared.utils import (NndctScreenLogger, create_work_dir,
                                set_option_value)
from pytorch_nndct.qproc.utils import (connect_module_with_graph,
                                       prepare_quantizable_module,
                                       register_output_hook,
                                       set_outputs_recorder_status,
                                       update_nndct_blob_data)
from pytorch_nndct.utils.module_util import to_device
from .dpu_partition import DPUPartition
from .device import DeviceType
from .target_helper import DPUTargetHelper
from vai_utf.python.target_factory import VAI_UTF as utf


class InspectContext(object):
  def __enter__(self):
    set_option_value("nndct_quant_off", True)
  def __exit__(self, *args):
    set_option_value("nndct_quant_off", False)

class InspectorImpl(object):
  def __init__(self, target):
    self._target = target
    self._partition = DPUPartition(self._target)
    self._node_msgs = defaultdict(set)
    self._module_root = ""
    self._graph = None
    
    NndctScreenLogger().info(f"=>Inspector is initialized successfully with target:")
    dpu_target = self._target.get_devices()[0].get_legacy_dpu_target()
    print(DPUTargetHelper.get_basic_info(dpu_target))

  @classmethod
  def create_by_DPU_arch_name(cls, name):
    target = utf().create_legacy_dpu_target(name, file_type="name")
    return cls(target)

  @classmethod
  def create_by_DPU_fingerprint(cls, fingerprint):
    target = utf().create_legacy_dpu_target(fingerprint, file_type="fingerprint")
    return cls(target)
  
  @classmethod
  def create_by_DPU_arch_json(cls, arch_json):
    target = utf().create_legacy_dpu_target(arch_json, file_type="json")
    return cls(target)


  def inspect(self, module, input_args, device, output_dir, verbose_level):
    with InspectContext():
      if isinstance(module, torch.nn.DataParallel):
        module = module.module
      copied_model = copy.deepcopy(module)
      copied_model.eval()
      # dpu_target = self._target.get_devices()[0].get_legacy_dpu_target()
      # print("Target Basic Inforamtion:")
      # print(DPUTargetHelper.get_basic_info(dpu_target))
      self._module_root = module.__class__.__module__.replace(".", "/")
      dev_graph = self._prepare_deployable_graph(copied_model, input_args, device, output_dir)
      self._partition.simple_allocate_op_device(dev_graph) 
      self._attach_extra_node_msg(dev_graph)
      self._show_partition_result_on_screen(dev_graph, output_dir, verbose_level)
      self._dump_txt(dev_graph, output_dir)
      self._graph = dev_graph

  def _prepare_deployable_graph(self, module, input_args, device, output_dir):
    module, input_args = to_device(module, input_args, device)
    quant_module, graph = prepare_quantizable_module(
        module=module,
        input_args=input_args,
        export_folder=output_dir,
        device=device)
    set_option_value("nndct_quant_off", True)
    register_output_hook(quant_module, record_once=True)
    set_outputs_recorder_status(quant_module, True)
    if isinstance(input_args, tuple):
      _ = quant_module.to(device)(*input_args)
    else:
      _ = quant_module.to(device)(input_args)
    g_optmizer = DevGraphOptimizer(graph)
    connect_module_with_graph(quant_module, g_optmizer.dev_graph, recover_param=False)
    update_nndct_blob_data(quant_module, g_optmizer.dev_graph)
    g_optmizer.strip_redundant_ops()
    g_optmizer.update_op_attrs()
    g_optmizer.constant_folding()
    g_optmizer.layout_tranform()    
    connect_module_with_graph(quant_module, g_optmizer.dev_graph, recover_param=False)
    update_nndct_blob_data(quant_module, g_optmizer.dev_graph)
    connect_module_with_graph(quant_module, graph, recover_param=False)
    return g_optmizer.dev_graph

  def _show_partition_result_on_screen(self, graph, output_dir, verbose_level):
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    # pd.set_option("max_colwidth", 100)
    # pd.set_option("display.width", 5000)
    target_name = DPUTargetHelper.get_name(self._target.get_devices()[0].get_legacy_dpu_target())
    if verbose_level == 0:
      return
    elif verbose_level == 1:
      d = []
      for node in graph.nodes:
        if node.op.type in [NNDCT_OP.RETURN, NNDCT_OP.INPUT]:
          continue
        if node.target_device is not None:
          if node.target_device.get_device_type() == DeviceType.CPU:
            d.append([node.name, node.op.type, node.target_device.get_filter_message()])
      if d:
        # df = pd.DataFrame(d, columns=["Node Name", "Op Type", "Hardware Constraints"])
        NndctScreenLogger().info(f"The operators assigned to the CPU are as follows(see more details in '{os.path.join(output_dir, f'inspect_{target_name}.txt')}'):")
        # print(df)
        print(tabulate(d, headers=["node name", "op Type", "hardware constraints"]))
      else:
        NndctScreenLogger().info(f"All the operators are assigned to the DPU(see more details in '{os.path.join(output_dir, f'inspect_{target_name}.txt')}')")
    
    elif verbose_level == 2:
      d = []
      for node in graph.nodes:
        if node.op.type in [NNDCT_OP.RETURN, NNDCT_OP.INPUT]:
          continue
        if node.target_device is not None:
          d.append([node.name, node.op.type, node.target_device.get_device_type().value])
      # df = pd.DataFrame(d, columns=["Node_Name", "Op_Type", "Assgined_Device"]) 
      NndctScreenLogger().info(f"Operator device allocation table(see more details in '{os.path.join(output_dir, 'inspect.txt')}'):")
      # print(df)
      print(tabulate(d, headers=["node name", "op type", "assgined device"]))

  def _dump_txt(self, graph, output_dir):
    target_name = DPUTargetHelper.get_name(self._target.get_devices()[0].get_legacy_dpu_target())
    file_name = os.path.join(output_dir, f"inspect_{target_name}.txt")
    with open(file_name, 'w') as f:
      self._dump_comment(f)
      self._dump_body(f, graph)

  def _get_subgraphs_and_output_boundaries(self, device_node_set, device_type):
    def get_set_id_from_nodeset(nodeset):
      return min([node.set_id for node in nodeset])

    def collect_node_set(node, visited):
      if not hasattr(node, "set_id"):
        node.set_id = set_id
      
      id2nodes[set_id].add(node)
      visited.add(node)
      
      for cn in node.owning_graph.children(node):
        if cn not in visited and cn in device_node_set:
          collect_node_set(cn, visited)  
    
    id2nodes = defaultdict(set)
    set_id = 0
    for node in device_node_set:
      if not node.in_nodes or all([pn.target_device and pn.target_device.get_device_type() !=
          device_type for pn in node.owning_graph.parents(node)]):
        visited = set()
        collect_node_set(node, visited)
        set_id += 1

    merged_id2nodes = defaultdict(set)
    for _, nodeset in id2nodes.items():
      set_id = get_set_id_from_nodeset(nodeset)
      merged_id2nodes[set_id].update(nodeset)
    
    for set_id, node_set in merged_id2nodes.items():
      for node in node_set:
        node.set_id = set_id

    boundary_nodes = []
    for node in device_node_set:
      for cn in node.owning_graph.children(node):
        if cn.op.type == NNDCT_OP.RETURN:
          continue
        if cn not in node_set or node.set_id != cn.set_id:
          boundary_nodes.append(node)
   
    subgraph_node_sets = []
    for _, nodes in merged_id2nodes.items():
      subgraph_node_sets.append(nodes)
    return subgraph_node_sets, boundary_nodes

  @staticmethod
  def _normalize(name):
    return name.split("::")[-1]

  def _create_cluster_graph(self, dot_graph, device_type, node_sets):
    subgraph_attr = {
      "dpu": {"color": "blue"},
      "cpu": {"color": "red"}
    }
    device_type_str = device_type.value.lower()
    assert device_type_str in subgraph_attr
    for i, node_set in enumerate(node_sets):
      name = f"{device_type_str}_subgraph_{i}"
      with dot_graph.subgraph(name=f"cluster_{device_type_str}_{i}") as subgraph:
        # subgraph.attr(color='blue')
        subgraph.attr(**subgraph_attr[device_type_str])
        subgraph.attr(label=name)
        for node in node_set:
            subgraph.node(self._normalize(node.name),
            label=f"type:{node.op.type}\nname:{node.name}\nout shape:{[out.shape for out in node.out_tensors]}")
        for node in node_set:
          for on in node.out_nodes:
            if node.owning_graph.node(on) in node_set:
              subgraph.edge(self._normalize(node.name), self._normalize(on))
  
  def _create_dot_graph(self, output_dir, device_type_subgraph_node_sets_map, boundaries):
    from graphviz import Digraph
    g = Digraph(self._graph.name)
    for device_type, subgraph_node_sets in device_type_subgraph_node_sets_map.items():
      self._create_cluster_graph(g, device_type, subgraph_node_sets)
    for b_node in boundaries:
      for on in b_node.out_nodes:
          g.edge(self._normalize(b_node.name), self._normalize(on))
    return g
  

  def _create_dot_graph_v2(self):
    from graphviz import Digraph
    g = Digraph(self._graph.name)
    device_node_attrs = {
      DeviceType.DPU: {"color": "blue"},
      DeviceType.CPU: {"color": "red"}
    }
    for node in self._graph.nodes:
      if node.op.type == NNDCT_OP.RETURN:
        continue
      g.node(self._normalize(node.name),
      label=f"type:{node.op.type}\nname:{node.name}\nassigned device:{node.target_device.get_device_type().value}\nout shape:{[out.shape for out in node.out_tensors]}",
      **device_node_attrs[node.target_device.get_device_type()])
    
    for node in self._graph.nodes:
      for on in node.out_nodes:
        if self._graph.node(on).op.type == NNDCT_OP.RETURN:
          continue
        g.edge(self._normalize(node.name), self._normalize(on))
    return g

  def export_dot_image_v2(self, output_dir, format):
    assert self._graph is not None
    target_name = DPUTargetHelper.get_name(self._target.get_devices()[0].get_legacy_dpu_target())
    file_name = os.path.join(output_dir, ".".join([f"inspect_{target_name}", format]))
    dot_graph = self._create_dot_graph_v2()
    dot_graph.render(outfile=file_name).replace('\\', '/')
    NndctScreenLogger().info(f"Dot image is generated.({file_name})")

  def export_dot_image(self, output_dir, format):
    assert self._graph is not None
    file_name = os.path.join(output_dir, ".".join(["inspect", format]))
    device_type_node_sets = defaultdict(list)
    for node in self._graph.nodes:    
      if node.op.type == NNDCT_OP.RETURN:
        continue
      if node.target_device:
        device_type_node_sets[node.target_device.get_device_type()].append(node)
      else:
        raise RuntimeError(f"{node}({node.op.type}) has no target device.")

    device_type_subgraph_node_sets = defaultdict(list)
    boundaries = []
    for device_type, node_set in device_type_node_sets.items():
      subgraph_node_sets, sub_boundaries = self._get_subgraphs_and_output_boundaries(node_set, device_type)
      device_type_subgraph_node_sets[device_type] = subgraph_node_sets
      boundaries += sub_boundaries

    dot_graph = self._create_dot_graph(output_dir, device_type_subgraph_node_sets, boundaries)
  
    dot_graph.render(outfile=file_name).replace('\\', '/')
    NndctScreenLogger().info(f"Dot image is generated.({file_name})")

  def _dump_comment(self, f):
    f.write("# The 'inspect.txt' file is used to show all the details of each operation in NN model.\n")
    f.write("# Field Description:\n")
    f.write("# target info: target device information.\n")
    f.write('# inspection summary: summary report of inspection')
    f.write("# graph name: The name of graph representing of the NN model.\n")
    f.write("# node name: The name of node in graph.\n")
    f.write("# input nodes: The parents of the node.\n")
    f.write("# output nodes: The children of node.\n")
    f.write("# op type: The type of operation.\n")
    f.write("# output shape: The shape of node output tensor(Data layout follows XIR requirements).\n")
    f.write("# op attributes: The attributes of operation.(The description is consistent with that of XIR)\n")
    f.write("# assigend device: The device type on which the operation execute.\n")
    f.write("# hardware constrains: If the operation is assigned to cpu. This filed will give some hits about why the DPU does not support this operation.\n")
    f.write("# node messages: This filed will give some extra information about the node.(For example, if quantizer need to insert a permute operation to convert data layout from 'NCHW' to 'NHWC' or from 'NCHW' to 'NHWC' for deployment. This message will be add to node_messages.)\n")
    # f.write("# scope: The scope of operation.(For example, if we defined a self.conv1 = nn.Conv2d(...) in MyModule, the scope of this oepration should look like 'MyModule/Conv2d[conv1]')\n")
    f.write("# source range: points to a source which is a stack track and helps to find the exact location of this operation in source code.\n\n")
    f.write("# Hints:\n")
    f.write("# Due to data layout difference between Pytorch('NCHW') and XIR('NHWC'), \n# if quantizer inserts some permutes(which the node message will inform us about),\n# these permutes may prevent the entire model from being deployed to the target device.\n# Sometimes, we can cancel out this automatically inserted permute by inserting a permute in the original float model,\n# sometimes, we can't.\n")
    f.write('# These two examples are used to demonstrated this problem:\n')
    f.write("# Example 1:\n")
    f.write("# Pytorch: conv:[1, 64, 1, 1] -> reshape(shape=(1, -1):[1, 64] =>\n")
    f.write("# Xmodel: conv:[1, 1, 1, 64] -> permute(order=(0, 3, 1, 2)):[1, 64, 1, 1] -> reshape(shape=(1, -1):[1, 64]\n")
    f.write("# Insert a permute in the original float model:\n")
    f.write("# Pytorch: conv:[1, 64, 1, 1] -> permute(order=(0, 2, 3, 1)):[1, 1, 1, 64] -> reshape(shape=(1, -1):[1, 64] =>\n")
    f.write("# Xmodel: conv:[1, 1, 1, 64] -> reshape(shape=(1, -1):[1, 64]\n")
    f.write("# In example 1, the permute inserted by quantizer can be canceled out by inserting a permute in float model. \n# After model modification, output shape and data memory layout are the same compared with before.\n")
    f.write("# Example 2:\n")
    f.write("# Pytorch: conv:[1, 3, 4, 4] -> reshape(shape=(1, -1):[1, 48] =>\n")
    f.write("# Xmodel: conv:[1, 4, 4, 3] -> permute(order=(0, 3, 1, 2)):[1, 3, 4, 4] -> reshape(shape=(1, -1):[1, 48]\n")
    f.write("# Insert a permute in the original float model:\n")
    f.write("# Pytorch: conv:[1, 3, 4, 4] -> permute(order=(0, 2, 3, 1)):[1, 4, 4, 3] -> reshape(shape=(1, -1):[1, 48] =>\n")
    f.write("# Xmodel: conv:[1, 4, 4, 3] -> reshape(shape=(1, -1):[1, 48]\n")
    f.write("# In example 2, the permute inserted by quantizer can't be canceled out by inserting a permute in float model. \n# After model modification, output data memory layout changed.\n")
    f.write("\n")   

  def _dump_body(self, f, graph):
    sep_num = 160
    sep_sym = "="
    sep_str = sep_num * sep_sym
    indent_str = 2 * " "
    f.write(sep_str + "\n")
    f.write(f"target info:\n")
    f.write(sep_str + "\n")
    f.write(f"{DPUTargetHelper.get_full_info(self._target.get_devices()[0].get_legacy_dpu_target())}\n\n")
    f.write(sep_str + "\n")
    f.write("inspection summary:\n")
    f.write(sep_str + "\n")
    d = []
    for node in graph.nodes:
      if node.op.type == NNDCT_OP.RETURN:
        continue
      if node.target_device is not None:
        if node.target_device.get_device_type() == DeviceType.CPU:
          d.append([node.name, node.op.type, node.target_device.get_filter_message()])
    if d:
      f.write(f"The operators assigned to the CPU are as follows:\n")
      f.write(tabulate(d, headers=["node name", "op Type", "hardware constraints"]))
      f.write("\n")
    else:
      f.write(f"All the operators are assigned to the DPU.\n")
    f.write(sep_str + "\n")
    f.write(f"graph name: {graph.name}\n")
    for node in graph.nodes:
      if node.op.type == NNDCT_OP.RETURN:
        continue
      f.write(sep_str + "\n")
      f.write(f"node name: {node.name}\n")
      f.write(f"input nodes: {node.in_nodes}\n")
      f.write(f"output nodes: {node.out_nodes}\n")
      f.write(f"op type: {node.op.type}\n")
      f.write(f"outputs shape: {[out.shape for out in node.out_tensors]}\n")
      f.write("op attributes:\n")
      for attr_name in node.op.attrs.keys():
        if node.op.is_xir_attr(attr_name):
          f.write(f"{indent_str}{attr_name.value}: {node.node_attr(attr_name)}\n")
        
      if node.target_device:
        f.write(f"assigned device: {node.target_device.get_device_type().value}\n")
        if node.target_device.get_filter_message():
          f.write(f"hardware constraints: {node.target_device.get_filter_message()}\n")
      
      if node in self._node_msgs:
        f.write("node messages:\n")
        for msg in self._node_msgs[node]:
          f.write(f"{indent_str}{msg}\n")

      # if node.scope_name:
      #   f.write(f"scope: {node.scope_name}\n")
      if node.source_range:
        f.write(f"source range:\n{node.source_range}\n")
        # src_list = [src.split(":")[0] for src in node.source_range.split("\n")]
        # src_loc = node.source_range
        # if self._module_root == "__main__":
        #   for src in src_list:
        #     if len(src.split("/")) == 1:
        #       src_loc = src
        #       break
        # else:
        #   for src in src_list:
        #     if self._module_root in src:
        #       src_loc = src
        #       break
        
  def _attach_extra_node_msg(self, graph):
    transpose_order_to_msg = {
      (0, 3, 1, 2): "from 'NHWC' to 'NCHW'",
      (0, 2, 3, 1): "from 'NCHW' to 'NHWC'"
    }
    for node in graph.nodes:
      if node.op.type == NNDCT_OP.PERMUTE and any([kw in node.name for kw in ["swim_transpose", "sink_transpose"]]):
        order = node.node_attr(node.op.AttrName.ORDER)
        self._node_msgs[node].add(f"quantizer insert this permute operation to convert data layout {transpose_order_to_msg[tuple(order)]} for deployment.")






  
