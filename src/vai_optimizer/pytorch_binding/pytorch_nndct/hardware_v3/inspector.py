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
import torch
from tabulate import tabulate
from collections import defaultdict
from nndct_shared.base import NNDCT_OP
from nndct_shared.utils import NndctScreenLogger, DeviceType, NndctOption, NndctDebugLogger
from nndct_shared.inspector.device_allocator import DPUAllocator
from nndct_shared.inspector.utils import log_debug_info
from .utils import prepare_deployable_graph


class InspectorImpl(object):
  def __init__(self, target):
    log_debug_info("####Inspector Debug Info:")
    self._target = target
    self._device_allocator = DPUAllocator(target)
    self._node_msgs = defaultdict(set)
    self._module_root = ""
    self._graph = None

  @classmethod
  def create_by_DPU_arch_name(cls, name):
    return cls(name)

  @classmethod
  def create_by_DPU_fingerprint(cls, fingerprint):
    return cls(fingerprint)
  

  def inspect(self, module, input_args, device, output_dir, verbose_level):
    if isinstance(module, torch.nn.DataParallel):
        module = module.module
    copied_model = copy.deepcopy(module)
    copied_model.eval()
    self._module_root = module.__class__.__module__.replace(".", "/")
    dev_graph, deploy_graphs = prepare_deployable_graph(copied_model, input_args, device, output_dir)
    self._device_allocator.process(dev_graph)

    for node in dev_graph.nodes:
      node.target_device = self._device_allocator.get_node_device(node.name)
    #   print(node.name, node.op.type, node.target_device.get_device_type())
    self._attach_extra_node_msg(dev_graph)
    self._show_partition_result_on_screen(dev_graph, output_dir, verbose_level)
    self._dump_txt(dev_graph, output_dir)
    self._graph = dev_graph
    self._graph.clean_tensors_data()
    """
    xir_convert = CompilerFactory.get_compiler("xmodel")
    xcompiler = CompilerFactory.get_compiler("xcompiler")
    fake_quant_config =  convert_quant_config_to_dict(quant_config)
    for graph in deploy_graphs[0]:
      xmodel_file_name = os.path.join("/tmp", graph.name)
      xgraph = xir_convert.do_compile(graph, output_file_name=xmodel_file_name, quant_config_info=fake_quant_config)
      compiled_graph = xcompiler.compile_and_reload(xmodel_file_name, self._target)
      for op in compiled_graph.get_ops():
        if op.get_type() in ["data-fix", "reshape-fix"]:
          continue
        if "-fix" in op.get_type():
          if not(any([fanout.get_attr("device") == "DPU" for fanout in op.get_fanout_ops()]) and op.get_input_ops() and all([inp.get_attr("device") == "DPU" or inp.get_type() == "float2fix" for inp in op.get_input_ops()["input"] ])):
            # print(op.get_name())
            assert op.get_attr("device") == "DPU"
    """

      


  
  def _create_dot_graph_v2(self):
    from graphviz import Digraph
    g = Digraph(self._graph.name)
    device_node_attrs = {
      DeviceType.DPU: {"color": "blue"},
      DeviceType.CPU: {"color": "red"},
      DeviceType.USER: {"color": "black"}
    }
    for node in self._graph.nodes:
      # if node.op.type == NNDCT_OP.RETURN:
      #   continue
      g.node(self._normalize(node.name),
      label=f"type:{node.op.type}\nname:{node.name}\nassigned device:{node.target_device.get_device_type().value}\nout shape:{[out.shape for out in node.out_tensors]}",
      **device_node_attrs[node.target_device.get_device_type()])
    
    for node in self._graph.nodes:
      for on in node.out_nodes:
        # if self._graph.node(on).op.type == NNDCT_OP.RETURN:
        #   continue
        g.edge(self._normalize(node.name), self._normalize(on))
    return g

  def export_dot_image_v2(self, output_dir, format):
    assert self._graph is not None
    # target_name = DPUTargetHelper.get_name(self._target.get_devices()[0].get_legacy_dpu_target())
    file_name = os.path.join(output_dir, ".".join([f"inspect_{self._target}", format]))
    dot_graph = self._create_dot_graph_v2()
    dot_graph.render(outfile=file_name).replace('\\', '/')
    NndctScreenLogger().info(f"Dot image is generated.({file_name})")

  
  @staticmethod
  def _normalize(name):
    return name.split("::")[-1]

  def _dump_txt(self, graph, output_dir):
    file_name = os.path.join(output_dir, f"inspect_{self._target}.txt")
    with open(file_name, 'w') as f:
      self._dump_comment(f)
      self._dump_body(f, graph)

  

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
    f.write('''# Explanation of some hardware constraints messages:
  "Try to assign {pattern name} to DPU failed.": The compiler refuses to deploy this pattern on DPU.
  "Convert nndct graph to XIR failed.": If you encounter this problem, please contact the developer.
  "{op type} can't be converted to XIR.": The operator cannot be represented by XIR.
  "{op type} can't be assigned to DPU.": Although the operator can be converted to XIR, it cannot be deployed on the DPU.
  ''')
    f.write("\n")   

  def _dump_body(self, f, graph):
    sep_num = 160
    sep_sym = "="
    sep_str = sep_num * sep_sym
    indent_str = 2 * " "
    f.write(sep_str + "\n")
    f.write(f"target info:\n")
    f.write(sep_str + "\n")
    f.write(f"{self._target}\n\n")
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
      output_shapes = ",".join([str(out.shape) for out in node.out_tensors])
      f.write(f"outputs shape: {output_shapes}\n")
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

  def _attach_extra_node_msg(self, graph):
    transpose_order_to_msg = {
      (0, 3, 1, 2): "from 'NHWC' to 'NCHW'",
      (0, 2, 3, 1): "from 'NCHW' to 'NHWC'",
      (0, 4, 3, 1, 2): "from 'NHWDC' to 'NCDHW'",
      (0, 3, 4, 2, 1): "from 'NCDHW' to 'NHWDC'"
    }
    for node in graph.nodes:
      if node.op.type == NNDCT_OP.PERMUTE and any([kw in node.name for kw in ["swim_transpose", "sink_transpose"]]):
        order = node.node_attr(node.op.AttrName.ORDER)
        self._node_msgs[node].add(f"quantizer insert this permute operation to convert data layout {transpose_order_to_msg[tuple(order)]} for deployment.")
    

  def _show_partition_result_on_screen(self, graph, output_dir, verbose_level):
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    # pd.set_option("max_colwidth", 100)
    # pd.set_option("display.width", 5000)
    target_name = self._target
    if verbose_level == 0:
      return
    elif verbose_level == 1:
      d = []
      for node in graph.nodes:
        if node.target_device is not None:
          if node.target_device.get_device_type() == DeviceType.CPU:
            assert node.target_device.get_filter_message(), f"{node.name}, {node.op.type}"
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
