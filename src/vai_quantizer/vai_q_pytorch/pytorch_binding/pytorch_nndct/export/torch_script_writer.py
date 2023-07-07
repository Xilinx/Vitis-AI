

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

import abc
from collections import OrderedDict
from typing import Any, Callable, Dict, List
import pytorch_nndct.utils as py_utils
from nndct_shared.base.key_names import FrameworkType, NNDCT_OP
from nndct_shared.nndct_graph import Graph, Node, Tensor
from pytorch_nndct.utils import TorchOpClassType, TorchSymbol

from .op_descriptor import MISC_OP_DISCR_MAP
from nndct_shared.utils import  NndctOption

class TorchBaseScriptWriter(metaclass=abc.ABCMeta):
  def __init__(self):
    self._tensor_output_map: Dict[str, str] = {}  # key: torch tensor name, value: output name
    self._output_name_alias: Dict[str, str] = {}  # key: output name , value: real output name
    self._reuse_node_output_map: Dict[str] = {}
    self._module_names = {}
    self._global_idx = 0
    
  def write(self, graph: Graph, file_path: str):
    with open(file_path, 'w') as f:
      self._do_write(f, graph)    
  
  def _do_write(self, f, graph: Graph):
    self._write_header(f, graph)
    self._write_graph(f, graph)
  
  def _write_header(self, f: Callable, graph: Graph):
    f.write(f"# GENETARED BY NNDCT, DO NOT EDIT!\n\n")
    f.write(f"import torch\nfrom torch import tensor\n")
  
  def _write_graph(self, f: Callable, graph: Graph):
    self._write_init(f, graph)
    self._write_forward(f, graph)

  def _write_init(self, f, graph: Graph):
    indent_str = 4 * " "
    f.write(indent_str + "def __init__(self):\n")
    indent_str += indent_str

    f.write(indent_str + "super({}, self).__init__()\n".format(graph.name))
    
    def _write_module_init(graph):
      for node in graph.nodes:
        if node.blocks:
          for block in node.blocks:
            _write_module_init(block)
        else:
          module_init_str = self._get_init_module_str(node)
          if module_init_str:
            f.write(indent_str + module_init_str + '\n')
    
      # generate global params init code
      for node in graph.nodes:
        param_init_str_lst = self._get_init_param_str(node)
        if param_init_str_lst:
          for parma_init_str in param_init_str_lst:
            f.write(indent_str + parma_init_str + '\n')
            
    _write_module_init(graph)
        
  def _collect_reuse_output(self, graph: Graph):
        
    def dfs_recursion(node, visited):
      visited.append(node)
      if len(node.out_tensors) == 1 \
      and graph.parents(node) \
      and len(graph.parents(node)[0].out_nodes) == 1 \
      and graph.parents(node)[0].out_tensors[0] not in graph.end_tensors:
        self._reuse_node_output_map[node.name] = graph.parents(node)[0].out_tensors[0].name
             
      for cn in graph.children(node):
        if cn not in visited:
          dfs_recursion(cn, visited)
        
    def dfs_iteration(node_in, visited_in):
      task_stack = [[node_in, visited_in]]
      while len(task_stack) > 0:
          node, visited = task_stack.pop()
          visited.append(node)
          if len(node.out_tensors) == 1 \
          and graph.parents(node) \
          and len(graph.parents(node)[0].out_nodes) == 1 \
          and graph.parents(node)[0].out_tensors[0] not in graph.end_tensors:
            self._reuse_node_output_map[node.name] = graph.parents(node)[0].out_tensors[0].name
          task_list = []
          for cn in graph.children(node):
            if cn not in visited:
                task_list.append([cn, visited])
          task_list.reverse()
          for k in task_list:
            task_stack.append(k)
    def dfs():
        pass
    if NndctOption.nndct_traversal_graph_mode.value == 1:
        dfs = dfs_recursion
    else:
        dfs = dfs_iteration    
          
    if len(graph.all_blocks()) > 1:
      return
    
    visited = []
    input_nodes = [node for node in graph.nodes if node.op.type == NNDCT_OP.INPUT]
    for node in input_nodes:
      dfs(node, visited)
      
    
    
  def _write_forward(self, f: Callable, graph: Graph):
    indent_str = 4 * " "
    f.write('\n' + indent_str + "@py_nndct.nn.forward_processor")
    f.write('\n' + indent_str + "def forward(self, *args):\n")
    indent_str += indent_str
    self._collect_reuse_output(graph)
    for node in graph.nodes:
      forward_str, output_str = self._get_forward_str(node)
      format_forward_str = self._append_indent(indent_str, forward_str)
      f.write(format_forward_str + '\n')

  def _get_init_param_str(self, node: Node) -> List[str]:
    str_list = []
    if node.has_bound_params():
      return str_list
    for param_type, param_tensor in node.op.params.items():
      if param_tensor.name not in self._tensor_output_map:
        param_name = param_type.value
        param_shape = tuple(param_tensor.shape)
        param_init_str = f"self.{param_name} = torch.nn.parameter.Parameter(torch.Tensor{param_shape})"
        str_list.append(param_init_str)
        self._tensor_output_map[param_tensor.name] = f"self.{param_name}"
    return str_list
  
  @abc.abstractmethod
  def _get_init_module_str(self, node: Node) -> str:
    pass
  
  @abc.abstractmethod
  def _get_forward_str(self, node: Node) -> str:
    pass
  
  def _get_module_attrs_map(self, node: Node, torch_op_type: str, torch_op_attrs: Dict[str, Any]):
    ordered_attrs = OrderedDict()
    attrs_template = torch_op_attrs if torch_op_attrs and torch_op_attrs != [
        'args', 'kwargs'
    ] else node.op.configs
    for name in attrs_template:
      if hasattr(node.op, name) or (hasattr(node.op, 'has_config') and node.op.has_config(name)):
        ordered_attrs[name] = node.node_config(name)

    return ordered_attrs
  
  def _get_module_output(self, node):
    output_list = []
    if node.name in self._reuse_node_output_map:
      name = self._tensor_output_map[self._reuse_node_output_map[node.name]]
      self._tensor_output_map[node.out_tensors[0].name] = name
      output_list.append(name)
    elif node.out_tensors and node.out_tensors[0].name in self._tensor_output_map:
      for tensor in node.out_tensors:
        output_list.append(self._tensor_output_map[tensor.name])
    else:
      if len(node.out_tensors) == 1:
        output_node = node.out_tensors[0].node
        if self._get_module_name(output_node) is not None:
          name_list = [TorchSymbol.MODULE_OUTPUT_PREFIX, self._get_module_name(output_node)]
        else:
          name_list = [TorchSymbol.MODULE_OUTPUT_PREFIX, str(self._global_idx)]
          self._global_idx += 1
          
        name = TorchSymbol.MODULE_NAME_SEPERATOR.join(name_list)
        self._tensor_output_map[node.out_tensors[0].name] = name
        output_list.append(name)
      else:
        for id, tensor in enumerate(node.out_tensors):
          output_node = tensor.node
          if self._get_module_name(output_node) is not None:
            name_list = [TorchSymbol.MODULE_OUTPUT_PREFIX, self._get_module_name(output_node)]  
          else:
            name_list = [TorchSymbol.MODULE_OUTPUT_PREFIX, str(self._global_idx)]
            self._global_idx += 1
            
          name = TorchSymbol.MODULE_NAME_SEPERATOR.join(
              name_list + [str(id)])
          
          self._tensor_output_map[tensor.name] = name
          output_list.append(name)

    return output_list
  
  def set_name_alias_for_output(self, name, alias_name):
    if name != alias_name:
      self._output_name_alias[name] = alias_name
  
  def get_output_tensor_name(self, tensor):
    if tensor is None:
        return 'None'
    output_tensor_name = self._tensor_output_map.get(tensor.name, tensor.name)
    while True:
      alias_name = self._output_name_alias.get(output_tensor_name)
      if alias_name is None:
        return output_tensor_name
      else:
        output_tensor_name = alias_name
        
  
  def _get_module_input(self, node):
    input_list = []
    for tensor in node.in_tensors:
      if node.has_bound_params() and tensor.is_param_tensor():
        continue
      input_list.append(self.get_output_tensor_name(tensor))
    return input_list
  
  def infer_attr_value(self, value):
    r'''
      Sometimes, the attribute value is a tuple or list, looks like ["111", 4],
      the "111" is a temp str value and means a output tensor of one node, the we need to replace "111"
      with output name of the node, then, the value will be looks like [output_module_..., 4]
      '''

    if isinstance(value, (tuple, list)):
      value_list: List[str] = []
      for element in value:
        str_element = self.infer_attr_value(element)
        value_list.append(str_element)
        # if isinstance(element, Tensor):
        #   str_value = self.get_output_tensor_name(element)
        #   value_list.append(str_value)
        # else:
        #   value_list.append(str(element))
      if isinstance(value, tuple):
        if len(value_list) == 1:
          return self._to_list_str(value_list)
        else:
          return '(' + self._to_list_str(value_list) + ')'
      else:
        return '[' + self._to_list_str(value_list) + ']'
    elif isinstance(value, Tensor):
      return self.get_output_tensor_name(value)
    else:
      return str(value)

  def _infer_attrs(self, attrs: dict):
    for name, value in attrs.items():
      value = self.infer_attr_value(value)
      attrs[name] = value
  
  @staticmethod
  def _to_list_str(input_list):
    input_list = [i if isinstance(i, str) else str(i) for i in input_list]
    if input_list:
      if len(input_list) == 1:
        return input_list[0]
      else:
        return ','.join(input_list)
    else:
      return ''
      #raise Exception('The input or output of modules can not be empty')

  @staticmethod
  def _to_map_str(input_map):
    str_list = []
    for k, v in input_map.items():
      string = '{key}={value}'.format(key=k, value=v)
      str_list.append(string)
    return ', '.join(str_list)

  @staticmethod
  def _to_dict_str(input_map):
    str_list = []
    for k, v in input_map.items():
      string = "'{key}': {value}".format(key=k, value=v)
      str_list.append(string)
    dict_str = ",".join(str_list)
    return "{" + dict_str + "}"

  def _gen_module_name(self, node: Node):
    module_name = TorchSymbol.MODULE_NAME_SEPERATOR.join(
        [TorchSymbol.MODULE_BASE_SYMBOL,
         str(self._global_idx)])
    node.idx = self._global_idx
    self._global_idx += 1
    self._module_names[node.name] = module_name
    return module_name
  
  def _get_module_name(self, node: Node):
    return self._module_names.get(node.name, None)
  
  def get_model_type(self):
    return FrameworkType.TORCH

  @staticmethod
  def _append_indent(indent_str: str, body_str: str) -> str:
    lines = []
    for line in body_str.split("\n"):
      lines.append(indent_str + line)
    return "\n".join(lines)

 
 
class TorchScriptWriter(TorchBaseScriptWriter):
  def __init__(self):
    super().__init__()
  
  def _get_init_module_str(self, node: Node) -> str:
    torch_op_type = py_utils.get_torch_op_type(node.op.type)
    torch_op_attr = py_utils.get_torch_op_attr(torch_op_type)
    if torch_op_attr.op_class_type == TorchOpClassType.NN_MODULE:
      attrs_str = self._to_map_str(self._get_module_attrs_map(node, torch_op_type, torch_op_attr.attrs))
      return 'self.{module_name} = {op_name}({attrs}) #{node_name}'.format(module_name=self._gen_module_name(node), 
                                                                           op_name=torch_op_attr.op_name, 
                                                                           attrs=attrs_str, node_name=node.name)
  
  def _get_forward_str(self, node: Node) -> str:
    output_str = self._to_list_str(self._get_module_output(node))
    torch_op_type = py_utils.get_torch_op_type(node.op.type)
    torch_op_attr = py_utils.get_torch_op_attr(torch_op_type)

    if torch_op_attr.op_class_type == TorchOpClassType.NN_MODULE:
      input_str = self._to_list_str(self._get_module_input(node)) 
      forward_str = "{output} = self.{module_name}({input})".format(output=output_str, 
                                                                    module_name=self._get_module_name(node), 
                                                                    input=input_str)

    elif (torch_op_attr.op_class_type == TorchOpClassType.NN_FUNCTION or 
          torch_op_attr.op_class_type == TorchOpClassType.TORCH_FUNCTION or
          torch_op_attr.op_class_type == TorchOpClassType.NN_CORE_FUNCTION):
      func_attrs = self._get_module_attrs_map(node, torch_op_type, torch_op_attr.attrs)
      self._infer_attrs(func_attrs)
      forward_str = "{output} = {op_name}({attrs}) #{node_name}".format(output=output_str, 
                                                                        op_name=torch_op_attr.op_name, 
                                                                        attrs=self._to_map_str(func_attrs), 
                                                                        node_name=node.name)

    elif torch_op_attr.op_class_type == TorchOpClassType.TENSOR:
        input = self._get_module_input(node)[0]
        func_attrs = self._get_module_attrs_map(node, torch_op_type, torch_op_attr.attrs)
        self._infer_attrs(func_attrs)
        if 'input' in func_attrs:
          del func_attrs['input']
          
        forward_str = "{output} = {input}.{op_name}({attrs}) #{node_name}".format(output=output_str, 
                                                                                  input=input, 
                                                                                  op_name=torch_op_attr.op_name, 
                                                                                  attrs=self._to_map_str(func_attrs), 
                                                                                  node_name=node.name)

    elif node.op.type in MISC_OP_DISCR_MAP:
      forward_str = MISC_OP_DISCR_MAP[node.op.type](self, node, output_str)
    
    else:
      raise RuntimeError('op_class_type of op is unknown, please check the operation: {}'.format(node.op.type))

    return forward_str, output_str
  
  def _write_header(self, f, graph):
    super()._write_header(f, graph)
    f.write("class {}(torch.nn.Module):\n".format(graph.name))
    
    
class TorchQuantScriptWriter(TorchBaseScriptWriter):

  def __init__(self):
    super().__init__()

  def _get_init_module_str(self, node: Node) -> str:
    torch_op_type = py_utils.get_torch_op_type(node.op.type)
    torch_op_attr = py_utils.get_torch_op_attr(torch_op_type)
    if torch_op_attr.op_class_type != TorchOpClassType.UNKNOWN:
      op_name, attrs_str = self._init_op_and_attrs_str(node)
      return 'self.{module_name} = {op_name}({attrs}) #{node_name}'.format(module_name=self._gen_module_name(node), 
                                                                          op_name=op_name, 
                                                                          attrs=attrs_str, 
                                                                          node_name=node.name)
    
  def _init_op_and_attrs_str(self, node: Node) -> str:
    torch_op_type = py_utils.get_torch_op_type(node.op.type)
    torch_op_attr = py_utils.get_torch_op_attr(torch_op_type)
    op_name = py_utils.get_defined_quant_module(torch_op_type) if not node.has_custom_op() or torch_op_attr.op_class_type in [TorchOpClassType.PRIMITIVE] else ''
    op_name, is_defined_op = (op_name, True) if op_name else (".".join([TorchSymbol.MODULE_PREFIX, "Module"]), False)
    attrs_str = ""     
    if torch_op_attr.op_class_type == TorchOpClassType.NN_MODULE:
      attrs_str = self._to_map_str(self._get_module_attrs_map(node, torch_op_type, torch_op_attr.attrs))
    
    if not is_defined_op:
      attrs_str = f"'{node.op.type}',{attrs_str}" if attrs_str else f"'{node.op.type}'"
    
    return op_name, attrs_str
    
  def _get_forward_str(self, node: Node) -> str:
    output_str = self._to_list_str(self._get_module_output(node))
    
    torch_op_type = py_utils.get_torch_op_type(node.op.type)
    torch_op_attr = py_utils.get_torch_op_attr(torch_op_type)

    if torch_op_attr.op_class_type == TorchOpClassType.NN_MODULE:
      input_str = self._to_list_str(self._get_module_input(node))
      forward_str = "{output} = self.{module_name}({input})".format(
          output=output_str,
          module_name=self._get_module_name(node),
          input=input_str)
    elif (torch_op_attr.op_class_type == TorchOpClassType.NN_FUNCTION or 
          torch_op_attr.op_class_type == TorchOpClassType.TORCH_FUNCTION or 
          torch_op_attr.op_class_type == TorchOpClassType.TENSOR or
          torch_op_attr.op_class_type == TorchOpClassType.PRIMITIVE or
          torch_op_attr.op_class_type == TorchOpClassType.NN_CORE_FUNCTION):
      func_attrs = self._get_module_attrs_map(node, torch_op_type,
                                              torch_op_attr.attrs)
      if not func_attrs:
        input_str = self._to_list_str(self._get_module_input(node))
        forward_str = "{output} = self.{module_name}({input})".format(
          output=output_str,
          module_name=self._get_module_name(node),
          input=input_str)
      else:
        self._infer_attrs(func_attrs)
        func_attrs_str = self._to_map_str(func_attrs)
        if torch_op_attr.op_class_type == TorchOpClassType.TENSOR and "input" not in func_attrs:
          input = self._get_module_input(node)[0]
          func_attrs_str = f"input={input}, {func_attrs_str}"
        
        forward_str = "{output} = self.{module_name}({attrs})".format(
            output=output_str,
            module_name=self._get_module_name(node),
            attrs=func_attrs_str)
      
    elif torch_op_attr.op_class_type == TorchOpClassType.UNKNOWN and node.op.type in MISC_OP_DISCR_MAP:
      forward_str = MISC_OP_DISCR_MAP[node.op.type](self, node, output_str)
    elif  torch_op_attr.op_class_type == TorchOpClassType.AUTO_INFER_OP:
      func_attrs = self._get_module_attrs_map(node, torch_op_type, torch_op_attr.attrs)
      self._infer_attrs(func_attrs)
      func_attrs_str = self._to_dict_str(func_attrs)
      forward_str = "{output} = self.{module_name}({attrs})".format(
            output=output_str,
            module_name=self._get_module_name(node),
            attrs=func_attrs_str)

    elif torch_op_attr.op_class_type in [TorchOpClassType.TORCH_SCRIPT_BUILTIN_FUNCTION,
                                         TorchOpClassType.MATH_BUILTIN_FUNCTION,
                                         TorchOpClassType.GLOBAL_BUILTIN_FUNCTION,
                                         TorchOpClassType.CUSTOM_FUNCTION]:
      func_attrs = self._get_module_attrs_map(node, torch_op_type, torch_op_attr.attrs)
      if not func_attrs:
        input_str = self._to_list_str(self._get_module_input(node))
        forward_str = "{output} = self.{module_name}({input})".format(
          output=output_str,
          module_name=self._get_module_name(node),
          input=input_str)
      else:
        self._infer_attrs(func_attrs)
        args = [arg_value for arg_value in func_attrs.values()]
        args_str = self._to_list_str(args)
        forward_str = "{output} = self.{module_name}({attrs})".format(
            output=output_str,
            module_name=self._get_module_name(node),
            attrs=args_str)
        
    else:
      raise RuntimeError(f'op_class_type of op ({torch_op_attr.op_class_type.value}) is unknown, please check the operation: {node.op.type}.')
     
    return forward_str, output_str

  def _write_header(self, f, graph):
    super()._write_header(f, graph)
    f.write(f"import pytorch_nndct as py_nndct\n")
    f.write("\nclass {}(py_nndct.nn.NndctQuantModel):\n".format(graph.name))
