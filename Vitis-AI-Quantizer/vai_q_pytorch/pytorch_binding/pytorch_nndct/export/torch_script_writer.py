from collections import OrderedDict
import math
import numpy as np

from .op_descriptor import OpDescriptor
from nndct_shared.base import NNDCT_OP, NNDCT_KEYS
from nndct_shared.base.key_names import FrameworkType
from nndct_shared.nndct_graph import Tensor, Graph
import pytorch_nndct.utils as py_utils
from pytorch_nndct.utils import TorchOpClassType, TorchSymbol

class TorchScriptWriter(object):

  def __init__(self):
    self.tensor_output_map = {}  # key: torch tensor name, value: output name

  def write(self, graph: Graph, file_path: str):
    with open(file_path, 'w') as f:
      self._do_write(f, graph)

  def _do_write(self, f, graph):
    #do the write
    self._write_header(f, graph)
    self._write_graph(f, graph)

  def get_model_type(self):
    return FrameworkType.TORCH

  def _get_op_name(self, torch_op_type):
    op_name = py_utils.get_defined_quant_module(torch_op_type)
    torch_op_name, is_defined_op = (op_name, True) if op_name else (".".join([TorchSymbol.MODULE_PREFIX,
    "Module"]), False)

    return torch_op_name, is_defined_op

  def _get_init_module_str(self, node):
    torch_op_type = py_utils.get_torch_op_type(node.op.type)

    # if torch_op_type in TORCH_UNSUPPORTED_NNDCTOPS:
    #   return ''

    torch_op_attr = py_utils.get_torch_op_attr(torch_op_type)
    op_name, is_defined_op = self._get_op_name(torch_op_type)
    
    def _init_attrs_str():
      attrs_str = ""     
      if torch_op_attr.op_class_type == TorchOpClassType.NN_MODULE:
        attrs_str = self._to_map_str(self._get_module_attrs_map(node, torch_op_type,torch_op_attr.attrs))
      
      if not is_defined_op:
        attrs_str = f"'{node.op.type}',{attrs_str}" if attrs_str else f"'{node.op.type}'"
      
      return attrs_str
        
    def _init_module_str():
      attrs_str = _init_attrs_str()
      return 'self.{module_name} = {op_name}({attrs}) #{node_name}'.format(
        module_name=self._get_module_name(node),
        op_name=op_name,
        attrs=attrs_str,
        node_name=node.name)
    
    return _init_module_str()
  
   
  @staticmethod
  def _get_module_name(node):
    return TorchSymbol.MODULE_NAME_SEPERATOR.join(
        [TorchSymbol.MODULE_BASE_SYMBOL,
         str(node.idx)])

  def _get_module_attrs_map(self, node, torch_op_type, torch_op_attrs):

    ordered_attrs = OrderedDict()
    attrs_template = torch_op_attrs if torch_op_attrs and torch_op_attrs != [
        'args', 'kwargs'
    ] else node.op.configs
    for name in attrs_template:
      if hasattr(node.op, name):
        ordered_attrs[name] = node.node_config(name)

    return ordered_attrs

  def _write_graph(self, f, graph: Graph):

    self._write_init(f, graph)
    self._write_forward(f, graph)

  def _write_init(self, f, graph: Graph):
    indent_str = 4 * " "
    f.write(indent_str + "def __init__(self):\n")
    indent_str += indent_str

    f.write(indent_str + "super({}, self).__init__()\n".format(graph.name))
    for node in graph.nodes:
      module_init_str = self._get_init_module_str(node)
      if module_init_str:
        f.write(indent_str + module_init_str + '\n')

  def _write_forward(self, f, graph: Graph):
    indent_str = 4 * " "
    f.write('\n' + indent_str + "def forward(self, *args):\n")
    indent_str += indent_str
    for node in graph.nodes:
      forward_str, output_str = self._get_forward_str(node)
      f.write(indent_str + forward_str + '\n')

    return_str = indent_str + 'return '
    for i, end_tensor in enumerate(graph.get_end_tensors()):
      if i > 0:
        return_str = ','.join(
            [return_str, self.tensor_output_map[end_tensor.name]])
      else:
        return_str += self.tensor_output_map[end_tensor.name]
       
    f.write(return_str + '\n')

  def _get_module_output(self, node):
    output_list = []
    if len(node.out_tensors) == 1:
      output_node = node.out_tensors[0].node
      name_list = [TorchSymbol.MODULE_OUTPUT_PREFIX, self._get_module_name(output_node)]
      name = 'self.' + TorchSymbol.MODULE_NAME_SEPERATOR.join(name_list)
      self.tensor_output_map[node.out_tensors[0].name] = name
      output_list.append(name)
    else:
      for id, tensor in enumerate(node.out_tensors):
        output_node = tensor.node
        name_list = [TorchSymbol.MODULE_OUTPUT_PREFIX, self._get_module_name(output_node)]  
        name = 'self.' + TorchSymbol.MODULE_NAME_SEPERATOR.join(
            name_list + [str(id)])
       
        self.tensor_output_map[tensor.name] = name
        output_list.append(name)

    return output_list

  def _get_module_input(self, node):
    input_list = []
    for tensor in node.in_tensors:
      input_list.append(self.tensor_output_map[tensor.name])
    return input_list

  @staticmethod
  def _to_list_str(input_list):
    input_list = [i if isinstance(i, str) else str(i) for i in input_list]
    if input_list:
      if len(input_list) == 1:
        return input_list[0]
      else:
        return ','.join(input_list)
    else:
      raise Exception('The input or output of modules can not be empty')

  @staticmethod
  def _to_map_str(input_map):
    str_list = []
    for k, v in input_map.items():
      string = '{key}={value}'.format(key=k, value=v)
      str_list.append(string)
    return ', '.join(str_list)

  def _infer_attr_value(self, value):
    r'''
        Sometimes, the attribute value is a tuple or list, looks like ["111", 4],
        the "111" is a temp str value and means a output tensor of one node, the we need to replace "111"
        with output name of the node, then, the value will be looks like [output_module_..., 4]
        '''

    if isinstance(value, (tuple, list)):
      value_list = []
      for element in value:
        if isinstance(element, Tensor):
          str_value = self.tensor_output_map.get(element.name, element.name)
          value_list.append(str_value)
        else:
          value_list.append(str(element))
      if isinstance(value, tuple):
        return '(' + self._to_list_str(value_list) + ')'
      else:
        return '[' + self._to_list_str(value_list) + ']'
    elif isinstance(value, Tensor):
      return self.tensor_output_map.get(value.name, value.name)
    else:
      return str(value)

  def _infer_attrs(self, attrs: dict):
    for name, value in attrs.items():
      value = self._infer_attr_value(value)
      attrs[name] = value

  def _get_forward_str(self, node):
    output_str = self._to_list_str(self._get_module_output(node))
    
    torch_op_type = py_utils.get_torch_op_type(node.op.type)
    torch_op_attr = py_utils.get_torch_op_attr(torch_op_type)

    if torch_op_attr.op_class_type == TorchOpClassType.NN_MODULE:
      input_str = self._to_list_str(self._get_module_input(node))
      forward_str = "{output} = self.{module_name}({input})".format(
          output=output_str,
          module_name=self._get_module_name(node),
          input=input_str)
    else:
      func_attrs = self._get_module_attrs_map(node, torch_op_type,
                                              torch_op_attr.attrs)
      
      self._infer_attrs(func_attrs)
      func_attrs_str = self._to_map_str(func_attrs)
      if torch_op_attr.op_class_type == TorchOpClassType.TENSOR:
        input = self._get_module_input(node)[0]
        func_attrs_str = f"input={input}, {func_attrs_str}"
      
      forward_str = "{output} = self.{module_name}({attrs})".format(
          output=output_str,
          module_name=self._get_module_name(node),
          attrs=func_attrs_str)
     

    return forward_str, output_str

  def _write_header(self, f, graph):
    f.write('''# GENETARED BY NNDCT, DO NOT EDIT!\n\n''')
    f.write('''import torch
import pytorch_nndct as py_nndct\n''')
    f.write("class {}(torch.nn.Module):\n".format(graph.name))

  
