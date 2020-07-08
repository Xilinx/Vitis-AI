import torch
import inspect
from .torch_const import TorchOpClassType
_TORCH_OP_ATTR_MAP = {}  #Dict(str, TorchOpAttr)



class TorchOpAttr:

  def __init__(self, op_name: str):
    self.op_name = op_name
    self.op_class_type = None
    self.attrs = []  #List[str]
    self.input_args = []  #List[str]

  def set_op_class_type(self):
    if self.op_name in dir(torch.nn):
      self.op_class_type = TorchOpClassType.NN_MODULE
      self.op_name = '.'.join(['torch', 'nn', self.op_name])

    elif self.op_name in dir(torch.nn.functional):
      self.op_class_type = TorchOpClassType.NN_FUNCTION
      self.op_name = '.'.join(['torch', 'nn', 'functional', self.op_name])

    elif self.op_name in dir(torch):
      self.op_class_type = TorchOpClassType.TORCH_FUNCTION
      self.op_name = '.'.join(['torch', self.op_name])

    elif self.op_name in dir(torch.Tensor):
      self.op_class_type = TorchOpClassType.TENSOR

    else:
      self.op_class_type = TorchOpClassType.PRIMITIVE

  def fill_in_attr_and_inputs_table(self):
    if self.op_class_type == TorchOpClassType.NN_MODULE:
      self.attrs[:] = list(
          eval('inspect.signature({}.__init__).parameters'.format(
              self.op_name)))[1:]
      self.input_args[:] = list(
          eval('inspect.signature({}.forward).parameters'.format(
              self.op_name)))[1:]

    elif self.op_class_type == TorchOpClassType.NN_FUNCTION:
      self.attrs[:] = list(
          eval('inspect.signature({}).parameters'.format(self.op_name)))[:]
      self.input_args[:] = self.attrs[:]

def gen_attr(torch_op: str):
  global _TORCH_OP_ATTR_MAP
  if torch_op not in _TORCH_OP_ATTR_MAP:
    op_attr = TorchOpAttr(torch_op)
    op_attr.set_op_class_type()
    op_attr.fill_in_attr_and_inputs_table()
    _TORCH_OP_ATTR_MAP[torch_op] = op_attr

def get_torch_op_attr_map():
  global _TORCH_OP_ATTR_MAP
  if len(_TORCH_OP_ATTR_MAP) == 0:
    raise Exception(
        'please build the torch_op_type - > torch_op_attributes map')
  return _TORCH_OP_ATTR_MAP

def get_torch_op_attr(torch_op_type):
  if get_torch_op_attr_map().get(torch_op_type, None) is None:
    raise Exception(
        'pleas check torch op attribute :"{}"'.format(torch_op_type))
  else:
    return get_torch_op_attr_map()[torch_op_type]
