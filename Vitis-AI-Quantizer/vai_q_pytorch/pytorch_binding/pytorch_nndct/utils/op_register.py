import inspect
import functools
from functools import wraps
from .nndct2torch_op_map import add_mapping_item
from .torch_op_attr import gen_attr
from .torch_const import TorchSymbol

def op_register(nndct_op: str, torch_op: str):
  add_mapping_item(nndct_op, torch_op)
  gen_attr(torch_op)



_QUANT_MODULES = []  #List[str]

def register_quant_op(func):
  if not inspect.isfunction(func):
    raise RuntimeError("Only decorate function")
  global _QUANT_MODULES
  _QUANT_MODULES.append(func.__name__)
  @functools.wraps(func)
  def innner(*args, **kwargs):
    return func(*args, **kwargs)
  return innner
    
    

def get_defined_quant_module(torch_op: str):
  quant_modules_lower = [module.lower() for module in _QUANT_MODULES]
  if torch_op.lower() in quant_modules_lower:
    index = quant_modules_lower.index(torch_op.lower())
    return ".".join([TorchSymbol.MODULE_PREFIX, _QUANT_MODULES[index]])



    
    
    