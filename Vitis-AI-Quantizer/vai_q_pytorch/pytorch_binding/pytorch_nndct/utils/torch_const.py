from enum import Enum, unique 

def get_enum_val(en):
  return en.value

class TorchSymbol():
  MODULE_OUTPUT_PREFIX = "output"
  MODULE_PREFIX = "py_nndct.nn"
  MODULE_NAME_SEPERATOR = "_"
  MODULE_BASE_SYMBOL = "module"
  SCRIPT_SUFFIX = ".py"
  
  
@unique
class TorchOpClassType(Enum):
  NN_MODULE = 'nn.module'
  NN_FUNCTION = 'nn.functional'
  TORCH_FUNCTION = 'function'
  TENSOR  = 'tensor'
  PRIMITIVE = 'primitive'