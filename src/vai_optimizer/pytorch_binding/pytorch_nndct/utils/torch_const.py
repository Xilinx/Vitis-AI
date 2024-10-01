

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

from enum import Enum, unique 

def get_enum_val(en):
  return en.value

class TorchSymbol():
  MODULE_OUTPUT_PREFIX = "output"
  MODULE_PREFIX = "py_nndct.nn"
  MODULE_NAME_SEPERATOR = "_"
  MODULE_BASE_SYMBOL = "module"
  SCRIPT_SUFFIX = ".py"
  
class TorchGraphSymbol():
  NODE_NAME_SEPERATOR = '/'
  GRAPH_SCOPE_SYM = '::'

@unique
class TorchOpClassType(Enum):
  NN_MODULE = 'nn.module'
  NN_FUNCTION = 'nn.functional'
  TORCH_FUNCTION = 'function'
  TENSOR = 'tensor'
  PRIMITIVE = 'primitive'
  UNKNOWN = 'unknown'
  AUTO_INFER_OP = 'auto_infer_op'
  TORCH_SCRIPT_BUILTIN_FUNCTION = 'torch_bultin_funcion'
  MATH_BUILTIN_FUNCTION = 'math_builtin'
  GLOBAL_BUILTIN_FUNCTION = 'global_builtin'
  NN_CORE_FUNCTION = 'torch._C.nn'
  CUSTOM_FUNCTION = 'custom_function'
  