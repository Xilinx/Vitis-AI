

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
  
  
@unique
class TorchOpClassType(Enum):
  NN_MODULE = 'nn.module'
  NN_FUNCTION = 'nn.functional'
  TORCH_FUNCTION = 'function'
  TENSOR = 'tensor'
  PRIMITIVE = 'primitive'
  UNKNOWN = 'unknown'
  