

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

from tensorflow.python.util import tf_inspect

from tf_nndct.utils import registry
from tf_nndct.layers import quantization

_quant_module_registry = registry.Registry('quant_op')

class QuantizedModule(object):
  """A decorator for registering a NNDCT op that can be quantized.
  The graph writer will write this op as the decorated class.

  The decorator argument `op_type` is the string type of an
  op which corresponds to the `NodeDef.op` field in the proto definition.
  """

  def __init__(self, op_type):
    """Creates a new decorator with `op_type` as the Operation type.

    Args:
      op_type: The type of an framework operation.

    Raises:
      TypeError: If `op_type` is not string.
    """
    self._op_type = op_type

  def __call__(self, cls):
    """Registers the class 'cls' as module class for writing op_type."""
    if not tf_inspect.isclass(cls):
      raise TypeError("cls must be a class.")
    quantized_cls = quantization.make_quantized(cls)
    _quant_module_registry.register(quantized_cls, self._op_type)
    return cls

def get_quant_module(op_type, default=None):
  if op_type not in _quant_module_registry.list():
    return default
  return _quant_module_registry.lookup(op_type)
