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

import torch.nn as nn
import torch.nn.functional as F

class QuantizedLinear(nn.Linear):
  """A QuantizedLinear module attached with FakeQuantizer module for weight,
    used for quantization aware training.

    The interface is adopted from `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.
    """
  _FLOAT_MODULE = nn.Linear

  def __init__(self, in_features, out_features, bias=True, qconfig=None):
    super().__init__(in_features, out_features, bias)
    assert qconfig, 'qconfig must be provided for quantized module'
    self.qconfig = qconfig

    self.weight_quantizer = qconfig.get_weight_quantizer('weight')
    if bias:
      self.bias_quantizer = qconfig.get_weight_quantizer('bias')

  def forward(self, input):
    weight = self.weight_quantizer(self.weight)
    bias = self.bias_quantizer(self.bias) if self.bias is not None else None
    return F.linear(input, weight, bias)

  @property
  def is_quantized(self):
    return True

  @classmethod
  def from_float(cls, mod, qconfig):
    """Create a quantized module from a float module.

    Args:
      mod: A float module of type torch.nn.Linear.
      qconfig (pytorch_nndct.quantization.config.RuntimeSpec):
          A qconfig object that saves the quantizers for the module.
    """

    assert qconfig, 'qconfig must be provided for quantized module'
    assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
        cls._FLOAT_MODULE.__name__

    linear = cls(
        mod.in_features,
        mod.out_features,
        bias=mod.bias is not None,
        qconfig=qconfig)
    linear.weight = mod.weight
    linear.bias = mod.bias
    return linear
