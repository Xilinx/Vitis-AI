
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

import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Linear):
  """A linear module attached with FakeQuantize modules for both output
    activation and weight, used for quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.

    """
  _FLOAT_MODULE = nn.Linear

  def __init__(self, in_features, out_features, bias=True, qconfig=None):
    super(Linear, self).__init__(in_features, out_features, bias)
    #assert qconfig, 'qconfig must be provided for QAT module'
    self.qconfig = qconfig

    self.weight_quantizer = qconfig.weight
    if bias:
      self.bias_quantizer = qconfig.bias

  def forward(self, input):
    weight = self.weight_quantizer(self.weight)
    bias = self.bias_quantizer(self.bias) if self.bias is not None else None
    return F.linear(input, weight, bias)

  def extra_repr(self):
    return super(Linear, self).extra_repr()

  @classmethod
  def from_float(cls, mod, qconfig):
    """Create a qat module from a float module or qparams_dict

    Args: `mod` a float module, either produced by torch.quantization utilities
        or directly from user
    """
    assert qconfig, 'Input float module must have a valid qconfig'
    assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
        cls._FLOAT_MODULE.__name__

    qat_linear = cls(
        mod.in_features,
        mod.out_features,
        bias=mod.bias is not None,
        qconfig=qconfig)
    qat_linear.weight = mod.weight
    qat_linear.bias = mod.bias
    return qat_linear
