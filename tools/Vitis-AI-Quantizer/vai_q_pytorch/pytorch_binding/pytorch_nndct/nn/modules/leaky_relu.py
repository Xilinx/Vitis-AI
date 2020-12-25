

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

import torch
from nndct_shared.quantization import maybe_get_quantizer
from nndct_shared.quantization import process_inputs_and_params
from nndct_shared.quantization import post_quant_process
import pytorch_nndct.utils as py_utils
from nndct_shared.utils import NndctOption

__all__ = ['LeakyReLU']


class deephi_LeakyReLU(torch.nn.LeakyReLU):
  r"""DeePhi LeakyReLU operation"""

  def __init__(self, *args, **kwargs):
    # only support the specified slope and inplace operation
    super().__init__(*args, **kwargs)
    self.negative_slope = 0.1015625
    self.quant_mode, self.quantizer = maybe_get_quantizer()
    self.node = None

  def forward(self, tensors):
    inputs, _ = process_inputs_and_params(
        self.node,
        self.quantizer,
        inputs=tensors)
    output = super().forward(inputs)
    [output] = post_quant_process(self.node, [output])
    return output


@py_utils.register_quant_op
def LeakyReLU(*args, **kwargs):
  quant_mode, _ = maybe_get_quantizer()
  if quant_mode is None or NndctOption.nndct_quant_off.value:
    return torch.nn.LeakyReLU(*args, **kwargs)
  return deephi_LeakyReLU(*args, **kwargs)
