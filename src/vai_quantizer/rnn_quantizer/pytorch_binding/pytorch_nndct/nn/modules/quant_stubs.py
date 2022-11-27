

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

from typing import Sequence

import torch

from .function import DeQuantStubF, QuantStubF


class QuantStub(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, inputs):
    if not isinstance(inputs, Sequence):
      return QuantStubF.apply(inputs) if isinstance(inputs, torch.Tensor) else inputs

    outputs = []
    for ip in inputs:
      # output = QuantStubF.apply(ip) if isinstance(ip, torch.Tensor) else ip
      output = self.forward(ip)
      outputs.append(output)
    if len(outputs) == 1:
      return outputs[0]
    else:
      return outputs
   

class DeQuantStub(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, inputs):
    if not isinstance(inputs, Sequence):
      return DeQuantStubF.apply(inputs) if isinstance(inputs, torch.Tensor) else inputs

    outputs = []
    for ip in inputs:
      # output = DeQuantStubF.apply(ip) if isinstance(ip, torch.Tensor) else ip
      output = self.forward(ip)
      outputs.append(output)
    if len(outputs) == 1:
      return outputs[0]
    else:
      return outputs
   