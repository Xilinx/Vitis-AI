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

import functools
import torch
from nndct_shared.base import GLOBAL_MAP, NNDCT_KEYS
from nndct_shared.utils import NndctScreenLogger
from pytorch_nndct.utils.module_util import to_device, collect_input_devices, get_flattened_input_args


def forward_processor(forward_func):
  @functools.wraps(forward_func)
  def wrapper(self, *args, **kwargs):
    def _check_input_args(input):
      quant_device = GLOBAL_MAP.get_ele(NNDCT_KEYS.QUANT_DEVICE)
      if quant_device is not None:
        input_devices = collect_input_devices(input)
        if any([device != quant_device.type for device in input_devices]):
          NndctScreenLogger().warning_once(f"The Device of input args mismatch with quantizer device type({quant_device.type}).")
          _, input = to_device(None, input, device=quant_device)
      if self.is_from_script is True:
        return input
      else:
        return get_flattened_input_args(input)
    self.set_inference_status(True)
    flatten_inputs = _check_input_args(args)
    return forward_func(self, *flatten_inputs)
  return wrapper


class NndctQuantModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self._nndct_inferenced = False
    self._nndct_from_script = False
  
  @property
  def is_inferenced(self):
    return self._nndct_inferenced
 
  def set_inference_status(self, flag):
    self._nndct_inferenced = flag
  
  @property
  def is_from_script(self):
    return self._nndct_from_script

  def from_script(self, flag):
    self._nndct_from_script = flag



