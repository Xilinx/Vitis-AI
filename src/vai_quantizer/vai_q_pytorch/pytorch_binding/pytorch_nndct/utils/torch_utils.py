# Copyright 2023 Xilinx Inc.
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

import torch

from torch import nn
from distutils.version import LooseVersion
from enum import unique, Enum

@unique
class CmpFlag(Enum):
  """
  Enum for comparison flags
  """
  EQUAL = 0
  LESS = 1
  LESS_EQUAL = 2
  GREATER = 3
  GREATER_EQUAL = 4
  NOT_EQUAL = 5

def compare_torch_version(compare_type:CmpFlag, version:str):
    if compare_type == CmpFlag.EQUAL:
      return LooseVersion(torch.__version__) == LooseVersion(version)
    if compare_type == CmpFlag.LESS:
      return LooseVersion(torch.__version__) < LooseVersion(version)
    if compare_type == CmpFlag.LESS_EQUAL:
      return LooseVersion(torch.__version__) <= LooseVersion(version)
    if compare_type == CmpFlag.GREATER:
      return LooseVersion(torch.__version__) > LooseVersion(version)
    if compare_type == CmpFlag.GREATER_EQUAL:
      return LooseVersion(torch.__version__) >= LooseVersion(version)
    if compare_type == CmpFlag.NOT_EQUAL:
      return LooseVersion(torch.__version__) != LooseVersion(version)

def strip_parallel(model):
  if isinstance(
      model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
    return model.module
  return model
