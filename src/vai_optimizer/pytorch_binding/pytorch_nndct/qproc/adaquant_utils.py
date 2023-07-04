
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


import  torch

_FLOAT_32 = 4
_KB = 1024
_MB = 1024 * _KB
_GB = 1024 * _MB


def tensor_size(tensor):
  assert isinstance(tensor, torch.Tensor)
  return torch.numel(tensor) * _FLOAT_32 / _GB   


def tensor_size_by_num(num):
  return num * _FLOAT_32 / _GB   