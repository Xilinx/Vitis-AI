# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from nndct_shared.utils.tensor_util import DataFormatMap
from typing import List

def num_remaining_channels(num_channels, ratio, channel_divisible):
  if num_channels <= channel_divisible:
    return num_channels

  value = int((1 - ratio) * num_channels)
  return max(
      channel_divisible,
      int(value + channel_divisible / 2) // channel_divisible *
      channel_divisible)

def out_in_axis(dim):
  layout = DataFormatMap.param_format('nndct', dim)
  return layout.index('O'), layout.index('I')

def generate_indices_group(indices: List[int], dim_size: int,
                           groups: int) -> List[List[int]]:
  indices_set = set(indices)
  interval: int = dim_size // groups
  start_idx = 0
  end_idx = interval
  ret: List[List[int]] = []
  while start_idx < dim_size:
    idx_group: List[int] = []
    for i in range(start_idx, end_idx):
      if i in indices_set:
        idx_group.append(i - start_idx)
    ret.append(idx_group)
    start_idx = end_idx
    end_idx += interval
  return ret
