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

from typing import List

class GroupSpec(object):
  def __init__(self, nodes: List[str], sparsity: float) -> None:
    self._nodes = nodes
    self._sparsity = sparsity
  
  @property
  def nodes(self) -> List[str]:
    return self._nodes
  
  @property
  def sparsity(self) -> float:
    return self._sparsity

  def __eq__(self, other) -> bool:
    if self.sparsity != other.sparsity:
      return False
    if len(self.nodes) != len(other.nodes):
      return False
    for n in self.nodes:
      if n not in other.nodes:
        return False
    return True


class PruningSpec(object):
  def __init__(self, channel_divisible: int=2, group_specs: List[GroupSpec]=[]) -> None:
    self._channel_divisible = channel_divisible
    self._group_specs = [g for g in group_specs]

  def add_group_spec(self, group_spec: GroupSpec):
    self._group_specs.append(group_spec)
  
  @property
  def channel_divisible(self) -> int:
    return self._channel_divisible

  @channel_divisible.setter
  def channel_divisible(self, c: int) -> None:
    self._channel_divisible = c
  
  @property
  def group_specs(self) -> List[GroupSpec]:
    return self._group_specs
  
  def __eq__(self, other) -> bool:
    if self.channel_divisible != other.channel_divisible:
      return False
    if len(self.group_specs) != len(other.group_specs):
      return False
    for group_spec in self.group_specs:
      found = False
      for other_group_spec in other.group_specs:
        if group_spec == other_group_spec:
          found = True
          break
      if not found:
        return False
    return True