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

from typing import List, Tuple
from tf1_nndct.optimization.spec import GroupSpec, PruningSpec
import json

class Sensitivity(object):
  def __init__(self, sparsity: float=0.0, val: float=None) -> None:
    self.sparsity = sparsity
    self.val = val
  
  def serialize(self):
    return {
      "sparsity": self.sparsity,
      "val": self.val
    }

  @classmethod
  def deserialize(cls, obj):
    return cls(obj["sparsity"], obj["val"])


class GroupSensitivity(object):
  def __init__(self, nodes: List[str]=[], sens: List[Sensitivity]=[]) -> None:
    self.nodes = [n for n in nodes]
    self.sens = [s for s in sens]
  
  def serialize(self):
    return {
      "nodes": self.nodes,
      "sens": [s.serialize() for s in self.sens]
    }

  @classmethod
  def deserialize(cls, obj):
    return cls(nodes=obj["nodes"], sens=[Sensitivity.deserialize(o) for o in obj["sens"]])


class NetSensitivity(object):
  def __init__(self, group_sens: List[GroupSensitivity]=[]) -> None:
    self._group_sens = [g for g in group_sens]
  
  def get_group_sens(self, idx: int) -> GroupSensitivity:
    return self._group_sens[idx]
  
  def add_group_sens(self, group_sens: GroupSensitivity) -> None:
    self._group_sens.append(group_sens)
  
  @property
  def group_sens(self) -> List[GroupSensitivity]:
    return self._group_sens
  
  def serialize(self):
    return [g.serialize() for g in self._group_sens]
  
  @classmethod
  def deserialize(cls, obj):
    return cls(group_sens=[GroupSensitivity.deserialize(o) for o in obj])


class SensAnalyzer(object):
  def __init__(self) -> None:
    self._net_sens = NetSensitivity()
  
  def add_group_sens(self, group_sens: GroupSensitivity) -> None:
    self._net_sens.add_group_sens(group_sens)
  
  def unfinished_specs(self) -> List[Tuple[Sensitivity, PruningSpec]]:
    ret = []
    for group_sens in self._net_sens.group_sens:
      for sens in group_sens.sens:
        if sens.val is None:
          ret.append((sens, PruningSpec(group_specs=[GroupSpec(group_sens.nodes, sens.sparsity)])))
    return ret

  def generate_spec_by_threshold(self, threshold: float) -> PruningSpec:
    pruning_spec = PruningSpec()
    for group_sens in self._net_sens.group_sens:
      for i in range(1, len(group_sens.sens)):
        relative_diff = abs(group_sens.sens[i].val - group_sens.sens[0].val) / group_sens.sens[0].val
        if relative_diff > threshold:
          if i > 1:
            pruning_spec.add_group_spec(GroupSpec(group_sens.nodes, group_sens.sens[i - 1].sparsity))
          break
        if i == len(group_sens.sens) - 1:
          pruning_spec.add_group_spec(GroupSpec(group_sens.nodes, group_sens.sens[i - 1].sparsity))
    return pruning_spec

  def save(self, path):
    with open(path, 'w') as f:
      json.dump(self._net_sens.serialize(), f, indent=2)
  
  def load(self, path):
    with open(path) as f:
      self._net_sens = NetSensitivity.deserialize(json.load(f))
