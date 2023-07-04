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

from nndct_shared.nndct_graph.base_tensor import Tensor
import json


class ExpandableGroup(object):
  def __init__(self, nodes: List[str], channel_divisible: int = 2) -> None:
    self._nodes: List[str] = nodes
    self._channel_divisible: int = channel_divisible

  @property
  def nodes(self) -> List[str]:
    return self._nodes

  @property
  def channel_divisible(self) -> int:
    return self._channel_divisible

  def serialize(self) -> str:
    return json.dumps({
      "nodes": self._nodes,
      "channel_divisible": self._channel_divisible
    })

  @classmethod
  def from_string(cls, s: str):
    json_obj = json.loads(s)
    return ExpandableGroup(json_obj["nodes"], json_obj["channel_divisible"])


class ExpandingSpec(object):
  def __init__(self, groups: List[ExpandableGroup] = None) -> None:
    self._groups: List[ExpandableGroup] = []
    if groups is not None:
      self._groups = groups

  @property
  def groups(self) -> List[ExpandableGroup]:
    return self._groups

  def add_group(self, group: ExpandableGroup):
    self._groups.append(group)

  def serialize(self) -> str:
    serialized_groups: List[str] = []
    for group in self._groups:
      serialized_groups.append(group.serialize())
    return json.dumps(serialized_groups)

  @classmethod
  def from_string(cls, s: str):
    serialized_groups: List[str] = json.loads(s)
    groups: List[ExpandableGroup] = [ExpandableGroup.from_string(s) for s in serialized_groups]
    return ExpandingSpec(groups)


class DataInsert(object):
  def __init__(self, position: int = 0, added_num_channels: int = 0, added_data: Tensor = None) -> None:
    self._position: int = position
    self._added_num_channels: int = added_num_channels
    self._added_data: Tensor = added_data

  @property
  def position(self) -> int:
    return self._position

  @property
  def added_num_channels(self) -> int:
    return self._added_num_channels

  @property
  def added_data(self) -> Tensor:
    return self._added_data

  @added_data.setter
  def added_data(self, data: Tensor) -> None:
    self._added_data = data


class StructuredExpanding(object):
  def __init__(self, node_name: str) -> None:
    self._node_name: str = node_name
    self._in_dim: int = 0
    self._out_dim: int = 0

  @property
  def node_name(self) -> str:
    return self._node_name

  @property
  def in_dim(self) -> int:
    return self._in_dim

  @in_dim.setter
  def in_dim(self, v: int) -> None:
    self._in_dim = v

  @property
  def out_dim(self) -> int:
    return self._out_dim

  @out_dim.setter
  def out_dim(self, v: int) -> None:
    self._out_dim = v

  @property
  def added_out_channel(self) -> int:
    raise NotImplementedError("method added_out_channel is not implemented")

  @property
  def added_in_channel(self) -> int:
    raise NotImplementedError("method added_in_channel is not implemented")

  @property
  def out_inserts(self) -> List[DataInsert]:
    raise NotImplementedError("method out_inserts is not implemented")


class WeightedNodeStructuredExpanding(StructuredExpanding):
  def __init__(self, node_name: str) -> None:
    super().__init__(node_name)
    self._weight_out_inserts: List[DataInsert] = []
    self._weight_in_inserts: List[DataInsert] = []
    self._bias_inserts: List[DataInsert] = []

  @property
  def added_out_channel(self) -> int:
    ret = 0
    for insert in self._weight_out_inserts:
      ret += insert.added_num_channels
    return ret

  @property
  def added_in_channel(self) -> int:
    ret = 0
    for insert in self._weight_in_inserts:
      ret += insert.added_num_channels
    return ret

  @property
  def weight_out_inserts(self) -> List[DataInsert]:
    return self._weight_out_inserts

  @weight_out_inserts.setter
  def weight_out_inserts(self, v: List[DataInsert]) -> None:
    self._weight_out_inserts = v

  @property
  def weight_in_inserts(self) -> List[DataInsert]:
    return self._weight_in_inserts

  @weight_in_inserts.setter
  def weight_in_inserts(self, v: List[DataInsert]) -> None:
    self._weight_in_inserts = v

  @property
  def bias_inserts(self) -> List[DataInsert]:
    return self._bias_inserts

  @bias_inserts.setter
  def bias_inserts(self, v: List[DataInsert]) -> None:
    self._bias_inserts = v

  @property
  def out_inserts(self) -> List[DataInsert]:
    return self._weight_out_inserts

  def add_weight_out_insert(self, weight_insert: DataInsert):
    self._weight_out_inserts.append(weight_insert)

  def add_weight_in_insert(self, weight_insert: DataInsert):
    self._weight_in_inserts.append(weight_insert)

  def add_bias_insert(self, bias_insert: DataInsert):
    self._bias_inserts.append(bias_insert)


class BatchNormStructuredExpanding(StructuredExpanding):
  def __init__(self, node_name: str) -> None:
    super().__init__(node_name)
    self._moving_mean_inserts: List[DataInsert] = []
    self._moving_var_inserts: List[DataInsert] = []
    self._beta_inserts: List[DataInsert] = []
    self._gamma_inserts: List[DataInsert] = []

  @property
  def added_out_channel(self) -> int:
    ret = 0
    for insert in self._moving_mean_inserts:
      ret += insert.added_num_channels
    return ret

  @property
  def added_in_channel(self) -> int:
    ret = 0
    for insert in self._moving_mean_inserts:
      ret += insert.added_num_channels
    return ret

  @property
  def moving_mean_inserts(self) -> List[DataInsert]:
    return self._moving_mean_inserts

  @moving_mean_inserts.setter
  def moving_mean_inserts(self, v: List[DataInsert]) -> None:
    self._moving_mean_inserts = v

  @property
  def moving_var_inserts(self) -> List[DataInsert]:
    return self._moving_var_inserts

  @moving_var_inserts.setter
  def moving_var_inserts(self, v: List[DataInsert]) -> None:
    self._moving_var_inserts = v

  @property
  def beta_inserts(self) -> List[DataInsert]:
    return self._beta_inserts

  @beta_inserts.setter
  def beta_inserts(self, v: List[DataInsert]) -> None:
    self._beta_inserts = v

  @property
  def gamma_inserts(self) -> List[DataInsert]:
    return self._gamma_inserts

  @gamma_inserts.setter
  def gamma_inserts(self, v: List[DataInsert]) -> None:
    self._gamma_inserts = v

  @property
  def out_inserts(self) -> List[DataInsert]:
    return self._moving_mean_inserts

  def add_moving_mean_insert(self, weight_insert: DataInsert):
    self._moving_mean_inserts.append(weight_insert)

  def add_moving_var_insert(self, weight_insert: DataInsert):
    self._moving_var_inserts.append(weight_insert)

  def add_beta_insert(self, weight_insert: DataInsert):
    self._beta_inserts.append(weight_insert)

  def add_gamma_insert(self, weight_insert: DataInsert):
    self._gamma_inserts.append(weight_insert)


class InstanceNormStructuredExpanding(StructuredExpanding):
  def __init__(self, node_name: str) -> None:
    super().__init__(node_name)
    self._beta_inserts: List[DataInsert] = []
    self._gamma_inserts: List[DataInsert] = []

  @property
  def added_out_channel(self) -> int:
    ret = 0
    for insert in self._gamma_inserts:
      ret += insert.added_num_channels
    return ret

  @property
  def added_in_channel(self) -> int:
    return self.added_out_channel

  @property
  def beta_inserts(self) -> List[DataInsert]:
    return self._beta_inserts

  @beta_inserts.setter
  def beta_inserts(self, v: List[DataInsert]) -> None:
    self._beta_inserts = v

  @property
  def gamma_inserts(self) -> List[DataInsert]:
    return self._gamma_inserts

  @gamma_inserts.setter
  def gamma_inserts(self, v: List[DataInsert]) -> None:
    self._gamma_inserts = v

  @property
  def out_inserts(self) -> List[DataInsert]:
    return self._gamma_inserts

  def add_beta_insert(self, weight_insert: DataInsert):
    self._beta_inserts.append(weight_insert)

  def add_gamma_insert(self, weight_insert: DataInsert):
    self._gamma_inserts.append(weight_insert)


class GenericStructuredExpanding(StructuredExpanding):
  def __init__(self, node_name: str) -> None:
    super().__init__(node_name)
    self._inserts: List[DataInsert] = []

  @property
  def added_out_channel(self) -> int:
    ret = 0
    for insert in self._inserts:
      ret += insert.added_num_channels
    return ret

  @property
  def added_in_channel(self) -> int:
    return self.added_out_channel

  @property
  def out_inserts(self) -> List[DataInsert]:
    return self._inserts

  def add_insert(self, insert: DataInsert):
    self._inserts.append(insert)
