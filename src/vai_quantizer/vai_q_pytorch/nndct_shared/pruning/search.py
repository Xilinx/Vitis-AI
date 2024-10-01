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

import collections
import json
import os

from nndct_shared.pruning.pruning_lib import PruningSpec, NodeGroup
from nndct_shared.pruning import errors
from nndct_shared.utils import io
from typing import List

class SubnetConfig(
    collections.namedtuple('SubnetConfig', ['ratios', 'score', 'macs'])):

  #def __init__(self, ratios, score):
  #  self.ratios = ratios
  #  self.score = score

  def serialize(self):
    return {'ratios': self.ratios, 'score': self.score, 'macs': self.macs}

  @classmethod
  def deserialize(cls, config):
    return cls(config['ratios'], config['score'], config['macs'])

  def __str__(self):
    return str(self.serialize())

class SubnetSearcher(object):

  def __init__(self, groups: List[NodeGroup]):
    self._groups = groups
    self._supernet = None
    self._subnets = []

  def set_supernet(self, score, macs=None):
    self._supernet = SubnetConfig(None, score, macs)

  def add_subnet(self, ratios, score, macs=None):
    self._subnets.append(SubnetConfig(ratios, score, macs))

  def _sorted_subnet(self):
    return sorted(self._subnets, key=lambda x: x.score)

  def subnet(self, index):
    if not self._subnets:
      raise errors.OptimizerSubnetError('No subnet candidates.')
    if index and index >= len(self._subnets):
      raise errors.OptimizerInvalidArgumentError(
          'Subnet index is out of range [0, {}]'.format(len(self._subnets) - 1))
    return self._subnets[index]

  def best_subnet(self):
    subnets = self._sorted_subnet()
    base_score = self._supernet.score
    b0 = abs(base_score - subnets[0].score)
    b1 = abs(base_score - subnets[-1].score)
    index = 0 if b0 < b1 else -1
    return subnets[index]

  def spec(self, ratios):
    return PruningSpec.from_node_groups(self._groups, ratios)

  @property
  def groups(self):
    return self._groups

  @property
  def config(self):
    return self._config

  @property
  def supernet(self):
    return self._supernet

  @supernet.setter
  def supernet(self, supernet):
    self._supernet = supernet

  def serialize(self):
    subnets = self._sorted_subnet()
    ratios = []
    macs = []
    scores = {}
    for index, subnet in enumerate(subnets):
      config = subnet.serialize()
      ratios.append(','.join([str(ratio) for ratio in subnet.ratios]))
      macs.append(subnet.macs)
      scores[index] = subnet.score
    return {
        'groups': [g.serialize() for g in self._groups],
        'supernet': self._supernet.serialize(),
        'ratios': ratios,
        'macs': macs,
        'scores': scores
    }

  @classmethod
  def deserialize(cls, data):
    instance = cls([NodeGroup.deserialize(item) for item in data['groups']])
    scores = data['scores']
    macs = data['macs']
    for index, ratio_str in enumerate(data['ratios']):
      ratio_strs = ratio_str.split(',')
      ratios = [float(ratio) for ratio in ratio_strs]
      instance.add_subnet(ratios, scores[str(index)], macs[index])
    instance.supernet = SubnetConfig.deserialize(data['supernet'])
    return instance

  def to_json(self):
    return json.dumps(self.serialize(), indent=2)

def save_searcher(searcher, filepath):
  io.create_work_dir(os.path.dirname(filepath))
  with open(filepath, 'w') as f:
    f.write(searcher.to_json())

def load_searcher(filepath):
  with open(filepath, 'r') as f:
    return SubnetSearcher.deserialize(json.load(f))
