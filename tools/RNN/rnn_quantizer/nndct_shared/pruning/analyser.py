

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

from nndct_shared.pruning import logging
from nndct_shared.pruning import pruning_lib

class AnaMetric(collections.namedtuple('AnaMetric', ['sparsity', 'value'])):
  pass

class ModelAnalyser(object):
  """Class for performing analysis on model."""

  ExpsPerGroup = 9

  def __init__(self, graph):
    self._cur_step = 0

    groups = pruning_lib.group_nodes(graph)
    self._total_steps = len(groups) * ModelAnalyser.ExpsPerGroup + 1
    self._groups = groups
    self._metrics = [None for i in range(self._total_steps)]

  def steps(self):
    #while self._cur_step < self._total_steps:
    #  step = self._cur_step
    #  self._cur_step += 1
    #  yield step
    return self._total_steps

  def _eval_plan(self, step):
    # Step 0 for baseline.
    if step == 0:
      return 0, None
    group, exp = divmod(step - 1, ModelAnalyser.ExpsPerGroup)
    return group, (exp + 1) * 0.1

  def spec(self, step):
    spec = pruning_lib.PruningSpec()
    if step > 0:
      group_idx, sparsity = self._eval_plan(step)
      nodes = self._groups[group_idx]
      spec.add_group(pruning_lib.PrunableGroup(nodes, sparsity))
    # Empty spec for baseline.
    return spec

  @abc.abstractmethod
  def task(self):
    pass

  def record(self, step, result):
    if step >= self._total_steps:
      raise IndexError

    logging.vlog(3, "step {} recodred as {}".format(step, result))
    _, sparsity = self._eval_plan(step)
    self._metrics[step] = AnaMetric(sparsity, result)

  def save(self):
    net_sens = pruning_lib.NetSens()
    start_index = 1
    end_index = start_index + ModelAnalyser.ExpsPerGroup
    for group in self._groups:
      metrics = [self._metrics[0]] + self._metrics[start_index:end_index]
      net_sens.groups.append(pruning_lib.GroupSens(group, metrics))
      start_index = end_index
      end_index += ModelAnalyser.ExpsPerGroup
    pruning_lib.write_sens(net_sens, '.ana')
