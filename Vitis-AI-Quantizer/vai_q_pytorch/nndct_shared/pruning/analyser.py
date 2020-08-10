from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from nndct_shared.pruning import logging
from nndct_shared.pruning import pruning_lib

class ModelAnalyser(object):
  """Class for performing analysis on model."""

  NUM_EXPS_PER_GROUP = 9

  def __init__(self, graph):
    self._cur_step = 0

    groups = pruning_lib.group_nodes(graph)
    self._total_steps = len(groups) * ModelAnalyser.NUM_EXPS_PER_GROUP + 1
    self._groups = groups
    self._results = [None for i in range(self._total_steps)]

  def steps(self):
    #while self._cur_step < self._total_steps:
    #  step = self._cur_step
    #  self._cur_step += 1
    #  yield step
    return self._total_steps

  def _group_and_exp(self, step):
    return divmod(step - 1, ModelAnalyser.NUM_EXPS_PER_GROUP)

  def spec(self, step):
    spec = pruning_lib.PruningSpec()
    group_idx, exp = self._group_and_exp(step)
    if group_idx >= 0:
      nodes = self._groups[group_idx]
      spec.add_group(pruning_lib.GroupSparsity(nodes, (exp + 1) * 0.1))
    return spec

  @abc.abstractmethod
  def task(self):
    pass

  def record(self, step, result):
    if step >= self._total_steps:
      raise IndexError

    logging.vlog(3, "step {} recodred as {}".format(step, result))
    self._results[step] = result

  def save(self):
    net_sens = pruning_lib.NetSens()
    start_index = 1
    end_index = start_index + ModelAnalyser.NUM_EXPS_PER_GROUP
    for group in self._groups:
      vals = [self._results[0]] + self._results[start_index:end_index]
      net_sens.groups.append(pruning_lib.GroupSens(group, vals))
      start_index = end_index
      end_index += ModelAnalyser.NUM_EXPS_PER_GROUP
    pruning_lib.write_sens(net_sens, '.ana')
