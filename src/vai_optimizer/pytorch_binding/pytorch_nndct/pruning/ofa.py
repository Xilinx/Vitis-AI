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

import copy
import json
import os
import random
import torch

from nndct_shared.pruning import errors
from nndct_shared.pruning import logging
from nndct_shared.pruning import pruning_lib
from nndct_shared.pruning import utils as nspu

from pytorch_nndct.nn.modules.dynamic_ops import DynamicBatchNorm2d
from pytorch_nndct.nn.modules.dynamic_ops import DynamicConv2d
from pytorch_nndct.nn.modules.dynamic_ops import DynamicConvTranspose2d
from pytorch_nndct.pruning import utils
from pytorch_nndct.utils import module_util as mod_util
from pytorch_nndct.utils import profiler

class OFAPruner(object):
  """Implements once-for-all pruning at the module level."""

  def __init__(self, model, inputs):
    """Concrete example:

    ```python
      inputs = torch.randn([1, 3, 224, 224], dtype=torch.float32).cuda()
      model = MyModel()
      ofa_pruner = OFAPruner(model, inputs)
      ofa_model = ofa_runner.ofa_model([0.5, 1], 8, None)
    ```

    Arguments:
      model (Module): Model to be pruned.
      input_specs(tuple or list): The specifications of model inputs.
    """

    if isinstance(model, torch.nn.DataParallel):
      raise errors.OptimizerDataParallelNotAllowedError(
          'DataParallel object is not allowed.')

    self._model = model
    self._inputs = inputs
    self._to_update_dict_list = []  # all module need to replace
    self._search_space_list = []  # dynamic conv layers search_space_list
    self._candidates_setting_list = []  # group node candidates setting list
    self._dynamic_subnet_setting = []  # dynamic subnet setting
    self._subnet_config = []  # static subnet config list

    self._graph = utils.parse_to_graph(model, inputs)

  def ofa_model(self,
                expand_ratio=[0.5, 0.75, 1],
                channel_divisble=8,
                excludes=None,
                auto_add_excludes=True,
                save_search_space=False):

    # find all nn.Conv2d \ nn.ConvTranspose2d \ nn.BatchNorm2d export excluded module
    model = copy.deepcopy(self._model)
    device = next(model.parameters()).device

    for n, m in model.named_modules():
      to_update_dict = {}
      if isinstance(
          m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d)):
        if excludes and (n in excludes):
          logging.info('Excluded modules:')
          logging.info(n)
        to_update_dict[n] = m
        self._to_update_dict_list.append(to_update_dict)

    logging.info('Modules to be replaced:')
    for i in (self._to_update_dict_list):
      logging.info(i)

    # replace all nn.Conv2d \ nn.ConvTranspose2d \ nn.BatchNorm2d  and reload ckpt
    for idx, to_update_dict in enumerate(self._to_update_dict_list):
      for name, sub_module in to_update_dict.items():

        search_space_dict = {}

        if isinstance(sub_module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
          in_channel_list = [
              nspu.num_remaining_channels(sub_module.in_channels, (1 - i),
                                          channel_divisble)
              for i in expand_ratio
          ]

          out_channel_list = [
              nspu.num_remaining_channels(sub_module.out_channels, (1 - i),
                                          channel_divisble)
              for i in expand_ratio
          ]

          if isinstance(sub_module, torch.nn.Conv2d):
            # make sure  max_cand_kernel_size %2 == 1
            kernel_size_list = []

            if sub_module.kernel_size[0] % 2 > 0 and sub_module.kernel_size[
                0] > 3 and sub_module.kernel_size[0] == sub_module.kernel_size[
                    1] and sub_module.dilation[
                        0] == 1 and sub_module.kernel_size[
                            0] // 2 == sub_module.padding[0]:  # odd number
              for i in range(3, sub_module.kernel_size[0] + 1):
                if i % 2 == 1 and i <= sub_module.kernel_size[0]:
                  kernel_size_list.append((i, i))
            else:
              kernel_size_list.append(sub_module.kernel_size)

            dynamic_modules = DynamicConv2d(
                in_channel_list,
                out_channel_list,
                kernel_size_list,
                sub_module.stride,
                sub_module.padding,
                sub_module.dilation,
                sub_module.groups,
                bias=True if sub_module.bias is not None else False,
            )

          elif isinstance(sub_module, torch.nn.ConvTranspose2d):

            kernel_size_list = [sub_module.kernel_size]

            dynamic_modules = DynamicConvTranspose2d(
                in_channel_list, out_channel_list, kernel_size_list,
                sub_module.stride, sub_module.padding,
                sub_module.output_padding, sub_module.groups,
                True if sub_module.bias is not None else False,
                sub_module.dilation)
          search_space_dict['conv_module_name'] = name
          search_space_dict['dynamic_conv_module_name'] = name
          search_space_dict['dynamic_conv_module_searchspace'] = {
              'name': dynamic_modules.__class__.__name__,
              'candidate_in_channel_list': in_channel_list,
              'candidate_out_channel_list': out_channel_list,
              'candidate_kernel_size_list': kernel_size_list,
          }
          self._search_space_list.append(search_space_dict)

        elif isinstance(sub_module, torch.nn.BatchNorm2d):

          dynamic_modules = DynamicBatchNorm2d(sub_module.num_features,
                                               sub_module.eps,
                                               sub_module.momentum,
                                               sub_module.affine,
                                               sub_module.track_running_stats)

        mod_util.replace_modules(model, name, dynamic_modules, copy_ckpt=True)

    self.get_config_list(excludes, auto_add_excludes)

    if save_search_space:
      self.save_subnet_config(self._candidates_setting_list,
                              'searchspace.config')

    logging.info('OFA module candidates setting (search space) list:')
    for i in self._candidates_setting_list:
      logging.info(i)

    return model.to(device)

  def reset_search_space(self, searchspace):
    for index, config in enumerate(searchspace):
      comp_output_channel = -1
      for idx, c in enumerate(config):
        for (key, value) in searchspace[index][idx].items():
          if key in self._candidates_setting_list[index][idx]:
            if isinstance(
                value,
                list) and (set(self._candidates_setting_list[index][idx][key])
                           >= set(value) == False):
              logging.error(searchspace[index][idx])
              logging.error(self._candidates_setting_list[index][idx])

              logging.error(
                  'reset_candidate_kernel_size_list / reset_candidate_out_channel_list should be subset of candidate_kernel_size_list / candidate_out_channel_list. Please check searchspace !'
              )
              return False

            elif not isinstance(
                value, list
            ) and self._candidates_setting_list[index][idx][key] != value:
              logging.error(searchspace[index][idx])
              logging.error(self._candidates_setting_list[index][idx])
              logging.error(
                  'The value of searchspace is error. Please check searchspace !'
              )
              return False
          else:
            logging.error(searchspace[index][idx])
            logging.error(self._candidates_setting_list[index][idx])
            logging.error(
                'The key of searchspace is error. Please check searchspace !')
            return False

        if idx == 0:
          comp_output_channel = searchspace[index][idx][
              'candidate_out_channel_list']
        else:
          if comp_output_channel != searchspace[index][idx][
              'candidate_out_channel_list']:
            logging.error(searchspace[index][idx])
            logging.error(self._candidates_setting_list[index][idx])
            logging.error(
                'The modules in the same group should set same candidate_out_channel_list. Please check searchspace !'
            )
            return False

    self._candidates_setting_list = searchspace

  def save_subnet_config(self, setting_config, file_name):
    # save candidates setting list 'searchspace.config'
    json.dump(setting_config, open(file_name, 'w'), indent=4)

  def load_subnet_config(self, file_name):
    # load candidates setting list
    setting_config = json.load(open(file_name))
    return setting_config

  def find_new_module(self, oldmodule_name):
    for search_space in self._search_space_list:
      if oldmodule_name == search_space['conv_module_name']:
        return search_space

  def get_config_list(self, excludes=None, auto_add_excludes=True):
    """Group nodes and remove excluded nodes. Get candidates setting list."""

    excluded_nodes = self._get_exclude_nodes(excludes) if excludes else []

    groups, is_mobile = pruning_lib.group_nodes_for_ofa_dynamic_conv(
        self._graph)

    first_conv_nodes, last_conv_nodes = pruning_lib.find_leaf_node(self._graph)

    if auto_add_excludes:
      for leaf_node in (first_conv_nodes + last_conv_nodes):
        leaf_module = mod_util.module_name_from_node(leaf_node)
        if leaf_module is not None:
          excluded_nodes.append(leaf_module)

    nodes_to_exclude = []

    # exclude node
    for index, group in enumerate(groups):
      skip = False
      if len(group) > 0:
        for excluded_node in excluded_nodes:
          for g in group:
            if excluded_node == mod_util.module_name_from_node(g):
              skip = True
              break
        if not skip:
          nodes_to_exclude.append(sorted(group))

    if len(nodes_to_exclude) == 0:
      logging.info('No module can be pruned!')
      return

    nodes_to_exclude = sorted(nodes_to_exclude)

    for group in nodes_to_exclude:
      comp_config_list = []
      for index, node in enumerate(group):
        comp_attr_config = {}
        oldmodule_name = mod_util.module_name_from_node(node)
        search_space = self.find_new_module(oldmodule_name)
        dynamic_conv_module_name = search_space['dynamic_conv_module_name']
        candidate_out_channel_list = search_space[
            'dynamic_conv_module_searchspace']['candidate_out_channel_list']
        candidate_kernel_size_list = search_space[
            'dynamic_conv_module_searchspace']['candidate_kernel_size_list']
        # keep last conv output channel
        for last_conv_node in last_conv_nodes:
          if node is last_conv_node.name:
            attrs = {
                name: last_conv_node.op.get_config(name)
                for name in last_conv_node.op.configs
            }
            candidate_out_channel_list = [attrs['out_channels']]
            break

        # Shrinking search space of 1x1 conv
        if is_mobile == 0 and len(candidate_out_channel_list) > 1:
          reset_searchspace = 0
          for n in group:
            oldmodule_name_tmp = mod_util.module_name_from_node(n)
            search_space_tmp = self.find_new_module(oldmodule_name_tmp)
            candidate_kernel_size_list_tmp = search_space_tmp[
                'dynamic_conv_module_searchspace']['candidate_kernel_size_list']
            if min(candidate_kernel_size_list_tmp[0]) == 1:
              reset_searchspace = 1
              break
          if reset_searchspace == 1:
            candidate_out_channel_list = sorted(
                candidate_out_channel_list)[len(candidate_out_channel_list) //
                                            2:]

        comp_attr_config['dynamic_conv_module_name'] = dynamic_conv_module_name
        comp_attr_config[
            'candidate_out_channel_list'] = candidate_out_channel_list
        comp_attr_config['active_out_channel'] = -1
        comp_attr_config[
            'candidate_kernel_size_list'] = candidate_kernel_size_list
        comp_attr_config['active_kernel_size'] = -1

        comp_config_list.append(comp_attr_config)
      self._candidates_setting_list.append(comp_config_list)

  def sample_subnet(self, model, kind):
    if kind not in ['random', 'max', 'min']:
      raise errors.OptimizerInvalidArgumentError(
          'Subnet must be one of "random", "max" and "min"')

    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DataParallel,
                          torch.nn.parallel.DistributedDataParallel)):
      model_ = model.module
    else:
      model_ = model

    self._dynamic_subnet_setting = copy.deepcopy(self._candidates_setting_list)

    for index, config in enumerate(self._candidates_setting_list):
      comp_output_channel = -1
      for idx, c in enumerate(config):
        if idx == 0:
          if kind == 'random':
            random_kernel = random.choice(c['candidate_kernel_size_list'])
            comp_output_channel = random.choice(c['candidate_out_channel_list'])
          elif kind == 'min':
            random_kernel = min(c['candidate_kernel_size_list'])
            comp_output_channel = min(c['candidate_out_channel_list'])
          elif kind == 'max':
            random_kernel = max(c['candidate_kernel_size_list'])
            comp_output_channel = max(c['candidate_out_channel_list'])

          mod_util.set_module(
              model_, c['dynamic_conv_module_name'] + '.active_kernel_size',
              random_kernel)
          mod_util.set_module(
              model_, c['dynamic_conv_module_name'] + '.active_out_channel',
              comp_output_channel)

          self._dynamic_subnet_setting[index][idx][
              'active_kernel_size'] = random_kernel
          self._dynamic_subnet_setting[index][idx][
              'active_out_channel'] = comp_output_channel

        else:
          if kind == 'random':
            random_kernel = random.choice(c['candidate_kernel_size_list'])
          elif kind == 'min':
            random_kernel = min(c['candidate_kernel_size_list'])
          elif kind == 'max':
            random_kernel = max(c['candidate_kernel_size_list'])

          mod_util.set_module(
              model_, c['dynamic_conv_module_name'] + '.active_kernel_size',
              random_kernel)
          mod_util.set_module(
              model_, c['dynamic_conv_module_name'] + '.active_out_channel',
              comp_output_channel)

          self._dynamic_subnet_setting[index][idx][
              'active_kernel_size'] = random_kernel
          self._dynamic_subnet_setting[index][idx][
              'active_out_channel'] = comp_output_channel

    return model, self._dynamic_subnet_setting

  def check_dynamic_subnet_setting(self, dynamic_subnet_setting):

    for index, config in enumerate(dynamic_subnet_setting):
      comp_output_channel = -1
      for idx, c in enumerate(config):
        if dynamic_subnet_setting[index][idx]['active_kernel_size'] not in dynamic_subnet_setting[index][idx]['candidate_kernel_size_list'] or \
            dynamic_subnet_setting[index][idx]['active_out_channel'] not in dynamic_subnet_setting[index][idx]['candidate_out_channel_list']:

          logging.error(
              'active_kernel_size / active_out_channel should in candidate_kernel_size_list / candidate_out_channel_list. Please check dynamic_subnet_setting !'
          )
          return False

        if idx == 0:
          comp_output_channel = dynamic_subnet_setting[index][idx][
              'active_out_channel']
        else:
          if comp_output_channel != dynamic_subnet_setting[index][idx][
              'active_out_channel']:
            logging.error(
                'The modules in the same group should set same active_out_channel. Please check dynamic_subnet_setting !'
            )
            return False

    return True

  def get_static_subnet(self, dynamic_subnet, dynamic_subnet_setting=None):
    self._subnet_config.clear()  # clear memory
    model = copy.deepcopy(dynamic_subnet)

    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DataParallel,
                          torch.nn.parallel.DistributedDataParallel)):
      model_ = model.module
    else:
      model_ = model

    sum_k = 0
    sum_c = 0

    if dynamic_subnet_setting is not None:
      assert self.check_dynamic_subnet_setting(dynamic_subnet_setting)

      for index, config in enumerate(dynamic_subnet_setting):
        for idx, c in enumerate(config):

          sum_k = sum_k + max(
              dynamic_subnet_setting[index][idx]['active_kernel_size'])
          sum_c = sum_c + dynamic_subnet_setting[index][idx][
              'active_out_channel']

          mod_util.set_module(
              model_,
              dynamic_subnet_setting[index][idx]['dynamic_conv_module_name'] +
              '.active_kernel_size',
              dynamic_subnet_setting[index][idx]['active_kernel_size'])
          mod_util.set_module(
              model_,
              dynamic_subnet_setting[index][idx]['dynamic_conv_module_name'] +
              '.active_out_channel',
              dynamic_subnet_setting[index][idx]['active_out_channel'])

    model_.eval()
    model_(self._inputs)
    model_.train()

    self._subnet_config.append({'net_id': f'net_sum_k_{sum_k}_sum_c_{sum_c}'})

    for idx, to_update_dict in enumerate(self._to_update_dict_list):
      subnet_config = {}
      for name, sub_module in to_update_dict.items():
        if isinstance(sub_module, torch.nn.Conv2d):
          subnet_config['module_name'] = name
          subnet_config['module_type'] = 'Conv2d'
          subnet_config['in_channels'] = mod_util.get_module(
              model_, name + '.active_in_channel')
          subnet_config['out_channels'] = mod_util.get_module(
              model_, name + '.active_out_channel')
          subnet_config['kernel_size'] = mod_util.get_module(
              model_, name + '.active_kernel_size')
          subnet_config['stride'] = sub_module.stride
          subnet_config['padding'] = mod_util.get_module(
              model_, name + '.padding')
          subnet_config['dilation'] = sub_module.dilation
          subnet_config['groups'] = mod_util.get_module(model_,
                                                        name + '.groups')
          subnet_config['bias'] = mod_util.get_module(model_, name + '.bias')
        elif isinstance(sub_module, torch.nn.ConvTranspose2d):
          subnet_config['module_name'] = name
          subnet_config['module_type'] = 'ConvTranspose2d'
          subnet_config['in_channels'] = mod_util.get_module(
              model_, name + '.active_in_channel')
          subnet_config['out_channels'] = mod_util.get_module(
              model_, name + '.active_out_channel')
          subnet_config['kernel_size'] = mod_util.get_module(
              model_, name + '.active_kernel_size')
          subnet_config['stride'] = sub_module.stride
          subnet_config['padding'] = mod_util.get_module(
              model_, name + '.padding')
          subnet_config['output_padding'] = sub_module.output_padding
          subnet_config['groups'] = mod_util.get_module(model_,
                                                        name + '.groups')
          subnet_config['bias'] = mod_util.get_module(model_, name + '.bias')
          subnet_config['dilation'] = sub_module.dilation
        elif isinstance(sub_module, torch.nn.BatchNorm2d):
          subnet_config['module_name'] = name
          subnet_config['module_type'] = 'BatchNorm2d'
          subnet_config['num_features'] = mod_util.get_module(
              model_, name + '.num_features')
          subnet_config['eps'] = sub_module.eps
          subnet_config['momentum'] = sub_module.momentum
          subnet_config['affine'] = sub_module.affine
          subnet_config['track_running_stats'] = sub_module.track_running_stats

      self._subnet_config.append(subnet_config)

    for indx, config in enumerate(self._subnet_config):
      if indx > 0:  # excluded net_id
        if config["module_type"] == 'Conv2d':
          modules = torch.nn.Conv2d(
              config['in_channels'],
              config['out_channels'],
              config['kernel_size'],
              config['stride'],
              config['padding'],
              config['dilation'],
              config['groups'],
              config['bias'],
          )
        elif config["module_type"] == 'ConvTranspose2d':
          modules = torch.nn.ConvTranspose2d(
              config['in_channels'],
              config['out_channels'],
              config['kernel_size'],
              config['stride'],
              config['padding'],
              config['output_padding'],
              config['groups'],
              config['bias'],
              config['dilation'],
          )
        elif config["module_type"] == 'BatchNorm2d':
          modules = torch.nn.BatchNorm2d(
              config['num_features'],
              config['eps'],
              config['momentum'],
              config['affine'],
              config['track_running_stats'],
          )

        mod_util.replace_modules(
            model_, config["module_name"], modules, copy_ckpt=True)

    macs, params = profiler.model_complexity(
        model_, self._inputs, readable=False)

    macs = macs / 1e6
    params = params / 1e6

    return model, self._subnet_config, macs, params

  def sample_random_subnet_within_range(self, model, min_macs, max_macs):
    while True:
      dynamic_subnet, dynamic_subnet_setting = self.sample_subnet(
          model, 'random')
      static_subnet, _, macs, params = self.get_static_subnet(dynamic_subnet)

      if macs >= min_macs and macs <= max_macs:
        return macs, dynamic_subnet_setting

  def sample_random_subnet_mutate_and_reset(self,
                                            dynamic_subnet_setting,
                                            prob=0.1):

    assert self.check_dynamic_subnet_setting(dynamic_subnet_setting)

    cfg = copy.deepcopy(dynamic_subnet_setting)
    pick_another = lambda x, candidates: x if len(candidates) == 1 or len(
        set(candidates)) == 1 else random.choice(
            [v for v in candidates if v != x])

    for index, config in enumerate(dynamic_subnet_setting):
      comp_output_channel = -1
      r = random.random()
      if r < prob:
        for idx, c in enumerate(config):
          if idx == 0:
            random_kernel = pick_another(cfg[index][idx]['active_kernel_size'],
                                         c['candidate_kernel_size_list'])
            comp_output_channel = pick_another(
                cfg[index][idx]['active_out_channel'],
                c['candidate_out_channel_list'])

            cfg[index][idx]['active_kernel_size'] = random_kernel
            cfg[index][idx]['active_out_channel'] = comp_output_channel

          else:
            random_kernel = pick_another(cfg[index][idx]['active_kernel_size'],
                                         c['candidate_kernel_size_list'])

            cfg[index][idx]['active_kernel_size'] = random_kernel
            cfg[index][idx]['active_out_channel'] = comp_output_channel

    return cfg

  def sample_random_subnet_crossover_and_reset(self, cfg1, cfg2, p=0.5):

    def cross_helper_fun(g1, g2, prob):
      assert type(g1) == type(g2)
      if isinstance(g1, (int, tuple)):
        return g1 if random.random() < prob else g2
      elif isinstance(g1, list):
        return [v1 if random.random() < prob else v2 for v1, v2 in zip(g1, g2)]

    cfg = copy.deepcopy(cfg1)

    for index, config in enumerate(self._candidates_setting_list):
      comp_output_channel = -1
      for idx, c in enumerate(config):
        if idx == 0:
          random_kernel = cross_helper_fun(
              cfg1[index][idx]['active_kernel_size'],
              cfg2[index][idx]['active_kernel_size'], p)
          comp_output_channel = cross_helper_fun(
              cfg1[index][idx]['active_out_channel'],
              cfg2[index][idx]['active_out_channel'], p)

          cfg[index][idx]['active_kernel_size'] = random_kernel
          cfg[index][idx]['active_out_channel'] = comp_output_channel

        else:
          random_kernel = cross_helper_fun(
              cfg1[index][idx]['active_kernel_size'],
              cfg2[index][idx]['active_kernel_size'], p)

          cfg[index][idx]['active_kernel_size'] = random_kernel
          cfg[index][idx]['active_out_channel'] = comp_output_channel

    return cfg

  def reset_bn_running_stats_for_calibration(self, model):
    for n, m in model.named_modules():
      if isinstance(m, torch.nn.BatchNorm2d) or isinstance(
          m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.SyncBatchNorm):
        m.training = True
        m.momentum = None
        m.reset_running_stats()

  def _get_exclude_nodes(self, excludes):
    return utils.excluded_node_names(self._model, self._graph, excludes)

  def run_evolutionary_search(self,
                              model,
                              calibration_fn,
                              calib_args,
                              eval_fn,
                              eval_args,
                              evaluation_metric,
                              min_or_max_metric,
                              min_macs,
                              max_macs,
                              macs_step=10,
                              parent_popu_size=16,
                              iteration=10,
                              mutate_size=8,
                              mutate_prob=0.2,
                              crossover_size=4):

    if isinstance(evaluation_metric, str) == False:
      evaluation_metric = str(evaluation_metric)

    assert min_or_max_metric in ['max', 'min']

    parent_popu = []

    for idx in range(parent_popu_size):
      macs_subnet = 0
      if idx == 0:
        dynamic_subnet, dynamic_subnet_setting = self.sample_subnet(
            model, 'min')
        subnet, _, macs, params = self.get_static_subnet(
            dynamic_subnet, dynamic_subnet_setting)
        macs_subnet = int(macs)
      else:
        macs, dynamic_subnet_setting = self.sample_random_subnet_within_range(
            model, min_macs, max_macs)
        macs_subnet = int(macs)
      dynamic_subnet_setting.append(
          {'net_id': f'net_evo_0_{idx}_macs_{macs_subnet}'})
      parent_popu.append(dynamic_subnet_setting)

    pareto_global = {}

    for evo in range(iteration):
      subnets_to_be_evaluated = {}
      for cfg in parent_popu:
        if cfg[-1]['net_id'].startswith(f'net_'):
          subnets_to_be_evaluated[cfg[-1]['net_id']] = cfg[:-1]

      eval_results = []

      for net_id in subnets_to_be_evaluated:
        dynamic_subnet_setting = subnets_to_be_evaluated[net_id]
        static_subnet, _, macs, params = self.get_static_subnet(
            model, dynamic_subnet_setting)

        calibration_fn(static_subnet, *(calib_args))
        metric = eval_fn(static_subnet, *(eval_args))

        summary = {
            'net_id': net_id,
            'mode': 'evaluate',
            evaluation_metric: metric,
            'macs': macs,
            'params': params,
            'subnet_setting': dynamic_subnet_setting
        }

        logging.info(summary)

        eval_results += [summary]

      for cfg in eval_results:
        f = round(cfg['macs'] / macs_step) * macs_step
        if min_or_max_metric == 'max':
          if str(f) not in pareto_global or pareto_global[str(
              f)][evaluation_metric] < cfg[evaluation_metric]:
            pareto_global[str(f)] = cfg  # net config + resulut config
        else:
          if str(f) not in pareto_global or pareto_global[str(
              f)][evaluation_metric] > cfg[evaluation_metric]:
            pareto_global[str(f)] = cfg  # net config + resulut config

      parent_popu = []
      # mutate
      for idx in range(mutate_size):
        while True:
          old_cfg = random.choice(list(
              pareto_global.values()))['subnet_setting']
          cfg = self.sample_random_subnet_mutate_and_reset(
              old_cfg, prob=mutate_prob)
          subnet, _, macs, params = self.get_static_subnet(model, cfg)
          if macs >= min_macs and macs <= max_macs:
            break
        cfg.append({'net_id': f'net_evo_{evo}_mutate_{idx}'})
        parent_popu.append(cfg)

      # cross over
      for idx in range(crossover_size):
        while True:
          cfg1 = random.choice(list(pareto_global.values()))['subnet_setting']
          cfg2 = random.choice(list(pareto_global.values()))['subnet_setting']
          cfg = self.sample_random_subnet_crossover_and_reset(cfg1, cfg2)
          subnet, _, macs, params = self.get_static_subnet(model, cfg)
          if macs >= min_macs and macs <= max_macs:
            break
        cfg.append({'net_id': f'net_evo_{evo}_crossover_{idx}'})
        parent_popu.append(cfg)

    return pareto_global
