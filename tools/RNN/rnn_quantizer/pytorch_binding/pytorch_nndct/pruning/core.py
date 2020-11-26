

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

import torch.multiprocessing as mp
import os
import shutil
import torch

from collections import OrderedDict

from nndct_shared.nndct_graph.base_node import Node
from nndct_shared.pruning import analyser as ana_lib
from nndct_shared.pruning import logging
from nndct_shared.pruning import pruner as pruner_lib
from nndct_shared.pruning import pruning_lib

from pytorch_nndct import parse
from pytorch_nndct.pruning import summary
from pytorch_nndct.pruning import utils

def parse_to_graph(module, input_specs):
  inputs = utils.dummy_inputs(input_specs)

  parser = parse.TorchParser()
  return parser(module._get_name(), module, *inputs)

class AnaTask(object):

  def __init__(self, module, input_specs, gpu, eval_fn, args=()):
    self.module = module
    self.input_specs = input_specs
    self.gpu = gpu
    self.eval_fn = eval_fn
    self.args = args

    # Must set env variable here, not in __call__ function.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)

  def __call__(self, input_queue, output_queue):
    # We have to reparse graph here to recreate the global variable
    # _NNDCT_OP_2_TORCH_OP. (obviously not a good idea)
    # TorchScriptWriter use this global map to generate the script.
    graph = parse_to_graph(self.module, self.input_specs)
    pruner = pruner_lib.ChannelPruner(graph)
    analyser = ana_lib.ModelAnalyser(graph)
    steps = analyser.steps()

    while not input_queue.empty():
      cur_step = input_queue.get()
      spec = analyser.spec(cur_step)
      pruned_graph, _ = pruner.prune(spec)

      rebuilt_module, _ = utils.rebuild_module(pruned_graph)
      module = rebuilt_module.cuda()
      module.eval()
      score = self.eval_fn(module, *self.args).item()
      output_queue.put((cur_step, score))
      logging.info('Analysis complete %d/%d' % (cur_step + 1, steps))

class Pruner(object):
  """Implements channel pruning at the module level."""

  def __init__(self, module, input_specs):
    """Concrete example:

    ```python
      model = MyModel()
      pruner = Pruner(model, InputSpec(shape=(3, 32, 32), dtype=torch.flot32))
      pruner.ana(ana_eval_fn, args=(val_loader, loss_fn))
      model = pruner.prune(ratio=0.2)
    ```

    Arguments:
      model (Module): Model to be pruned.
      input_specs(tuple or list): The specifications of model inputs.
    """

    self._module = module
    if not isinstance(input_specs, (tuple, list)):
      input_specs = [input_specs]
    self._input_specs = input_specs

    self._graph = parse_to_graph(module, self._input_specs)

  def _prune(self, graph, pruning_spec, output_script=None):
    """Use `ChannelPruner` to perform pruning on the given graph by given spec.

    Arguments:
      graph: A `NndctGraph` to be pruned.
      pruning_spec: A `PruningSpec` object indicates how to prune the model.
      output_script: Filepath that saves the generated script used for
        rebuilding model. If None, then the generated script will be written
        to a tempfile.

    Returns:
      A `torch.nn.Module` object rebuilt from the pruned `NndctGraph` model.
      A pruned nndct graph.
      A dict of `NodePruningResult` that indicates how each node is pruned.
    """

    pruner = pruner_lib.ChannelPruner(graph)
    pruned_graph, pruning_info = pruner.prune(pruning_spec)

    rebuilt_module, filename = utils.rebuild_module(pruned_graph)
    if output_script:
      shutil.move(filename, output_script)

    # NOTE: The current approach relies on TorchParser's implementation.
    # If the tensor's name no longer comes from the original module's
    # state dict key, then the following code will not work.
    for node in pruned_graph.nodes:
      node_pruning = pruning_info[node.name]
      node_pruning.state_dict_keys = []
      for tensor in node.op.params.values():
        # tensor.name -> state_dict_key
        # ResNet::conv1.weight -> conv1.weight
        node_pruning.state_dict_keys.append(
            tensor.name.lstrip(pruned_graph.name + '::'))

    return rebuilt_module, pruned_graph, pruning_info

  def ana(self, eval_fn, args=(), gpus=None):
    """Performs model analysis.
    Arguments:
      eval_fn: Callable object that takes a `torch.nn.Module` object as its
        first argument and returns the evaluation score.
      args: A tuple of arguments that will be passed to eval_fn.
      gpus: A tuple or list of gpu indices used for model analysis. If not set,
        the default gpu will be used.
    """
    num_parallel = len(gpus) if gpus else 1
    if num_parallel > 1:
      self._parallel_ana(gpus, eval_fn, args)
    else:
      self._ana(eval_fn, args)

  def _ana(self, eval_fn, args=()):
    analyser = ana_lib.ModelAnalyser(self._graph)
    steps = analyser.steps()

    for step in range(steps):
      spec = analyser.spec(step)
      model, _, _ = self._prune(self._graph, spec)

      model.eval()
      model = model.cuda()
      score = eval_fn(model, *args).item()
      analyser.record(step, score)
      logging.info('Analysis complete %d/%d' % (cur_step + 1, steps))

    analyser.save()

  def _parallel_ana(self, gpus, eval_fn, args=()):
    graph = parse_to_graph(self._module, self._input_specs)
    analyser = ana_lib.ModelAnalyser(graph)
    steps = analyser.steps()

    mp.set_start_method('spawn')
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    for step in range(steps):
      input_queue.put(step)

    processes = []
    for rank in range(len(gpus)):
      p = mp.Process(
          target=AnaTask(
              self._module, self._input_specs, gpus[rank], eval_fn, args=args),
          args=(input_queue, output_queue))
      p.start()
      processes.append(p)

    for p in processes:
      p.join()

    while not output_queue.empty():
      analyser.record(*output_queue.get())

    analyser.save()

  def prune(self,
            ratio=None,
            threshold=None,
            excludes=[],
            output_script='graph.py'):
    """Prune the network by given ratio or threshold.

      Arguments:
        ratio: The expected percentage of FLOPs reduction. This is just a hint
          value, the actual FLOPs drop not necessarily strictly to this value
          after pruning.
        threshold: Relative proportion of model performance loss
          that can be tolerated.
        excludes: Modules that need to prevent from pruning.
        output_script: Filepath that saves the generated script
          used for rebuilding model.

      Return:
        A `PruningModule` object works like a normal torch.nn.Module with
          addtional pruning info.
    """
    sens_path = pruning_lib.sens_path(self._graph)
    if not os.path.exists(sens_path):
      raise RuntimeError("Must call ana() before runnig prune.")
    net_sens = pruning_lib.read_sens(sens_path)

    excluded_nodes = []
    if excludes:
      module_to_node = utils.map_original_module_to_node(
          self._module, self._graph)
      for module in excludes:
        excluded_nodes.append(module_to_node[id(module)])

    if threshold:
      logging.info('start pruning by threshold = {}'.format(threshold))
      spec = self._spec_from_threshold(net_sens, threshold, excluded_nodes)
    elif ratio:
      logging.info('start pruning by ratio = {}'.format(ratio))
      spec = self._spec_from_ratio(net_sens, ratio, excluded_nodes)
    else:
      raise ValueError("One of 'ratio' or 'threshold' must be given.")

    pruned_model, pruned_graph, pruning_info = self._prune(
        self._graph, spec, output_script)
    return PruningModule(pruned_model, pruned_graph, pruning_info)

  def _spec_from_threshold(self, net_sens, threshold, excludes=[]):
    groups = pruning_lib.prunable_groups_by_threshold(net_sens, threshold,
                                                      excludes)
    spec = pruning_lib.PruningSpec()
    for group in groups:
      spec.add_group(group)
    return spec

  def _spec_from_ratio(self, net_sens, ratio, excludes=[]):
    logging.info('searching for appropriate sparsity for each layer...')

    inputs = utils.dummy_inputs(self._input_specs)
    flops, _ = summary.model_complexity(self._module, inputs)
    expected_flops = (1 - ratio) * flops

    flops_tolerance = 1e-2
    min_th = 1e-5
    max_th = 1 - min_th
    num_attempts = 0
    max_attempts = 100

    prev_spec = None
    cur_spec = None
    while num_attempts < max_attempts:
      prev_spec = cur_spec
      num_attempts += 1
      threshold = (min_th + max_th) / 2
      cur_spec = self._spec_from_threshold(net_sens, threshold, excludes)
      if prev_spec and prev_spec == cur_spec:
        continue

      pruned_model, _, _ = self._prune(self._graph, cur_spec)
      current_flops, _ = summary.model_complexity(pruned_model, inputs)
      error = abs(expected_flops - current_flops) / expected_flops
      if error < flops_tolerance:
        break
      if current_flops < expected_flops:
        max_th = threshold
      else:
        min_th = threshold

    return cur_spec

  def summary(self, pruned_model):
    # TODO: more detailed summary
    inputs = utils.dummy_inputs(self._input_specs)
    orig_flops, orig_params = summary.model_complexity(self._module, inputs)
    pruned_flops, pruned_params = summary.model_complexity(pruned_model, inputs)
    flops_fields = [
        'FLOPs', '{}({})'.format(utils.readable_num(orig_flops), orig_flops),
        '{}({})'.format(utils.readable_num(pruned_flops), pruned_flops)
    ]
    params_fields = [
        'Params', '{}({})'.format(utils.readable_num(orig_params), orig_params),
        '{}({})'.format(utils.readable_num(pruned_params), pruned_params)
    ]

    column_width = []
    for i in range(len(flops_fields)):
      column_width.append(max(len(flops_fields[i]), len(params_fields[i])))

    spaces_between_columns = 1
    current_pos = 0
    column_positions = []
    for i in range(len(column_width)):
      column_positions.append(current_pos + column_width[i] +
                              spaces_between_columns)
      current_pos = column_positions[-1]
    line_length = column_positions[-1]

    def print_row(fields, positions):
      line = ''
      for i in range(len(fields)):
        if i > 0:
          line = line[:-1] + ' '
        line += str(fields[i])
        line = line[:positions[i]]
        line += ' ' * (positions[i] - len(line))
      print(line)

    print('=' * line_length)
    print_row(['Metric', 'Baseline', 'Pruned'], column_positions)
    print('=' * line_length)
    print_row(flops_fields, column_positions)
    print('-' * line_length)
    print_row(params_fields, column_positions)
    print('-' * line_length)

class InputSpec(object):

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

class PruningModule(torch.nn.Module):

  def __init__(self, module, graph, pruning_info):

    super(PruningModule, self).__init__()

    self._module = module
    self._graph = graph
    self._pruning_info = pruning_info

  def forward(self, *inputs, **kwargs):
    return self._module.forward(*inputs, **kwargs)

  def save(self, path):
    torch.save(self.padded_state_dict(), path)

  def state_dict(self, destination=None, prefix='', keep_vars=False):
    if destination is None:
      destination = OrderedDict()

    state_dict = self._module.state_dict(None, prefix, keep_vars)
    for name, value in state_dict.items():
      # The prefix comes from self._module attribute setting  in __init__.
      prefix = '_module.'
      if name.startswith(prefix):
        name = name[len(prefix):]
      destination[name] = value
    return destination

  def padded_state_dict(self):
    destination = OrderedDict()
    module_to_node = utils.map_rebuilt_module_to_node(self._module, self._graph)
    for module_name, module in self._module.named_children():
      node = module_to_node[module_name]
      for key, value in module.state_dict().items():
        # raw param name: weight, bias ...
        param_name = utils.raw_param_name(key)
        for tensor in node.op.params.values():
          if param_name in tensor.name:
            orig_key = tensor.name.lstrip(self._graph.name + '::')
            pruning_info = self._pruning_info.get(node.name, None)
            if pruning_info:
              value = utils.pad_to_sparse_tensor(value, pruning_info)
            destination[orig_key] = value

    return destination

  @property
  def module(self):
    return self._module

  @property
  def pruning_info(self):
    return self._pruning_info
