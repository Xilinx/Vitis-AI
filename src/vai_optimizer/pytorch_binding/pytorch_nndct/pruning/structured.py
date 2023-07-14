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
import copy
import json
import numpy as np
import os
import random
import torch
import torch.multiprocessing as mp
import types

from typing import List

from nndct_shared.base import NNDCT_OP as OpTypes
from nndct_shared.pruning import errors
from nndct_shared.pruning import logging
from nndct_shared.pruning import pruner as pruner_lib
from nndct_shared.pruning import pruning_lib
from nndct_shared.pruning import search
from nndct_shared.pruning import sensitivity as sens
from nndct_shared.pruning import utils as spu
from nndct_shared.pruning.pruning_lib import is_grouped_conv, is_depthwise_conv
from nndct_shared.pruning.utils import generate_indices_group
from nndct_shared.utils import common

from pytorch_nndct import parse
from pytorch_nndct.pruning import utils
from pytorch_nndct.utils import module_util as mod_util
from pytorch_nndct.utils import profiler
from pytorch_nndct.utils import torch_const
from pytorch_nndct.utils.calibration import calibrate_sens, calibrate_spec

_WEIGHTED_OPS = [
    OpTypes.CONV2D,
    OpTypes.CONV3D,
    OpTypes.CONVTRANSPOSE2D,
    OpTypes.CONVTRANSPOSE3D,
    OpTypes.DEPTHWISE_CONV2D,
    OpTypes.DEPTHWISE_CONV3D,
    OpTypes.DEPTHWISE_CONVTRANSPOSE2D,
    OpTypes.DEPTHWISE_CONVTRANSPOSE3D,
    OpTypes.DENSE,
    OpTypes.BATCH_NORM,
    OpTypes.INSTANCE_NORM,
]

_CONVTRANSPOSE_OPS = [OpTypes.CONVTRANSPOSE2D, OpTypes.CONVTRANSPOSE3D]

_VAI_DIR = '.vai'

class AnaTask(object):

  def __init__(self,
               model,
               inputs,
               excludes,
               gpu,
               eval_fn,
               eval_fn_args=(),
               with_group_conv: bool = False):
    self.model = model
    self.inputs = inputs
    self.excludes = excludes
    self.gpu = gpu
    self.eval_fn = eval_fn
    self.eval_fn_args = eval_fn_args
    self._with_group_conv = with_group_conv

    os.environ['CUDA_VISIBLE_DEVICES'] = str(utils.get_actual_device(gpu))

  def __call__(self, input_queue, output_queue):
    # We have to reparse the nndct graph here to recreate
    # _NNDCT_OP_2_TORCH_OP, as TorchScriptWriter use this global map to
    # generate script for rebuilding.
    pruner = PruningRunner(self.model, self.inputs)

    analyser = sens.ModelAnalyser(pruner._graph, self.excludes,
                                  self._with_group_conv)
    total_steps = analyser.steps()

    while not input_queue.empty():
      cur_step = input_queue.get()
      spec = analyser.spec(cur_step)
      model, pruned_graph, pruning_info = pruner._prune(spec, mode='sparse')

      model.eval()
      model.cuda()
      score = self.eval_fn(model, *self.eval_fn_args)
      if isinstance(score, torch.Tensor):
        score = score.item()
      output_queue.put((cur_step, score))
      logging.info('Analysis complete %d/%d' % (cur_step + 1, total_steps))

class PruningRunner(object):
  """Implements channel pruning at the model level."""

  def __init__(self, model, input_signature):
    """
    model: Model for pruning.
    input_signature: Input needed by pruning.
    """
    self._model = model
    self._input_signature = input_signature

    self._graph = utils.parse_to_graph(model, input_signature,
                                       utils.is_debug_mode())
    self._node_name_to_module_name = {}

  def _prune(self, pruning_spec, mode='sparse'):
    """Perform pruning on the given graph by given spec.

    Arguments:
      pruning_spec: A `PruningSpec` object indicates how to prune the model.
      mode: 'sparse' or 'slim' mode.

    Returns:
      A `torch.nn.Module` object modified by pruning info from original model.
      A pruned nndct graph.
      A dict of `StructuredPruning` that indicates how each node is pruned.
    """
    assert mode in ['sparse', 'slim']
    pruner = pruner_lib.ChannelPruner(self._graph)
    pruned_graph, pruning_info = pruner.prune(
        calibrate_spec(pruning_spec, self._graph))
    func = self._generate_sparse_model if mode == 'sparse' else self._generate_slim_model
    model = func(pruned_graph, pruning_info)

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
            tensor.name.lstrip(pruned_graph.name +
                               torch_const.TorchGraphSymbol.GRAPH_SCOPE_SYM))

    return model, pruned_graph, pruning_info

  def _generate_slim_model(self, pruned_graph, pruning_info):
    """Generate pruned slim model by replace original modules with slim modules."""
    model = copy.deepcopy(self._model)
    for node in pruned_graph.nodes:
      node_pruning = pruning_info[node.name]
      if not node_pruning.removed_outputs and not node_pruning.removed_inputs:
        continue
      if node.op.type not in _WEIGHTED_OPS:
        if len(list(node.op.params)) != 0:
          raise errors.OptimizerUnSupportedOpError(
              'Unsupported op with parameters ({})'.format(node.name))
        continue

      module_name = mod_util.module_name_from_node(node)
      if not module_name:
        raise errors.OptimizerTorchModuleError(
            ('Prunable operation "{}" must be instance of '
             '"torch.nn.Module" (Node: {})').format(node.op.type, node.name))
      module = mod_util.get_module(model, module_name)
      pruned_module = mod_util.create_module_by_node(type(module), node)
      pruned_module.to(next(module.parameters()).device)
      mod_util.replace_modules(model, module_name, pruned_module)
      if node.name not in self._node_name_to_module_name:
        self._node_name_to_module_name[node.name] = module_name

    return model

  def _generate_sparse_model(self, pruned_graph, pruning_info):
    model = copy.deepcopy(self._model)
    state_dict = model.state_dict()
    for node in self._graph.nodes:
      node_pruning = pruning_info[node.name]
      if not node_pruning.removed_outputs and not node_pruning.removed_inputs:
        continue

      if node.op.type not in _WEIGHTED_OPS:
        if len(list(node.op.params)) != 0:
          raise errors.OptimizerUnSupportedOpError(
              'Unsupported op with parameters ({})'.format(node.name))
        continue

      module_name = mod_util.module_name_from_node(node)
      if not module_name:
        raise errors.OptimizerTorchModuleError(
            ('Prunable operation "{}" must be instance of '
             '"torch.nn.Module" (Node: {})').format(node.op.type, node.name))
      module = mod_util.get_module(model, module_name)
      for param, tensor in node.op.params.items():
        key = utils.state_dict_key_from_tensor(tensor)
        name = key.split('.')[-1]

        weight = state_dict[key]
        mask = torch.ones_like(weight)

        out_dims = node_pruning.removed_outputs
        in_dims = node_pruning.removed_inputs

        # Conv2d: [out_channels, in_channels // groups, *kernel_size]
        # DepthwiseConv2d: [out_channels, 1, *kernel_size]
        # ConvTranspose2d: [in_channels, out_channels // groups, *kernel_size]
        # DepthwiseConvTranspose2d: [in_channels, 1, *kernel_size]
        # No need to swap out/in dim for DepthwiseConv and DepthwiseConvTranspose.
        if (node.op.type in _CONVTRANSPOSE_OPS and
            param == node.op.ParamName.WEIGHTS):
          out_dims, in_dims = in_dims, out_dims
        # Depthwise conv's in_dim is 1.
        elif pruning_lib.is_depthwise_conv(node.op) and \
            param == node.op.ParamName.WEIGHTS:
          in_dims = []

        getattr(module, name).data = _sparsify_tensor(
            weight, out_dims, in_dims, node.op.attr['group'] if
            is_grouped_conv(node.op) and not is_depthwise_conv(node.op) else 1)
        logging.vlog(
            1, 'Register buffer for {}: {}'.format(module_name, name + '_mask'))
        module.register_buffer(
            name + '_mask',
            _sparsify_tensor(
                mask, out_dims, in_dims,
                node.op.attr['group'] if is_grouped_conv(node.op) and
                not is_depthwise_conv(node.op) else 1))

        tensor_names = getattr(module, '_tensor_names', [])
        tensor_names.append(name)
        setattr(module, '_tensor_names', tensor_names)
        module.register_forward_pre_hook(_apply_mask)

    return model

  def _get_exclude_nodes(self, excludes):
    return utils.excluded_node_names(self._model, self._graph, excludes)

  def _summary(self, pruned_model):
    orig_macs, orig_params = profiler.model_complexity(
        self._model, self._input_signature, readable=True)
    pruned_macs, pruned_params = profiler.model_complexity(
        pruned_model, self._input_signature, readable=True)

    header_fields = ['Metric', 'Baseline', 'Pruned']
    macs_fields = ['MACs', orig_macs, pruned_macs]
    params_fields = ['Params', orig_params, pruned_params]

    common.print_table(header_fields, [macs_fields, params_fields])

  def generate_pruning_info(self, filename, pruning_info):
    pruner_lib.save_pruning_info(self._node_name_to_module_name, pruning_info,
                                 filename)

class IterativePruningRunner(PruningRunner):

  def __init__(self, model, input_signature):
    super(IterativePruningRunner, self).__init__(model, input_signature)

    self._sens_path = os.path.join(_VAI_DIR, self._graph.name + '.sens')

  def _load_analysis_result(self):
    return sens.load_sens(self._sens_path) if os.path.exists(
        self._sens_path) else None

  def ana(self,
          eval_fn,
          args=(),
          gpus=None,
          excludes=None,
          forced=False,
          with_group_conv: bool = False):
    """Performs model analysis.
    Arguments:
      eval_fn: Callable object that takes a `torch.nn.Module` object as its
        first argument and returns the evaluation score.
      args: A tuple of arguments that will be passed to eval_fn.
      gpus: A tuple or list of gpu indices used for model analysis. If not set,
        the default gpu will be used.
      excludes: A list of node name or torch module to be excluded from pruning.
    """
    net_sens = None if forced else self._load_analysis_result()
    excluded_nodes = self._get_exclude_nodes(excludes) if excludes else []
    analyser = sens.ModelAnalyser(self._graph, excluded_nodes, with_group_conv)

    if net_sens:
      analyser.recover_state(calibrate_sens(net_sens, self._graph))
    uncompleted_steps = analyser.uncompleted_steps()
    if len(uncompleted_steps) == 0:
      logging.info(
          'Skip model analysis and use cached result, if you do not want to use it, set forced=True'
      )
      return

    gpu = gpus[0] if gpus else 0
    self._ana_pre_check(gpu, eval_fn, args, excluded_nodes, with_group_conv)
    num_parallel = len(gpus) if gpus else 1
    if num_parallel > 1:
      self._parallel_ana(gpus, eval_fn, args, analyser, excluded_nodes,
                         with_group_conv)
    else:
      self._ana(gpu, eval_fn, args, analyser)

  def _ana_pre_check(self,
                     gpu,
                     eval_fn,
                     args,
                     excludes,
                     with_group_conv: bool = False):
    """Prune model but not test it to check if all pruning steps can pass."""

    logging.info('Pre-checking for analysis...')
    groups = pruning_lib.group_nodes(self._graph, excludes, with_group_conv)

    spec = pruning_lib.PruningSpec.from_node_groups(groups, 0.9)
    model, pruned_graph, pruning_info = self._prune(spec, mode='sparse')
    current_env = copy.deepcopy(os.environ)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.get_actual_device(gpu))
    model.eval()
    model.cuda()
    eval_fn(model, *args)
    os.environ = current_env

  def _ana(self, gpu, eval_fn, args, analyser):
    uncompleted_steps = copy.copy(analyser.uncompleted_steps())
    total_steps = analyser.steps()
    try:
      for step in uncompleted_steps:
        spec = analyser.spec(step)
        model, _, _ = self._prune(spec, mode='sparse')

        current_env = copy.deepcopy(os.environ)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.get_actual_device(gpu))
        model.eval()
        model.cuda()
        eval_res = eval_fn(model, *args)
        os.environ = current_env
        analyser.record(
            step,
            eval_res.item() if isinstance(eval_res, torch.Tensor) else eval_res)
        analyser.save(self._sens_path)
        logging.info('Analysis complete %d/%d' % (step + 1, total_steps))
    finally:
      analyser.save(self._sens_path)

  def _parallel_ana(self,
                    gpus,
                    eval_fn,
                    args,
                    analyser,
                    excludes,
                    with_group_conv: bool = False):
    graph = utils.parse_to_graph(self._model, self._input_signature)
    total_steps = analyser.steps()
    uncompleted_steps = copy.copy(analyser.uncompleted_steps())

    ctx = mp.get_context('spawn')
    error_queues = []
    input_queue = ctx.Queue()
    output_queue = ctx.Queue()
    for step in uncompleted_steps:
      input_queue.put(step)

    try:
      processes = []
      # To avoid CUDA-out-of-memory error. Main process should execute one copy of task
      # using the same GPU on which we do torch tracing
      input_signature_device = self._input_signature.device
      for rank in range(len(gpus)):
        # For the GPU on which we do torch tracing, the task should be executed by main process instead of sub-processes
        if input_signature_device.type == "cuda" and input_signature_device.index == int(
            gpus[rank]):
          continue
        current_env = copy.deepcopy(os.environ)
        p = ctx.Process(
            target=AnaTask(
                self._model,
                self._input_signature,
                excludes,
                gpus[rank],
                eval_fn,
                eval_fn_args=args,
                with_group_conv=with_group_conv),
            args=(input_queue, output_queue))
        p.start()
        os.environ = current_env
        processes.append(p)
      # Main process executes one copy of task
      if input_signature_device.type == "cuda":
        current_env = copy.deepcopy(os.environ)
        ana_task = AnaTask(
            self._model,
            self._input_signature,
            excludes,
            input_signature_device.index,
            eval_fn,
            eval_fn_args=args)
        ana_task(input_queue, output_queue)
        os.environ = current_env
      for p in processes:
        p.join()
    finally:
      while not output_queue.empty():
        cur_step, score = output_queue.get()
        analyser.record(cur_step, score)
      analyser.save(self._sens_path)

  def prune(self,
            removal_ratio=None,
            threshold=None,
            spec_path=None,
            excludes=None,
            mode='sparse',
            pruning_info_path=None,
            channel_divisible=2):
    """Prune the network by given removal_ratio or threshold.

      Arguments:
        removal_ratio: The expected percentage of macs reduction. This is just a hint
          value, the actual macs drop not necessarily strictly to this value.
        threshold: Relative proportion of model performance loss
          that can be tolerated.
        spec_path: Pre-defined pruning specification.
        excludes: Modules that need to excludes from pruning.
        mode: One of ['sparse', 'slim'].
        channel_divisible: The number of remaining channels in the pruned layer
          can be divided by channel_divisble.

      Return:
        A `torch.nn.Module` object with addtional pruning info.
    """
    if not isinstance(removal_ratio, float):
      raise errors.OptimizerInvalidArgumentError(
          'Expected float "ratio", but got {}({})'.format(
              removal_ratio, type(removal_ratio)))

    net_sens = self._load_analysis_result()
    if net_sens is None:
      raise errors.OptimizerNoAnaResultsError(
          "Must call ana() before model pruning.")

    excluded_nodes = self._get_exclude_nodes(excludes) if excludes else []

    training_flag_info = {}
    for name, module in self._model.named_modules():
      training_flag_info[name] = module.training

    if removal_ratio:
      logging.info('Pruning ratio = {}'.format(removal_ratio))
      spec = self._spec_from_ratio(net_sens, removal_ratio, excluded_nodes)
      spec.channel_divisible = channel_divisible
    elif threshold:
      logging.info('Pruning threshold = {}'.format(threshold))
      spec = self._spec_from_threshold(net_sens, threshold, excluded_nodes)
      spec.channel_divisible = channel_divisible
    elif spec_path:
      logging.info('Pruning specification = {}'.format(spec_path))
      spec = pruning_lib.PruningSpec.deserialize(
          json.load(open(spec_path, 'r')))
    else:
      raise errors.OptimizerInvalidArgumentError(
          'One of [ratio, threshold, spec_path] must be given.')

    spec_path = os.path.join(
        _VAI_DIR, '{}_ratio_{}.spec'.format(self._graph.name, removal_ratio))
    with open(spec_path, 'w') as f:
      json.dump(spec.serialize(), f, indent=2)
    logging.info('Pruning spec saves in {}'.format(spec_path))

    model, pruned_graph, pruning_info = self._prune(spec, mode)
    pruning_info_path = pruning_info_path if pruning_info_path else spec_path.split(
        '.spec')[0] + '_pruning_info.json'
    self.generate_pruning_info(pruning_info_path, pruning_info)
    model._graph = pruned_graph
    model._pruning_info = pruning_info
    if mode == 'sparse':
      model._register_state_dict_hook(_remove_mask)
      model.slim_state_dict = types.MethodType(slim_state_dict, model)
    else:
      model.sparse_state_dict = types.MethodType(sparse_state_dict, model)

    for name, module in model.named_modules():
      module.training = training_flag_info[name]

    logging.info('Pruning summary:')
    slim_model = self._prune(spec, 'slim')[0]
    self._summary(slim_model)
    return model

  def transform(self, spec_path, script_path='pruned_model.py'):
    with open(spec_path, 'r') as f:
      spec = json.load(f)
    _, pruned_graph, _ = self._prune(spec, 'slim')
    pruned_model, _ = utils.rebuild_model(pruned_graph)
    return pruned_model

  def _spec_from_threshold(self, net_sens, threshold, excludes):
    groups = net_sens.prunable_groups_by_threshold(threshold, excludes)
    return pruning_lib.PruningSpec(groups)

  def _spec_from_ratio(self, net_sens, ratio, excludes):
    logging.info('Searching for appropriate ratio for each layer...')

    macs, _ = profiler.model_complexity(self._model, self._input_signature)
    expected_macs = (1 - ratio) * macs

    macs_tolerance = 1e-2
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

      pruned_model = self._prune(cur_spec, 'slim')[0]
      current_macs, _ = profiler.model_complexity(pruned_model,
                                                  self._input_signature)
      error = abs(expected_macs - current_macs) / expected_macs
      if error < macs_tolerance:
        break
      if current_macs < expected_macs:
        max_th = threshold
      else:
        min_th = threshold

    return cur_spec

class RandomSearchTask(object):

  def __init__(self,
               model,
               input_signature,
               removal_ratio,
               num_subnet,
               orig_macs,
               excluded_nodes,
               gpu,
               eval_fn,
               calibration_fn,
               eval_args=(),
               calib_args=(),
               with_group_conv: bool = False):
    self.model = model
    self.input_signature = input_signature
    self.excluded_nodes = excluded_nodes
    self.gpu = gpu
    self.eval_fn = eval_fn
    self.eval_args = eval_args
    self.num_subnet = num_subnet
    self.orig_macs = orig_macs
    self.removal_ratio = removal_ratio
    self.calibration_fn = calibration_fn
    self.calib_args = calib_args
    self.with_group_conv = with_group_conv

    # Must set env variable here, not in __call__ function.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(utils.get_actual_device(gpu))

  def __call__(self, input_queue, output_queue):
    pruner = PruningRunner(self.model, self.input_signature)
    groups = pruning_lib.group_nodes(pruner._graph, self.excluded_nodes,
                                     self.with_group_conv)
    ratio_min = 0.05
    ratio_max = 0.9
    epsilon = 0.01

    while not input_queue.empty():
      ratios = []
      spec = pruning_lib.PruningSpec()
      for group in groups:
        if self.removal_ratio >= 0.5:
          mu = random.uniform(self.removal_ratio / 2, self.removal_ratio)
        else:
          mu = random.uniform(0, self.removal_ratio / 2)
        sigma = random.uniform(0, 0.3)
        pruning_ratio = random.gauss(mu, sigma)
        pruning_ratio = np.clip(pruning_ratio, ratio_min, ratio_max)
        ratios.append(pruning_ratio)
        spec.add_group(
            pruning_lib.PrunableGroup(group.nodes, pruning_ratio,
                                      group.num_groups))

      #logging.info("ratios={}".format(ratios))
      pruned_model, pruned_graph, pruning_info = pruner._prune(
          spec, mode='slim')
      pruned_model.cuda()
      macs, params = profiler.model_complexity(
          pruned_model, self.input_signature, readable=False)
      cur_removal_ratio = 1 - macs / self.orig_macs
      if abs(self.removal_ratio - cur_removal_ratio) < epsilon:
        self.calibration_fn(pruned_model, *(self.calib_args))
        score = self.eval_fn(pruned_model, *(self.eval_args))
        if isinstance(score, torch.Tensor):
          score = score.item()
        output_queue.put((ratios, score, 1 - cur_removal_ratio))
        if input_queue.empty():
          del (pruned_model)
          break
        index = input_queue.get()
        logging.info('Search complete %d/%d' % (index + 1, self.num_subnet))
      del (pruned_model)

class OneStepPruningRunner(PruningRunner):
  """EagleEye."""

  def __init__(self, model, input_signature):
    super(OneStepPruningRunner, self).__init__(model, input_signature)

    self._searcher_saved_path = os.path.join(_VAI_DIR,
                                             self._graph.name + '.search')

  def search(self,
             gpus=['0'],
             calibration_fn=None,
             calib_args=(),
             num_subnet=200,
             removal_ratio=0.5,
             excludes=[],
             eval_fn=None,
             eval_args=(),
             forced=False,
             with_group_conv: bool = False):
    """
    Perform pruned candidates search.

    Arguments:
      gpus: A tuple or list of gpu indices used for model analysis. If not set,
        the default gpu will be used.
      calibration_fn: Callable object that takes a `torch.nn.Module` object as
        its first argument. It's for calibrating BN layer's statistics.
      calib_args: A tuple of arguments that will be passed to calibration_fn.
      num_subnet: The number of subnets needed to search matching the macs
        requirement.
      removal_ratio: The expected percentage of macs reduction.
      excludes: Modules that need to exclude from pruning.
      eval_fn: Callable object that takes a `torch.nn.Module` object as its
        first argument and returns the evaluation score.
      eval_args: A tuple of arguments that will be passed to eval_fn.
    """
    self._searcher_saved_path = os.path.join(
        _VAI_DIR, '{}_ratio_{}.search'.format(self._graph.name, removal_ratio))
    excluded_nodes = self._get_exclude_nodes(excludes) if excludes else []
    orig_macs, orig_params = profiler.model_complexity(
        self._model, self._input_signature, readable=False)
    searcher = None
    if not forced and os.path.exists(self._searcher_saved_path):
      searcher = search.load_searcher(self._searcher_saved_path)
      if len(searcher._subnets) >= num_subnet:
        logging.info(
            'Skip subnet search and use cached result, if you do not want to use it, set forced=True'
        )
        return
    if not searcher:
      groups = pruning_lib.group_nodes(self._graph, excluded_nodes,
                                       with_group_conv)
      searcher = search.SubnetSearcher(groups)
      base_score = eval_fn(self._model, *eval_args)
      if isinstance(base_score, torch.Tensor):
        base_score = base_score.item()
      searcher.set_supernet(base_score, orig_macs)
    self._search_random_precheck(calibration_fn, eval_fn, removal_ratio,
                                 excluded_nodes, calib_args, eval_args,
                                 orig_macs, with_group_conv)
    num_parallel = len(gpus) if gpus else 1
    if num_parallel > 1:
      self._parallel_search_random(gpus, calibration_fn, eval_fn, num_subnet,
                                   removal_ratio, excluded_nodes, calib_args,
                                   eval_args, searcher, orig_macs,
                                   with_group_conv)
    else:
      self._search_random(calibration_fn, eval_fn, num_subnet, removal_ratio,
                          excluded_nodes, calib_args, eval_args, searcher,
                          orig_macs, with_group_conv)

  def _parallel_search_random(self,
                              gpus,
                              calibration_fn,
                              eval_fn,
                              num_subnet,
                              removal_ratio,
                              excluded_nodes,
                              calib_args,
                              eval_args,
                              searcher,
                              orig_macs,
                              with_group_conv: bool = False):
    ctx = mp.get_context('spawn')
    output_queue = ctx.Queue()
    input_queue = ctx.Queue()
    for index in range(len(searcher._subnets), num_subnet):
      input_queue.put(index)
    try:
      processes = []
      # To avoid CUDA-out-of-memory error. Main process should execute one copy of task
      # using the same GPU on which we do torch tracing
      input_signature_device = self._input_signature.device
      for rank in range(len(gpus)):
        # For the GPU on which we do torch tracing, the task should be executed by main process instead of sub-processes
        if input_signature_device.type == "cuda" and input_signature_device.index == int(
            gpus[rank]):
          continue
        current_env = copy.deepcopy(os.environ)
        p = ctx.Process(
            target=RandomSearchTask(
                self._model,
                self._input_signature,
                removal_ratio,
                num_subnet,
                orig_macs,
                excluded_nodes,
                gpus[rank],
                eval_fn,
                calibration_fn,
                eval_args=eval_args,
                calib_args=calib_args,
                with_group_conv=with_group_conv),
            args=(input_queue, output_queue))
        p.start()
        os.environ = current_env
        processes.append(p)
      # Main process executes one copy of task
      if input_signature_device.type == "cuda":
        current_env = copy.deepcopy(os.environ)
        search_task = RandomSearchTask(
            self._model,
            self._input_signature,
            removal_ratio,
            num_subnet,
            orig_macs,
            excluded_nodes,
            input_signature_device.index,
            eval_fn,
            calibration_fn,
            eval_args=eval_args,
            calib_args=calib_args,
            with_group_conv=with_group_conv)
        search_task(input_queue, output_queue)
        os.environ = current_env
      for p in processes:
        p.join()
    finally:
      while not output_queue.empty():
        searcher.add_subnet(*output_queue.get())
      search.save_searcher(searcher, self._searcher_saved_path)

  def _search_random_precheck(self,
                              calibration_fn,
                              eval_fn,
                              removal_ratio,
                              excluded_nodes,
                              calib_args,
                              eval_args,
                              orig_macs,
                              with_group_conv: bool = False):
    groups = pruning_lib.group_nodes(self._graph, excluded_nodes,
                                     with_group_conv)
    spec = pruning_lib.PruningSpec.from_node_groups(groups, 0.9)
    pruned_model, pruned_graph, pruning_info = self._prune(spec, mode='slim')
    macs, params = profiler.model_complexity(
        pruned_model, self._input_signature, readable=False)
    cur_removal_ratio = 1 - macs / orig_macs
    calibration_fn(pruned_model, *(calib_args))
    score = eval_fn(pruned_model, *(eval_args))
    if isinstance(score, torch.Tensor):
      score = score.item()
    logging.info('ratios={}, score={}, cur_removal_ratio={}'.format(
        removal_ratio, score, cur_removal_ratio))

  def _search_random(self,
                     calibration_fn,
                     eval_fn,
                     num_subnet,
                     removal_ratio,
                     excluded_nodes,
                     calib_args,
                     eval_args,
                     searcher,
                     orig_macs,
                     with_group_conv: bool = False):
    groups = pruning_lib.group_nodes(self._graph, excluded_nodes,
                                     with_group_conv)
    ratio_min = 0.05
    ratio_max = 0.9
    epsilon = 0.01
    index = len(searcher._subnets) + 1
    try:
      while True:
        ratios = []
        spec = pruning_lib.PruningSpec()
        for group in groups:
          if removal_ratio >= 0.5:
            mu = random.uniform(removal_ratio / 2, removal_ratio)
          else:
            mu = random.uniform(0, removal_ratio / 2)
          sigma = random.uniform(0, 0.3)
          pruning_ratio = random.gauss(mu, sigma)
          pruning_ratio = np.clip(pruning_ratio, ratio_min, ratio_max)
          ratios.append(pruning_ratio)
          spec.add_group(
              pruning_lib.PrunableGroup(group.nodes, pruning_ratio,
                                        group.num_groups))
        pruned_model, pruned_graph, pruning_info = self._prune(
            spec, mode='slim')
        macs, params = profiler.model_complexity(
            pruned_model, self._input_signature, readable=False)
        cur_removal_ratio = 1 - macs / orig_macs
        if abs(removal_ratio - cur_removal_ratio) < epsilon:
          calibration_fn(pruned_model, *(calib_args))
          score = eval_fn(pruned_model, *(eval_args))
          if isinstance(score, torch.Tensor):
            score = score.item()
          searcher.add_subnet(ratios, score, 1 - cur_removal_ratio)
          search.save_searcher(searcher, self._searcher_saved_path)
          logging.info('Search complete %d/%d' % (index, num_subnet))
          #logging.info('Index={}, ratios={}, score={}, cur_removal_ratio={}'.format(
          #    index, ratios, score, cur_removal_ratio))
          index += 1
        #logging.info('cur_removal_ratio={}'.format(cur_removal_ratio))
        if index > num_subnet:
          break
    finally:
      search.save_searcher(searcher, self._searcher_saved_path)
      logging.info('Search %d subnets!' % (len(searcher._subnets)))

  def prune(self,
            mode='slim',
            removal_ratio=None,
            index=None,
            pruning_info_path=None,
            channel_divisible=2):
    """Get pruned candidate subnet of the specific index.

      Arguments:
        mode: One of ['sparse', 'slim'].
        removal_ratio: Pruning ratio for model.
        index: Subnet index. By default, the optimal subnet is selected automatically.

      Return:
        A `torch.nn.Module` object with addtional pruning info.
    """
    assert removal_ratio, 'Pruning ratio for model is needed!'
    self._searcher_saved_path = os.path.join(
        _VAI_DIR, '{}_ratio_{}.search'.format(self._graph.name, removal_ratio))
    searcher = search.load_searcher(self._searcher_saved_path)
    groups = searcher.groups
    if index:
      subnet = searcher.subnet(index)
    else:
      subnet = searcher.best_subnet()
    logging.info('Sparsity={}, score={}'.format(1 - subnet.macs, subnet.score))
    assert len(groups) == len(subnet.ratios)
    spec = searcher.spec(subnet.ratios)
    spec_path = self._searcher_saved_path.split(
        '.search')[0] + '_{}.spec'.format(index if index else 'best')
    with open(spec_path, 'w') as f:
      json.dump(spec.serialize(), f, indent=2)
    logging.info('Pruning spec saves in {}'.format(spec_path))

    spec.channel_divisible = channel_divisible
    model, pruned_graph, pruning_info = self._prune(spec, mode=mode)
    pruning_info_path = pruning_info_path if pruning_info_path else self._searcher_saved_path.split(
        '.search')[0] + '_pruning_info.json'
    self.generate_pruning_info(pruning_info_path, pruning_info)
    model._graph = pruned_graph
    model._pruning_info = pruning_info
    model._register_state_dict_hook(_remove_mask)
    if mode == 'sparse':
      model.slim_state_dict = types.MethodType(slim_state_dict, model)
    else:
      model.sparse_state_dict = types.MethodType(sparse_state_dict, model)
    return model

def get_pruning_runner(model, input_signature, method):
  assert method in ['iterative', 'one_step']
  cls = IterativePruningRunner if method == 'iterative' else OneStepPruningRunner
  return cls(model, input_signature)

def sparse_state_dict(self, destination=None, prefix='', keep_vars=False):
  state_dict = self.state_dict(destination, prefix, keep_vars)
  for node in self._graph.nodes:
    pruning_info = self._pruning_info.get(node.name, None)
    if not pruning_info:
      continue
    for tensor in node.op.params.values():
      # ResNet::layer1.1.conv1.weight -> layer1.1.conv1.weight
      key = tensor.name.lstrip(self._graph.name +
                               torch_const.TorchGraphSymbol.GRAPH_SCOPE_SYM)
      value = utils.pad_to_sparse_tensor(state_dict[key], pruning_info)
      state_dict[key] = value
  return state_dict

def slim_state_dict(self, destination=None, prefix='', keep_vars=False):
  """Returns a slim state dict in which the weight names are same with the
  original model and the tensors are pruned to slim ones.
  """
  assert prefix == ''

  if destination is None:
    destination = collections.OrderedDict()
    destination._metadata = collections.OrderedDict()

  state_dict = self.state_dict(None, prefix, keep_vars)
  for node in self._graph.nodes:
    node_pruning = self._pruning_info.get(node.name, None)
    for param, tensor in node.op.params.items():
      # ResNet::layer1.1.conv1.weight -> layer1.1.conv1.weight
      key = utils.state_dict_key_from_tensor(tensor)
      value = state_dict[key]

      if not node_pruning:
        destination[key] = value
      else:
        out_dims = node_pruning.removed_outputs
        in_dims = node_pruning.removed_inputs
        if (node.op.type in _CONVTRANSPOSE_OPS and
            param == node.op.ParamName.WEIGHTS):
          out_dims, in_dims = in_dims, out_dims

        destination[key] = _prune_tensor(value, out_dims, in_dims)
  # The model during train/inference procedure may be differnt.
  # 'self._graph' is generated according to inference model,
  # so some keys in state_dict may not in destination.
  for key in state_dict:
    if key not in destination:
      destination[key] = state_dict[key]

  return destination

def _sparsify_tensor(tensor: torch.Tensor,
                     out_dims: List[int],
                     in_dims: List[int],
                     groups: int = 1):
  """Fill 0 in removed channels."""
  device = tensor.device
  tensor = tensor.detach().cpu().clone(memory_format=torch.contiguous_format)
  dim_size = len(tensor.shape)
  if groups == 1:
    if out_dims:
      tensor[out_dims, ...] = 0
    # data format in pytorch: OIHW
    if in_dims and dim_size > 1:
      tensor[:, in_dims, ...] = 0
  else:
    out_dims_group = generate_indices_group(out_dims, tensor.size(0), groups)
    in_dims_group = generate_indices_group(
        in_dims,
        tensor.size(1) * groups, groups) if dim_size > 1 else [[]] * groups
    parts = tensor.split(tensor.size(0) // groups, dim=0)
    sparse_parts: List[torch.Tensor] = []
    for part, o, i in zip(parts, out_dims_group, in_dims_group):
      sparse_parts.append(_sparsify_tensor(part, o, i))
    tensor = torch.cat(sparse_parts, dim=0)
  return tensor.to(device)

def _prune_tensor(tensor, out_dims, in_dims):
  """Remove pruned channels of given tensor and returns a slim tensor."""
  dim_size = len(tensor.shape)
  ndarray = tensor.detach().cpu().numpy()
  out_axis, in_axis = 0, 1
  if out_dims:
    ndarray = np.delete(ndarray, out_dims, axis=out_axis)
  if in_dims and dim_size > in_axis:
    ndarray = np.delete(ndarray, in_dims, axis=in_axis)
  return torch.from_numpy(ndarray)

def _apply_mask(module, inputs):
  for tensor_name in module._tensor_names:
    weight = getattr(module, tensor_name)
    mask = getattr(module, tensor_name + '_mask')
    weight.data = mask.to(dtype=weight.dtype) * weight

def _remove_mask(self, destination=None, prefix='', keep_vars=False):
  keys = list(destination.keys())
  for key in keys:
    if '_mask' in key:
      del destination[key]
  return destination
