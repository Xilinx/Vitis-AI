from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp
import multiprocessing as mp
import os
import sys
import torch

from nndct_shared.base import key_names
from nndct_shared.pruning import analyser as ana_lib
from nndct_shared.pruning import logging
from nndct_shared.pruning import pruner as pruner_lib
from nndct_shared.pruning import pruning_lib
from nndct_shared.utils import tensor_util

from pytorch_nndct.nndct import parse
from pytorch_nndct.nndct.pruning import utils

class AnaTask(object):

  def __init__(self, eval_fn, args=()):
    pass

  def __call__(self):
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class Pruner(object):
  """Implements channel pruning at the module level."""

  def __init__(self, module, input_shapes):
    """Concrete example:

    ```python
      model = MyModel()
      pruner = pytorch_nndct.nndct.pruning.Pruner(model, (3, 32, 32))
      model = pruner.prune(model, (3, 32, 32), 0.5)
    ```

    Arguments:
      model (Module): Model to be pruned.
      input_shapes (tuple or list): Shape of inputs.
      sparsity (float): Expected sparsity of pruned network.

    Returns:
      A module that has been pruned with updated attributes.

    Raises:
      TypeError: In case of improper type of `input_shapes`.
    """

    self._graph = self._parse_to_graph(module, input_shapes)
    self._module = module

  def _parse_to_graph(self, module, input_shapes):
    if isinstance(input_shapes, tuple):
      inputs = torch.randn([1, *input_shapes])
    elif isinstance(input_shapes, list):
      inputs = []
      for shape in input_shapes:
        inputs.append(torch.randn([1, *shape]))
    else:
      raise TypeError('`input_shapes` must be a list or tuple')

    parser = parse.TorchParser()
    return parser(module._get_name(), module, inputs)

  def _prune(self, graph, pruning_spec):
    """A pruning runner function that generates a `PruningSpec` by given
    threshold and use `ChannelPruner` to perform pruning on the given model.

    Arguments:
      graph: A `NndctGraph` to be pruned.
      pruning_spec: A `PruningSpec` object indicates how to prune the model.

    Returns:
      A `torch.nn.Module` object rebuilt from the pruned `NndctGraph` model.
      A dict of `NodePruningInfo` which the key is the name of the node.
    """

    pruner = pruner_lib.ChannelPruner(graph)
    pruned_graph, pruning_info = pruner.prune(pruning_spec)

    # TODO(yuwang): Support user-provided path.
    rebuilt_module = utils.rebuild_module(pruned_graph, './graph.py')

    # NodePruningInfo.state_dict_keys is only used to validate pruning result,
    # it has no effect on pruned graph.
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

    return rebuilt_module, pruning_info

  def ana(self, eval_fn, args=()):
    """Performs model analysis.
    Args:
      eval_fn: Callable object that takes a `torch.nn.Module` object as its
        first argument and returns the evaluation score.
      args: A tuple of arguments that will be passed to eval_fn.
    """
    analyser = ana_lib.ModelAnalyser(self._graph)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #num_parallel = torch.cuda.device_count() if use_cuda else 1
    num_parallel = 1

    mp.set_start_method('spawn')
    cur_step = 0
    steps = analyser.steps()
    while cur_step < steps:
      batch_steps = min(num_parallel, steps - cur_step)
      results = []
      models = []
      for bs in range(batch_steps):
        spec = analyser.spec(cur_step + bs)
        model, _ = self._prune(self._graph, spec)
        models.append(model)

      #pool = mp.Pool(num_parallel)
      #for i, model in enumerate(models):
      #  model.eval()
      #  model = model.to('cuda:%s' % str(i))
      #  # TODO(yuwang): AnaTask
      #  results.append(pool.apply_async(eval_fn, args=((model,) + args)))
      #pool.close()
      #pool.join()
      for model in models:
        results.append(eval_fn(model, *args))

      for bs in range(batch_steps):
        #analyser.record(cur_step, results[bs].get())
        # TODO:
        analyser.record(cur_step, results[bs].item())
        cur_step += 1

    analyser.save()

  def prune(self, threshold=None, sparsity=None):
    spec = pruning_lib.PruningSpec()
    if threshold:
      sens_path = pruning_lib.sens_path(self._graph)
      if not os.path.exists(sens_path):
        raise RuntimeError("Must call ana() before runnig prune.")
      net_sens = pruning_lib.read_sens(sens_path)

      # TODO(yuwang): Support excludes: important to detection net.
      net_sparsity = pruning_lib.get_sparsity_by_threshold(net_sens, threshold)
      logging.vlog(1, 'NetSparsity: \n{}'.format(
          '\n'.join([str(group) for group in net_sparsity])))
      for group_sparsity in net_sparsity:
        spec.add_group(group_sparsity)
    elif sparsity:
      groups = pruning_lib.group_nodes(self._graph)
      for group in groups:
        spec.add_group(pruning_lib.GroupSparsity(group, sparsity))
    else:
      raise ValueError("At least one of 'sparsity' or 'threshold' to be set")

    pruned_model, pruning_info = self._prune(self._graph, spec)
    return PruningModule(pruned_model, pruning_info)

class PruningModule(torch.nn.Module):

  def __init__(self, module, pruning_info):

    super(PruningModule, self).__init__()

    self._module = module
    self._pruning_info = pruning_info

  def forward(self, *inputs, **kwargs):
    return self._module.forward(*inputs, **kwargs)

  def save(self):
    pass

  @property
  def pruning_info(self):
    return self._pruning_info
