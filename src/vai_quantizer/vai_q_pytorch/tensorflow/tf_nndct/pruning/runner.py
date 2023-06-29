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
import gc
import inspect
import json
import numpy as np
import os
import random
import tensorflow as tf
import types

from tensorflow.python.distribute import distribution_strategy_context as ds_context

from nndct_shared.pruning import errors
from nndct_shared.pruning import pruner as pruner_lib
from nndct_shared.pruning import pruning_lib
from nndct_shared.pruning import search
from nndct_shared.pruning import sensitivity as sens
from nndct_shared.utils import common
from tf_nndct.graph import parser
from tf_nndct.graph.ops import OpTypes
from tf_nndct.pruning import pruning_impl
from tf_nndct.utils import generic_utils
from tf_nndct.utils import keras_utils as ku
from tf_nndct.utils import logging
from tf_nndct.utils import tensor_utils
from nndct_shared.pruning.utils import generate_indices_group
from nndct_shared.pruning.pruning_lib import is_grouped_conv, is_depthwise_conv

keras = tf.keras

_VAI_DIR = '.vai'

def _is_debug():
  return os.environ.get('VAI_OPTIMIZER_DEBUG', None) == '1'

def add_pruning_mask(instance):

  def build(self, input_shape):
    type(self).build(self, input_shape)
    weight_vars, mask_vars, = [], []
    # For each of the weights, add mask variables.
    for weight in self.weights:
      # res2a_branch2a/kernel:0 -> kernel
      weight_name = weight.name.split('/')[-1].split(':')[0]
      mask = self.add_weight(
          weight_name + '_mask',
          shape=weight.shape,
          initializer=keras.initializers.get('ones'),
          dtype=weight.dtype,
          trainable=False,
          aggregation=tf.VariableAggregation.MEAN)

      weight_vars.append(weight)
      mask_vars.append(mask)

    pruning_vars = list(zip(weight_vars, mask_vars))
    # Create a pruning object
    self.pruning_obj = pruning_impl.Pruning(pruning_vars)

  def call(self, inputs, training=None, **kwargs):
    if training is None:
      training = keras.backend.learning_phase()

    # Always execute the op that performs weights = weights * mask
    # Relies on UpdatePruningStep callback to ensure the weights
    # are sparse after the final backpropagation.
    #
    # self.add_update does nothing during eager execution.
    self.add_update(self.pruning_obj.weight_mask_op())
    # TODO(evcu) remove this check after dropping py2 support. In py3 getargspec
    # is deprecated.
    layer_call = type(self).call
    if hasattr(inspect, 'getfullargspec'):
      args = inspect.getfullargspec(layer_call).args
    else:
      args = inspect.getargspec(layer_call).args
    # Propagate the training bool to the underlying layer if it accepts
    # training as an arg.
    if 'training' in args:
      return layer_call(self, inputs, training=training, **kwargs)

    return layer_call(self, inputs, **kwargs)

  instance.build = types.MethodType(build, instance)
  instance.call = types.MethodType(call, instance)
  return instance

class PruningRunner(object):

  def __init__(self, model: keras.Model, input_signature: tf.TensorSpec):
    if not isinstance(model, keras.Model):
      raise errors.OptimizerKerasModelError(
          '"model" must be an instance of keras.Model')

    # Check whether the model is a subclass model.
    if (not model._is_graph_network and
        not isinstance(model, keras.models.Sequential)):
      raise errors.OptimizerKerasModelError(
          'Subclassed models are not supported currently.')

    self._model = model
    self._input_signature = input_signature

    self._graph = parser.from_keras_model(model, input_signature)

    if _is_debug():
      #from nndct_shared.utils import saving
      from tf_nndct.utils import viz
      from tf_nndct.graph import writer
      graph_name = self._graph.name
      #utils.write_graph_script(self._graph, '{}_baseline.py'.format(graph_name))
      #saving.save_graph(self._graph, hdf5_path='{}.hdf5'.format(graph_name))

      #writer.GraphCodeGenerator(self._graph, class_spec).write('{}_baseline.py'.format(graph_name))
      viz.export_to_netron('{}.pb'.format(graph_name), self._graph)

  def _get_exclude_nodes(self, excludes):
    layer_to_node = {}
    for node in self._graph.nodes:
      if node.layer_name:
        layer_to_node[node.layer_name] = node.name

    if not generic_utils.is_list_or_tuple(excludes):
      excludes = generic_utils.to_list(excludes)
    excluded_nodes = []
    for exclude in excludes:
      if isinstance(exclude, str):
        excluded_nodes.append(exclude)
      elif isinstance(exclude, keras.layers.Layer):
        for layer in [exclude] + list(ku.gather_layers(exclude)):
          if layer.name in layer_to_node:
            excluded_nodes.append(layer_to_node[layer.name])
      else:
        raise errors.OptimizerInvalidArgumentError(
            'Excludes must be list of either string or keras.layers.Layer')
    return excluded_nodes

  def _prune(self, pruning_spec, mode='slim'):
    """Perform pruning by given specification.

    Arguments:
      pruning_spec: A `PruningSpec` object indicates how to prune the model.
      mode: 'sparse' or 'slim' mode.

    Returns:
      A keras.Model object cloned and modified from original baseline model.
      If mode is 'sparse', returns a model with the same architecture as the
      baseline model, with the weights sparsified.
      If mode is 'slim', returns a slim model with the pruned channels
      removed from the baseline model.
    """
    assert mode in ['sparse', 'slim']
    pruner = pruner_lib.ChannelPruner(self._graph)
    pruned_graph, net_pruning = pruner.prune(pruning_spec)

    if mode == 'sparse':
      return self._get_sparse_model(net_pruning)
    else:
      return self._get_slim_model(pruned_graph, net_pruning)

  def _get_sparse_model(self, net_pruning):

    def mask_pruned_layer(layer):
      if isinstance(layer, keras.Model):
        return keras.models.clone_model(
            layer, input_tensors=None, clone_function=mask_pruned_layer)

      # We don't want to modify the original layer so clone it first and then
      # make decorating on the cloned one.
      cloned_layer = layer.__class__.from_config(layer.get_config())
      if cloned_layer.__class__.__name__ in ['TensorFlowOpLayer', 'TFOpLambda']:
        # the op e.g.['+' tf.concat] will be converted to TensorFlowOpLayer or TFOpLambda layer
        # TODO: these layer like relu not have parm whether need add_pruning_mask()?
        return cloned_layer
      if layer.name in to_prune:
        logging.vlog(3, 'Add pruning mask: {}'.format(layer.name))
        # Ensure that layer.build() can be called to add mask weight.
        cloned_layer = add_pruning_mask(cloned_layer)
      return cloned_layer

    to_prune = set()
    layer_weights = {}
    for node in self._graph.nodes:
      node_pruning = net_pruning[node.name]
      weight_vals, mask_vals = [], []
      if node_pruning.removed_outputs or node_pruning.removed_inputs:
        to_prune.add(node.layer_name)
        for param, tensor in node.op.params.items():
          if pruning_lib.is_transpose_conv(node.op):
            # weight for tf transpose conv layer: transpoe the O <-> I
            weight = tensor_utils.transposeconv_weight_dim_trans(\
                      tensor_utils.param_to_tf_numpy(tensor))
          else:  # for normal conv/linear
            weight = tensor_utils.param_to_tf_numpy(tensor)

          mask = np.ones_like(weight)
          removed_out_channels = node_pruning.removed_outputs
          removed_in_channels = node_pruning.removed_inputs

          # only prune input_dim for depthwise conv
          if pruning_lib.is_depthwise_conv(node.op) and len(weight.shape) >= 4:
            # depthwise kernel shape: [*kernel_size, input_dim, depthwise_multipler]
            removed_out_channels = []
          elif pruning_lib.is_transpose_conv(
              node.op) and len(weight.shape) >= 4:
            removed_out_channels, removed_in_channels = removed_in_channels, removed_out_channels
          # Doesn't prune depthwise weight of separate conv
          elif pruning_lib.is_separable_conv(node.op) and \
               pruning_lib.is_separable_conv_depthwise_weight(param):
            removed_out_channels = []
          elif pruning_lib.is_separable_conv(node.op) and \
                pruning_lib.is_separable_conv_pointwise_weight(param):
            removed_in_channels = node_pruning.removed_separableconv_pointwise_inputs
          weight_vals.append(
              _sparsify_tensor(
                  weight, removed_out_channels, removed_in_channels,
                  node.op.attr['group'] if is_grouped_conv(node.op) and
                  not is_depthwise_conv(node.op) else 1))
          mask_vals.append(
              _sparsify_tensor(
                  mask, removed_out_channels, removed_in_channels,
                  node.op.attr['group'] if is_grouped_conv(node.op) and
                  not is_depthwise_conv(node.op) else 1))
      else:
        weight_vals = tensor_utils.layer_weights_from_node(node)
      layer_weights[node.layer_name] = (weight_vals, mask_vals)

    strategy = self._model._distribution_strategy or ds_context.get_strategy()
    with strategy.scope():
      model = mask_pruned_layer(self._model)

    for layer in ku.gather_layers(model):
      if not layer.weights:
        continue
      # Since mask variables were added after the weights are created,
      # the weights of a decorated layer is like:
      # [kernel, bias, kernel_mask, bias_mask]
      weights, masks = layer_weights[layer.name]
      layer.set_weights(weights + masks)

    return model

  def _get_slim_model(self, pruned_graph, net_pruning):

    def create_from_config(layer):
      if isinstance(layer, keras.Model):
        return keras.models.clone_model(
            layer, input_tensors=None, clone_function=create_from_config)

      if layer.name in layer_config:
        config = layer_config[layer.name]
      else:
        config = layer.get_config()
      return layer.__class__.from_config(config)

    def update_input_shape(model):
      '''Update input layer's shape for nested Functional models.

      As keras.InputLayer will not be cloned by its own config instead of
      provided clone function 'create_from_config', so we have to update
      input layer's shape to match the latest shape after pruning.
      See https://github.com/keras-team/keras/blob/v2.8.0/keras/models.py#L241
      '''

      inbound_layers = ku.get_actual_inbound_layers(model)
      for layer in model.layers:
        if not hasattr(layer, '_input_layers'):
          continue

        assert layer.name in inbound_layers and len(
            inbound_layers[layer.name]) == len(layer._input_layers)
        for index, input_layer in enumerate(layer._input_layers):
          inbound_layer_name = inbound_layers[layer.name][index]
          if inbound_layer_name not in layer_to_node:
            continue
          node_pruning = net_pruning[layer_to_node[inbound_layer_name].name]
          input_layer._batch_input_shape = input_layer._batch_input_shape[:-1] + (
              node_pruning.out_dim,)

    layer_config, layer_to_node = {}, {}
    for node in pruned_graph.nodes:
      node_pruning = net_pruning[node.name]

      if node.layer_name:
        layer_to_node[node.layer_name] = node

      if not node_pruning.removed_outputs and not node_pruning.removed_inputs:
        continue

      if node.op.type not in pruning_lib.OPS_WITH_PARAMETERS and len(
          list(node.op.params)) != 0:
        raise errors.OptimizerUnSupportedOpError(
            'Unsupported op with parameters: {}({})'.format(
                node.name, node.op.type))

      config = {}
      for name in node.op.configs:
        config[name] = node.op.get_config(name)
      # Parser reset activation to None in op's config and saved the original
      # value to op's attribute. Here we use this attribute value to build
      # the config for recreating model.
      if node.op.has_attr('activation'):
        config['activation'] = node.op.attr['activation']
      layer_config[node.layer_name] = config

    update_input_shape(self._model)
    strategy = self._model._distribution_strategy or ds_context.get_strategy()
    with strategy.scope():
      model = create_from_config(self._model)
      for layer in ku.gather_layers(model):
        if not layer.weights:
          continue
        weights = tensor_utils.layer_weights_from_node(
            layer_to_node[layer.name])
        layer.set_weights(weights)
    return model

class IterativePruningRunner(PruningRunner):

  def __init__(self, model: keras.Model, input_signature: tf.TensorSpec):
    super(IterativePruningRunner, self).__init__(model, input_signature)

    self._sens_path = os.path.join(_VAI_DIR, self._graph.name + '.sens')
    self._latest_spec = os.path.join(_VAI_DIR, 'latest_spec')

  def ana(self,
          eval_fn,
          excludes=None,
          forced=False,
          with_group_conv: bool = False):
    """Performs model analysis. The analysis result will be saved in '.vai'
    directory and this cached result will be used directly in subsequent
    calls unless 'forced' is set to True.

    Arguments:
      eval_fn: Callable object that takes a keras.Model object as its first
        argument and returns the evaluation score.
      excludes: excludes: A list of node name or torch module to be excluded
        from pruning.
      forced: When set to True, forced to run model analysis instead of using
        cached analysis result.
    """
    if not forced:
      net_sens = self._load_analysis_result()
      if net_sens is not None:
        logging.info(
            'Using cached analysis result. If you want to re-analyze the model, set forced=True'
        )
        return

    excluded_nodes = self._get_exclude_nodes(excludes) if excludes else []
    self._ana_pre_check(eval_fn, excluded_nodes, with_group_conv)

    self._ana(eval_fn, excluded_nodes, with_group_conv)

  def prune(self,
            ratio=None,
            threshold=None,
            spec_path=None,
            excludes=None,
            mode='sparse',
            channel_divisible=2):
    """Prune the baseline model and returns a sparse model. The degree of model
    reduction can be specified in three ways: ratio, threshold or pruning
    specification. The first method should be used in preference, the latter
    two are more suitable for experiments with manual tuning.

    Arguments:
      ratio: The expected percentage of MACs reduction of baseline model.
        This is just a hint value and the actual MACs reduction not
        strictly equals to this value.
      threshold: Relative proportion of model performance loss between
        baseline model and the pruned model.
      spec_path: Pruning specfication path used to prune the model.
      excludes: A list of layer name or layer instance to be excluded from pruning.
      mode: In which mode the pruned model is generated. Should be either
        'sparse' or 'slim'. Must be 'sparse' in iterative pruning loop.
      channel_divisible: The number of remaining channels in the pruned layer
        can be divided by channel_divisble.

    Returns:
      A sparse or a slim model according to given 'mode'.
    """

    if ratio or threshold:
      net_sens = self._load_analysis_result()
      if net_sens is None:
        raise errors.OptimizerNoAnaResultsError(
            "Must call ana() before model pruning.")

    excluded_nodes = self._get_exclude_nodes(excludes) if excludes else []

    if ratio:
      if not isinstance(ratio, float):
        raise errors.OptimizerInvalidArgumentError(
            'Expected "ratio" to be float, but got {}({})'.format(
                ratio, type(ratio)))
      logging.info('Pruning ratio = {}'.format(ratio))
      spec = self._spec_from_ratio(net_sens, ratio, excluded_nodes)
      spec.channel_divisible = channel_divisible
      target = ('ratio', ratio)
    elif threshold:
      if not isinstance(threshold, float):
        raise errors.OptimizerInvalidArgumentError(
            'Expected "threshold" to be float, but got {}({})'.format(
                threshold, type(threshold)))
      logging.info('Pruning threshold = {}'.format(threshold))
      spec = self._spec_from_threshold(net_sens, threshold, excluded_nodes)
      spec.channel_divisible = channel_divisible
      target = ('threshold', threshold)
    elif spec_path:
      logging.info('Pruning specification = {}'.format(spec_path))
      spec = self._spec_from_path(spec_path)
    else:
      raise errors.OptimizerInvalidArgumentError(
          'One of [ratio, threshold, spec_path] must be given.')

    if ratio or threshold:
      filename = '{}_{}_{}.spec'.format(self._graph.name, *target)
      spec_path = os.path.join(_VAI_DIR, filename)
      json.dump(spec.serialize(), open(spec_path, 'w'), indent=2)
      logging.info('Pruning specification saves in {}'.format(spec_path))

    with open(self._latest_spec, 'w') as f:
      json.dump(spec_path, f)

    if mode == 'slim':
      logging.warn(
          ('UserWarning: slim model can not be used for the next iteration. '
           'Set "mode=sparse" for iterative purpose.'))

    model = self._prune(spec, mode)
    if mode == 'sparse':
      model.save_weights = types.MethodType(save_weights, model)

    logging.info('Pruning summary:')
    self._summary(spec)
    return model

  def get_slim_model(self, spec_path=None):
    """Get a slim model from a sparse model. Use the latest pruning
    specification to do this transformation by default. If the sparse model
    was not generated from the latest specification, a specification path
    can be provided explicitly.

    Arguments:
      spec_path: Path of pruning specification used to transform a sparse
      model to a slim model.

    Returns:
      A shrinked slim model.
    """

    spec_path = spec_path or json.load(open(self._latest_spec))
    logging.info('Get slim model from specification {}'.format(spec_path))
    return self._prune(self._spec_from_path(spec_path), 'slim')

  def _load_analysis_result(self):
    return sens.load_sens(self._sens_path) if os.path.exists(
        self._sens_path) else None

  def _ana_pre_check(self, eval_fn, excludes, with_group_conv: bool = False):
    """Prune model but not test it to check if all pruning steps can pass."""

    logging.info('Pre-checking for analysis...')
    groups = pruning_lib.group_nodes(self._graph, excludes, with_group_conv)
    spec = pruning_lib.PruningSpec.from_node_groups(groups, 0.9)

    pruner = pruner_lib.ChannelPruner(self._graph)
    pruned_graph, net_pruning = pruner.prune(spec)

    model = self._get_slim_model(pruned_graph, net_pruning)
    eval_fn(model)

  def _ana(self, eval_fn, excludes, with_group_conv: bool = False):
    analyser = sens.ModelAnalyser(self._graph, excludes, with_group_conv)

    steps = analyser.steps()
    for step in range(steps):
      model = self._prune(analyser.spec(step))

      eval_res = eval_fn(model)
      if not isinstance(eval_res, (int, float)):
        raise errors.OptimizerInvalidArgumentError(
            'int or float expected, but got {}'.format(type(eval_res)))
      analyser.record(step, eval_res)
      logging.info('Analysis complete %d/%d' % (step + 1, steps))

      del model
      keras.backend.clear_session()
      tf.compat.v1.reset_default_graph()

    analyser.save(self._sens_path)

  def _spec_from_ratio(self, net_sens, ratio, excludes):
    logging.info('Searching for appropriate ratio for each layer...')

    flops = ku.try_count_flops(self._model)
    target_flops = (1 - ratio) * flops

    flops_tolerance = 1e-2
    min_th = 1e-5
    max_th = 1 - min_th
    num_attempts, max_attempts = 0, 100

    prev_spec = None
    cur_spec = None
    while num_attempts < max_attempts:
      prev_spec = cur_spec
      num_attempts += 1
      threshold = (min_th + max_th) / 2
      cur_spec = self._spec_from_threshold(net_sens, threshold, excludes)

      slim_model = self._prune(cur_spec, mode='slim')
      current_flops = ku.try_count_flops(slim_model)
      error = abs(target_flops - current_flops) / target_flops
      if error < flops_tolerance:
        break
      if current_flops < target_flops:
        max_th = threshold
      else:
        min_th = threshold
    return cur_spec

  def _spec_from_threshold(self, net_sens, threshold, excludes):
    groups = net_sens.prunable_groups_by_threshold(threshold, excludes)
    return pruning_lib.PruningSpec(groups)

  def _spec_from_path(self, path):
    return pruning_lib.PruningSpec.deserialize(json.load(open(path, 'r')))

  def _summary(self, spec):
    orig_flops = common.readable_num(ku.try_count_flops(self._model))
    orig_params = common.readable_num(ku.try_count_params(self._model))

    slim_model = self._prune(spec, 'slim')
    current_flops = common.readable_num(ku.try_count_flops(slim_model))
    current_params = common.readable_num(ku.try_count_params(slim_model))

    header_fields = ['Metric', 'Baseline', 'Pruned']
    flops_fields = ['FLOPs', orig_flops, current_flops]
    params_fields = ['Params', orig_params, current_params]
    common.print_table(header_fields, [flops_fields, params_fields])

class OneStepPruningRunner(PruningRunner):
  """Implements channel pruning at the model level."""

  def __init__(self, model: keras.Model, input_signature: tf.TensorSpec):
    """Concrete example:

    ```python
      model = MyModel()
      pruner = OneStepPruningPruner(model, tf.TensorSpec(input_shape, tf.float32))
      model = pruner.search_subnets(0.2, train_fn, eval_fn, 1000)
    ```

    Arguments:
      model (keras.Model): Model to prune.
      input_signature(tuple or list): The input specifications of model.
    """
    super(OneStepPruningRunner, self).__init__(model, input_signature)

  def _searcher_saved_path(self, ratio):
    return os.path.join(_VAI_DIR, self._graph.name + '_search_{}'.format(ratio))

  def _random_ratios(self, count):
    return [random.random() for _ in range(count)]

  def search_subnets(self,
                     ratio,
                     train_fn,
                     eval_fn,
                     num_iterations,
                     excludes=None,
                     config=None,
                     with_group_conv: bool = False):
    if not isinstance(ratio, float):
      raise errors.OptimizerInvalidArgumentError(
          'Expect "ratio" to be float, but got {}({})'.format(
              ratio, type(ratio)))

    excluded_nodes = self._get_exclude_nodes(excludes) if excludes else []
    groups = pruning_lib.group_nodes(self._graph, excluded_nodes,
                                     with_group_conv)
    searcher = search.SubnetSearcher(groups)

    score = eval_fn(self._model)
    base_flops = ku.try_count_flops(self._model)
    searcher.set_supernet(score, base_flops)
    searcher_saved_path = self._searcher_saved_path(ratio)

    for i in range(num_iterations):
      ratios = self._random_ratios(len(groups))
      model = self._prune(searcher.spec(ratios))

      current_flops = ku.try_count_flops(model)
      flops_ratio = 1 - current_flops / base_flops
      print('Iter {}: ratios={}, flops_ratio = {}'.format(
          i, ratios, flops_ratio))
      eps = 0.05
      if flops_ratio > ratio - eps and flops_ratio < ratio + eps:
        train_fn(model)
        score = eval_fn(model)
        searcher.add_subnet(ratios, score, current_flops)
        search.save_searcher(searcher, searcher_saved_path)
        logging.info('Found subnet: ratios={}, score={}'.format(ratios, score))

      del model
      #gc.collect()
      keras.backend.clear_session()
      tf.compat.v1.reset_default_graph()

    logging.info('Search results saved in {}'.format(searcher_saved_path))

  def get_subnet(self, ratio, index=None):
    searcher = search.load_searcher(self._searcher_saved_path(ratio))
    subnet = searcher.subnet(index) if index else searcher.best_subnet()
    print('best_subnet:', subnet)
    spec = searcher.spec(subnet.ratios)
    logging.vlog(1, 'Get subnet from spec:\n{}'.format(spec))

    return self._prune(spec)

def save_weights(model,
                 filepath,
                 overwrite=True,
                 save_format=None,
                 options=None):

  filepath = generic_utils.path_to_string(filepath)
  filepath_is_h5 = (
      filepath.endswith('.h5') or filepath.endswith('.keras') or
      filepath.endswith('.hdf5'))
  if save_format is None:
    if filepath_is_h5:
      save_format = 'h5'
    else:
      save_format = 'tf'
  else:
    user_format = save_format.lower().strip()
    if user_format in ('tensorflow', 'tf'):
      save_format = 'tf'
    elif user_format in ('hdf5', 'h5', 'keras'):
      save_format = 'h5'
    else:
      raise errors.OptimizerDataFormatError(
          'Unknown format "%s". Was expecting one of {"tf", "h5"}.' %
          (save_format,))
  if save_format == 'h5':
    raise errors.OptimizerDataFormatError((
        'HDF5 format is not allowed for sparse model, please '
        'use "tf" format. See '
        'https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights'
    ))

  keras.Model.save_weights(model, filepath, overwrite, save_format, options)

def _sparsify_tensor(t: np.ndarray, out_channels, in_channels, groups: int = 1):
  """Fill 0 in removed channels."""
  tensor = copy.deepcopy(t)
  dim_size = len(tensor.shape)
  if dim_size == 0:
    # e.g layers.Normalization the param 'count' is a num so the shape = 0
    return tensor
  assert dim_size in [1, 2, 4, 5]  # 5: 3D conv
  # weight format in tensorflow: HWIO/IO/DHWIO
  if groups == 1:
    if out_channels:
      tensor[..., out_channels] = 0.0
    if in_channels and dim_size > 1:
      if dim_size == 2:
        tensor[in_channels, :] = 0.0
      else:
        tensor[..., in_channels, :] = 0.0
  else:
    out_dims_group = generate_indices_group(out_channels, tensor.shape[-1],
                                            groups)
    in_dims_group = generate_indices_group(
        in_channels, tensor.shape[-2] *
        groups, groups) if dim_size > 1 else [[]] * groups
    parts = np.split(tensor, groups, axis=-1)
    sparse_parts: List[torch.Tensor] = []
    for part, o, i in zip(parts, out_dims_group, in_dims_group):
      sparse_parts.append(_sparsify_tensor(part, o, i))
    tensor = np.concatenate(sparse_parts, axis=-1)
  return tensor

def _prune_tensor(t: np.ndarray, out_channels, in_channels):
  """Remove dimensions by giving channels."""
  tensor = copy.deepcopy(t)
  dim_size = len(tensor.shape)
  # weight format in tensorflow: HWIO/IO
  assert dim_size in [1, 2, 4]
  if dim_size == 1:
    out_axis, in_axis = 0, None
  elif dim_size == 2:
    out_axis, in_axis = 1, 0
  else:
    out_axis, in_axis = 3, 2

  if out_channels:
    tensor = np.delete(tensor, out_channels, axis=out_axis)
  if in_channels and in_axis is not None:
    tensor = np.delete(tensor, in_channels, axis=in_axis)
  return tensor

def get_pruning_runner(model, input_signature, method='one_step'):
  assert method in ['iterative', 'one_step']
  cls = IterativePruningRunner if method == 'iterative' else OneStepPruningRunner
  return cls(model, input_signature)
