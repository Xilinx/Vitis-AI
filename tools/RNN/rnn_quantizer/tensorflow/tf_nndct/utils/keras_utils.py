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

import collections
import tensorflow as tf

from distutils.version import LooseVersion
from typing import Any, Callable, Dict, List, Optional, Union

from tensorflow.keras import layers
from tensorflow.python.eager import def_function
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest

from tf_nndct.utils import logging
from tf_nndct.utils import tf_utils

_is_tf_later_than_220 = tf_utils.tf_version() >= LooseVersion('2.2')
if _is_tf_later_than_220:
  from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

keras = tf.keras

def data_format():
  return keras.backend.image_data_format()

def _keras_weight_name(name):
  # Given 'dense/kernel:0', return 'kernel'.
  return name.split('/')[-1].rsplit(':', 1)[0]

def get_named_weights(layer):
  params = collections.OrderedDict()
  weights = layer.get_weights()
  for i, weight in enumerate(layer.weights):
    name = _keras_weight_name(weight.name)
    # For repeated weights, append index to its name.
    # Usually this happens in a RNN layer.
    if name in params:
      name = '{}_{}'.format(name, i)
    params[name] = weights[i]
  return params

def is_stacked_rnn_cells(layer):
  return isinstance(layer, layers.StackedRNNCells)

def is_sequential_or_functional(model):
  return isinstance(model, keras.Model) and (isinstance(model, keras.Sequential)
                                             or model._is_graph_network)

def is_subclassing(model):
  return isinstance(model, keras.Model) and not is_sequential_or_functional(model)

def get_layers(model):
  layers = []
  if hasattr(model, 'layers'):
    layers = model.layers
  elif hasattr(model, '_layers'):
    sub_layers = model._layers
    if len(sub_layers) > 0:
      layers = sub_layers[0].layers if isinstance(
          sub_layers[0], data_structures.ListWrapper) else sub_layers
  else:
    pass
  return layers

# Copy from https://github.com/tensorflow/tensorflow/blob/r2.3/tensorflow/python/keras/engine/base_layer.py#L2847
def flatten_layers(layer, recursive=True, include_self=True):
  if include_self:
    yield layer

  # Only instantiate set and deque if needed.
  layers_or_containers = getattr(layer, '_layers', None)
  if layers_or_containers:
    seen_object_ids = set()
    deque = collections.deque(layers_or_containers)
    while deque:
      layer_or_container = deque.popleft()

      layer_or_container_id = id(layer_or_container)
      if layer_or_container_id in seen_object_ids:
        continue
      seen_object_ids.add(layer_or_container_id)

      if isinstance(layer_or_container, layers.Layer):
        yield layer_or_container
        # Introspect recursively through sublayers.
        if recursive:
          sublayers = getattr(layer_or_container, '_layers', None)
          if sublayers:
            deque.extendleft(reversed(sublayers))
      elif isinstance(layer_or_container,
                      data_structures.TrackableDataStructure):
        # Data structures are introspected even with `recursive=False`.
        tracked_values = layer_or_container._values
        if tracked_values:
          deque.extendleft(reversed(tracked_values))

def gather_layers(layer, include_container=False):
  """Gather all sub layers from given model.

    Args:
      layer: An instance of keras.Layer
      include_container: Whether to include layer container
  """
  if not isinstance(layer, keras.Model):
    return []

  gathered_layers = []
  to_visit = collections.deque([layer])
  while to_visit:
    obj = to_visit.popleft()
    if isinstance(obj, keras.Model):
      if include_container:
        gathered_layers.append(obj)
      to_visit.extendleft(reversed(obj.layers))
    else:
      gathered_layers.append(obj)
  return gathered_layers

# Modified from tensorflow/python/keras/saving/saving_utils.py:trace_model_call
def trace_model_call(model, input_signature=None):
  """Trace the model call to create a tf.function for exporting a Keras model.

  Args:
    model: A Keras model.
    input_signature: optional, a list of tf.TensorSpec objects specifying the
      inputs to the model.

  Returns:
    A tf.function wrapping the model's call function with input signatures set.

  Raises:
    ValueError: if input signature cannot be inferred from the model.
  """
  if input_signature is None and isinstance(model.call, def_function.Function):
    input_signature = model.call.input_signature

  @def_function.function(input_signature=input_signature, autograph=False)
  def _wrapped_model(*args):
    """A concrete tf.function that wraps the model's call function."""
    # When given a single input, Keras models will call the model on the tensor
    # rather than a list consisting of the single tensor.
    # inputs = args[0] if len(input_signature) == 1 else list(args)
    inputs = args[0] if len(args) == 1 else list(args)

    with base_layer_utils.call_context().enter(
        model, inputs=inputs, build_graph=False, training=False, saving=True):
      return model(*args, training=False)
      #outputs_list = nest.flatten(model(*args, training=False))

    #try:
    #  output_names = model.output_names
    #except AttributeError:
    #  from tensorflow.python.keras.engine import training_utils  # pylint: disable=g-import-not-at-top
    #  output_names = training_utils.generic_output_names(outputs_list)
    #return {name: output for name, output in zip(output_names, outputs_list)}

  return _wrapped_model

# https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/python/keras/engine/base_layer_utils.py#L261
def unnest_if_single_tensor(input_tensors):
  # Preserve compatibility with older configs
  flat_input_tensors = nest.flatten(input_tensors)
  # If this is a single element but not a dict, unwrap. If this is a dict,
  # assume the first layer expects a dict (as is the case with a
  # DenseFeatures layer); pass through.
  if not isinstance(input_tensors, dict) and len(flat_input_tensors) == 1:
    input_tensors = flat_input_tensors[0]
  return input_tensors

def try_count_params(model: Union[tf.Module, tf.keras.Model],
                     trainable_only: bool = False):
  """Count the number of parameters if model is possible.
  Args:
    model: Try to count the number of params in this model.
    trainable_only: Whether to calculate trainable params only. This flag is
      not used when the model has `count_params` attribute.
  Returns:
    The number of parameters or None.
  """
  if hasattr(model, 'count_params'):
    try:
      return model.count_params()
    except ValueError:
      logging.info('Number of trainable params unknown, because the build() '
                   'methods in keras layers were not called. This is probably '
                   'because the model was not feed any input, e.g., the max '
                   'train step already reached before this run.')
      return None
  else:
    total_params = 0
    variables = model.trainable_variables if trainable_only else model.variables
    for var in variables:
      shape = tf.shape(var)
      total_params += tf.math.reduce_prod(shape).numpy()
  return total_params

def try_count_flops(model: Union[tf.Module, tf.keras.Model],
                    inputs_kwargs: Optional[Dict[str, Any]] = None):
  """Counts and returns model FLOPs.
  Args:
    model: A model instance.
    inputs_kwargs: An optional dictionary of argument pairs specifying inputs'
      shape specifications to getting corresponding concrete function.
  Returns:
    The model's FLOPs.
  """
  if hasattr(model, 'inputs'):
    try:
      # Get input shape and set batch size to 1.
      if model.inputs:
        inputs = [
            tf.TensorSpec([1] + input.shape[1:], input.dtype)
            for input in model.inputs
        ]
        concrete_func = tf.function(model).get_concrete_function(inputs)
      # If model.inputs is invalid, try to use the input to get concrete
      # function for model.call (subclass model).
      else:
        concrete_func = tf.function(
            model.call).get_concrete_function(**inputs_kwargs)

      if _is_tf_later_than_220:
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(
            concrete_func)
      else:
        frozen_func = tf_utils.convert_to_constants(
            concrete_func, lower_control_flow=False)

      # Calculate FLOPs.
      run_meta = tf.compat.v1.RunMetadata()
      opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
      opts['output'] = 'none'
      flops = tf.compat.v1.profiler.profile(
          graph=frozen_func.graph, run_meta=run_meta, options=opts)
      return flops.total_float_ops
    except Exception as e:  # pylint: disable=broad-except
      logging.info(
          'Failed to count model FLOPs with error %s, because the build() '
          'methods in keras layers were not called. This is probably because '
          'the model was not feed any input, e.g., the max train step already '
          'reached before this run.', e)
      return None
  return None
