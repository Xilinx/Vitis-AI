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

import tensorflow as tf

from collections import OrderedDict
from tensorflow.python.util import nest
from tensorflow.python.eager import def_function
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras import layers
from tensorflow.python.training.tracking import data_structures as tf_ds
from tensorflow.python.ops import array_ops

def _keras_weight_name(name):
  # Given 'dense/kernel:0', return 'kernel'.
  return name.split('/')[-1].rsplit(':', 1)[0]

def keras_layer_params(layer):
  params = OrderedDict()
  weights = layer.get_weights()
  for i, weight in enumerate(layer.weights):
    name = _keras_weight_name(weight.name)
    # For repeated weights, append index to its name.
    # Usually this happens in a RNN layer.
    if name in params:
      name = '{}_{}'.format(name, i)
    params[name] = weights[i]
  return params

def is_lstm(layer):
  return isinstance(layer, (layers.recurrent.LSTM, layers.recurrent_v2.LSTM))

def is_stacked_rnn_cells(layer):
  return isinstance(layer, layers.StackedRNNCells)

def _is_subclassing(model):
  return not (model._is_graph_network or isinstance(model, tf.keras.Sequential))

def get_layers(model):
  layers = []
  if hasattr(model, 'layers'):
    layers = model.layers
  elif hasattr(model, '_layers'):
    sub_layers = model._layers
    if len(sub_layers) > 0:
      layers = sub_layers[0].layers if isinstance(
          sub_layers[0], tf_ds.ListWrapper) else sub_layers
  else:
    pass
  return layers

def gather_layers(model):
  layers = get_layers(model)

  all_layers = layers[:]
  for layer in layers:
    all_layers.extend(gather_layers(layer))
  return all_layers

#def trace_lstm_cell(cell, input_signature=None):
#  """Trace the model call to create a tf.function for exporting a Keras model.
#
#  Args:
#    model: A Keras model.
#    input_signature: optional, a list of tf.TensorSpec objects specifying the
#      inputs to the model.
#
#  Returns:
#    A tf.function wrapping the model's call function with input signatures set.
#
#  Raises:
#    ValueError: if input signature cannot be inferred from the model.
#  """
#  @def_function.function(input_signature=input_signature, autograph=False)
#  def _wrapped_model(*args):
#    """A concrete tf.function that wraps the model's call function."""
#    # When given a single input, Keras models will call the model on the tensor
#    # rather than a list consisting of the single tensor.
#    input_dim = cell.kernel_i.shape[0]
#    dtype = cell.kernel_i.dtype
#    inputs = array_ops.ones(shape=(1, input_dim), dtype=dtype)
#    states = cell.get_initial_state(inputs)
#    with base_layer_utils.call_context().enter(
#        cell, inputs=inputs, build_graph=False, training=False, saving=True):
#      outputs_list = nest.flatten(cell(inputs=inputs, states=states, training=False))
#
#    try:
#      output_names = cell.output_names
#    except AttributeError:
#      print('AttributeError: no output_names')
#      from tensorflow.python.keras.engine import training_utils  # pylint: disable=g-import-not-at-top
#      output_names = training_utils.generic_output_names(outputs_list)
#    return {name: output for name, output in zip(output_names, outputs_list)}
#
#  return _wrapped_model

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

  #if input_signature is None:
  #  input_signature = model_input_signature(model)

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
