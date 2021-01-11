# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Test utility to generate models for testing."""

import tempfile
import numpy as np

import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

keras = tf.keras
l = keras.layers


def _build_mnist_layer_list():
  return [
      l.Conv2D(
          32, 5, padding='same', activation='relu', input_shape=(28, 28, 1)),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.BatchNormalization(),
      l.Conv2D(64, 5, padding='same', activation='relu'),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.Flatten(),
      l.Dense(1024, activation='relu'),
      l.Dropout(0.4),
      l.Dense(10, activation='softmax')
  ]


def _build_mnist_sequential_model():
  return keras.Sequential(_build_mnist_layer_list())


def _build_mnist_functional_model():
  # pylint: disable=missing-docstring
  inp = keras.Input(shape=(28, 28, 1))
  x = l.Conv2D(32, 5, padding='same', activation='relu')(inp)
  x = l.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
  x = l.BatchNormalization()(x)
  x = l.Conv2D(64, 5, padding='same', activation='relu')(x)
  x = l.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
  x = l.Flatten()(x)
  x = l.Dense(1024, activation='relu')(x)
  x = l.Dropout(0.4)(x)
  out = l.Dense(10, activation='softmax')(x)

  return keras.models.Model([inp], [out])


def _build_mnist_layerwise_pruned_model(pruning_params):
  if pruning_params is None:
    raise ValueError('pruning_params should be provided.')

  return keras.Sequential([
      prune.prune_low_magnitude(
          l.Conv2D(32, 5, padding='same', activation='relu'),
          input_shape=(28, 28, 1),
          **pruning_params),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.BatchNormalization(),
      prune.prune_low_magnitude(
          l.Conv2D(64, 5, padding='same', activation='relu'), **pruning_params),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.Flatten(),
      prune.prune_low_magnitude(
          l.Dense(1024, activation='relu'), **pruning_params),
      l.Dropout(0.4),
      prune.prune_low_magnitude(
          l.Dense(10, activation='softmax'), **pruning_params)
  ])


def build_mnist_model(model_type, pruning_params=None):
  return {
      'sequential': _build_mnist_sequential_model(),
      'functional': _build_mnist_functional_model(),
      'layer_list': _build_mnist_layer_list(),
      'layer_wise': _build_mnist_layerwise_pruned_model(pruning_params),
  }[model_type]


def model_type_keys():
  return ['sequential', 'functional', 'layer_list', 'layer_wise']


def list_to_named_parameters(param_name, options):
  """Convert list of options for parameter to input to @parameterized.named_parameters.

  Arguments:
    param_name: name of parameter
    options: list of options for parameter

  Returns:
    named_params: input to @parameterized.named_parameters

  Needed to stack multiple parameters (e.g. with keras run_all_modes).
  """

  def snakecase_to_camelcase(value):
    # Non-comprensive check for camelcase already.
    if value[0].isupper() and '_' not in value:
      return value

    camelcase = ''
    for s in value.split('_'):
      camelcase += s.capitalize()
    return camelcase

  def name(s):
    if isinstance(s, str):
      return s

    return s.__name__

  named_params = []
  for key in options:
    named_params.append({
        'testcase_name': snakecase_to_camelcase(name(key)),
        param_name: key
    })
  return named_params


def _save_restore_keras_model(model):
  _, keras_file = tempfile.mkstemp('.h5')
  keras.models.save_model(model, keras_file)

  with prune.prune_scope():
    loaded_model = keras.models.load_model(keras_file)

  return loaded_model


def _save_restore_tf_model(model):
  tmpdir = tempfile.mkdtemp()
  tf.keras.models.save_model(model, tmpdir, save_format='tf')
  with prune.prune_scope():
    loaded_model = tf.keras.models.load_model(tmpdir)
  return loaded_model


def save_restore_fns():
  return [_save_restore_keras_model, _save_restore_tf_model]


# Assertion/Sparsity Verification functions.


def _get_sparsity(weights):
  return 1.0 - np.count_nonzero(weights) / float(weights.size)


def assert_model_sparsity(test_case, sparsity, model, rtol=1e-6, atol=1e-6):
  for layer in model.layers:
    if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      for weight in layer.layer.get_prunable_weights():
        test_case.assertAllClose(
            sparsity, _get_sparsity(tf.keras.backend.get_value(weight)), rtol=rtol, atol=atol)


# Check if model does not have target sparsity.
def is_model_sparsity_not(sparsity, model):
  for layer in model.layers:
    if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      for weight in layer.layer.get_prunable_weights():
        if sparsity != _get_sparsity(tf.keras.backend.get_value(weight)):
          return True
  return False
