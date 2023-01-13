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
"""Keras callbacks for pruning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import g3
import numpy as np
import six
import tensorflow as tf

from tensorflow_model_optimization.python.core.keras import compat
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

K = tf.keras.backend
callbacks = tf.keras.callbacks


def _collect_prunable_layers(model):
  """Recursively collect the prunable layers in the model."""
  prunable_layers = []
  for layer in model.layers:
    # A keras model may have other models as layers.
    if isinstance(layer, tf.keras.Model):
      prunable_layers += _collect_prunable_layers(layer)
    if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      prunable_layers.append(layer)

  return prunable_layers


class UpdatePruningStep(callbacks.Callback):
  """Keras callback which updates pruning wrappers with the optimizer step.

  This callback must be used when training a model which needs to be pruned. Not
  doing so will throw an error.

  Example:

  ```python
  model.fit(x, y,
      callbacks=[UpdatePruningStep()])
  ```
  """

  def __init__(self):
    super(UpdatePruningStep, self).__init__()
    self.prunable_layers = []

  def on_train_begin(self, logs=None):
    # Collect all the prunable layers in the model.
    self.prunable_layers = _collect_prunable_layers(self.model)
    self.step = K.get_value(self.model.optimizer.iterations)

  def on_train_batch_begin(self, batch, logs=None):
    tuples = []
    for layer in self.prunable_layers:
      tuples.append((layer.pruning_step, self.step))

    K.batch_set_value(tuples)
    self.step = self.step + 1

  def on_epoch_end(self, batch, logs=None):
    # At the end of every epoch, remask the weights. This ensures that when
    # the model is saved after completion, the weights represent mask*weights.
    weight_mask_ops = []

    for layer in self.prunable_layers:
      if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        if tf.executing_eagerly():
          layer.pruning_obj.weight_mask_op()
        else:
          weight_mask_ops.append(layer.pruning_obj.weight_mask_op())

    K.batch_get_value(weight_mask_ops)


class PruningSummaries(callbacks.TensorBoard):
  """A Keras callback for adding pruning summaries to tensorboard.

  Logs the sparsity(%) and threshold at a given iteration step.
  """

  def __init__(self, log_dir, update_freq='epoch', **kwargs):
    if not isinstance(log_dir, six.string_types) or not log_dir:
      raise ValueError(
          '`log_dir` must be a non-empty string. You passed `log_dir`='
          '{input}.'.format(input=log_dir))

    super(PruningSummaries, self).__init__(
        log_dir=log_dir, update_freq=update_freq, **kwargs)
    if not compat.is_v1_apis():  # TF 2.X
      log_dir = self.log_dir + '/metrics'
      self._file_writer = tf.summary.create_file_writer(log_dir)

  def _log_pruning_metrics(self, logs, prefix, step):
    if compat.is_v1_apis():
      # Safely depend on TF 1.X private API given
      # no more 1.X releases.
      self._write_custom_summaries(step, logs)
    else:
      with self._file_writer.as_default():
        for name, value in logs.items():
          tf.summary.scalar(name, value, step=step)

        self._file_writer.flush()

  def on_epoch_begin(self, epoch, logs=None):
    if logs is not None:
      super(PruningSummaries, self).on_epoch_begin(epoch, logs)

    pruning_logs = {}
    params = []
    prunable_layers = _collect_prunable_layers(self.model)
    for layer in prunable_layers:
      for _, mask, threshold in layer.pruning_vars:
        params.append(mask)
        params.append(threshold)

    params.append(self.model.optimizer.iterations)

    values = K.batch_get_value(params)
    iteration = values[-1]
    del values[-1]
    del params[-1]

    param_value_pairs = list(zip(params, values))

    for mask, mask_value in param_value_pairs[::2]:
      pruning_logs.update({
          mask.name + '/sparsity': 1 - np.mean(mask_value)
      })

    for threshold, threshold_value in param_value_pairs[1::2]:
      pruning_logs.update({threshold.name + '/threshold': threshold_value})

    self._log_pruning_metrics(pruning_logs, '', iteration)
