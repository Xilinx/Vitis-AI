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
"""Tests for Pruning callbacks."""

import os
import tempfile

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# TODO(b/139939526): move to public API.
from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.keras import test_utils as keras_test_utils
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks

keras = tf.keras
errors_impl = tf.errors


class PruneCallbacksTest(tf.test.TestCase, parameterized.TestCase):

  _BATCH_SIZE = 20

  def _assertLogsExist(self, log_dir):
    self.assertNotEmpty(os.listdir(log_dir))

  def _pruned_model_setup(self, custom_training_loop=False):
    pruned_model = prune.prune_low_magnitude(
        keras_test_utils.build_simple_dense_model())

    x_train = np.random.rand(self._BATCH_SIZE, 10)
    y_train = keras.utils.to_categorical(
        np.random.randint(5, size=(self._BATCH_SIZE, 1)), 5)

    loss = keras.losses.categorical_crossentropy
    optimizer = keras.optimizers.SGD()

    if custom_training_loop:
      return pruned_model, loss, optimizer, x_train, y_train
    else:
      pruned_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
      return pruned_model, x_train, y_train

  @keras_parameterized.run_all_keras_modes
  def testUpdatePruningStepsAndLogsSummaries(self):
    log_dir = tempfile.mkdtemp()
    pruned_model, x_train, y_train = self._pruned_model_setup()
    pruned_model.fit(
        x_train,
        y_train,
        batch_size=self._BATCH_SIZE,
        epochs=3,
        callbacks=[
            pruning_callbacks.UpdatePruningStep(),
            pruning_callbacks.PruningSummaries(log_dir=log_dir)
        ])

    self.assertEqual(
        2, tf.keras.backend.get_value(pruned_model.layers[0].pruning_step))
    self.assertEqual(
        2, tf.keras.backend.get_value(pruned_model.layers[1].pruning_step))

    self._assertLogsExist(log_dir)

  # This style of custom training loop isn't available in graph mode.
  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def testUpdatePruningStepsAndLogsSummaries_CustomTrainingLoop(self):
    log_dir = tempfile.mkdtemp()
    pruned_model, loss, optimizer, x_train, y_train = self._pruned_model_setup(
        custom_training_loop=True)

    unused_arg = -1

    step_callback = pruning_callbacks.UpdatePruningStep()
    log_callback = pruning_callbacks.PruningSummaries(log_dir=log_dir)
    # TODO(tfmot): we need a separate API for custom training loops
    # that doesn't rely on users setting the model and optimizer.
    #
    # Example is currently based on callbacks.py configure_callbacks
    # and model.compile internals.
    step_callback.set_model(pruned_model)
    log_callback.set_model(pruned_model)
    pruned_model.optimizer = optimizer

    step_callback.on_train_begin()
    for _ in range(3):
      log_callback.on_epoch_begin(epoch=unused_arg)
      # only one batch given batch_size = 20 and input shape.
      step_callback.on_train_batch_begin(batch=unused_arg)
      inp = np.reshape(x_train,
                       [self._BATCH_SIZE, 10])  # original shape: from [10].
      with tf.GradientTape() as tape:
        logits = pruned_model(inp, training=True)
        loss_value = loss(y_train, logits)
        grads = tape.gradient(loss_value, pruned_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, pruned_model.trainable_variables))
      step_callback.on_epoch_end(batch=unused_arg)

    self.assertEqual(
        2, tf.keras.backend.get_value(pruned_model.layers[0].pruning_step))
    self.assertEqual(
        2, tf.keras.backend.get_value(pruned_model.layers[1].pruning_step))
    self._assertLogsExist(log_dir)

  @keras_parameterized.run_all_keras_modes
  def testPruneTrainingRaisesError_PruningStepCallbackMissing(self):
    pruned_model, x_train, y_train = self._pruned_model_setup()

    # Throws an error since UpdatePruningStep is missing.
    with self.assertRaises(errors_impl.InvalidArgumentError):
      pruned_model.fit(x_train, y_train)

  # This style of custom training loop isn't available in graph mode.
  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  def testPruneTrainingLoopRaisesError_PruningStepCallbackMissing_CustomTrainingLoop(
      self):
    pruned_model, _, _, x_train, _ = self._pruned_model_setup(
        custom_training_loop=True)

    # Throws an error since UpdatePruningStep is missing.
    with self.assertRaises(errors_impl.InvalidArgumentError):
      inp = np.reshape(x_train, [self._BATCH_SIZE, 10])  # original shape: [10].
      with tf.GradientTape():
        pruned_model(inp, training=True)

  @keras_parameterized.run_all_keras_modes
  def testPruningSummariesRaisesError_LogDirNotNonEmptyString(self):
    with self.assertRaises(ValueError):
      pruning_callbacks.PruningSummaries(log_dir='')

    with self.assertRaises(ValueError):
      pruning_callbacks.PruningSummaries(log_dir=None)

    with self.assertRaises(ValueError):
      pruning_callbacks.PruningSummaries(log_dir=object())


if __name__ == '__main__':
  tf.test.main()
