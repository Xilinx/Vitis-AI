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
"""End to End tests for the Pruning API."""

import tempfile

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# TODO(b/139939526): move to public API.
from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.keras import test_utils as keras_test_utils
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow_model_optimization.python.core.sparsity.keras import test_utils


keras = tf.keras
layers = keras.layers

list_to_named_parameters = test_utils.list_to_named_parameters
ModelCompare = keras_test_utils.ModelCompare


@keras_parameterized.run_all_keras_modes
class PruneIntegrationTest(tf.test.TestCase, parameterized.TestCase,
                           ModelCompare):

  # Fetch all the prunable layers from the registry.
  _PRUNABLE_LAYERS = [
      layer for layer, weights in
      prune_registry.PruneRegistry._LAYERS_WEIGHTS_MAP.items()
        if (weights and layer != tf.keras.layers.Conv3DTranspose
                    and layer != tf.keras.layers.Conv2DTranspose)
  ]

  # Fetch all the non-prunable layers from the registry.
  _NON_PRUNABLE_LAYERS = [
      layer for layer, weights in
      prune_registry.PruneRegistry._LAYERS_WEIGHTS_MAP.items()
      if not weights
  ]

  @staticmethod
  def _batch(dims, batch_size):
    """Adds provided batch_size to existing dims.

    If dims is (None, 5, 2), returns (batch_size, 5, 2)

    Args:
      dims: Dimensions
      batch_size: batch_size

    Returns:
      dims with batch_size added as first parameter of list.
    """
    if dims[0] is None:
      dims[0] = batch_size
    return dims

  @staticmethod
  def _get_params_for_layer(layer_type):
    """Returns the arguments required to construct a layer.

    For a given `layer_type`, return ( [params], (input_shape) )

    Args:
      layer_type: Type of layer, Dense, Conv2D etc.

    Returns:
      Arguments to construct layer as a 2-tuple. First value is the list of
      params required to construct the layer. Second value is the input_shape
      for the layer.
    """
    return {
        layers.Conv1D: ([4, 2], (3, 6)),
        layers.Conv2D: ([4, (2, 2)], (4, 6, 1)),
        # TODO(tf-mot): fix for Conv2DTranspose on some form of eager,
        # with or without functions. The weights become nan (though the
        # mask seems fine still).
        #layers.Conv2DTranspose: ([2, (3, 3)], (7, 6, 3)),
        layers.Conv3D: ([2, (3, 3, 3)], (5, 7, 6, 3)),
        # TODO(tf-mot): fix for Conv3DTranspose on some form of eager,
        # with or without functions. The weights become nan (though the
        # mask seems fine still).
        #layers.Conv3DTranspose: ([2, (3, 3, 3)], (5, 7, 6, 3)),
        layers.SeparableConv1D: ([4, 3], (3, 6)),
        layers.SeparableConv2D: ([4, (2, 2)], (4, 6, 1)),
        layers.Dense: ([4], (6,)),
        layers.LocallyConnected1D: ([4, 2], (3, 6)),
        layers.LocallyConnected2D: ([4, (2, 2)], (4, 6, 1)),

        # Embedding has a separate test since training it is not
        # feasible as a single layer.
        layers.Embedding: (None, None),
    }[layer_type]

  def setUp(self):
    super(PruneIntegrationTest, self).setUp()
    self.params = {
        'pruning_schedule': pruning_schedule.ConstantSparsity(0.5, 0, -1, 1),
        # TODO(pulkitb): Add tests for block sparsity.
        'block_size': (1, 1),
        'block_pooling_type': 'AVG'
    }

  # TODO(pulkitb): Also assert correct weights are pruned.
  # TODO(tfmot): this should not be verified in all the unit tests.
  # As long as there are a few unit tests for strip_pruning,
  # these checks are redundant.
  def _check_strip_pruning_matches_original(
      self, model, sparsity, input_data=None):
    stripped_model = prune.strip_pruning(model)
    test_utils.assert_model_sparsity(self, sparsity, stripped_model)

    if input_data is None:
      input_data = np.random.randn(
          *self._batch(model.input.get_shape().as_list(), 1))

    model_result = model.predict(input_data)
    stripped_model_result = stripped_model.predict(input_data)
    np.testing.assert_almost_equal(model_result, stripped_model_result)

  @staticmethod
  def _is_pruned(model):
    for layer in model.layers:
      if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
        return True

  @staticmethod
  def _train_model(model, epochs=1, x_train=None, y_train=None, callbacks=None):
    if x_train is None:
      x_train = np.random.rand(20, 10),
    if y_train is None:
      y_train = keras.utils.to_categorical(
          np.random.randint(5, size=(20, 1)), 5)

    if model.optimizer is None:
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])

    if callbacks is None:
      callbacks = []
      if PruneIntegrationTest._is_pruned(model):
        callbacks = [pruning_callbacks.UpdatePruningStep()]

    model.fit(
        x_train, y_train, epochs=epochs, batch_size=20, callbacks=callbacks)

  def _get_pretrained_model(self):
    model = keras_test_utils.build_simple_dense_model()
    self._train_model(model, epochs=1)
    return model

  ###################################################################
  # Sanity checks and special cases for training with pruning.

  def testPrunesZeroSparsity_IsNoOp(self):
    model = keras_test_utils.build_simple_dense_model()

    model2 = keras_test_utils.build_simple_dense_model()
    model2.set_weights(model.get_weights())

    params = self.params
    params['pruning_schedule'] = pruning_schedule.ConstantSparsity(
        target_sparsity=0, begin_step=0, frequency=1)
    pruned_model = prune.prune_low_magnitude(model2, **params)

    x_train = np.random.rand(20, 10),
    y_train = keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5)

    self._train_model(model, epochs=1, x_train=x_train, y_train=y_train)
    self._train_model(pruned_model, epochs=1, x_train=x_train, y_train=y_train)

    self._assert_weights_different_objects(model, pruned_model)
    self._assert_weights_same_values(model, pruned_model)

  # TODO(tfmot): https://github.com/tensorflow/model-optimization/issues/215
  def testPruneWithHighSparsity_Fails(self):
    params = self.params
    params['pruning_schedule'] = pruning_schedule.ConstantSparsity(
        target_sparsity=0.99, begin_step=0, frequency=1)

    model = prune.prune_low_magnitude(
        keras_test_utils.build_simple_dense_model(), **params)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self._train_model(model, epochs=1)

  ###################################################################
  # Tests for training with pruning with pretrained models or weights.

  def testPrunePretrainedModel_RemovesOptimizer(self):
    model = self._get_pretrained_model()

    self.assertIsNotNone(model.optimizer)
    pruned_model = prune.prune_low_magnitude(model, **self.params)
    self.assertIsNone(pruned_model.optimizer)

  def testPrunePretrainedModel_PreservesWeightObjects(self):
    model = self._get_pretrained_model()

    pruned_model = prune.prune_low_magnitude(model, **self.params)
    self._assert_weights_same_objects(model, pruned_model)

  def testPrunePretrainedModel_SameInferenceWithoutTraining(self):
    model = self._get_pretrained_model()
    pruned_model = prune.prune_low_magnitude(model, **self.params)

    input_data = np.random.rand(10, 10)

    out = model.predict(input_data)
    pruned_out = pruned_model.predict(input_data)

    self.assertTrue((out == pruned_out).all())

  def testLoadTFWeightsThenPrune_SameInferenceWithoutTraining(self):
    model = self._get_pretrained_model()

    _, tf_weights = tempfile.mkstemp('.tf')
    model.save_weights(tf_weights)

    # load weights into model then prune.
    same_architecture_model = keras_test_utils.build_simple_dense_model()
    same_architecture_model.load_weights(tf_weights)
    pruned_model = prune.prune_low_magnitude(same_architecture_model,
                                             **self.params)

    input_data = np.random.rand(10, 10)

    out = model.predict(input_data)
    pruned_out = pruned_model.predict(input_data)

    self.assertTrue((out == pruned_out).all())

  # Test this and _DifferentInferenceWithoutTraining
  # because pruning and then loading pretrained weights
  # is unusual behavior and extra coverage is safer.
  def testPruneThenLoadTFWeights_DoesNotPreserveWeights(self):
    model = self._get_pretrained_model()

    _, tf_weights = tempfile.mkstemp('.tf')
    model.save_weights(tf_weights)

    # load weights into pruned model.
    same_architecture_model = keras_test_utils.build_simple_dense_model()
    pruned_model = prune.prune_low_magnitude(same_architecture_model,
                                             **self.params)
    pruned_model.load_weights(tf_weights)

    self._assert_weights_different_values(model, pruned_model)

  def testPruneThenLoadTFWeights_DifferentInferenceWithoutTraining(self):
    model = self._get_pretrained_model()

    _, tf_weights = tempfile.mkstemp('.tf')
    model.save_weights(tf_weights)

    # load weights into pruned model.
    same_architecture_model = keras_test_utils.build_simple_dense_model()
    pruned_model = prune.prune_low_magnitude(same_architecture_model,
                                             **self.params)
    pruned_model.load_weights(tf_weights)

    input_data = np.random.rand(10, 10)

    out = model.predict(input_data)
    pruned_out = pruned_model.predict(input_data)

    self.assertFalse((out == pruned_out).any())

  def testPruneThenLoadsKerasWeights_Fails(self):
    model = self._get_pretrained_model()

    _, keras_weights = tempfile.mkstemp('.h5')
    model.save_weights(keras_weights)

    # load weights into pruned model.
    same_architecture_model = keras_test_utils.build_simple_dense_model()
    pruned_model = prune.prune_low_magnitude(same_architecture_model,
                                             **self.params)

    # error since number of keras_weights is fewer than weights in pruned model
    # because pruning introduces weights.
    with self.assertRaises(ValueError):
      pruned_model.load_weights(keras_weights)

  ###################################################################
  # Tests for training with pruning from scratch.

  @parameterized.parameters(_PRUNABLE_LAYERS)
  def testPrunesSingleLayer_ReachesTargetSparsity(self, layer_type):
    model = keras.Sequential()
    args, input_shape = self._get_params_for_layer(layer_type)
    if args is None:
      return  # Test for layer not supported yet.
    model.add(prune.prune_low_magnitude(
        layer_type(*args), input_shape=input_shape, **self.params))

    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    test_utils.assert_model_sparsity(self, 0.0, model)
    model.fit(
        np.random.randn(*self._batch(model.input.get_shape().as_list(), 32)),
        np.random.randn(*self._batch(model.output.get_shape().as_list(), 32)),
        callbacks=[pruning_callbacks.UpdatePruningStep()])

    test_utils.assert_model_sparsity(self, 0.5, model)

    self._check_strip_pruning_matches_original(model, 0.5)

  @parameterized.parameters(prune_registry.PruneRegistry._RNN_LAYERS -
                            {keras.layers.RNN})
  def testRNNLayersSingleCell_ReachesTargetSparsity(self, layer_type):
    model = keras.Sequential()
    model.add(
        prune.prune_low_magnitude(
            layer_type(10), input_shape=(3, 4), **self.params))

    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    test_utils.assert_model_sparsity(self, 0.0, model)
    model.fit(
        np.random.randn(*self._batch(model.input.get_shape().as_list(), 32)),
        np.random.randn(*self._batch(model.output.get_shape().as_list(), 32)),
        callbacks=[pruning_callbacks.UpdatePruningStep()])

    test_utils.assert_model_sparsity(self, 0.5, model)

    self._check_strip_pruning_matches_original(model, 0.5)

  def testRNNLayersWithRNNCellParams_ReachesTargetSparsity(self):
    model = keras.Sequential()
    model.add(
        prune.prune_low_magnitude(
            keras.layers.RNN([
                layers.LSTMCell(10),
                layers.GRUCell(10),
                tf.keras.experimental.PeepholeLSTMCell(10),
                layers.SimpleRNNCell(10)
            ]),
            input_shape=(3, 4),
            **self.params))

    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    test_utils.assert_model_sparsity(self, 0.0, model)
    model.fit(
        np.random.randn(*self._batch(model.input.get_shape().as_list(), 32)),
        np.random.randn(*self._batch(model.output.get_shape().as_list(), 32)),
        callbacks=[pruning_callbacks.UpdatePruningStep()])

    test_utils.assert_model_sparsity(self, 0.5, model)

    self._check_strip_pruning_matches_original(model, 0.5)

  def testPrunesEmbedding_ReachesTargetSparsity(self):
    model = keras.Sequential()
    model.add(
        prune.prune_low_magnitude(
            layers.Embedding(input_dim=10, output_dim=3),
            input_shape=(5,),
            **self.params))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    test_utils.assert_model_sparsity(self, 0.0, model)
    model.fit(
        np.random.randint(10, size=(32, 5)),
        np.random.randint(2, size=(32, 1)),
        callbacks=[pruning_callbacks.UpdatePruningStep()])

    test_utils.assert_model_sparsity(self, 0.5, model)

    input_data = np.random.randint(10, size=(32, 5))
    self._check_strip_pruning_matches_original(model, 0.5, input_data)

  @parameterized.parameters(test_utils.model_type_keys())
  def testPrunesMnist_ReachesTargetSparsity(self, model_type):
    model = test_utils.build_mnist_model(model_type, self.params)
    if model_type == 'layer_list':
      model = keras.Sequential(prune.prune_low_magnitude(model, **self.params))
    elif model_type in ['sequential', 'functional']:
      model = prune.prune_low_magnitude(model, **self.params)

    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    test_utils.assert_model_sparsity(self, 0.0, model, rtol=1e-4, atol=1e-4)
    model.fit(
        np.random.rand(32, 28, 28, 1),
        keras.utils.to_categorical(np.random.randint(10, size=(32, 1)), 10),
        callbacks=[pruning_callbacks.UpdatePruningStep()])

    test_utils.assert_model_sparsity(self, 0.5, model, rtol=1e-4, atol=1e-4)

    self._check_strip_pruning_matches_original(model, 0.5)

  ###################################################################
  # Tests for pruning with checkpointing.

  # TODO(tfmot): https://github.com/tensorflow/model-optimization/issues/206.
  #
  # Note the following:
  # 1. This test doesn't exactly reproduce bug. Test should sometimes
  # pass when ModelCheckpoint save_freq='epoch'. The behavior was seen when
  # training mobilenet.
  # 2. testPruneStopAndRestart_PreservesSparsity passes, indicating
  # checkpointing in general works. Just don't use the checkpoint for
  # serving.
  def testPruneCheckpoints_CheckpointsNotSparse(self):
    is_model_sparsity_not_list = []

    # Run multiple times since problem doesn't always happen.
    for _ in range(3):
      model = keras_test_utils.build_simple_dense_model()
      pruned_model = prune.prune_low_magnitude(model, **self.params)

      checkpoint_dir = tempfile.mkdtemp()
      checkpoint_path = checkpoint_dir + '/weights.{epoch:02d}.tf'

      callbacks = [
          pruning_callbacks.UpdatePruningStep(),
          tf.keras.callbacks.ModelCheckpoint(
              filepath=checkpoint_path, save_weights_only=True, save_freq=1)
      ]

      # Train one step. Sparsity reaches final sparsity.
      self._train_model(pruned_model, epochs=1, callbacks=callbacks)
      test_utils.assert_model_sparsity(self, 0.5, pruned_model)

      latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

      same_architecture_model = keras_test_utils.build_simple_dense_model()
      pruned_model = prune.prune_low_magnitude(same_architecture_model,
                                               **self.params)

      # Sanity check.
      test_utils.assert_model_sparsity(self, 0, pruned_model)

      pruned_model.load_weights(latest_checkpoint)
      is_model_sparsity_not_list.append(
          test_utils.is_model_sparsity_not(0.5, pruned_model))

    self.assertTrue(any(is_model_sparsity_not_list))

  @parameterized.parameters(test_utils.save_restore_fns())
  def testPruneStopAndRestart_PreservesSparsity(self, save_restore_fn):
    # TODO(tfmot): renable once SavedModel preserves step again.
    # This existed in TF 2.0 and 2.1 and should be reenabled in
    # TF 2.3. b/151755698
    if save_restore_fn.__name__ == '_save_restore_tf_model':
      return

    begin_step, end_step = 0, 4
    params = self.params
    params['pruning_schedule'] = pruning_schedule.PolynomialDecay(
        0.2, 0.6, begin_step, end_step, 3, 1)

    model = prune.prune_low_magnitude(
        keras_test_utils.build_simple_dense_model(), **params)
    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # Model hasn't been trained yet. Sparsity 0.0
    test_utils.assert_model_sparsity(self, 0.0, model)

    # Train only 1 step. Sparsity 0.2 (initial_sparsity)
    self._train_model(model, epochs=1)
    test_utils.assert_model_sparsity(self, 0.2, model)

    model = save_restore_fn(model)

    # Training has run all 4 steps. Sparsity 0.6 (final_sparsity)
    self._train_model(model, epochs=3)
    test_utils.assert_model_sparsity(self, 0.6, model)

    self._check_strip_pruning_matches_original(model, 0.6)

  @parameterized.parameters(test_utils.save_restore_fns())
  def testPruneWithPolynomialDecayPastEndStep_PreservesSparsity(
      self, save_restore_fn):
    # TODO(tfmot): renable once SavedModel preserves step again.
    # This existed in TF 2.0 and 2.1 and should be reenabled in
    # TF 2.3. b/151755698
    if save_restore_fn.__name__ == '_save_restore_tf_model':
      return

    begin_step, end_step = 0, 2
    params = self.params
    params['pruning_schedule'] = pruning_schedule.PolynomialDecay(
        0.2, 0.6, begin_step, end_step, 3, 1)

    model = prune.prune_low_magnitude(
        keras_test_utils.build_simple_dense_model(), **params)
    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # Model hasn't been trained yet. Sparsity 0.0
    test_utils.assert_model_sparsity(self, 0.0, model)

    # Train 3 steps, past end_step. Sparsity 0.6 (final_sparsity)
    self._train_model(model, epochs=3)
    test_utils.assert_model_sparsity(self, 0.6, model)

    model = save_restore_fn(model)

    # Ensure sparsity is preserved.
    test_utils.assert_model_sparsity(self, 0.6, model)

    # Train one more step to ensure nothing happens that brings sparsity
    # back below 0.6.
    self._train_model(model, epochs=1)
    test_utils.assert_model_sparsity(self, 0.6, model)

    self._check_strip_pruning_matches_original(model, 0.6)


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class PruneIntegrationCustomTrainingLoopTest(tf.test.TestCase,
                                             parameterized.TestCase):

  def testPrunesModel_CustomTrainingLoop_ReachesTargetSparsity(self):
    pruned_model = prune.prune_low_magnitude(
        keras_test_utils.build_simple_dense_model())

    batch_size = 20
    x_train = np.random.rand(20, 10)
    y_train = keras.utils.to_categorical(
        np.random.randint(5, size=(batch_size, 1)), 5)

    loss = keras.losses.categorical_crossentropy
    optimizer = keras.optimizers.SGD()

    unused_arg = -1

    step_callback = pruning_callbacks.UpdatePruningStep()
    step_callback.set_model(pruned_model)
    pruned_model.optimizer = optimizer

    step_callback.on_train_begin()
    # 2 epochs
    for _ in range(2):
      step_callback.on_train_batch_begin(batch=unused_arg)
      inp = np.reshape(x_train, [batch_size, 10])  # original shape: from [10].
      with tf.GradientTape() as tape:
        logits = pruned_model(inp, training=True)
        loss_value = loss(y_train, logits)
        grads = tape.gradient(loss_value, pruned_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, pruned_model.trainable_variables))
      step_callback.on_epoch_end(batch=unused_arg)

    test_utils.assert_model_sparsity(self, 0.5, pruned_model)


if __name__ == '__main__':
  tf.test.main()
