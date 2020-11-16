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
"""Distributed pruning test."""

import tempfile
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_model_optimization.python.core.keras import test_utils as keras_test_utils
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import test_utils

keras = tf.keras


def _distribution_strategies():
  return [
      tf.distribute.experimental.MultiWorkerMirroredStrategy(),
      tf.distribute.MirroredStrategy(),
      # TODO(pulkitb): Add parameter_server
      # tf.distribute.experimental.ParameterServerStrategy,
      tf.distribute.OneDeviceStrategy('/cpu:0'),
  ]


class PruneDistributedTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(PruneDistributedTest, self).setUp()
    self.params = {
        'pruning_schedule': pruning_schedule.ConstantSparsity(0.5, 0, -1, 1),
        'block_size': (1, 1),
        'block_pooling_type': 'AVG'
    }

  @parameterized.parameters(_distribution_strategies())
  def testPrunesSimpleDenseModel(self, distribution):
    with distribution.scope():
      model = prune.prune_low_magnitude(
          keras_test_utils.build_simple_dense_model(), **self.params)
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])

    # Model hasn't been trained yet. Sparsity 0.0
    test_utils.assert_model_sparsity(self, 0.0, model)

    # Simple unpruned model. No sparsity.
    model.fit(
        np.random.rand(20, 10),
        keras.utils.to_categorical(np.random.randint(5, size=(20, 1)), 5),
        epochs=2,
        callbacks=[pruning_callbacks.UpdatePruningStep()],
        batch_size=20)
    model.predict(np.random.rand(20, 10))
    test_utils.assert_model_sparsity(self, 0.5, model)

    _, keras_file = tempfile.mkstemp('.h5')
    keras.models.save_model(model, keras_file)

    with prune.prune_scope():
      loaded_model = keras.models.load_model(keras_file)

    test_utils.assert_model_sparsity(self, 0.5, loaded_model)


if __name__ == '__main__':
  tf.test.main()
