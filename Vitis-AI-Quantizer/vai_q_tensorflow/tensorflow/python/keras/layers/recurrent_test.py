# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for recurrent layers functionality other than GRU, LSTM, SimpleRNN.

See also: lstm_test.py, gru_test.py, simplernn_test.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.layers import recurrent as rnn_v1
from tensorflow.python.keras.layers import recurrent_v2 as rnn_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity

# Used for nested input/output/state RNN test.
NestedInput = collections.namedtuple('NestedInput', ['t1', 't2'])
NestedState = collections.namedtuple('NestedState', ['s1', 's2'])


@keras_parameterized.run_all_keras_modes
class RNNTest(keras_parameterized.TestCase):

  def test_minimal_rnn_cell_non_layer(self):

    class MinimalRNNCell(object):

      def __init__(self, units, input_dim):
        self.units = units
        self.state_size = units
        self.kernel = keras.backend.variable(
            np.random.random((input_dim, units)))

      def call(self, inputs, states):
        prev_output = states[0]
        output = keras.backend.dot(inputs, self.kernel) + prev_output
        return output, [output]

    # Basic test case.
    cell = MinimalRNNCell(32, 5)
    x = keras.Input((None, 5))
    layer = keras.layers.RNN(cell)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

    # Test stacking.
    cells = [MinimalRNNCell(8, 5),
             MinimalRNNCell(32, 8),
             MinimalRNNCell(32, 32)]
    layer = keras.layers.RNN(cells)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

  def test_minimal_rnn_cell_non_layer_multiple_states(self):

    class MinimalRNNCell(object):

      def __init__(self, units, input_dim):
        self.units = units
        self.state_size = (units, units)
        self.kernel = keras.backend.variable(
            np.random.random((input_dim, units)))

      def call(self, inputs, states):
        prev_output_1 = states[0]
        prev_output_2 = states[1]
        output = keras.backend.dot(inputs, self.kernel)
        output += prev_output_1
        output -= prev_output_2
        return output, [output * 2, output * 3]

    # Basic test case.
    cell = MinimalRNNCell(32, 5)
    x = keras.Input((None, 5))
    layer = keras.layers.RNN(cell)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

    # Test stacking.
    cells = [MinimalRNNCell(8, 5),
             MinimalRNNCell(16, 8),
             MinimalRNNCell(32, 16)]
    layer = keras.layers.RNN(cells)
    self.assertEqual(layer.cell.state_size, ((8, 8), (16, 16), (32, 32)))
    self.assertEqual(layer.cell.output_size, 32)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

  def test_minimal_rnn_cell_layer(self):

    class MinimalRNNCell(keras.layers.Layer):

      def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

      def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

      def call(self, inputs, states):
        prev_output = states[0]
        h = keras.backend.dot(inputs, self.kernel)
        output = h + keras.backend.dot(prev_output, self.recurrent_kernel)
        return output, [output]

      def get_config(self):
        config = {'units': self.units}
        base_config = super(MinimalRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # Test basic case.
    x = keras.Input((None, 5))
    cell = MinimalRNNCell(32)
    layer = keras.layers.RNN(cell)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

    # Test basic case serialization.
    x_np = np.random.random((6, 5, 5))
    y_np = model.predict(x_np)
    weights = model.get_weights()
    config = layer.get_config()
    with keras.utils.CustomObjectScope({'MinimalRNNCell': MinimalRNNCell}):
      layer = keras.layers.RNN.from_config(config)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.set_weights(weights)
    y_np_2 = model.predict(x_np)
    self.assertAllClose(y_np, y_np_2, atol=1e-4)

    # Test stacking.
    cells = [MinimalRNNCell(8),
             MinimalRNNCell(12),
             MinimalRNNCell(32)]
    layer = keras.layers.RNN(cells)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

    # Test stacked RNN serialization.
    x_np = np.random.random((6, 5, 5))
    y_np = model.predict(x_np)
    weights = model.get_weights()
    config = layer.get_config()
    with keras.utils.CustomObjectScope({'MinimalRNNCell': MinimalRNNCell}):
      layer = keras.layers.RNN.from_config(config)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.set_weights(weights)
    y_np_2 = model.predict(x_np)
    self.assertAllClose(y_np, y_np_2, atol=1e-4)

  def test_minimal_rnn_cell_abstract_rnn_cell(self):

    class MinimalRNNCell(keras.layers.AbstractRNNCell):

      def __init__(self, units, **kwargs):
        self.units = units
        super(MinimalRNNCell, self).__init__(**kwargs)

      @property
      def state_size(self):
        return self.units

      def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

      def call(self, inputs, states):
        prev_output = states[0]
        h = keras.backend.dot(inputs, self.kernel)
        output = h + keras.backend.dot(prev_output, self.recurrent_kernel)
        return output, output

      @property
      def output_size(self):
        return self.units

    cell = MinimalRNNCell(32)
    x = keras.Input((None, 5))
    layer = keras.layers.RNN(cell)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer="rmsprop",
        loss="mse",
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

    # Test stacking.
    cells = [MinimalRNNCell(8),
             MinimalRNNCell(16),
             MinimalRNNCell(32)]
    layer = keras.layers.RNN(cells)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

  def test_rnn_with_time_major(self):
    batch = 10
    time_step = 5
    embedding_dim = 4
    units = 3

    # Test basic case.
    x = keras.Input((time_step, embedding_dim))
    time_major_x = keras.layers.Lambda(
        lambda t: array_ops.transpose(t, [1, 0, 2]))(x)
    layer = keras.layers.SimpleRNN(
        units, time_major=True, return_sequences=True)
    self.assertEqual(
        layer.compute_output_shape((time_step, None,
                                    embedding_dim)).as_list(),
        [time_step, None, units])
    y = layer(time_major_x)
    self.assertEqual(layer.output_shape, (time_step, None, units))

    y = keras.layers.Lambda(lambda t: array_ops.transpose(t, [1, 0, 2]))(y)

    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        np.zeros((batch, time_step, embedding_dim)),
        np.zeros((batch, time_step, units)))

    # Test stacking.
    x = keras.Input((time_step, embedding_dim))
    time_major_x = keras.layers.Lambda(
        lambda t: array_ops.transpose(t, [1, 0, 2]))(x)
    cell_units = [10, 8, 6]
    cells = [keras.layers.SimpleRNNCell(cell_units[i]) for i in range(3)]
    layer = keras.layers.RNN(cells, time_major=True, return_sequences=True)
    y = layer(time_major_x)
    self.assertEqual(layer.output_shape, (time_step, None, cell_units[-1]))

    y = keras.layers.Lambda(lambda t: array_ops.transpose(t, [1, 0, 2]))(y)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        np.zeros((batch, time_step, embedding_dim)),
        np.zeros((batch, time_step, cell_units[-1])))

    # Test masking.
    x = keras.Input((time_step, embedding_dim))
    time_major = keras.layers.Lambda(
        lambda t: array_ops.transpose(t, [1, 0, 2]))(x)
    mask = keras.layers.Masking()(time_major)
    rnn = keras.layers.SimpleRNN(
        units, time_major=True, return_sequences=True)(mask)
    y = keras.layers.Lambda(lambda t: array_ops.transpose(t, [1, 0, 2]))(rnn)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        np.zeros((batch, time_step, embedding_dim)),
        np.zeros((batch, time_step, units)))

    # Test layer output
    x = keras.Input((time_step, embedding_dim))
    rnn_1 = keras.layers.SimpleRNN(units, return_sequences=True)
    y = rnn_1(x)

    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        np.zeros((batch, time_step, embedding_dim)),
        np.zeros((batch, time_step, units)))

    x_np = np.random.random((batch, time_step, embedding_dim))
    y_np_1 = model.predict(x_np)

    time_major = keras.layers.Lambda(
        lambda t: array_ops.transpose(t, [1, 0, 2]))(x)
    rnn_2 = keras.layers.SimpleRNN(
        units, time_major=True, return_sequences=True)
    y_2 = rnn_2(time_major)
    y_2 = keras.layers.Lambda(
        lambda t: array_ops.transpose(t, [1, 0, 2]))(y_2)

    model_2 = keras.models.Model(x, y_2)
    rnn_2.set_weights(rnn_1.get_weights())

    y_np_2 = model_2.predict(x_np)
    self.assertAllClose(y_np_1, y_np_2, atol=1e-4)

  def test_rnn_cell_with_constants_layer(self):
    # Test basic case.
    x = keras.Input((None, 5))
    c = keras.Input((3,))
    cell = RNNCellWithConstants(32, constant_size=3)
    layer = keras.layers.RNN(cell)
    y = layer(x, constants=c)

    model = keras.models.Model([x, c], y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        [np.zeros((6, 5, 5)), np.zeros((6, 3))],
        np.zeros((6, 32))
    )

    # Test basic case serialization.
    x_np = np.random.random((6, 5, 5))
    c_np = np.random.random((6, 3))
    y_np = model.predict([x_np, c_np])
    weights = model.get_weights()
    config = layer.get_config()
    custom_objects = {'RNNCellWithConstants': RNNCellWithConstants}
    with keras.utils.CustomObjectScope(custom_objects):
      layer = keras.layers.RNN.from_config(config.copy())
    y = layer(x, constants=c)
    model = keras.models.Model([x, c], y)
    model.set_weights(weights)
    y_np_2 = model.predict([x_np, c_np])
    self.assertAllClose(y_np, y_np_2, atol=1e-4)

    # test flat list inputs.
    with keras.utils.CustomObjectScope(custom_objects):
      layer = keras.layers.RNN.from_config(config.copy())
    y = layer([x, c])
    model = keras.models.Model([x, c], y)
    model.set_weights(weights)
    y_np_3 = model.predict([x_np, c_np])
    self.assertAllClose(y_np, y_np_3, atol=1e-4)

    # Test stacking.
    cells = [keras.layers.recurrent.GRUCell(8),
             RNNCellWithConstants(12, constant_size=3),
             RNNCellWithConstants(32, constant_size=3)]
    layer = keras.layers.recurrent.RNN(cells)
    y = layer(x, constants=c)
    model = keras.models.Model([x, c], y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        [np.zeros((6, 5, 5)), np.zeros((6, 3))],
        np.zeros((6, 32))
    )

    # Test GRUCell reset_after property.
    x = keras.Input((None, 5))
    c = keras.Input((3,))
    cells = [keras.layers.recurrent.GRUCell(32, reset_after=True)]
    layer = keras.layers.recurrent.RNN(cells)
    y = layer(x, constants=c)
    model = keras.models.Model([x, c], y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        [np.zeros((6, 5, 5)), np.zeros((6, 3))],
        np.zeros((6, 32))
    )

    # Test stacked RNN serialization
    x_np = np.random.random((6, 5, 5))
    c_np = np.random.random((6, 3))
    y_np = model.predict([x_np, c_np])
    weights = model.get_weights()
    config = layer.get_config()
    with keras.utils.CustomObjectScope(custom_objects):
      layer = keras.layers.recurrent.RNN.from_config(config.copy())
    y = layer(x, constants=c)
    model = keras.models.Model([x, c], y)
    model.set_weights(weights)
    y_np_2 = model.predict([x_np, c_np])
    self.assertAllClose(y_np, y_np_2, atol=1e-4)

  def test_rnn_cell_with_non_keras_constants(self):
    # Test basic case.
    x = keras.Input((None, 5))
    c = array_ops.zeros([6, 3], dtype=dtypes.float32)
    cell = RNNCellWithConstants(32, constant_size=3)
    layer = keras.layers.RNN(cell)
    y = layer(x, constants=c)

    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

    # Test stacking.
    cells = [keras.layers.recurrent.GRUCell(8),
             RNNCellWithConstants(12, constant_size=3),
             RNNCellWithConstants(32, constant_size=3)]
    layer = keras.layers.recurrent.RNN(cells)
    y = layer(x, constants=c)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

  def test_rnn_cell_with_constants_layer_passing_initial_state(self):
    # Test basic case.
    x = keras.Input((None, 5))
    c = keras.Input((3,))
    s = keras.Input((32,))
    cell = RNNCellWithConstants(32, constant_size=3)
    layer = keras.layers.RNN(cell)
    y = layer(x, initial_state=s, constants=c)
    model = keras.models.Model([x, s, c], y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        [np.zeros((6, 5, 5)), np.zeros((6, 32)), np.zeros((6, 3))],
        np.zeros((6, 32))
    )

    # Test basic case serialization.
    x_np = np.random.random((6, 5, 5))
    s_np = np.random.random((6, 32))
    c_np = np.random.random((6, 3))
    y_np = model.predict([x_np, s_np, c_np])
    weights = model.get_weights()
    config = layer.get_config()
    custom_objects = {'RNNCellWithConstants': RNNCellWithConstants}
    with keras.utils.CustomObjectScope(custom_objects):
      layer = keras.layers.RNN.from_config(config.copy())
    y = layer(x, initial_state=s, constants=c)
    model = keras.models.Model([x, s, c], y)
    model.set_weights(weights)
    y_np_2 = model.predict([x_np, s_np, c_np])
    self.assertAllClose(y_np, y_np_2, atol=1e-4)

    # verify that state is used
    y_np_2_different_s = model.predict([x_np, s_np + 10., c_np])
    with self.assertRaises(AssertionError):
      self.assertAllClose(y_np, y_np_2_different_s, atol=1e-4)

    # test flat list inputs
    with keras.utils.CustomObjectScope(custom_objects):
      layer = keras.layers.RNN.from_config(config.copy())
    y = layer([x, s, c])
    model = keras.models.Model([x, s, c], y)
    model.set_weights(weights)
    y_np_3 = model.predict([x_np, s_np, c_np])
    self.assertAllClose(y_np, y_np_3, atol=1e-4)

  def test_rnn_cell_with_non_keras_constants_and_initial_state(self):
    # Test basic case.
    x = keras.Input((None, 5))
    c = array_ops.zeros([6, 3], dtype=dtypes.float32)
    s = array_ops.zeros([6, 32], dtype=dtypes.float32)
    cell = RNNCellWithConstants(32, constant_size=3)
    layer = keras.layers.RNN(cell)
    y = layer(x, initial_state=s, constants=c)

    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

    # Test stacking.
    cells = [keras.layers.recurrent.GRUCell(8),
             RNNCellWithConstants(12, constant_size=3),
             RNNCellWithConstants(32, constant_size=3)]
    layer = keras.layers.recurrent.RNN(cells)
    s = [array_ops.zeros([6, 8], dtype=dtypes.float32),
         array_ops.zeros([6, 12], dtype=dtypes.float32),
         array_ops.zeros([6, 32], dtype=dtypes.float32)]
    y = layer(x, initial_state=s, constants=c)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

  def test_stacked_rnn_attributes(self):
    if context.executing_eagerly():
      self.skipTest('reduce_sum is not available in eager mode.')

    cells = [keras.layers.LSTMCell(1),
             keras.layers.LSTMCell(1)]
    layer = keras.layers.RNN(cells)
    layer.build((None, None, 1))

    # Test weights
    self.assertEqual(len(layer.trainable_weights), 6)
    cells[0].trainable = False
    self.assertEqual(len(layer.trainable_weights), 3)
    self.assertEqual(len(layer.non_trainable_weights), 3)

    # Test `get_losses_for` and `losses`
    x = keras.Input((None, 1))
    loss_1 = math_ops.reduce_sum(x)
    loss_2 = math_ops.reduce_sum(cells[0].kernel)
    cells[0].add_loss(loss_1, inputs=x)
    cells[0].add_loss(loss_2)
    self.assertEqual(len(layer.losses), 2)
    self.assertEqual(layer.get_losses_for(None), [loss_2])
    self.assertEqual(layer.get_losses_for(x), [loss_1])

    # Test `get_updates_for` and `updates`
    cells = [keras.layers.LSTMCell(1),
             keras.layers.LSTMCell(1)]
    layer = keras.layers.RNN(cells)
    x = keras.Input((None, 1))
    _ = layer(x)

    update_1 = state_ops.assign_add(cells[0].kernel,
                                    x[0, 0, 0] * cells[0].kernel)
    update_2 = state_ops.assign_add(cells[0].kernel,
                                    array_ops.ones_like(cells[0].kernel))
    # TODO(b/128682878): Remove when RNNCells are __call__'d.
    with base_layer_utils.call_context().enter(layer, x, True, None):
      cells[0].add_update(update_1, inputs=x)
      cells[0].add_update(update_2)
    self.assertEqual(len(layer.updates), 2)
    self.assertEqual(len(layer.get_updates_for(None)), 1)
    self.assertEqual(len(layer.get_updates_for(x)), 1)

  def test_rnn_dynamic_trainability(self):
    layer_class = keras.layers.SimpleRNN
    embedding_dim = 4
    units = 3

    layer = layer_class(units)
    layer.build((None, None, embedding_dim))
    self.assertEqual(len(layer.weights), 3)
    self.assertEqual(len(layer.trainable_weights), 3)
    self.assertEqual(len(layer.non_trainable_weights), 0)
    layer.trainable = False
    self.assertEqual(len(layer.weights), 3)
    self.assertEqual(len(layer.trainable_weights), 0)
    self.assertEqual(len(layer.non_trainable_weights), 3)
    layer.trainable = True
    self.assertEqual(len(layer.weights), 3)
    self.assertEqual(len(layer.trainable_weights), 3)
    self.assertEqual(len(layer.non_trainable_weights), 0)

  def test_state_reuse_with_dropout(self):
    layer_class = keras.layers.SimpleRNN
    embedding_dim = 4
    units = 3
    timesteps = 2
    num_samples = 2

    input1 = keras.Input(batch_shape=(num_samples, timesteps, embedding_dim))
    layer = layer_class(units,
                        return_state=True,
                        return_sequences=True,
                        dropout=0.2)
    state = layer(input1)[1:]

    input2 = keras.Input(batch_shape=(num_samples, timesteps, embedding_dim))
    output = layer_class(units)(input2, initial_state=state)
    model = keras.Model([input1, input2], output)

    inputs = [np.random.random((num_samples, timesteps, embedding_dim)),
              np.random.random((num_samples, timesteps, embedding_dim))]
    model.predict(inputs)

  def test_builtin_rnn_cell_serialization(self):
    for cell_class in [keras.layers.SimpleRNNCell,
                       keras.layers.GRUCell,
                       keras.layers.LSTMCell]:
      # Test basic case.
      x = keras.Input((None, 5))
      cell = cell_class(32)
      layer = keras.layers.RNN(cell)
      y = layer(x)
      model = keras.models.Model(x, y)
      model.compile(
          optimizer='rmsprop',
          loss='mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

      # Test basic case serialization.
      x_np = np.random.random((6, 5, 5))
      y_np = model.predict(x_np)
      weights = model.get_weights()
      config = layer.get_config()
      layer = keras.layers.RNN.from_config(config)
      y = layer(x)
      model = keras.models.Model(x, y)
      model.set_weights(weights)
      y_np_2 = model.predict(x_np)
      self.assertAllClose(y_np, y_np_2, atol=1e-4)

      # Test stacking.
      cells = [cell_class(8),
               cell_class(12),
               cell_class(32)]
      layer = keras.layers.RNN(cells)
      y = layer(x)
      model = keras.models.Model(x, y)
      model.compile(
          optimizer='rmsprop',
          loss='mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

      # Test stacked RNN serialization.
      x_np = np.random.random((6, 5, 5))
      y_np = model.predict(x_np)
      weights = model.get_weights()
      config = layer.get_config()
      layer = keras.layers.RNN.from_config(config)
      y = layer(x)
      model = keras.models.Model(x, y)
      model.set_weights(weights)
      y_np_2 = model.predict(x_np)
      self.assertAllClose(y_np, y_np_2, atol=1e-4)

  @parameterized.named_parameters(
      *test_util.generate_combinations_with_testcase_name(
          layer=[rnn_v1.SimpleRNN, rnn_v1.GRU, rnn_v1.LSTM,
                 rnn_v2.GRU, rnn_v2.LSTM],
          unroll=[True, False]))
  def test_rnn_dropout(self, layer, unroll):
    rnn_layer = layer(3, dropout=0.1, recurrent_dropout=0.1, unroll=unroll)
    if not unroll:
      x = keras.Input((None, 5))
    else:
      x = keras.Input((5, 5))
    y = rnn_layer(x)
    model = keras.models.Model(x, y)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    x_np = np.random.random((6, 5, 5))
    y_np = np.random.random((6, 3))
    model.train_on_batch(x_np, y_np)

  @parameterized.named_parameters(
      *test_util.generate_combinations_with_testcase_name(
          cell=[keras.layers.SimpleRNNCell, keras.layers.GRUCell,
                keras.layers.LSTMCell],
          unroll=[True, False]))
  def test_stacked_rnn_dropout(self, cell, unroll):
    cells = [cell(3, dropout=0.1, recurrent_dropout=0.1),
             cell(3, dropout=0.1, recurrent_dropout=0.1)]
    layer = keras.layers.RNN(cells, unroll=unroll)

    if not unroll:
      x = keras.Input((None, 5))
    else:
      x = keras.Input((5, 5))
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(
        'sgd',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    x_np = np.random.random((6, 5, 5))
    y_np = np.random.random((6, 3))
    model.train_on_batch(x_np, y_np)

  def test_dropout_mask_reuse(self):
    # The layer is created with recurrent_initializer = zero, so that the
    # the recurrent state won't affect the output. By doing this, we can verify
    # the output and see if the same mask is applied to for each timestep.
    rnn = keras.layers.SimpleRNN(3,
                                 dropout=0.5,
                                 kernel_initializer='ones',
                                 recurrent_initializer='zeros',
                                 return_sequences=True,
                                 unroll=True)

    inputs = constant_op.constant(1.0, shape=(6, 2, 5))
    out = rnn(inputs, training=True)
    if not context.executing_eagerly():
      self.evaluate(variables_lib.global_variables_initializer())
    batch_1 = self.evaluate(out)
    batch_1_t0, batch_1_t1 = batch_1[:, 0, :], batch_1[:, 1, :]
    self.assertAllClose(batch_1_t0, batch_1_t1)

    # This simulate the layer called with multiple batches in eager mode
    if context.executing_eagerly():
      out2 = rnn(inputs, training=True)
    else:
      out2 = out
    batch_2 = self.evaluate(out2)
    batch_2_t0, batch_2_t1 = batch_2[:, 0, :], batch_2[:, 1, :]
    self.assertAllClose(batch_2_t0, batch_2_t1)

    # Also validate that different dropout is used by between batches.
    self.assertNotAllClose(batch_1_t0, batch_2_t0)
    self.assertNotAllClose(batch_1_t1, batch_2_t1)

  def test_stacked_rnn_compute_output_shape(self):
    cells = [keras.layers.LSTMCell(3),
             keras.layers.LSTMCell(6)]
    embedding_dim = 4
    timesteps = 2
    layer = keras.layers.RNN(cells, return_state=True, return_sequences=True)
    output_shape = layer.compute_output_shape((None, timesteps, embedding_dim))
    expected_output_shape = [(None, timesteps, 6),
                             (None, 3),
                             (None, 3),
                             (None, 6),
                             (None, 6)]
    self.assertEqual(
        [tuple(o.as_list()) for o in output_shape],
        expected_output_shape)

    # Test reverse_state_order = True for stacked cell.
    stacked_cell = keras.layers.StackedRNNCells(
        cells, reverse_state_order=True)
    layer = keras.layers.RNN(
        stacked_cell, return_state=True, return_sequences=True)
    output_shape = layer.compute_output_shape((None, timesteps, embedding_dim))
    expected_output_shape = [(None, timesteps, 6),
                             (None, 6),
                             (None, 6),
                             (None, 3),
                             (None, 3)]
    self.assertEqual(
        [tuple(o.as_list()) for o in output_shape],
        expected_output_shape)

  def test_trackable_dependencies(self):
    rnn = keras.layers.SimpleRNN
    x = np.random.random((2, 2, 2))
    y = np.random.random((2, 2))
    model = keras.models.Sequential()
    model.add(rnn(2))
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.fit(x, y, epochs=1, batch_size=1)

    # check whether the model variables are present in the
    # trackable list of objects
    checkpointed_objects = object_identity.ObjectIdentitySet(
        trackable_util.list_objects(model))
    for v in model.variables:
      self.assertIn(v, checkpointed_objects)

  def test_high_dimension_RNN(self):
    # Basic test case.
    unit_a = 10
    unit_b = 20
    input_a = 5
    input_b = 10
    batch = 32
    time_step = 4

    cell = Minimal2DRNNCell(unit_a, unit_b)
    x = keras.Input((None, input_a, input_b))
    layer = keras.layers.RNN(cell)
    y = layer(x)

    self.assertEqual(cell.state_size.as_list(), [unit_a, unit_b])

    if not context.executing_eagerly():
      init_state = layer.get_initial_state(x)
      self.assertEqual(len(init_state), 1)
      self.assertEqual(init_state[0].shape.as_list(), [None, unit_a, unit_b])

    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        np.zeros((batch, time_step, input_a, input_b)),
        np.zeros((batch, unit_a, unit_b)))
    self.assertEqual(model.output_shape, (None, unit_a, unit_b))

    # Test stacking.
    cells = [
        Minimal2DRNNCell(unit_a, unit_b),
        Minimal2DRNNCell(unit_a * 2, unit_b * 2),
        Minimal2DRNNCell(unit_a * 4, unit_b * 4)
    ]
    layer = keras.layers.RNN(cells)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        np.zeros((batch, time_step, input_a, input_b)),
        np.zeros((batch, unit_a * 4, unit_b * 4)))
    self.assertEqual(model.output_shape, (None, unit_a * 4, unit_b * 4))

  def test_high_dimension_RNN_with_init_state(self):
    unit_a = 10
    unit_b = 20
    input_a = 5
    input_b = 10
    batch = 32
    time_step = 4

    # Basic test case.
    cell = Minimal2DRNNCell(unit_a, unit_b)
    x = keras.Input((None, input_a, input_b))
    s = keras.Input((unit_a, unit_b))
    layer = keras.layers.RNN(cell)
    y = layer(x, initial_state=s)

    model = keras.models.Model([x, s], y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch([
        np.zeros((batch, time_step, input_a, input_b)),
        np.zeros((batch, unit_a, unit_b))
    ], np.zeros((batch, unit_a, unit_b)))
    self.assertEqual(model.output_shape, (None, unit_a, unit_b))

    # Bad init state shape.
    bad_shape_a = unit_a * 2
    bad_shape_b = unit_b * 2
    cell = Minimal2DRNNCell(unit_a, unit_b)
    x = keras.Input((None, input_a, input_b))
    s = keras.Input((bad_shape_a, bad_shape_b))
    layer = keras.layers.RNN(cell)
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             'however `cell.state_size` is'):
      layer(x, initial_state=s)

  def test_inconsistent_output_state_size(self):
    batch = 32
    time_step = 4
    state_size = 5
    input_size = 6
    cell = PlusOneRNNCell(state_size)
    x = keras.Input((None, input_size))
    layer = keras.layers.RNN(cell)
    y = layer(x)

    self.assertEqual(cell.state_size, state_size)
    if not context.executing_eagerly():
      init_state = layer.get_initial_state(x)
      self.assertEqual(len(init_state), 1)
      self.assertEqual(init_state[0].shape.as_list(), [None, state_size])

    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        np.zeros((batch, time_step, input_size)),
        np.zeros((batch, input_size)))
    self.assertEqual(model.output_shape, (None, input_size))

  def test_get_initial_state(self):
    cell = keras.layers.SimpleRNNCell(5)
    with self.assertRaisesRegexp(ValueError,
                                 'batch_size and dtype cannot be None'):
      cell.get_initial_state(None, None, None)

    if not context.executing_eagerly():
      inputs = keras.Input((None, 10))
      initial_state = cell.get_initial_state(inputs, None, None)
      self.assertEqual(initial_state.shape.as_list(), [None, 5])
      self.assertEqual(initial_state.dtype, inputs.dtype)

      batch = array_ops.shape(inputs)[0]
      dtype = inputs.dtype
      initial_state = cell.get_initial_state(None, batch, dtype)
      self.assertEqual(initial_state.shape.as_list(), [None, 5])
      self.assertEqual(initial_state.dtype, inputs.dtype)
    else:
      batch = 8
      inputs = np.random.random((batch, 10))
      initial_state = cell.get_initial_state(inputs, None, None)
      self.assertEqual(initial_state.shape.as_list(), [8, 5])
      self.assertEqual(initial_state.dtype, inputs.dtype)

      dtype = inputs.dtype
      initial_state = cell.get_initial_state(None, batch, dtype)
      self.assertEqual(initial_state.shape.as_list(), [batch, 5])
      self.assertEqual(initial_state.dtype, inputs.dtype)

  def test_nested_input_output(self):
    batch = 10
    t = 5
    i1, i2, i3 = 3, 4, 5
    o1, o2, o3 = 2, 3, 4

    cell = NestedCell(o1, o2, o3)
    rnn = keras.layers.RNN(cell)

    input_1 = keras.Input((t, i1))
    input_2 = keras.Input((t, i2, i3))

    outputs = rnn((input_1, input_2))

    self.assertEqual(len(outputs), 2)
    self.assertEqual(outputs[0].shape.as_list(), [None, o1])
    self.assertEqual(outputs[1].shape.as_list(), [None, o2, o3])

    model = keras.models.Model((input_1, input_2), outputs)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        [np.zeros((batch, t, i1)), np.zeros((batch, t, i2, i3))],
        [np.zeros((batch, o1)), np.zeros((batch, o2, o3))])
    self.assertEqual(model.output_shape, [(None, o1), (None, o2, o3)])

    cell = NestedCell(o1, o2, o3, use_tuple=True)

    rnn = keras.layers.RNN(cell)

    input_1 = keras.Input((t, i1))
    input_2 = keras.Input((t, i2, i3))

    outputs = rnn(NestedInput(t1=input_1, t2=input_2))

    self.assertEqual(len(outputs), 2)
    self.assertEqual(outputs[0].shape.as_list(), [None, o1])
    self.assertEqual(outputs[1].shape.as_list(), [None, o2, o3])

    model = keras.models.Model([input_1, input_2], outputs)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        [np.zeros((batch, t, i1)),
         np.zeros((batch, t, i2, i3))],
        [np.zeros((batch, o1)), np.zeros((batch, o2, o3))])
    self.assertEqual(model.output_shape, [(None, o1), (None, o2, o3)])

  def test_nested_input_output_with_state(self):
    batch = 10
    t = 5
    i1, i2, i3 = 3, 4, 5
    o1, o2, o3 = 2, 3, 4

    cell = NestedCell(o1, o2, o3)
    rnn = keras.layers.RNN(cell, return_sequences=True, return_state=True)

    input_1 = keras.Input((t, i1))
    input_2 = keras.Input((t, i2, i3))

    output1, output2, s1, s2 = rnn((input_1, input_2))

    self.assertEqual(output1.shape.as_list(), [None, t, o1])
    self.assertEqual(output2.shape.as_list(), [None, t, o2, o3])
    self.assertEqual(s1.shape.as_list(), [None, o1])
    self.assertEqual(s2.shape.as_list(), [None, o2, o3])

    model = keras.models.Model([input_1, input_2], [output1, output2])
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        [np.zeros((batch, t, i1)),
         np.zeros((batch, t, i2, i3))],
        [np.zeros((batch, t, o1)),
         np.zeros((batch, t, o2, o3))])
    self.assertEqual(model.output_shape, [(None, t, o1), (None, t, o2, o3)])

    cell = NestedCell(o1, o2, o3, use_tuple=True)

    rnn = keras.layers.RNN(cell, return_sequences=True, return_state=True)

    input_1 = keras.Input((t, i1))
    input_2 = keras.Input((t, i2, i3))

    output1, output2, s1, s2 = rnn(NestedInput(t1=input_1, t2=input_2))

    self.assertEqual(output1.shape.as_list(), [None, t, o1])
    self.assertEqual(output2.shape.as_list(), [None, t, o2, o3])
    self.assertEqual(s1.shape.as_list(), [None, o1])
    self.assertEqual(s2.shape.as_list(), [None, o2, o3])

    model = keras.models.Model([input_1, input_2], [output1, output2])
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        [np.zeros((batch, t, i1)),
         np.zeros((batch, t, i2, i3))],
        [np.zeros((batch, t, o1)),
         np.zeros((batch, t, o2, o3))])
    self.assertEqual(model.output_shape, [(None, t, o1), (None, t, o2, o3)])

  def test_nest_input_output_with_init_state(self):
    batch = 10
    t = 5
    i1, i2, i3 = 3, 4, 5
    o1, o2, o3 = 2, 3, 4

    cell = NestedCell(o1, o2, o3)
    rnn = keras.layers.RNN(cell, return_sequences=True, return_state=True)

    input_1 = keras.Input((t, i1))
    input_2 = keras.Input((t, i2, i3))
    init_s1 = keras.Input((o1,))
    init_s2 = keras.Input((o2, o3))

    output1, output2, s1, s2 = rnn((input_1, input_2),
                                   initial_state=(init_s1, init_s2))

    self.assertEqual(output1.shape.as_list(), [None, t, o1])
    self.assertEqual(output2.shape.as_list(), [None, t, o2, o3])
    self.assertEqual(s1.shape.as_list(), [None, o1])
    self.assertEqual(s2.shape.as_list(), [None, o2, o3])

    model = keras.models.Model([input_1, input_2, init_s1, init_s2],
                               [output1, output2])
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        [np.zeros((batch, t, i1)),
         np.zeros((batch, t, i2, i3)),
         np.zeros((batch, o1)),
         np.zeros((batch, o2, o3))],
        [np.zeros((batch, t, o1)),
         np.zeros((batch, t, o2, o3))])
    self.assertEqual(model.output_shape, [(None, t, o1), (None, t, o2, o3)])

    cell = NestedCell(o1, o2, o3, use_tuple=True)

    rnn = keras.layers.RNN(cell, return_sequences=True, return_state=True)

    input_1 = keras.Input((t, i1))
    input_2 = keras.Input((t, i2, i3))
    init_s1 = keras.Input((o1,))
    init_s2 = keras.Input((o2, o3))
    init_state = NestedState(s1=init_s1, s2=init_s2)

    output1, output2, s1, s2 = rnn(NestedInput(t1=input_1, t2=input_2),
                                   initial_state=init_state)

    self.assertEqual(output1.shape.as_list(), [None, t, o1])
    self.assertEqual(output2.shape.as_list(), [None, t, o2, o3])
    self.assertEqual(s1.shape.as_list(), [None, o1])
    self.assertEqual(s2.shape.as_list(), [None, o2, o3])

    model = keras.models.Model([input_1, input_2, init_s1, init_s2],
                               [output1, output2])
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        [np.zeros((batch, t, i1)),
         np.zeros((batch, t, i2, i3)),
         np.zeros((batch, o1)),
         np.zeros((batch, o2, o3))],
        [np.zeros((batch, t, o1)),
         np.zeros((batch, t, o2, o3))])
    self.assertEqual(model.output_shape, [(None, t, o1), (None, t, o2, o3)])

  def test_peephole_lstm_cell(self):

    def _run_cell(cell_fn, **kwargs):
      inputs = array_ops.one_hot([1, 2, 3, 4], 4)
      cell = cell_fn(5, **kwargs)
      cell.build(inputs.shape)
      initial_state = cell.get_initial_state(
          inputs=inputs, batch_size=4, dtype=dtypes.float32)
      inputs, _ = cell(inputs, initial_state)
      output = inputs
      if not context.executing_eagerly():
        self.evaluate(variables_lib.global_variables_initializer())
        output = self.evaluate(output)
      return output

    random_seed.set_random_seed(12345)
    # `recurrent_activation` kwarg is set to sigmoid as that is hardcoded into
    # rnn_cell.LSTMCell.
    no_peephole_output = _run_cell(
        keras.layers.LSTMCell,
        kernel_initializer='ones',
        recurrent_activation='sigmoid',
        implementation=1)
    first_implementation_output = _run_cell(
        keras.layers.PeepholeLSTMCell,
        kernel_initializer='ones',
        recurrent_activation='sigmoid',
        implementation=1)
    second_implementation_output = _run_cell(
        keras.layers.PeepholeLSTMCell,
        kernel_initializer='ones',
        recurrent_activation='sigmoid',
        implementation=2)
    tf_lstm_cell_output = _run_cell(
        rnn_cell.LSTMCell,
        use_peepholes=True,
        initializer=init_ops.ones_initializer)
    self.assertNotAllClose(first_implementation_output, no_peephole_output)
    self.assertAllClose(first_implementation_output,
                        second_implementation_output)
    self.assertAllClose(first_implementation_output, tf_lstm_cell_output)

  def test_masking_rnn_with_output_and_states(self):

    class Cell(keras.layers.Layer):

      def __init__(self):
        self.state_size = None
        self.output_size = None
        super(Cell, self).__init__()

      def build(self, input_shape):
        self.state_size = input_shape[-1]
        self.output_size = input_shape[-1]

      def call(self, inputs, states):
        return inputs, [s + 1 for s in states]

    x = keras.Input((3, 1), name='x')
    x_masked = keras.layers.Masking()(x)
    s_0 = keras.Input((1,), name='s_0')
    y, s = keras.layers.RNN(
        Cell(), return_state=True)(x_masked, initial_state=s_0)
    model = keras.models.Model([x, s_0], [y, s])
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    # last time step masked
    x_np = np.array([[[1.], [2.], [0.]]])
    s_0_np = np.array([[10.]])
    y_np, s_np = model.predict([x_np, s_0_np])

    # 1 is added to initial state two times
    self.assertAllClose(s_np, s_0_np + 2)
    # Expect last output to be the same as last output before masking
    self.assertAllClose(y_np, x_np[:, 1, :])

  def test_zero_output_for_masking(self):

    for unroll in [True, False]:
      cell = keras.layers.SimpleRNNCell(5)
      x = keras.Input((5, 5))
      mask = keras.layers.Masking()
      layer = keras.layers.RNN(
          cell, return_sequences=True, zero_output_for_mask=True, unroll=unroll)
      masked_input = mask(x)
      y = layer(masked_input)
      model = keras.models.Model(x, y)
      model.compile(
          optimizer='rmsprop',
          loss='mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

      np_x = np.ones((6, 5, 5))
      result_1 = model.predict(np_x)

      # set the time 4 and 5 for last record to be zero (masked).
      np_x[5, 3:] = 0
      result_2 = model.predict(np_x)

      # expect the result_2 has same output, except the time 4,5 for last
      # record.
      result_1[5, 3:] = 0
      self.assertAllClose(result_1, result_2)

  def test_unroll_single_step(self):
    """Even if the time dimension is only one, we should be able to unroll."""
    cell = keras.layers.SimpleRNNCell(5)
    x = keras.Input((1, 5))
    layer = keras.layers.RNN(cell, return_sequences=True, unroll=True)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())

    np_x = np.ones((6, 1, 5))
    result = model.predict(np_x)
    self.assertEqual((6, 1, 5), result.shape)

  def test_unroll_zero_step(self):
    """If the time dimension is None, we should fail to unroll."""
    cell = keras.layers.SimpleRNNCell(5)
    x = keras.Input((None, 5))
    layer = keras.layers.RNN(cell, return_sequences=True, unroll=True)
    with self.assertRaisesRegexp(ValueError, 'Cannot unroll a RNN.*'):
      layer(x)

  def test_full_input_spec(self):
    # See https://github.com/tensorflow/tensorflow/issues/25985
    inputs = keras.layers.Input(batch_shape=(1, 1, 1))
    state_h = keras.layers.Input(batch_shape=(1, 1))
    state_c = keras.layers.Input(batch_shape=(1, 1))
    states = [state_h, state_c]
    decoder_out = keras.layers.LSTM(1, stateful=True)(
        inputs,
        initial_state=states
    )
    model = keras.Model([inputs, state_h, state_c], decoder_out)
    model.reset_states()

  def test_reset_states(self):
    # See https://github.com/tensorflow/tensorflow/issues/25852
    with self.assertRaisesRegexp(ValueError, 'it needs to know its batch size'):
      simple_rnn = keras.layers.SimpleRNN(1, stateful=True)
      simple_rnn.reset_states()

    with self.assertRaisesRegexp(ValueError, 'it needs to know its batch size'):
      cell = Minimal2DRNNCell(1, 2)
      custom_rnn = keras.layers.RNN(cell, stateful=True)
      custom_rnn.reset_states()

  @parameterized.parameters(
      [keras.layers.SimpleRNNCell, keras.layers.GRUCell, keras.layers.LSTMCell])
  def test_stateful_rnn_with_stacking(self, cell):
    # See https://github.com/tensorflow/tensorflow/issues/28614.
    batch = 12
    timesteps = 10
    input_dim = 8
    output_dim = 64
    cells = [cell(32), cell(64)]
    x = keras.Input(batch_shape=(batch, None, input_dim))
    layer = keras.layers.RNN(cells, stateful=True)
    y = layer(x)

    model = keras.Model(x, y)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        np.zeros((batch, timesteps, input_dim)),
        np.zeros((batch, output_dim)))
    model.predict(np.ones((batch, timesteps, input_dim)))

    model.reset_states()
    model.predict(np.ones((batch, timesteps, input_dim)))

    new_states = nest.map_structure(lambda s: np.ones((batch, s)),
                                    layer.cell.state_size)
    layer.reset_states(new_states)
    model.predict(np.ones((batch, timesteps, input_dim)))

  def test_input_dim_length(self):
    simple_rnn = keras.layers.SimpleRNN(5, input_length=10, input_dim=8)
    self.assertEqual(simple_rnn._batch_input_shape, (None, 10, 8))

    simple_rnn = keras.layers.SimpleRNN(5, input_dim=8)
    self.assertEqual(simple_rnn._batch_input_shape, (None, None, 8))

    simple_rnn = keras.layers.SimpleRNN(5, input_length=10)
    self.assertEqual(simple_rnn._batch_input_shape, (None, 10, None))

  @parameterized.parameters(
      [keras.layers.SimpleRNNCell, keras.layers.GRUCell, keras.layers.LSTMCell])
  def test_state_spec_with_stack_cell(self, cell):
    # See https://github.com/tensorflow/tensorflow/issues/27817 for more detail.
    batch = 12
    timesteps = 10
    input_dim = 8
    output_dim = 8

    def create_cell():
      return [cell(output_dim),
              cell(output_dim),
              cell(output_dim)]

    inputs = keras.Input((timesteps, input_dim))
    encoder_output = keras.layers.RNN(create_cell(), return_state=True)(inputs)

    states = encoder_output[1:]

    decoder_output = keras.layers.RNN(
        create_cell())(inputs, initial_state=states)

    model = keras.models.Model(inputs, decoder_output)
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function())
    model.train_on_batch(
        np.zeros((batch, timesteps, input_dim)),
        np.zeros((batch, output_dim)))
    model.predict(np.ones((batch, timesteps, input_dim)))


class RNNCellWithConstants(keras.layers.Layer):

  def __init__(self, units, constant_size, **kwargs):
    self.units = units
    self.state_size = units
    self.constant_size = constant_size
    super(RNNCellWithConstants, self).__init__(**kwargs)

  def build(self, input_shape):
    self.input_kernel = self.add_weight(
        shape=(input_shape[-1], self.units),
        initializer='uniform',
        name='kernel')
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units),
        initializer='uniform',
        name='recurrent_kernel')
    self.constant_kernel = self.add_weight(
        shape=(self.constant_size, self.units),
        initializer='uniform',
        name='constant_kernel')
    self.built = True

  def call(self, inputs, states, constants):
    [prev_output] = states
    [constant] = constants
    h_input = keras.backend.dot(inputs, self.input_kernel)
    h_state = keras.backend.dot(prev_output, self.recurrent_kernel)
    h_const = keras.backend.dot(constant, self.constant_kernel)
    output = h_input + h_state + h_const
    return output, [output]

  def get_config(self):
    config = {'units': self.units, 'constant_size': self.constant_size}
    base_config = super(RNNCellWithConstants, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Minimal2DRNNCell(keras.layers.Layer):
  """The minimal 2D RNN cell is a simple combination of 2 1-D RNN cell.

  Both internal state and output have 2 dimensions and are orthogonal
  between each other.
  """

  def __init__(self, unit_a, unit_b, **kwargs):
    self.unit_a = unit_a
    self.unit_b = unit_b
    self.state_size = tensor_shape.as_shape([unit_a, unit_b])
    self.output_size = tensor_shape.as_shape([unit_a, unit_b])
    super(Minimal2DRNNCell, self).__init__(**kwargs)

  def build(self, input_shape):
    input_a = input_shape[-2]
    input_b = input_shape[-1]
    self.kernel = self.add_weight(
        shape=(input_a, input_b, self.unit_a, self.unit_b),
        initializer='uniform',
        name='kernel')
    self.recurring_kernel = self.add_weight(
        shape=(self.unit_a, self.unit_b, self.unit_a, self.unit_b),
        initializer='uniform',
        name='recurring_kernel')
    self.bias = self.add_weight(
        shape=(self.unit_a, self.unit_b), initializer='uniform', name='bias')
    self.built = True

  def call(self, inputs, states):
    prev_output = states[0]
    h = special_math_ops.einsum('bij,ijkl->bkl', inputs, self.kernel)
    h += array_ops.expand_dims(self.bias, axis=0)
    output = h + special_math_ops.einsum('bij,ijkl->bkl', prev_output,
                                         self.recurring_kernel)
    return output, [output]


class PlusOneRNNCell(keras.layers.Layer):
  """Add one to the input and state.

  This cell is used for testing state_size and output_size."""

  def __init__(self, num_unit, **kwargs):
    self.state_size = num_unit
    super(PlusOneRNNCell, self).__init__(**kwargs)

  def build(self, input_shape):
    self.output_size = input_shape[-1]

  def call(self, inputs, states):
    return inputs + 1, [states[0] + 1]


class NestedCell(keras.layers.Layer):

  def __init__(self, unit_1, unit_2, unit_3, use_tuple=False, **kwargs):
    self.unit_1 = unit_1
    self.unit_2 = unit_2
    self.unit_3 = unit_3
    self.use_tuple = use_tuple
    super(NestedCell, self).__init__(**kwargs)
    # A nested state.
    if use_tuple:
      self.state_size = NestedState(
          s1=unit_1, s2=tensor_shape.TensorShape([unit_2, unit_3]))
    else:
      self.state_size = (unit_1, tensor_shape.TensorShape([unit_2, unit_3]))
    self.output_size = (unit_1, tensor_shape.TensorShape([unit_2, unit_3]))

  def build(self, inputs_shape):
    # expect input_shape to contain 2 items, [(batch, i1), (batch, i2, i3)]
    if self.use_tuple:
      input_1 = inputs_shape.t1[1]
      input_2, input_3 = inputs_shape.t2[1:]
    else:
      input_1 = inputs_shape[0][1]
      input_2, input_3 = inputs_shape[1][1:]

    self.kernel_1 = self.add_weight(
        shape=(input_1, self.unit_1), initializer='uniform', name='kernel_1')
    self.kernel_2_3 = self.add_weight(
        shape=(input_2, input_3, self.unit_2, self.unit_3),
        initializer='uniform',
        name='kernel_2_3')

  def call(self, inputs, states):
    # inputs should be in [(batch, input_1), (batch, input_2, input_3)]
    # state should be in shape [(batch, unit_1), (batch, unit_2, unit_3)]
    flatten_inputs = nest.flatten(inputs)
    s1, s2 = states

    output_1 = math_ops.matmul(flatten_inputs[0], self.kernel_1)
    output_2_3 = special_math_ops.einsum('bij,ijkl->bkl', flatten_inputs[1],
                                         self.kernel_2_3)
    state_1 = s1 + output_1
    state_2_3 = s2 + output_2_3

    output = [output_1, output_2_3]
    new_states = NestedState(s1=state_1, s2=state_2_3)

    return output, new_states


if __name__ == '__main__':
  test.main()
