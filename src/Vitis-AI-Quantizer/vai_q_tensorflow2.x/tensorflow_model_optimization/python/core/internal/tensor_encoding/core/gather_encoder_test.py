# Copyright 2019, The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# TODO(b/139939526): Move to public API.
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import core_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import encoding_stage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import gather_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing import test_utils

# Abbreviated constants used in tests.
TENSORS = gather_encoder._TENSORS

P1_VALS = test_utils.PlusOneEncodingStage.ENCODED_VALUES_KEY
T2_VALS = test_utils.TimesTwoEncodingStage.ENCODED_VALUES_KEY
SL_VALS = test_utils.SimpleLinearEncodingStage.ENCODED_VALUES_KEY
SIF_SIGNS = test_utils.SignIntFloatEncodingStage.ENCODED_SIGNS_KEY
SIF_INTS = test_utils.SignIntFloatEncodingStage.ENCODED_INTS_KEY
SIF_FLOATS = test_utils.SignIntFloatEncodingStage.ENCODED_FLOATS_KEY
PN_VALS = test_utils.PlusOneOverNEncodingStage.ENCODED_VALUES_KEY


class GatherEncoderTest(tf.test.TestCase, parameterized.TestCase):

  @tf_test_util.run_all_in_graph_and_eager_modes
  def test_basic_encode_decode(self):
    """Tests basic encoding and decoding works as expected."""
    x_fn = lambda: tf.random.uniform((12,))
    encoder = gather_encoder.GatherEncoder.from_encoder(
        core_encoder.EncoderComposer(
            test_utils.PlusOneOverNEncodingStage()).make(),
        tf.TensorSpec.from_tensor(x_fn()))

    num_summands = 3
    iteration = _make_iteration_function(encoder, x_fn, num_summands)
    state = encoder.initial_state()

    for i in range(1, 5):
      data = self.evaluate(iteration(state))
      for j in range(num_summands):
        self.assertAllClose(
            data.x[j] + 1 / i,
            _encoded_x_field(data.encoded_x[j], [TENSORS, PN_VALS]))
      self.assertEqual((i,), data.initial_state)
      self.assertEqual((i + 1,), data.updated_state)
      state = data.updated_state

  @tf_test_util.run_all_in_graph_and_eager_modes
  def test_composite_encoder(self):
    """Tests functionality with a general, composite `Encoder`."""
    x_fn = lambda: tf.constant(1.2)
    encoder = core_encoder.EncoderComposer(
        test_utils.SignIntFloatEncodingStage())
    encoder.add_child(test_utils.TimesTwoEncodingStage(), SIF_SIGNS)
    encoder.add_child(test_utils.PlusOneEncodingStage(), SIF_INTS)
    encoder.add_child(test_utils.TimesTwoEncodingStage(), SIF_FLOATS).add_child(
        test_utils.PlusOneOverNEncodingStage(), T2_VALS)
    encoder = gather_encoder.GatherEncoder.from_encoder(
        encoder.make(), tf.TensorSpec.from_tensor(x_fn()))

    num_summands = 3
    iteration = _make_iteration_function(encoder, x_fn, num_summands)
    state = encoder.initial_state()

    for i in range(1, 5):
      data = self.evaluate(iteration(state))
      for j in range(num_summands):
        self.assertAllClose(
            2.0,
            _encoded_x_field(data.encoded_x[j], [TENSORS, SIF_SIGNS, T2_VALS]))
        self.assertAllClose(
            2.0,
            _encoded_x_field(data.encoded_x[j], [TENSORS, SIF_INTS, P1_VALS]))
        self.assertAllClose(
            0.4 + 1 / i,
            _encoded_x_field(data.encoded_x[j],
                             [TENSORS, SIF_FLOATS, T2_VALS, PN_VALS]))
        self.assertAllClose(data.x[j], data.part_decoded_x[j])
        self.assertAllClose(data.x[j] * num_summands,
                            data.summed_part_decoded_x)
        self.assertAllClose(data.x[j] * num_summands, data.decoded_x)

      self.assertEqual((i,), data.initial_state)
      self.assertEqual((i + 1,), data.updated_state)
      state = data.updated_state

  @tf_test_util.run_all_in_graph_and_eager_modes
  def test_none_state_equal_to_initial_state(self):
    """Tests that not providing state is the same as initial_state."""
    x_fn = lambda: tf.constant(1.0)
    encoder = gather_encoder.GatherEncoder.from_encoder(
        core_encoder.EncoderComposer(
            test_utils.PlusOneOverNEncodingStage()).make(),
        tf.TensorSpec.from_tensor(x_fn()))

    num_summands = 3
    stateful_iteration = _make_iteration_function(encoder, x_fn, num_summands)
    state = encoder.initial_state()
    stateless_iteration = _make_stateless_iteration_function(
        encoder, x_fn, num_summands)

    stateful_data = self.evaluate(stateful_iteration(state))
    stateless_data = self.evaluate(stateless_iteration())

    self.assertAllClose(stateful_data.encoded_x, stateless_data.encoded_x)
    self.assertAllClose(stateful_data.decoded_x, stateless_data.decoded_x)

  def test_python_constants_not_exposed(self):
    """Tests that only TensorFlow values are exposed to users."""
    x_fn = lambda: tf.constant(1.0)
    tensorspec = tf.TensorSpec.from_tensor(x_fn())
    encoder_py = gather_encoder.GatherEncoder.from_encoder(
        core_encoder.EncoderComposer(
            test_utils.SimpleLinearEncodingStage(2.0, 3.0)).add_parent(
                test_utils.PlusOneEncodingStage(), P1_VALS).add_parent(
                    test_utils.SimpleLinearEncodingStage(2.0, 3.0),
                    SL_VALS).make(), tensorspec)
    a_var = tf.compat.v1.get_variable('a_var', initializer=2.0)
    b_var = tf.compat.v1.get_variable('b_var', initializer=3.0)
    encoder_tf = gather_encoder.GatherEncoder.from_encoder(
        core_encoder.EncoderComposer(
            test_utils.SimpleLinearEncodingStage(a_var, b_var)).add_parent(
                test_utils.PlusOneEncodingStage(), P1_VALS).add_parent(
                    test_utils.SimpleLinearEncodingStage(a_var, b_var),
                    SL_VALS).make(), tensorspec)

    (encode_params_py, decode_before_sum_params_py,
     decode_after_sum_params_py) = encoder_py.get_params()
    (encode_params_tf, decode_before_sum_params_tf,
     decode_after_sum_params_tf) = encoder_tf.get_params()

    # Params that are Python constants -- not tf.Tensors -- should be hidden
    # from the user, and made statically available at appropriate locations.
    self.assertLen(encode_params_py, 1)
    self.assertLen(encode_params_tf, 5)
    self.assertLen(decode_before_sum_params_py, 1)
    self.assertLen(decode_before_sum_params_tf, 3)
    self.assertEmpty(decode_after_sum_params_py)
    self.assertLen(decode_after_sum_params_tf, 2)

  @tf_test_util.run_all_in_graph_and_eager_modes
  def test_decode_needs_input_shape(self):
    """Tests that mechanism for passing input shape works."""
    x_fn = lambda: tf.reshape(list(range(15)), [3, 5])
    encoder = gather_encoder.GatherEncoder.from_encoder(
        core_encoder.EncoderComposer(
            test_utils.ReduceMeanEncodingStage()).make(),
        tf.TensorSpec.from_tensor(x_fn()))

    iteration = _make_iteration_function(encoder, x_fn, 1)
    data = self.evaluate(iteration(encoder.initial_state()))
    self.assertAllEqual([[7.0] * 5] * 3, data.decoded_x)

  @tf_test_util.run_all_in_graph_and_eager_modes
  def test_commutativity_with_sum(self):
    """Tests that encoder that commutes with sum works."""
    x_fn = lambda: tf.constant([1.0, 3.0])
    encoder = gather_encoder.GatherEncoder.from_encoder(
        core_encoder.EncoderComposer(test_utils.TimesTwoEncodingStage()).make(),
        tf.TensorSpec.from_tensor(x_fn()))

    for num_summands in [1, 3, 7]:
      iteration = _make_iteration_function(encoder, x_fn, num_summands)
      data = self.evaluate(iteration(encoder.initial_state()))
      for i in range(num_summands):
        self.assertAllClose([1.0, 3.0], data.x[i])
        self.assertAllClose(
            [2.0, 6.0], _encoded_x_field(data.encoded_x[i], [TENSORS, T2_VALS]))
        self.assertAllClose(list(data.part_decoded_x[i].values())[0],
                            list(data.encoded_x[i].values())[0])
      self.assertAllClose(np.array([2.0, 6.0]) * num_summands,
                          list(data.summed_part_decoded_x.values())[0])
      self.assertAllClose(np.array([1.0, 3.0]) * num_summands, data.decoded_x)

  @tf_test_util.run_all_in_graph_and_eager_modes
  def test_full_commutativity_with_sum(self):
    """Tests that fully commutes with sum property works."""
    spec = tf.TensorSpec((2,), tf.float32)

    encoder = gather_encoder.GatherEncoder.from_encoder(
        core_encoder.EncoderComposer(test_utils.TimesTwoEncodingStage()).make(),
        spec)
    self.assertTrue(encoder.fully_commutes_with_sum)

    encoder = gather_encoder.GatherEncoder.from_encoder(
        core_encoder.EncoderComposer(
            test_utils.TimesTwoEncodingStage()).add_parent(
                test_utils.TimesTwoEncodingStage(), T2_VALS).make(), spec)
    self.assertTrue(encoder.fully_commutes_with_sum)

    encoder = core_encoder.EncoderComposer(
        test_utils.SignIntFloatEncodingStage())
    encoder.add_child(test_utils.TimesTwoEncodingStage(), SIF_SIGNS)
    encoder.add_child(test_utils.PlusOneEncodingStage(), SIF_INTS)
    encoder.add_child(test_utils.TimesTwoEncodingStage(), SIF_FLOATS).add_child(
        test_utils.PlusOneOverNEncodingStage(), T2_VALS)
    encoder = gather_encoder.GatherEncoder.from_encoder(encoder.make(), spec)
    self.assertFalse(encoder.fully_commutes_with_sum)

  @tf_test_util.run_all_in_graph_and_eager_modes
  def test_state_aggregation_modes(self):
    """Tests that all state updates tensors can be aggregated."""
    x_fn = lambda: tf.random.uniform((5,))
    encoder = gather_encoder.GatherEncoder.from_encoder(
        core_encoder.EncoderComposer(
            test_utils.StateUpdateTensorsEncodingStage()).make(),
        tf.TensorSpec.from_tensor(x_fn()))

    iteration = _make_iteration_function(encoder, x_fn, 3)
    data = self.evaluate(iteration(encoder.initial_state()))

    expected_sum = np.sum(data.x)
    expected_min = np.amin(data.x)
    expected_max = np.amax(data.x)
    expected_stack_values = 15  # 3 values of shape 5.
    expected_state = [
        expected_sum, expected_min, expected_max, expected_stack_values
    ]
    # We are not in control of ordering of the elements in state tuple.
    self.assertAllClose(sorted(expected_state), sorted(data.updated_state))

  def test_input_tensorspec(self):
    """Tests input_tensorspec property."""
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    encoder = gather_encoder.GatherEncoder.from_encoder(
        core_encoder.EncoderComposer(
            test_utils.PlusOneOverNEncodingStage()).make(),
        tf.TensorSpec.from_tensor(x))
    self.assertTrue(encoder.input_tensorspec.is_compatible_with(x))

  def test_not_fully_defined_shape_raises(self):
    """Tests tensorspec without fully defined shape."""
    encoder = core_encoder.EncoderComposer(
        test_utils.PlusOneOverNEncodingStage()).make()
    with self.assertRaisesRegex(TypeError, 'fully defined'):
      gather_encoder.GatherEncoder.from_encoder(
          encoder, tf.TensorSpec((None,), tf.float32))

  @parameterized.parameters([1.0, 'str', object])
  def test_not_an_encoder_raises(self, not_an_encoder):
    """Tests invalid type encoder argument."""
    tensorspec = tf.TensorSpec((1,), tf.float32)
    with self.assertRaisesRegex(TypeError, 'Encoder'):
      gather_encoder.GatherEncoder.from_encoder(not_an_encoder, tensorspec)

  @parameterized.parameters([1.0, 'str', object])
  def test_not_a_tensorspec_raises(self, not_a_tensorspec):
    """Tests invalid type of tensorspec argument."""
    encoder = core_encoder.EncoderComposer(
        test_utils.PlusOneOverNEncodingStage()).make()
    with self.assertRaisesRegex(TypeError, 'TensorSpec'):
      gather_encoder.GatherEncoder.from_encoder(encoder, not_a_tensorspec)


TestData = collections.namedtuple('TestData', [
    'x',
    'encoded_x',
    'part_decoded_x',
    'summed_part_decoded_x',
    'decoded_x',
    'initial_state',
    'updated_state',
])


def _make_iteration_function(encoder, x_fn, num_summands):
  """Returns a tf.function utility for testing."""

  assert isinstance(encoder, gather_encoder.GatherEncoder)

  @tf.function
  def iteration(initial_state):
    x = []
    encoded_x = []
    part_decoded_x = []
    state_update_tensors = []

    encode_params, decode_before_sum_params, decode_after_sum_params = (
        encoder.get_params(initial_state))
    for _ in range(num_summands):
      x_value = x_fn()
      enc_x, sut = encoder.encode(x_value, encode_params)
      part_dec_x = encoder.decode_before_sum(enc_x, decode_before_sum_params)
      x.append(x_value)
      encoded_x.append(enc_x)
      part_decoded_x.append(part_dec_x)
      state_update_tensors.append(sut)

    summed_part_decoded_x = part_decoded_x[0]
    for addend in part_decoded_x[1:]:
      summed_part_decoded_x = tf.nest.map_structure(lambda x, y: x + y,
                                                    summed_part_decoded_x,
                                                    addend)

    decoded_x = encoder.decode_after_sum(summed_part_decoded_x,
                                         decode_after_sum_params, num_summands)

    aggregated_state_update_tensors = _aggregate_structure(
        state_update_tensors, encoder.state_update_aggregation_modes)
    updated_state = encoder.update_state(initial_state,
                                         aggregated_state_update_tensors)
    return TestData(x, encoded_x, part_decoded_x, summed_part_decoded_x,
                    decoded_x, initial_state, updated_state)

  return iteration


def _make_stateless_iteration_function(encoder, x_fn, num_summands):
  """Returns a tf.function utility for testing, which does not use state."""

  assert isinstance(encoder, gather_encoder.GatherEncoder)

  @tf.function
  def iteration():
    x = []
    encoded_x = []
    part_decoded_x = []

    encode_params, decode_before_sum_params, decode_after_sum_params = (
        encoder.get_params())
    for _ in range(num_summands):
      x_value = x_fn()
      enc_x, _ = encoder.encode(x_value, encode_params)
      part_dec_x = encoder.decode_before_sum(enc_x, decode_before_sum_params)
      x.append(x_value)
      encoded_x.append(enc_x)
      part_decoded_x.append(part_dec_x)

    summed_part_decoded_x = part_decoded_x[0]
    for addend in part_decoded_x[1:]:
      summed_part_decoded_x = tf.nest.map_structure(lambda x, y: x + y,
                                                    summed_part_decoded_x,
                                                    addend)

    decoded_x = encoder.decode_after_sum(summed_part_decoded_x,
                                         decode_after_sum_params, num_summands)

    dummy = tf.constant(0.0)  # Avoids having to separate TF/PY values.
    return TestData(x, encoded_x, part_decoded_x, summed_part_decoded_x,
                    decoded_x, dummy, dummy)

  return iteration


def _aggregate_one(values, mode):
  if mode == encoding_stage.StateAggregationMode.SUM:
    return tf.reduce_sum(tf.stack(values), axis=0)
  elif mode == encoding_stage.StateAggregationMode.MIN:
    return tf.reduce_min(tf.stack(values), axis=0)
  elif mode == encoding_stage.StateAggregationMode.MAX:
    return tf.reduce_max(tf.stack(values), axis=0)
  elif mode == encoding_stage.StateAggregationMode.STACK:
    return tf.stack(values)


def _aggregate_structure(state_update_tensors, state_update_aggregation_modes):
  aggregated_state_update_tensors = []
  for i, mode in enumerate(state_update_aggregation_modes):
    values = [t[i] for t in state_update_tensors]
    aggregated_state_update_tensors.append(_aggregate_one(values, mode))
  return tuple(aggregated_state_update_tensors)


def _encoded_x_field(encoded_x, path):
  """Returns a field from `encoded_x` returned by the `encode` method.

  In order to test the correctness of encoding, we also need to access the
  encoded objects, which in turns depends on an implementation detail (the
  specific use of `nest.flatten_with_joined_string_paths`). This dependence is
  constrained to a single place in this utility.

  Args:
    encoded_x: The structure returned by the `encode` method.
    path: A list of keys corresponding to the path in the nested dictionary
      before it was flattened.

  Returns:
    A value from `encoded_x` corresponding to the `path`.
  """
  return encoded_x['/'.join(path)]


if __name__ == '__main__':
  tf.test.main()
