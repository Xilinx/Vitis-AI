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

from absl.testing import parameterized
import tensorflow as tf

# TODO(b/139939526): Move to public API.
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import core_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import simple_encoder
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing import test_utils

# Abbreviated constants used in tests.
TENSORS = simple_encoder._TENSORS

P1_VALS = test_utils.PlusOneEncodingStage.ENCODED_VALUES_KEY
T2_VALS = test_utils.TimesTwoEncodingStage.ENCODED_VALUES_KEY
SIF_SIGNS = test_utils.SignIntFloatEncodingStage.ENCODED_SIGNS_KEY
SIF_INTS = test_utils.SignIntFloatEncodingStage.ENCODED_INTS_KEY
SIF_FLOATS = test_utils.SignIntFloatEncodingStage.ENCODED_FLOATS_KEY
PN_VALS = test_utils.PlusOneOverNEncodingStage.ENCODED_VALUES_KEY


class SimpleEncoderTest(tf.test.TestCase, parameterized.TestCase):

  @tf_test_util.run_all_in_graph_and_eager_modes
  def test_basic_encode_decode(self):
    """Tests basic encoding and decoding works as expected."""
    x = tf.constant(1.0, tf.float32)
    encoder = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.PlusOneOverNEncodingStage()).make(),
        tf.TensorSpec.from_tensor(x))

    state = encoder.initial_state()
    iteration = _make_iteration_function(encoder)
    for i in range(1, 10):
      x, encoded_x, decoded_x, state = self.evaluate(iteration(x, state))
      self.assertAllClose(x, decoded_x)
      self.assertAllClose(1.0 + 1 / i,
                          _encoded_x_field(encoded_x, [TENSORS, PN_VALS]))

  @tf_test_util.run_all_in_graph_and_eager_modes
  def test_composite_encoder(self):
    """Tests functionality with a general, composite `Encoder`."""
    x = tf.constant(1.2)
    encoder = core_encoder.EncoderComposer(
        test_utils.SignIntFloatEncodingStage())
    encoder.add_child(test_utils.TimesTwoEncodingStage(), SIF_SIGNS)
    encoder.add_child(test_utils.PlusOneEncodingStage(), SIF_INTS)
    encoder.add_child(test_utils.TimesTwoEncodingStage(), SIF_FLOATS).add_child(
        test_utils.PlusOneOverNEncodingStage(), T2_VALS)
    encoder = simple_encoder.SimpleEncoder(encoder.make(),
                                           tf.TensorSpec.from_tensor(x))

    state = encoder.initial_state()
    iteration = _make_iteration_function(encoder)
    for i in range(1, 10):
      x, encoded_x, decoded_x, state = self.evaluate(iteration(x, state))
      self.assertAllClose(x, decoded_x)
      self.assertAllClose(
          2.0, _encoded_x_field(encoded_x, [TENSORS, SIF_SIGNS, T2_VALS]))
      self.assertAllClose(
          2.0, _encoded_x_field(encoded_x, [TENSORS, SIF_INTS, P1_VALS]))
      self.assertAllClose(
          0.4 + 1 / i,
          _encoded_x_field(encoded_x, [TENSORS, SIF_FLOATS, T2_VALS, PN_VALS]))

  @tf_test_util.run_all_in_graph_and_eager_modes
  def test_none_state_equal_to_initial_state(self):
    """Tests that not providing state is the same as initial_state."""
    x = tf.constant(1.0)
    encoder = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.PlusOneOverNEncodingStage()).make(),
        tf.TensorSpec.from_tensor(x))

    state = encoder.initial_state()
    stateful_iteration = _make_iteration_function(encoder)

    @tf.function
    def stateless_iteration(x):
      encoded_x, _ = encoder.encode(x)
      decoded_x = encoder.decode(encoded_x)
      return encoded_x, decoded_x

    _, encoded_x_stateful, decoded_x_stateful, _ = self.evaluate(
        stateful_iteration(x, state))
    encoded_x_stateless, decoded_x_stateless = self.evaluate(
        stateless_iteration(x))

    self.assertAllClose(encoded_x_stateful, encoded_x_stateless)
    self.assertAllClose(decoded_x_stateful, decoded_x_stateless)

  @tf_test_util.run_all_in_graph_and_eager_modes
  def test_python_constants_not_exposed(self):
    """Tests that only TensorFlow values are exposed to users."""
    x = tf.constant(1.0)
    tensorspec = tf.TensorSpec.from_tensor(x)
    encoder_py = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.SimpleLinearEncodingStage(2.0, 3.0)).make(), tensorspec)
    a_var = tf.compat.v1.get_variable('a_var', initializer=2.0)
    b_var = tf.compat.v1.get_variable('b_var', initializer=3.0)
    encoder_tf = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.SimpleLinearEncodingStage(a_var, b_var)).make(),
        tensorspec)

    state_py = encoder_py.initial_state()
    state_tf = encoder_tf.initial_state()
    iteration_py = _make_iteration_function(encoder_py)
    iteration_tf = _make_iteration_function(encoder_tf)

    self.evaluate(tf.compat.v1.global_variables_initializer())
    _, encoded_x_py, decoded_x_py, _ = self.evaluate(iteration_py(x, state_py))
    _, encoded_x_tf, decoded_x_tf, _ = self.evaluate(iteration_tf(x, state_tf))

    # The encoded_x_tf should have two elements that encoded_x_py does not.
    # These correspond to the two variables created passed on to constructor of
    # encoder_tf, which are exposed as params. For encoder_py, these are python
    # integers, and should thus be hidden from users.
    self.assertLen(encoded_x_tf, len(encoded_x_py) + 2)

    # Make sure functionality is still the same.
    self.assertAllClose(x, decoded_x_tf)
    self.assertAllClose(x, decoded_x_py)

  @tf_test_util.run_all_in_graph_and_eager_modes
  def test_decode_needs_input_shape_static(self):
    """Tests that mechanism for passing input shape works with static shape."""
    x = tf.reshape(list(range(15)), [3, 5])
    encoder = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.ReduceMeanEncodingStage()).make(),
        tf.TensorSpec.from_tensor(x))

    state = encoder.initial_state()
    iteration = _make_iteration_function(encoder)
    _, _, decoded_x, _ = self.evaluate(iteration(x, state))
    self.assertAllEqual([[7.0] * 5] * 3, decoded_x)

  def test_not_fully_defined_shape_raises(self):
    """Tests tensorspec without fully defined shape."""
    encoder = core_encoder.EncoderComposer(
        test_utils.PlusOneOverNEncodingStage()).make()
    with self.assertRaisesRegex(TypeError, 'fully defined'):
      simple_encoder.SimpleEncoder(encoder, tf.TensorSpec((None,), tf.float32))

  @tf_test_util.run_all_in_graph_and_eager_modes
  def test_input_signature_enforced(self):
    """Tests that encode/decode input signature is enforced."""
    x = tf.constant(1.0)
    encoder = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.PlusOneOverNEncodingStage()).make(),
        tf.TensorSpec.from_tensor(x))

    state = encoder.initial_state()
    with self.assertRaises(ValueError):
      bad_x = tf.stack([x, x])
      encoder.encode(bad_x, state)
    with self.assertRaises(ValueError):
      bad_state = state + (x,)
      encoder.encode(x, bad_state)
    encoded_x = encoder.encode(x, state)
    with self.assertRaises(ValueError):
      bad_encoded_x = dict(encoded_x)
      bad_encoded_x.update({'x': x})
      encoder.decode(bad_encoded_x)

  def test_input_tensorspec(self):
    """Tests input_tensorspec property."""
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    encoder = simple_encoder.SimpleEncoder(
        core_encoder.EncoderComposer(
            test_utils.PlusOneOverNEncodingStage()).make(),
        tf.TensorSpec.from_tensor(x))
    self.assertTrue(encoder.input_tensorspec.is_compatible_with(x))

  @parameterized.parameters([1.0, 'str', object])
  def test_not_an_encoder_raises(self, not_an_encoder):
    """Tests invalid encoder argument."""
    tensorspec = tf.TensorSpec((1,), tf.float32)
    with self.assertRaisesRegex(TypeError, 'Encoder'):
      simple_encoder.SimpleEncoder(not_an_encoder, tensorspec)

  @parameterized.parameters([1.0, 'str', object])
  def test_not_a_tensorspec_raises(self, not_a_tensorspec):
    """Tests invalid type of tensorspec argument."""
    encoder = core_encoder.EncoderComposer(
        test_utils.PlusOneOverNEncodingStage()).make()
    with self.assertRaisesRegex(TypeError, 'TensorSpec'):
      simple_encoder.SimpleEncoder(encoder, not_a_tensorspec)


def _make_iteration_function(encoder):
  assert isinstance(encoder, simple_encoder.SimpleEncoder)

  @tf.function
  def iteration(x, state):
    encoded_x, new_state = encoder.encode(x, state)
    decoded_x = encoder.decode(encoded_x)
    return x, encoded_x, decoded_x, new_state

  return iteration


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
