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
"""Tests mixed precision works correctly with Keras layers and models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision.experimental import policy
from tensorflow.python.keras.mixed_precision.experimental import test_util as mp_test_util
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.tracking import util as trackable_utils
from tensorflow.python.util import nest


class AssertTypeLayer(base_layer.Layer):
  """A layer which asserts it's inputs are a certain type."""

  def __init__(self, assert_type=None, **kwargs):
    self._assert_type = assert_type
    super(AssertTypeLayer, self).__init__(**kwargs)

  def assert_input_types(self, inputs):
    """Asserts `inputs` are of the correct type. Should be called in call()."""
    if self._assert_type:
      inputs_flattened = nest.flatten(inputs)
      for inp in inputs_flattened:
        assert inp.dtype.base_dtype == self._assert_type, (
            'Input tensor has type %s which does not match assert type %s' %
            (inp.dtype.name, self._assert_type.name))


class AddLayer(AssertTypeLayer):
  """A layer which adds it's input to a scalar variable."""

  def __init__(self,
               regularizer=None,
               use_operator=False,
               var_name='v',
               **kwargs):
    """Initializes the AddLayer.

    Args:
      regularizer: The regularizer on the scalar variable.
      use_operator: If True, add using the + operator. If False, add using
        tf.add.
      var_name: The name of the variable. It can be useful to pass a name other
        than 'v', to test having the attribute name (self.v) being different
        from the variable name.
      **kwargs: Passed to AssertTypeLayer constructor.
    """
    self._regularizer = regularizer
    self._use_operator = use_operator
    self._var_name = var_name
    super(AddLayer, self).__init__(**kwargs)

  def build(self, _):
    self.v = self.add_weight(
        self._var_name, (), initializer='ones', regularizer=self._regularizer)
    self.built = True

  def call(self, inputs):
    self.assert_input_types(inputs)
    assert inputs.dtype == self.v.dtype
    return self._add(inputs, self.v)

  def _add(self, x, y):
    if self._use_operator:
      return x + y
    else:
      return math_ops.add(x, y)


class AddLayerWithoutAutoCast(AddLayer):
  """Same as AddLayer, but does not use AutoCastVariables."""

  def build(self, _):
    dtype = self.dtype
    if dtype in ('float16', 'bfloat16'):
      dtype = 'float32'
    self.v = self.add_weight(
        'v', (),
        initializer='ones',
        dtype=dtype,
        experimental_autocast=False,
        regularizer=self._regularizer)
    self.built = True

  def call(self, inputs):
    self.assert_input_types(inputs)
    assert self.v.dtype in (dtypes.float32, dtypes.float64)
    return self._add(inputs, math_ops.cast(self.v, inputs.dtype))


class AddLayerWithFunction(AddLayer):
  """Same as AddLayer, but _add is decorated with a tf.function."""

  @def_function.function
  def _add(self, x, y):
    return super(AddLayerWithFunction, self)._add(x, y)


class IdentityRegularizer(regularizers.Regularizer):

  def __call__(self, x):
    assert x.dtype == dtypes.float32
    return array_ops.identity(x)


# If called outside any strategy.scope() calls, this will return the default
# strategy.
default_strategy_fn = distribution_strategy_context.get_strategy


def create_mirrored_strategy():
  if context.num_gpus() >= 1:
    return mirrored_strategy.MirroredStrategy(['cpu:0', 'gpu:0'])
  else:
    return mirrored_strategy.MirroredStrategy(['cpu:0'])


TESTCASES = ({
    'testcase_name': 'base',
    'strategy_fn': default_strategy_fn
}, {
    'testcase_name': 'distribute',
    'strategy_fn': create_mirrored_strategy
})


class KerasLayerTest(keras_parameterized.TestCase):
  """Test mixed precision with Keras layers."""

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_infer_with_float32_vars(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope(), policy.policy_scope('infer_float32_vars'):
      layer = AddLayer(assert_type=dtypes.float16)
      self.assertEqual(layer.dtype, dtypes.float32)
      y = layer(x)
      self.assertEqual(layer.v.dtype, dtypes.float32)
      self.assertEqual(y.dtype, dtypes.float16)
      self.assertEqual(layer.dtype, dtypes.float32)
      self.assertEqual(layer._dtype_policy._name, 'float16_with_float32_vars')
      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(self.evaluate(y), 2.)

      if base_layer_utils.v2_dtype_behavior_enabled():
        # Layer should now cast inputs to float16
        x = constant_op.constant([1.], dtype=dtypes.float32)
        y = layer(x)
        self.assertEqual(y.dtype, dtypes.float16)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  @testing_utils.enable_v2_dtype_behavior
  def test_floating_point_policies_with_float32_vars(self, strategy_fn):
    for dtype in 'bfloat16', 'float16', 'float64':
      x = constant_op.constant([1.])
      policy_name = dtype + '_with_float32_vars'
      with strategy_fn().scope(), policy.policy_scope(policy_name):
        layer = AddLayer(assert_type=dtype)
        self.assertEqual(layer.dtype, dtypes.float32)
        self.assertEqual(layer._dtype_policy._name, policy_name)
        y = layer(x)
        self.assertEqual(layer.v.dtype, dtypes.float32)
        self.assertEqual(y.dtype, dtype)
        self.assertEqual(layer.dtype, dtypes.float32)
        self.assertEqual(layer._dtype_policy._name, policy_name)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(y), 2.)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  @testing_utils.enable_v2_dtype_behavior
  def test_int32_with_float32_vars(self, strategy_fn):

    # The policy int32_with_float32_vars is not useful at all (nor is any other
    # non-float policy with float32 variables), but we have it for consistency,
    # and so we test it.

    class IdentityLayerWithVar(base_layer.Layer):

      def build(self, _):
        self.v = self.add_weight('v', ())

      def call(self, inputs):
        # Variables are only casted to other floats, not ints
        assert array_ops.identity(self.v).dtype == 'float32'
        return array_ops.identity(inputs)

    x = constant_op.constant([1])
    with strategy_fn().scope(), policy.policy_scope('int32_with_float32_vars'):
      layer = IdentityLayerWithVar()
      self.assertEqual(layer.dtype, dtypes.float32)
      self.assertEqual(layer._dtype_policy._name, 'int32_with_float32_vars')
      y = layer(x)
      self.assertEqual(layer.v.dtype, dtypes.float32)
      self.assertEqual(y.dtype, dtypes.int32)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_layer_with_non_autocast_variable(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope():
      with policy.policy_scope('infer_float32_vars'):
        layer = AddLayerWithoutAutoCast(assert_type=dtypes.float16)
        y = layer(x)
        self.assertEqual(layer.v.dtype, dtypes.float32)
        self.assertEqual(y.dtype, dtypes.float16)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(y), 2.)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_layer_calling_tf_function(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope():
      with policy.policy_scope('infer_float32_vars'):
        layer = AddLayerWithFunction(assert_type=dtypes.float16)
        y = layer(x)
        self.assertEqual(layer.v.dtype, dtypes.float32)
        self.assertEqual(y.dtype, dtypes.float16)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(y), 2.)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_layer_regularizer_runs_in_var_dtype(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope():
      with policy.policy_scope('infer_float32_vars'):
        # Test on AddLayer
        layer = AddLayer(
            assert_type=dtypes.float16, regularizer=IdentityRegularizer())
        layer(x)
        (regularizer_loss,) = layer.losses
        self.assertEqual(regularizer_loss.dtype, dtypes.float32)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(regularizer_loss), 1.)

        # Test on AddLayerWithoutAutoCast
        layer = AddLayerWithoutAutoCast(
            assert_type=dtypes.float16, regularizer=IdentityRegularizer())
        layer(x)
        (regularizer_loss,) = layer.losses
        self.assertEqual(regularizer_loss.dtype, dtypes.float32)
        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(self.evaluate(regularizer_loss), 1.)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_passing_policy_to_layer(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope():
      # Passing a Policy to 'dtype' sets the policy for that layer.
      layer = AddLayer(
          assert_type=dtypes.float16, dtype=policy.Policy('infer_float32_vars'))
      # layer.dtype refers to the variable dtype
      self.assertEqual(layer.dtype, dtypes.float32)
      layer(x)
      self.assertEqual(layer.v.dtype, dtypes.float32)
      with policy.policy_scope('infer_float32_vars'):
        # Passing a Policy to dtype overrides the global Policy
        layer = AddLayer(
            assert_type=dtypes.float16, dtype=policy.Policy('infer'))
        # layer dtype is not yet known
        self.assertEqual(layer.dtype, None)
        layer(x)
        self.assertEqual(layer.v.dtype, dtypes.float16)
        self.assertEqual(layer.dtype, dtypes.float16)

  @test_util.run_in_graph_and_eager_modes
  def test_error_passing_policy_string_to_layer(self):
    with self.assertRaisesRegexp(
        TypeError, "Cannot convert value 'float16_with_float32_vars' to a "
                   "TensorFlow DType"):
      # This is not allowed, as otherwise a "float16_with_float32_vars" policy
      # could be created without an API call that has the name "experimental" in
      # it.
      AddLayer(dtype='float16_with_float32_vars')

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_gradient(self, strategy_fn):
    x = constant_op.constant([1.], dtype=dtypes.float16)
    with strategy_fn().scope() as strategy:
      with policy.policy_scope('infer_float32_vars'):
        layer = AddLayer(assert_type=dtypes.float16)

        def run_fn():
          with backprop.GradientTape() as tape:
            y = layer(x)
            # Divide by num_replicas_in_sync, as the effective total loss is the
            # sum of each of the replica's losses.
            y /= strategy.num_replicas_in_sync

          # Learning rate is small enough that if applied to a float16 variable,
          # the variable will not change. So this tests the learning rate is not
          # applied to a float16 value, but instead the float32 variable.
          opt = gradient_descent.SGD(2**-14)
          grad = tape.gradient(y, layer.v)
          return opt.apply_gradients([(grad, layer.v)])

        op = strategy.experimental_run(run_fn)
        if not context.executing_eagerly():
          self.evaluate(variables.global_variables_initializer())
          self.evaluate(op)
        # The gradient with respective to the variable is 1. Since the
        # variable is initialized with 1 and the learning rate is 2**-14, the
        # new variable value should be: init_val - gradient * learning_rate,
        # which is  1 - 1 * 2**-14
        self.assertEqual(self.evaluate(layer.v), 1 - 2**-14)

  def _test_checkpointing_layer_weights(self, strategy_fn,
                                        mixed_prec_when_saving,
                                        mixed_prec_when_loading):
    # In this test, we potentially save with mixed precision enabled and load
    # with mixed precision disabled, or vice versa. This is possible because
    # variables are float32 regardless of whether mixed precision is enabled.
    save_policy = 'infer_float32_vars' if mixed_prec_when_saving else 'infer'
    load_policy = 'infer_float32_vars' if mixed_prec_when_loading else 'infer'
    save_input_dtype = 'float16' if mixed_prec_when_saving else 'float32'
    load_input_dtype = 'float16' if mixed_prec_when_loading else 'float32'

    # Create a layer and save a checkpoint.
    x = constant_op.constant([1.], dtype=save_input_dtype)
    with strategy_fn().scope():
      with policy.policy_scope(save_policy):
        layer = AddLayer(assert_type=save_input_dtype)
        layer(x)  # Build layer
    layer.set_weights([np.array(100.)])
    self.assertEqual(self.evaluate(layer(x)), 101.)
    checkpoint = trackable_utils.Checkpoint(layer=layer)
    prefix = os.path.join(self.get_temp_dir(), 'ckpt')
    save_path = checkpoint.save(prefix)

    # Create a new layer and restore the checkpoint.
    x = constant_op.constant([1.], dtype=load_input_dtype)
    with strategy_fn().scope():
      with policy.policy_scope(load_policy):
        layer = AddLayer(assert_type=load_input_dtype)
        layer(x)  # Build layer
    layer.set_weights([np.array(200.)])
    self.assertEqual(self.evaluate(layer(x)), 201.)
    checkpoint = trackable_utils.Checkpoint(layer=layer)
    checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    self.assertEqual(layer.get_weights(), [100.])
    self.assertEqual(self.evaluate(layer(x)), 101.)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_checkpointing_layer_weights(self, strategy_fn):
    self._test_checkpointing_layer_weights(
        strategy_fn, mixed_prec_when_saving=True, mixed_prec_when_loading=True)
    self._test_checkpointing_layer_weights(
        strategy_fn, mixed_prec_when_saving=True, mixed_prec_when_loading=False)
    self._test_checkpointing_layer_weights(
        strategy_fn, mixed_prec_when_saving=False, mixed_prec_when_loading=True)


class KerasModelTest(keras_parameterized.TestCase):
  """Test mixed precision with Keras models."""

  def _is_strategy_supported(self, strategy_fn, check_model_type=False):
    if (strategy_fn != default_strategy_fn and
        (testing_utils.should_run_eagerly() or
         (check_model_type and testing_utils.get_model_type() == 'subclass'))):
      # Distribution strategies do not support subclassed models or running with
      # `run_eagerly=True`.
      return False
    else:
      return True

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
          'strategy_fn': default_strategy_fn
      }, {
          'testcase_name': 'distribute',
          'strategy_fn': create_mirrored_strategy,
      }, {
          'testcase_name': 'operator',
          'strategy_fn': create_mirrored_strategy,
          'use_operator': True
      }, {
          'testcase_name': 'regularizer',
          'strategy_fn': create_mirrored_strategy,
          'use_regularizer': True
      }, {
          'testcase_name': 'infer',
          'strategy_fn': create_mirrored_strategy,
          'policy_name': 'mixed_float16'
      }, {
          'testcase_name': 'norun_distributed',
          'strategy_fn': create_mirrored_strategy,
          'experimental_run_tf_function': False
      })
  @testing_utils.enable_v2_dtype_behavior
  def test_model(self,
                 strategy_fn,
                 use_operator=False,
                 use_regularizer=False,
                 policy_name='mixed_float16',
                 experimental_run_tf_function=True):
    if not self._is_strategy_supported(strategy_fn, check_model_type=True):
      return
    regularizer = IdentityRegularizer() if use_regularizer else None
    with strategy_fn().scope():
      # Pass loss_scale=None, as this test will fail if the DynamicLossScale
      # skips applying gradients for a step
      with policy.policy_scope(policy.Policy(policy_name, loss_scale=None)):
        layer_list = []
        if testing_utils.get_model_type() == 'subclass':
          # Subclassed models do not have an Input layer, so the model does not
          # cast inputs to the Input layer's dtype. Therefore, we need to
          # manually insert a float16 cast.
          cast_f16_layer = layers.Lambda(
              lambda x: math_ops.cast(x, 'float16'), input_shape=(1,))
          layer_list.append(cast_f16_layer)
        layer = AddLayer(
            assert_type=dtypes.float16,
            use_operator=use_operator,
            regularizer=regularizer,
            input_shape=(1,))
        cast_f32_layer = layers.Lambda(lambda x: math_ops.cast(x, 'float32'))
        layer_list += [layer, cast_f32_layer]
        model = testing_utils.get_model_from_layers(
            layer_list, input_shape=(1,), input_dtype=dtypes.float16)

        def loss_fn(y_true, y_pred):
          del y_true
          return math_ops.reduce_mean(y_pred)

        # Learning rate is small enough that if applied to a float16 variable,
        # the variable will not change. So this tests the learning rate not
        # applied to a float16 value, but instead the float32 variable.
        opt = gradient_descent.SGD(2**-14)
        model.compile(
            opt,
            loss=loss_fn,
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones((2, 1))
    y = np.ones((2, 1))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(2)
    model.fit(dataset)
    # Variable starts at 1, and should have gradient of 2 ** -14 subtracted
    # from it.
    expected = 1 - 2**-14
    if use_regularizer:
      # Regularizer adds another 2 ** -14 to the gradient.
      expected -= 2**-14
    self.assertEqual(backend.eval(layer.v), expected)

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
          'strategy_fn': default_strategy_fn
      }, {
          'testcase_name': 'distribute',
          'strategy_fn': create_mirrored_strategy,
      }, {
          'testcase_name': 'norun_distributed',
          'strategy_fn': create_mirrored_strategy,
          'experimental_run_tf_function': False,
      })
  def test_fixed_loss_scaling(self,
                              strategy_fn,
                              experimental_run_tf_function=True):
    # Note: We do not test mixed precision in this method, only loss scaling.
    if not self._is_strategy_supported(strategy_fn):
      return
    loss_scale = 8.
    batch_size = 4
    with strategy_fn().scope():
      x = layers.Input(shape=(1,), batch_size=batch_size)
      layer = AddLayer()
      y = layer(x)

      # The gradient of 'y' at this point is 1. With loss scaling, the gradient
      # is 'loss_scale'. We divide by the batch size since the loss is averaged
      # across batch elements.
      expected_gradient = loss_scale / batch_size
      identity_with_grad_check_fn = (
          mp_test_util.create_identity_with_grad_check_fn([expected_gradient]))
      y = core.Lambda(identity_with_grad_check_fn)(y)
      model = models.Model(inputs=x, outputs=y)

      def loss_fn(y_true, y_pred):
        del y_true
        return math_ops.reduce_mean(y_pred)

      opt = gradient_descent.SGD(1.)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
      model.compile(
          opt,
          loss=loss_fn,
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

    self.assertEqual(backend.eval(layer.v), 1)
    x = np.ones((batch_size, 1))
    y = np.ones((batch_size, 1))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    model.fit(dataset)
    # Variable starts at 1, and should have gradient of 1 subtracted from it.
    expected = 0
    self.assertEqual(backend.eval(layer.v), expected)

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
          'strategy_fn': default_strategy_fn
      }, {
          'testcase_name': 'distribute',
          'strategy_fn': create_mirrored_strategy,
      }, {
          'testcase_name': 'loss_scaling',
          'strategy_fn': create_mirrored_strategy,
          'use_loss_scaling': True
      })
  @testing_utils.enable_v2_dtype_behavior
  def test_advanced_model(self, strategy_fn, use_loss_scaling=False):
    # The advanced model tests mixed-precision-related features that would occur
    # in a resnet50 model. It tests a model that has:
    #  * Multiple layers, some which use auto-cast variables and some which do
    #    not
    #  * Regularization on some variables and not others.
    #  * A fixed loss scale (if use_loss_scaling is True)

    if not self._is_strategy_supported(strategy_fn):
      return
    strategy = strategy_fn()
    if use_loss_scaling:
      loss_scale = 8.
    else:
      loss_scale = None
    learning_rate = 2**-14

    with strategy.scope():
      with policy.policy_scope(policy.Policy('mixed_float16',
                                             loss_scale=loss_scale)):
        x = layers.Input(shape=(1,), batch_size=2)
        layer1 = AddLayer(
            assert_type=dtypes.float16,
            regularizer=IdentityRegularizer(),
            use_operator=True)
        layer2 = AddLayerWithoutAutoCast(
            assert_type=dtypes.float16, use_operator=True)
        layer3 = AddLayer(assert_type=dtypes.float16, use_operator=False)
        layer4 = AddLayerWithoutAutoCast(
            assert_type=dtypes.float16,
            regularizer=IdentityRegularizer(),
            use_operator=False)
        y = layer1(x)
        y = layer2(y)
        y = layer3(y)
        y = layer4(y)
        if use_loss_scaling:
          # The gradient of 'y' at this point is 1. With loss scaling, the
          # gradient is 'loss_scale'. We divide by the batch size of 2 since the
          # loss is averaged across batch elements.
          expected_gradient = loss_scale / 2
          identity_with_grad_check_fn = (
              mp_test_util.create_identity_with_grad_check_fn(
                  expected_dtype=dtypes.float16,
                  expected_gradient=[expected_gradient]))
          y = core.Lambda(identity_with_grad_check_fn)(y)
        y = math_ops.cast(y, dtypes.float32)
        model = models.Model(inputs=x, outputs=y)

        def loss_fn(y_true, y_pred):
          self.assertEqual(y_true.dtype, dtypes.float32)
          self.assertEqual(y_pred.dtype, dtypes.float32)
          return math_ops.reduce_mean(y_pred)

        opt = gradient_descent.SGD(learning_rate)
        model.compile(
            opt,
            loss=loss_fn,
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function())

    x = np.ones((2, 1))
    y = np.ones((2, 1))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(2)
    model.fit(dataset)
    for layer in (layer1, layer2, layer3, layer4):
      if layer.losses:
        # Layer has weight regularizer
        self.assertEqual(backend.eval(layer.v), 1 - 2 * learning_rate)
      else:
        # Layer does not have weight regularizer
        self.assertEqual(backend.eval(layer.v), 1 - learning_rate)

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
          'strategy_fn': default_strategy_fn
      }, {
          'testcase_name': 'distribute',
          'strategy_fn': create_mirrored_strategy,
      }, {
          'testcase_name': 'pass_loss_scale_to_policy',
          'strategy_fn': create_mirrored_strategy,
          'pass_loss_scale_to_policy': True,
      }, {
          'testcase_name': 'norun_distributed',
          'strategy_fn': create_mirrored_strategy,
          'experimental_run_tf_function': False,
      })
  def test_dynamic_loss_scaling(self,
                                strategy_fn,
                                pass_loss_scale_to_policy=False,
                                experimental_run_tf_function=True):
    if not self._is_strategy_supported(strategy_fn):
      return
    strategy = strategy_fn()
    initial_loss_scale = 2.
    batch_size = 4
    loss_scale = loss_scale_module.DynamicLossScale(
        initial_loss_scale=initial_loss_scale, increment_period=2)
    expected_gradient = backend.variable([initial_loss_scale / batch_size],
                                         dtype=dtypes.float16)
    # If this variable is set to True, the model below will have NaN gradients
    have_nan_gradients = backend.variable(False, dtype=dtypes.bool)
    with strategy.scope():
      opt = gradient_descent.SGD(1.)
      if pass_loss_scale_to_policy:
        p = policy.Policy('infer_float32_vars', loss_scale=loss_scale)
      else:
        p = policy.Policy('infer_float32_vars')
        opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
      with policy.policy_scope(p):
        x = layers.Input(
            shape=(1,), batch_size=batch_size, dtype=dtypes.float16)
        layer = AddLayer(assert_type=dtypes.float16)
        y = layer(x)
        identity_with_nan_grads = (
            mp_test_util.create_identity_with_nan_gradients_fn(
                have_nan_gradients))
        y = core.Lambda(identity_with_nan_grads)(y)
        identity_with_grad_check_fn = (
            mp_test_util.create_identity_with_grad_check_fn(
                expected_dtype=dtypes.float16,
                expected_gradient=expected_gradient))
        y = core.Lambda(identity_with_grad_check_fn)(y)
        y = math_ops.cast(y, dtypes.float32)
        model = models.Model(inputs=x, outputs=y)

        def loss_fn(y_true, y_pred):
          del y_true
          return math_ops.reduce_mean(y_pred)

        model.compile(
            opt,
            loss=loss_fn,
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function())

    self.assertEqual(backend.eval(layer.v), 1)
    x = np.ones((batch_size, 1))
    y = np.ones((batch_size, 1))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    model.fit(dataset)
    # The variables starts with 1 and has a gradient of 1, so will go down by 1
    # each step.
    self.assertEqual(backend.eval(layer.v), 0)

    model.fit(dataset)
    self.assertEqual(backend.eval(layer.v), -1)

    # There have been two steps without NaNs, so the loss scale will double
    backend.set_value(expected_gradient,
                      backend.get_value(expected_gradient * 2))
    model.fit(dataset)
    self.assertEqual(backend.eval(layer.v), -2)

    # Next test with NaN gradients.
    backend.set_value(have_nan_gradients, True)
    model.fit(dataset)
    # Variable should not be updated
    self.assertEqual(backend.eval(layer.v), -2)

    # Test with finite gradients again
    backend.set_value(have_nan_gradients, False)
    # The loss scale will be halved due to the NaNs, so the gradient will also
    # be halved
    backend.set_value(expected_gradient,
                      backend.get_value(expected_gradient / 2))
    model.fit(dataset)
    self.assertEqual(backend.eval(layer.v), -3)

  @test_util.run_in_graph_and_eager_modes
  @testing_utils.enable_v2_dtype_behavior
  def test_loss_scale_optimizer_overrides_policy_loss_scale(self):
    with policy.policy_scope(policy.Policy('float32', loss_scale=10.)):
      opt = gradient_descent.SGD(1.)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale=5.)
      x = layers.Input(shape=(1,))
      y = AddLayer()(x)
      model = models.Model(x, y)
      model.compile(opt, loss='mse')
      self.assertEqual(self.evaluate(model.optimizer.loss_scale()), 5.)

  @test_util.run_in_graph_and_eager_modes
  @testing_utils.enable_v2_dtype_behavior
  def test_pass_invalid_optimizer_with_loss_scaling(self):
    with policy.policy_scope(policy.Policy('float32', loss_scale=10.)):
      x = layers.Input(shape=(1,))
      y = AddLayer()(x)
      model = models.Model(x, y)
      with self.assertRaisesRegexp(ValueError,
                                   'optimizer" must be an instance of '):
        model.compile(optimizers.SGD(1.), 'mse')

  @test_util.run_in_graph_and_eager_modes
  @testing_utils.enable_v2_dtype_behavior
  def test_functional_model_loss_dtype(self):
    with policy.policy_scope('float16'):
      x = layers.Input(shape=(1,))
      y = AddLayer()(x)
      model = models.Model(x, y)
      model.add_loss(math_ops.cast(y, 'float32'))
      # The loss should not be casted to the policy's dtype.
      self.assertEqual(model.losses[0].dtype, 'float32')

  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
          'strategy_fn': default_strategy_fn,
      }, {
          'testcase_name': 'distribute',
          'strategy_fn': create_mirrored_strategy,
      }, {
          'testcase_name': 'base_h5',
          'strategy_fn': default_strategy_fn,
          'h5': True,
      }, {
          'testcase_name': 'distribute_h5',
          'strategy_fn': create_mirrored_strategy,
          'h5': True,
      })
  @test_util.run_in_graph_and_eager_modes
  def test_save_weights_with_autocast_vars(self, strategy_fn, h5=False):
    with strategy_fn().scope():
      with policy.policy_scope('infer_float32_vars'):
        x = layers.Input(shape=(1,), batch_size=2, dtype=dtypes.float16)
        layer = AddLayer(assert_type=dtypes.float16)
        y = layer(x)
        y = math_ops.cast(y, dtypes.float32)
        model = models.Model(inputs=x, outputs=y)

    model.set_weights([np.array(100.)])
    x = np.ones((2, 1), dtype=np.float16)
    self.assertAllClose(backend.get_value(model(x)), x + 100.)
    suffix = '.h5' if h5 else ''
    weights_file = os.path.join(self.get_temp_dir(), 'weights' + suffix)
    model.save_weights(weights_file)

    model.set_weights([np.array(200.)])
    self.assertAllClose(backend.get_value(model(x)), x + 200.)
    model.load_weights(weights_file)
    self.assertAllClose(backend.get_value(model(x)), x + 100.)
    self.assertEqual(model.get_weights(), [np.array(100.)])

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      {
          'testcase_name': 'base',
          'strategy_fn': default_strategy_fn,
      }, {
          'testcase_name': 'distribute',
          'strategy_fn': create_mirrored_strategy,
      }, {
          'testcase_name': 'different_var_name',
          'strategy_fn': default_strategy_fn,
          'var_name': 'w'
      }, {
          'testcase_name': 'different_var_name_distribute',
          'strategy_fn': create_mirrored_strategy,
          'var_name': 'w'
      })
  def test_save_slot_variables_with_autocast_vars(self,
                                                  strategy_fn,
                                                  var_name='v'):
    if not self._is_strategy_supported(strategy_fn):
      return
    with strategy_fn().scope(), policy.policy_scope('infer_float32_vars'):
      x = layers.Input(shape=(2,), batch_size=2, dtype=dtypes.float16)
      # Having a var_name other than 'v' tests that a fixed bug (b/134713714)
      # does not reoccur. The bug was that a crash would occur when saving a
      # checkpoint where an AutoCastVariable with a slot variable would have a
      # different name than the layer attribute's name (layer.v in this case).
      layer = AddLayer(assert_type=dtypes.float16, var_name=var_name)
      y = layer(x)
      y = math_ops.cast(y, dtypes.float32)
      model = models.Model(inputs=x, outputs=y)
      opt = gradient_descent.SGD(1., 1.)
      model.compile(
          optimizer=opt,
          loss='mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())

    model.fit(np.zeros((2, 2)), np.zeros((2, 2)), batch_size=2)
    weights_file = os.path.join(self.get_temp_dir(), 'weights')
    model.save_weights(weights_file)
    saved_slot = backend.get_value(opt.get_slot(layer.v, 'momentum'))

    model.fit(np.zeros((2, 2)), np.zeros((2, 2)), batch_size=2)
    new_slot = backend.get_value(opt.get_slot(layer.v, 'momentum'))
    self.assertNotEqual(new_slot, saved_slot)

    model.load_weights(weights_file)
    restored_slot = backend.get_value(opt.get_slot(layer.v, 'momentum'))
    self.assertEqual(restored_slot, saved_slot)

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(*TESTCASES)
  def test_save_weights_with_dynamic_loss_scaling(self, strategy_fn):
    if not self._is_strategy_supported(strategy_fn):
      return
    strategy = strategy_fn()
    if (isinstance(strategy, mirrored_strategy.MirroredStrategy) and
        not context.executing_eagerly()):
      # TODO(b/121381184): Enable running the test in this case.
      return

    # Create and run model.
    with strategy.scope():
      x = layers.Input(shape=(2,), batch_size=2, dtype=dtypes.float32)
      y = AddLayer(assert_type=dtypes.float32)(x)
      model = models.Model(inputs=x, outputs=y)

      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=1., increment_period=2., multiplier=2.)
      opt = gradient_descent.SGD(1.)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
      model.compile(
          optimizer=opt,
          loss='mse',
          run_eagerly=testing_utils.should_run_eagerly(),
          experimental_run_tf_function=testing_utils.should_run_tf_function())
    # Run for 3 steps (6 examples with a batch size of 2)
    model.fit(np.zeros((6, 2)), np.zeros((6, 2)), batch_size=2)
    self.assertEqual(backend.get_value(loss_scale()), 2)
    self.assertEqual(backend.get_value(loss_scale._num_good_steps), 1)

    # Save model weights.
    save_prefix = os.path.join(self.get_temp_dir(), 'ckpt')
    model.save_weights(save_prefix)

    # Run model again for 1 step (2 examples with a batch size of 2)
    model.fit(np.zeros((2, 2)), np.zeros((2, 2)), batch_size=2)
    self.assertEqual(backend.get_value(loss_scale()), 4)
    self.assertEqual(backend.get_value(loss_scale._num_good_steps), 0)

    # Load model weights and ensure loss scale weights are restored.
    model.load_weights(save_prefix)
    self.assertEqual(backend.get_value(loss_scale()), 2)
    self.assertEqual(backend.get_value(loss_scale._num_good_steps), 1)


class RnnTest(keras_parameterized.TestCase):
  """Test mixed precision with RNNs."""

  # TODO(b/136512020): Support and test recurrent_v2.GRU.
  @parameterized.named_parameters(
      {
          'testcase_name': 'base_simple',
          'strategy_fn': default_strategy_fn,
          'rnn_class': recurrent.SimpleRNN,
      }, {
          'testcase_name': 'distribute_simple',
          'strategy_fn': create_mirrored_strategy,
          'rnn_class': recurrent.SimpleRNN,
      }, {
          'testcase_name': 'base_gru',
          'strategy_fn': default_strategy_fn,
          'rnn_class': recurrent.GRU,
      }, {
          'testcase_name': 'distribute_gru',
          'strategy_fn': create_mirrored_strategy,
          'rnn_class': recurrent.GRU,
      })
  @test_util.run_in_graph_and_eager_modes
  # RNNs do not work properly with GradientTape in graph mode when V1 control
  # flow is used.
  @test_util.enable_control_flow_v2
  def test_rnn(self, strategy_fn, rnn_class):
    x = array_ops.ones((2, 3, 4), dtype=dtypes.float16)
    strategy = strategy_fn()
    with strategy.scope(), policy.policy_scope('infer_float32_vars'):
      layer = rnn_class(units=4)

      def run_fn():
        with backprop.GradientTape() as tape:
          y = layer(x)
          self.assertEqual(y.dtype, dtypes.float16)
        opt = gradient_descent.SGD(1.)
        grads = tape.gradient(y, layer.trainable_weights)
        return opt.apply_gradients(zip(grads, layer.trainable_weights))

      op = strategy.experimental_run(run_fn)
      if not context.executing_eagerly():
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(op)

      for v in layer.weights:
        self.assertEqual(v.dtype, dtypes.float32)


if __name__ == '__main__':
  test.main()
