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
"""Tests for LossScaleOptimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision.experimental import test_util as mp_test_util
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.tracking import util as trackable_utils


# If called outside any strategy.scope() calls, this will return the default
# strategy.
default_strategy_fn = distribution_strategy_context.get_strategy


def create_mirrored_strategy():
  if context.num_gpus() >= 1:
    return mirrored_strategy.MirroredStrategy(['cpu:0', 'gpu:0'])
  else:
    return mirrored_strategy.MirroredStrategy(['cpu:0'])


TESTCASES = ({
    'testcase_name': 'Base',
    'strategy_fn': default_strategy_fn
}, {
    'testcase_name': 'Distribute',
    'strategy_fn': create_mirrored_strategy
})


@test_util.with_control_flow_v2
class LossScaleOptimizerTest(test.TestCase, parameterized.TestCase):

  def _run_if_in_graph_mode(self, val):
    # Running only in graph mode is useful, because optimizers sometimes return
    # a value that, in Graph mode, is runnable with self.evaluate. But in Eager
    # mode, the optimizer already does the computations and the return value
    # cannot be run.
    if not context.executing_eagerly():
      self.evaluate(val)

  def _run_fn_with_grad_check(self, strategy, var, opt, expected_grad):
    grad_check_fn = mp_test_util.create_identity_with_grad_check_fn(
        expected_grad)
    loss = lambda: grad_check_fn(var) / strategy.num_replicas_in_sync
    return lambda: opt.minimize(loss, var_list=[var])

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testFixedLossScaleAppliedToLossWithMinimize(self, strategy_fn):
    with strategy_fn().scope() as strategy:
      var = variables.Variable([5.0])
      opt = gradient_descent.SGD(2.0)
      loss_scale = 10.
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
      # We need num_replicas_in_sync to divide loss_scale, otherwise loss_scale
      # / strategy.num_replicas_in_sync will not be exact, which could lead to
      # assertion failures due to rounding issues.
      self.assertEqual(loss_scale % strategy.num_replicas_in_sync, 0)
      run_fn = self._run_fn_with_grad_check(
          strategy, var, opt, loss_scale / strategy.num_replicas_in_sync)
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # The loss is the identity of the variable. Therefore the gradient is 1,
      # and so the variable will be init_val - grad * lr == 5 - 1 * 2 == 3
      self.assertAllClose([3.], self.evaluate(var))

  @test_util.deprecated_graph_mode_only
  def testFixedLossScaleAppliedToLossWithGetGradients(self):
    var = variables.Variable([2.0])
    opt = gradient_descent.SGD(1.0)
    loss_scale = 10.
    opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
    grad_check_fn = mp_test_util.create_identity_with_grad_check_fn(loss_scale)
    loss = grad_check_fn(var)
    run_op = opt.get_gradients(loss, [var])
    self.evaluate(variables.global_variables_initializer())
    # This will cause an assertion to run, as
    # mp_test_util.create_identity_with_grad_check_fn added an assertion op.
    self.evaluate(run_op)

  @test_util.run_in_graph_and_eager_modes
  def testGetScaledLoss(self):
    opt = gradient_descent.SGD(2.0)
    opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale=2.)
    loss = ops.convert_to_tensor(5.)
    self.assertEqual(10., self.evaluate(opt.get_scaled_loss(loss)))
    self.assertEqual(10., self.evaluate(opt.get_scaled_loss(lambda: loss)()))
    loss = ops.convert_to_tensor(5., dtype='float16')
    self.assertEqual(10., self.evaluate(opt.get_scaled_loss(loss)))
    self.assertEqual(10., self.evaluate(opt.get_scaled_loss(lambda: loss)()))

  @test_util.run_in_graph_and_eager_modes
  def testGetUnscaledGradients(self):
    opt = gradient_descent.SGD(2.0)
    opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale=2)
    scaled_grads = [
        ops.convert_to_tensor(3.),
        None,
        ops.convert_to_tensor(-4., dtype='float16')
    ]
    grads = opt.get_unscaled_gradients(scaled_grads)
    grads = [self.evaluate(g) if g is not None else g for g in grads]
    self.assertEqual([1.5, None, -2.], grads)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testDynamicLossScale(self, strategy_fn):
    strategy = strategy_fn()
    learning_rate = 2.
    expected_gradient = resource_variable_ops.ResourceVariable(
        learning_rate / strategy.num_replicas_in_sync)
    with strategy.scope():
      var = variables.Variable([5.0])
      opt = gradient_descent.SGD(learning_rate)
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=2, increment_period=1, multiplier=2)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
      self.assertEqual(
          loss_scale.initial_loss_scale % strategy.num_replicas_in_sync, 0)

      run_fn = self._run_fn_with_grad_check(strategy, var, opt,
                                            expected_gradient)
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # The loss is the identity of the variable. Therefore the gradient is 1,
      # and so the variable will be init_val - grad * lr == 5 - 1 * 2 == 3
      self.assertAllClose([3.], self.evaluate(var))

      # Loss scale will be double, so the expected gradient is also doubled.
      self.evaluate(expected_gradient.assign(
          2 * learning_rate / strategy.num_replicas_in_sync))
      run_op = strategy.experimental_run(run_fn)
      self._run_if_in_graph_mode(run_op)
      # As before, the 2 is subtracted from the variable, making it's new value
      # 1.
      self.assertAllClose([1.], self.evaluate(var))

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testDynamicUpdate(self, strategy_fn):
    with strategy_fn().scope() as strategy:
      var = variables.Variable([1.0, 2.0])
      opt = gradient_descent.SGD(1.0)
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=2, increment_period=1, multiplier=2)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)

      # Test optimizer with finite gradients
      loss = lambda: var * 2.0 / strategy.num_replicas_in_sync
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # Gradient is 2, so variable will have 2 subtracted from it
      self.assertAllClose([-1.0, 0.0], self.evaluate(var))
      # Loss scale has doubled from 2 to 4
      self.assertEqual(4., self.evaluate(opt.loss_scale()))

      # Test optimizer with NaN gradients
      loss = lambda: var * float('NaN')
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self._run_if_in_graph_mode(run_op)
      # Variable should not change from before, due to NaN gradients.
      self.assertAllClose(self.evaluate(var), [-1.0, 0.0])
      # Loss scale should half due to NaN gradients.
      self.assertEqual(2., self.evaluate(opt.loss_scale()))

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testDynamicLossScaleWithFloat16Loss(self, strategy_fn):
    strategy = strategy_fn()
    learning_rate = 2.
    with strategy.scope():
      var = variables.Variable([5.0])
      opt = gradient_descent.SGD(learning_rate)
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=2, increment_period=1, multiplier=2)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)

      def loss():
        return math_ops.cast(var / strategy.num_replicas_in_sync, 'float16')
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # The loss is the identity of the variable. Therefore the gradient is 1,
      # and so the variable will be init_val - grad * lr == 5 - 1 * 2 == 3
      self.assertAllClose([3.], self.evaluate(var))

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testDynamicLossScaleWithSlots(self, strategy_fn):
    strategy_obj = strategy_fn()
    if (isinstance(strategy_obj, mirrored_strategy.MirroredStrategy) and
        control_flow_v2_toggles.control_flow_v2_enabled() and
        not context.executing_eagerly()):
      self.skipTest('b/138667997')
    with strategy_obj.scope() as strategy:
      var = variables.Variable([1.0, 2.0])
      # An SGD optimizer with momentum has slot variables.
      opt = gradient_descent.SGD(1.0, momentum=1.)
      initial_loss_scale = 2.
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=initial_loss_scale, increment_period=1,
          multiplier=4)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
      loss = lambda: var / strategy.num_replicas_in_sync
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # The momentum accumulator starts at 0 and the gradient is 1. The
      # accumulator is incremented by the gradient, so it is now 1. Then the
      # variable is subtracted by the accumulator, so the variable is subtracted
      # by 1.
      self.assertAllClose([0.0, 1.0], self.evaluate(var))
      self.assertEqual(self.evaluate(opt.loss_scale()), initial_loss_scale * 4)

      run_op = strategy.experimental_run(run_fn)
      self._run_if_in_graph_mode(run_op)
      # The momentum accumulator was 1 before this step and the gradient is 1.
      # The accumulator is incremented by the gradient, so it is now 2. Then the
      # variable is subtracted by the accumulator, so the variable is subtracted
      # by 2.
      self.assertAllClose([-2., -1.], self.evaluate(var))
      self.assertEqual(self.evaluate(opt.loss_scale()),
                       initial_loss_scale * 16)

  @test_util.run_in_graph_and_eager_modes
  def testIterations(self):
    opt = gradient_descent.SGD(2.0)
    lso = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale=10.)
    lso.iterations = 7
    self.assertEqual(lso.iterations, 7)
    self.assertEqual(opt.iterations, 7)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testGettingAndSettingLearningRate(self, strategy_fn):
    with strategy_fn().scope() as strategy:
      var = variables.Variable([5.0])
      opt = adam.Adam(learning_rate=1.0)
      loss = lambda: var * 2.0
      run_fn = lambda: opt.minimize(loss, [var])
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)

      lr = self.evaluate(opt.lr)
      self.assertEqual(1.0, lr)

      opt.lr = 2.0
      lr = self.evaluate(opt.lr)
      self.assertEqual(2.0, lr)

      self.evaluate(opt.lr.assign(3.0))
      lr = self.evaluate(opt.lr)
      self.assertEqual(3.0, lr)

      with self.assertRaises(AttributeError):
        opt.not_an_attr += 3

  @test_util.run_in_graph_and_eager_modes
  def testArbitraryAttributesNotExposed(self):
    opt = adam.Adam(learning_rate=1.0)
    # Test that Adam has attributes 'epsilon' and 'beta1'
    opt.epsilon  # pylint: disable=pointless-statement
    opt.beta_1  # pylint: disable=pointless-statement
    opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale=10.)
    # Test that attributes defined by OptimizerV2 subclasses are not exposed in
    # LossScaleOptimizer, and that the error message is sensible.
    with self.assertRaisesRegexp(
        AttributeError,
        "'LossScaleOptimizer' object has no attribute 'epsilon'"):
      opt.epsilon  # pylint: disable=pointless-statement
    with self.assertRaisesRegexp(
        AttributeError,
        "'LossScaleOptimizer' object has no attribute 'beta_1'"):
      opt.beta_1  # pylint: disable=pointless-statement

  @test_util.run_in_graph_and_eager_modes
  def testApplyGradientsGetsUnwrappedTensors(self):
    # Tests that gradients passed to apply_gradients are not wrapped in a
    # DistributionStrategy wrapper, such as PerReplica, but instead are raw
    # Tensors. Optimizer subclasses that override apply_gradients() expect raw
    # Tensors, even though the base Optimizer can handle PerReplica gradients.

    outer_self = self

    class MyOptimizer(gradient_descent.SGD):

      def apply_gradients(self, grads_and_vars, name=None):
        for grad, _ in grads_and_vars:
          outer_self.assertIsInstance(grad, ops.Tensor)
        return super(MyOptimizer, self).apply_gradients(grads_and_vars, name)

    with create_mirrored_strategy().scope() as strategy:
      var = variables.Variable([5.0])
      opt = MyOptimizer(learning_rate=1.0)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale=1)
      loss = lambda: var * 2.0
      run_fn = lambda: opt.minimize(loss, [var])
      strategy.experimental_run(run_fn)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testCheckpoint(self, strategy_fn):
    strategy = strategy_fn()
    if (isinstance(strategy, mirrored_strategy.MirroredStrategy) and
        not context.executing_eagerly()):
      # TODO(b/121381184): Enable running the test in this case.
      return

    with self.test_session(), strategy.scope():
      # Build and run a simple model.
      var = variables.Variable([2.0])
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=1., increment_period=2.,
          multiplier=2.)
      opt = gradient_descent.SGD(1., momentum=1.)
      opt = loss_scale_optimizer.LossScaleOptimizer(opt, loss_scale)
      run_fn = lambda: opt.minimize(lambda: var + 1., var_list=[var])
      opt_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(opt_op)
      self.assertEqual(self.evaluate(loss_scale()), 1.)
      self.assertEqual(self.evaluate(loss_scale._num_good_steps), 1)
      slot_var = opt._optimizer.get_slot(var, 'momentum')
      slot_value = self.evaluate(slot_var).item()

      # Save a checkpoint.
      checkpoint = trackable_utils.Checkpoint(optimizer=opt, var=var)
      prefix = os.path.join(self.get_temp_dir(), 'ckpt')
      save_path = checkpoint.save(prefix)

      # Run model again.
      self.evaluate(strategy.experimental_run(run_fn))
      self.assertEqual(self.evaluate(loss_scale()), 2.)
      self.assertEqual(self.evaluate(loss_scale._num_good_steps), 0)
      self.assertNotAlmostEqual(self.evaluate(slot_var).item(), slot_value)

      # Load checkpoint and ensure loss scale is back to it's original value.
      status = checkpoint.restore(save_path)
      status.assert_consumed()
      status.run_restore_ops()
      self.assertEqual(self.evaluate(loss_scale()), 1.)
      self.assertEqual(self.evaluate(loss_scale._num_good_steps), 1)
      self.assertAlmostEqual(self.evaluate(slot_var).item(), slot_value)


if __name__ == '__main__':
  test.main()
