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
"""Test MirroredVariable in MirroredStrategy and MultiWorkerMirroredStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.layers import core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables


def _replica_id():
  replica_id = ds_context.get_replica_context().replica_id_in_sync_group
  if not isinstance(replica_id, ops.Tensor):
    replica_id = constant_op.constant(replica_id)
  return replica_id


def _mimic_two_cpus():
  cpus = config.list_physical_devices("CPU")

  config.set_virtual_device_configuration(cpus[0], [
      context.VirtualDeviceConfiguration(),
      context.VirtualDeviceConfiguration(),
  ])


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
            combinations.NamedDistribution(
                "Collective2CPUs",
                # pylint: disable=g-long-lambda
                lambda: collective_all_reduce_strategy.
                CollectiveAllReduceStrategy._from_local_devices((
                    "/device:CPU:0", "/device:CPU:1")),
                required_gpus=0)
        ],
        mode=["graph", "eager"]))
class MirroredVariableCreationTest(test.TestCase):
  """Base class that tests mirrored variable creator.

  Currently it assumes all strategy objects have two replicas.
  """

  @classmethod
  def setUpClass(cls):
    _mimic_two_cpus()

  # TODO(priyag): Modify more tests to use this helper and check more
  # properties.
  def _test_mv_properties(self, var, name, strategy):
    self.assertIsInstance(var, values.MirroredVariable)
    self.assertEqual(name, var.name)
    self.assertIs(strategy, var.distribute_strategy)
    for d in var.devices:
      self.assertEqual(d, var.get(d).device)
      self.assertIs(strategy, var.get(d)._distribute_strategy)  # pylint: disable=protected-access

  def testVariableInFuncGraph(self, distribution):

    def model_fn():
      v = variable_scope.variable(2.0, name="bar")
      ds_context.get_replica_context().merge_call(lambda _: _)
      return v

    with func_graph.FuncGraph("fg").as_default(), distribution.scope():
      v1 = variable_scope.variable(1.0, name="foo")
      v2 = distribution.extended.call_for_each_replica(model_fn)

    self._test_mv_properties(v1, "foo:0", distribution)
    self._test_mv_properties(v2, "bar:0", distribution)

  def testVariableWithTensorInitialValueInFunction(self, distribution):
    if not context.executing_eagerly():
      self.skipTest("`tf.function` is an eager-only feature")

    v = [None]

    def model_fn():
      if v[0] is None:
        init_val = array_ops.zeros([])
        v[0] = variables.Variable(init_val)
      ds_context.get_replica_context().merge_call(lambda _: _)
      return v[0]

    @def_function.function(autograph=False)
    def make_v1():
      return distribution.experimental_local_results(
          distribution.extended.call_for_each_replica(model_fn))

    self.assertAllEqual([0, 0], make_v1())

  def testSingleVariable(self, distribution):

    def model_fn():
      # This variable should be created only once across the threads because of
      # special variable_creator functions used by
      # `distribution.extended.call_for_each_replica`.
      v = variable_scope.variable(1.0, name="foo")
      ds_context.get_replica_context().merge_call(lambda _: _)
      return v

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      self._test_mv_properties(result, "foo:0", distribution)

  def testUnnamedVariable(self, distribution):

    def model_fn():
      v = variable_scope.variable(1.0)
      ds_context.get_replica_context().merge_call(lambda _: _)
      return v

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      self._test_mv_properties(result, "Variable:0", distribution)

  def testMultipleVariables(self, distribution):

    def model_fn():
      vs = []
      for i in range(5):
        vs.append(variable_scope.variable(1.0, name="foo" + str(i)))
      ds_context.get_replica_context().merge_call(lambda _: _)
      return vs

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      for i, v in enumerate(result):
        self._test_mv_properties(v, "foo" + str(i) + ":0", distribution)

  def testMultipleVariablesWithSameCanonicalName(self, distribution):

    def model_fn():
      vs = []
      vs.append(variable_scope.variable(1.0, name="foo/bar"))
      vs.append(variable_scope.variable(1.0, name="foo_1/bar"))
      vs.append(variable_scope.variable(1.0, name="foo_1/bar_1"))
      vs.append(variable_scope.variable(1.0, name="foo/bar_1"))
      ds_context.get_replica_context().merge_call(lambda _: _)
      return vs

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      for v in result:
        self.assertIsInstance(v, values.MirroredVariable)
      self.assertEqual(4, len(result))
      self.assertEqual("foo/bar:0", result[0].name)
      self.assertEqual("foo_1/bar:0", result[1].name)
      self.assertEqual("foo_1/bar_1:0", result[2].name)
      self.assertEqual("foo/bar_1:0", result[3].name)

  def testVariableWithSameCanonicalNameAcrossThreads(self, distribution):

    def model_fn():
      replica_id = self.evaluate(_replica_id())
      v = variable_scope.variable(1.0, name="foo_" + str(replica_id))
      ds_context.get_replica_context().merge_call(lambda _: _)
      return v

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      self.assertIsInstance(result, values.MirroredVariable)
      # The resulting mirrored variable will use the name from the first device.
      self.assertEqual("foo_0:0", result.name)

  def testWithLayers(self, distribution):

    def model_fn(features):
      with variable_scope.variable_scope("common"):
        layer1 = core.Dense(1)
        layer1(features)
        layer2 = core.Dense(1)
        layer2(features)
        # This will pause the current thread, and execute the other thread.
        ds_context.get_replica_context().merge_call(lambda _: _)
        layer3 = core.Dense(1)
        layer3(features)
        return [(layer1.kernel, layer1.bias), (layer2.kernel, layer2.bias),
                (layer3.kernel, layer3.bias)]

    iterator = distribution.make_input_fn_iterator(
        lambda _: dataset_ops.Dataset.from_tensors([[1.]]).repeat(10))
    self.evaluate(iterator.initialize())
    features = iterator.get_next()

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(
          model_fn, args=(features,))
      suffixes = ["", "_1", "_2"]
      for (kernel, bias), suffix in zip(result, suffixes):
        self.assertIsInstance(kernel, values.MirroredVariable)
        self.assertEqual("common/dense" + suffix + "/kernel:0", kernel.name)
        self.assertIsInstance(bias, values.MirroredVariable)
        self.assertEqual("common/dense" + suffix + "/bias:0", bias.name)

  def testWithVariableAndVariableScope(self, distribution):

    def model_fn():
      v0 = variable_scope.variable(1.0, name="var0", aggregation=None)
      with variable_scope.variable_scope("common"):
        v1 = variable_scope.variable(1.0, name="var1")
        # This will pause the current thread, and execute the other thread.
        ds_context.get_replica_context().merge_call(lambda _: _)
        v2 = variable_scope.variable(
            1.0,
            name="var2",
            synchronization=variable_scope.VariableSynchronization.ON_READ,
            aggregation=variable_scope.VariableAggregation.SUM)
        v3 = variable_scope.variable(
            1.0,
            name="var3",
            synchronization=variable_scope.VariableSynchronization.ON_WRITE,
            aggregation=variable_scope.VariableAggregation.MEAN)

      return v0, v1, v2, v3

    with distribution.scope():
      v = variable_scope.variable(1.0, name="var-main0")
      self.assertEqual("var-main0:0", v.name)

      result = distribution.extended.call_for_each_replica(model_fn)
      self.assertEqual(4, len(result))
      v0, v1, v2, v3 = result
      self.assertIsInstance(v0, values.MirroredVariable)
      self.assertEqual("var0:0", v0.name)
      self.assertIsInstance(v1, values.MirroredVariable)
      self.assertEqual("common/var1:0", v1.name)
      self.assertIsInstance(v2, values.SyncOnReadVariable)
      self.assertEqual("common/var2:0", v2.name)
      self.assertEqual(variable_scope.VariableAggregation.SUM, v2.aggregation)
      self.assertIsInstance(v3, values.MirroredVariable)
      self.assertEqual("common/var3:0", v3.name)
      self.assertEqual(variable_scope.VariableAggregation.MEAN, v3.aggregation)

  def testWithGetVariableAndVariableScope(self, distribution):

    def model_fn():
      v0 = variable_scope.get_variable("var0", [1])
      with variable_scope.variable_scope("common"):
        v1 = variable_scope.get_variable("var1", [1])
        # This will pause the current thread, and execute the other thread.
        ds_context.get_replica_context().merge_call(lambda _: _)
        v2 = variable_scope.get_variable(
            "var2", [1],
            synchronization=variable_scope.VariableSynchronization.ON_READ,
            aggregation=variable_scope.VariableAggregation.SUM)
        v3 = variable_scope.get_variable(
            "var3", [1],
            synchronization=variable_scope.VariableSynchronization.ON_WRITE,
            aggregation=variable_scope.VariableAggregation.MEAN)

      return v0, v1, v2, v3

    with distribution.scope():
      with variable_scope.variable_scope("main"):
        v = variable_scope.get_variable("var-main0", [1])
        self.assertEqual("main/var-main0:0", v.name)

        result = distribution.extended.call_for_each_replica(model_fn)
        self.assertEqual(4, len(result))
        v0, v1, v2, v3 = result
        self.assertIsInstance(v0, values.MirroredVariable)
        self.assertEqual("main/var0:0", v0.name)
        self.assertIsInstance(v1, values.MirroredVariable)
        self.assertEqual("main/common/var1:0", v1.name)
        self.assertIsInstance(v2, values.SyncOnReadVariable)
        self.assertEqual("main/common/var2:0", v2.name)
        self.assertEqual(variable_scope.VariableAggregation.SUM, v2.aggregation)
        self.assertIsInstance(v3, values.MirroredVariable)
        self.assertEqual("main/common/var3:0", v3.name)
        self.assertEqual(variable_scope.VariableAggregation.MEAN,
                         v3.aggregation)

  def testOnlyFirstReplicaUpdatesVariables(self, distribution):

    def create_fn():
      aggregation = variable_scope.VariableAggregation.ONLY_FIRST_REPLICA
      v0 = variable_scope.variable(
          2.0,
          name="on_read",
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=aggregation)
      v1 = variable_scope.variable(
          3.0,
          name="on_write",
          synchronization=variable_scope.VariableSynchronization.ON_WRITE,
          aggregation=aggregation)
      return v0, v1

    devices = distribution.extended.worker_devices
    with distribution.scope():
      v0, v1 = distribution.extended.call_for_each_replica(create_fn)
      self.evaluate(v0.initializer)
      self.assertEqual(2.0, self.evaluate(v0.get(devices[0])))
      self.assertEqual(2.0, self.evaluate(v0.get(devices[1])))
      self.assertEqual(2.0, self.evaluate(distribution.extended.read_var(v0)))
      self.evaluate(v1.initializer)
      self.assertEqual(3.0, self.evaluate(v1.get(devices[0])))
      self.assertEqual(3.0, self.evaluate(v1.get(devices[1])))
      self.assertEqual(3.0, self.evaluate(distribution.extended.read_var(v1)))

      def replica_id_plus_one():
        return math_ops.cast(_replica_id() + 1, dtype=dtypes.float32)

      # Update using the assign_add member function.
      def update_member_fn():
        update0 = v0.assign_add(5.0 * replica_id_plus_one())
        update1 = v1.assign_add(7.0 * replica_id_plus_one())
        return update0, update1

      update0a, update1a = distribution.extended.call_for_each_replica(
          update_member_fn)

      # Update "sync on read" variable.
      self.evaluate(distribution.group(update0a))
      self.assertEqual(2.0 + 5.0, self.evaluate(v0.get(devices[0])))
      # Writes are not synchronized for "sync on read" variables,
      # so device[1] can end up with a different value.
      self.assertEqual(2.0 + 2 * 5.0, self.evaluate(v0.get(devices[1])))
      # Always reads from device 0.
      self.assertEqual(2.0 + 5.0,
                       self.evaluate(distribution.extended.read_var(v0)))

      # Update "sync on write" variable.
      self.evaluate(distribution.group(update1a))
      self.assertEqual(3.0 + 7.0, self.evaluate(v1.get(devices[0])))
      # Writes are synchronized for v1, only the argument to assign_add on
      # device[0] is used.
      self.assertEqual(3.0 + 7.0, self.evaluate(v1.get(devices[1])))
      self.assertEqual(3.0 + 7.0,
                       self.evaluate(distribution.extended.read_var(v1)))

      # Update using state_ops.assign_add global function.
      def update_state_ops_fn():
        update0 = state_ops.assign_add(v0, 11.0 * replica_id_plus_one())
        update1 = state_ops.assign_add(v1, 13.0 * replica_id_plus_one())
        return update0, update1

      update0b, update1b = distribution.extended.call_for_each_replica(
          update_state_ops_fn)
      self.evaluate(distribution.group(update0b))

      # Update "sync on read" variable.
      self.assertEqual(2.0 + 5.0 + 11.0, self.evaluate(v0.get(devices[0])))
      self.assertEqual(2.0 + 2 * 5.0 + 2 * 11.0,
                       self.evaluate(v0.get(devices[1])))
      self.assertEqual(2.0 + 5.0 + 11.0,
                       self.evaluate(distribution.extended.read_var(v0)))

      # Update "sync on write" variable.
      self.evaluate(distribution.group(update1b))
      self.assertEqual(3.0 + 7.0 + 13.0, self.evaluate(v1.get(devices[0])))
      self.assertEqual(3.0 + 7.0 + 13.0, self.evaluate(v1.get(devices[1])))
      self.assertEqual(3.0 + 7.0 + 13.0,
                       self.evaluate(distribution.extended.read_var(v1)))

  def testNoneSynchronizationWithGetVariable(self, distribution):
    with distribution.scope():
      with self.assertRaisesRegexp(
          ValueError, "`NONE` variable synchronization mode is not "
          "supported with `Mirrored` distribution strategy. Please change "
          "the `synchronization` for variable: v"):
        variable_scope.get_variable(
            "v", [1],
            synchronization=variable_scope.VariableSynchronization.NONE)

  def testNoneSynchronizationWithVariable(self, distribution):
    with distribution.scope():
      with self.assertRaisesRegexp(
          ValueError, "`NONE` variable synchronization mode is not "
          "supported with `Mirrored` distribution strategy. Please change "
          "the `synchronization` for variable: v"):
        variable_scope.variable(
            1.0,
            name="v",
            synchronization=variable_scope.VariableSynchronization.NONE)

  def testInvalidSynchronizationWithVariable(self, distribution):
    with distribution.scope():
      with self.assertRaisesRegexp(
          ValueError, "Invalid variable synchronization mode: Invalid for "
          "variable: v"):
        variable_scope.variable(1.0, name="v", synchronization="Invalid")

  def testInvalidAggregationWithGetVariable(self, distribution):
    with distribution.scope():
      with self.assertRaisesRegexp(
          ValueError, "Invalid variable aggregation mode: invalid for "
          "variable: v"):
        variable_scope.get_variable(
            "v", [1],
            synchronization=variable_scope.VariableSynchronization.ON_WRITE,
            aggregation="invalid")

  def testInvalidAggregationWithVariable(self, distribution):
    with distribution.scope():
      with self.assertRaisesRegexp(
          ValueError, "Invalid variable aggregation mode: invalid for "
          "variable: v"):
        variable_scope.variable(
            1.0,
            name="v",
            synchronization=variable_scope.VariableSynchronization.ON_WRITE,
            aggregation="invalid")

  def testNonMatchingVariableCreation(self, distribution):
    self.skipTest("b/123075960")

    def model_fn(name):
      v = variable_scope.variable(1.0, name=name)
      ds_context.get_replica_context().merge_call(lambda _: _)
      return v

    with distribution.scope():
      device_map = values.ReplicaDeviceMap(distribution.extended.worker_devices)
      names = values.DistributedValues(device_map, ("foo", "bar"))
      with self.assertRaises(RuntimeError):
        _ = distribution.extended.call_for_each_replica(model_fn, args=(names,))

  def testSyncOnReadVariable(self, distribution):
    if context.executing_eagerly():
      self.skipTest("Skip the test due to b/137400477.")

    all_v_sum = {}
    all_v_mean = {}
    components_sum = {}
    components_mean = {}

    def model_fn():
      replica_id = self.evaluate(_replica_id())
      v_sum = variable_scope.variable(
          1.0,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.SUM)
      v_mean = variable_scope.variable(
          4.0,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.MEAN)
      self.assertIsInstance(v_sum, values.SyncOnReadVariable)
      self.assertIsInstance(v_mean, values.SyncOnReadVariable)
      updates = [
          v_sum.assign_add(2.0 + replica_id),
          v_mean.assign(6.0 * replica_id)
      ]
      all_v_sum[replica_id] = v_sum
      all_v_mean[replica_id] = v_mean
      c_sum = v_sum.get()
      c_mean = v_mean.get()
      components_sum[replica_id] = c_sum
      components_mean[replica_id] = c_mean
      self.assertIsNot(v_sum, c_sum)
      self.assertIsNot(v_mean, c_mean)
      return updates, v_sum, v_mean, c_sum, c_mean

    with distribution.scope():
      # Create "sum" and "mean" versions of SyncOnReadVariables.
      ret_ops, ret_v_sum, ret_v_mean, regrouped_sum, regrouped_mean = (
          distribution.extended.call_for_each_replica(model_fn))
      # Should see the same wrapping instance in all replicas.
      self.assertIs(all_v_sum[0], ret_v_sum)
      self.assertIs(all_v_mean[0], ret_v_mean)
      self.assertIs(all_v_sum[0], all_v_sum[1])
      self.assertIs(all_v_mean[0], all_v_mean[1])

      # Regroup should recover the same wrapper.
      self.assertIs(ret_v_sum, regrouped_sum)
      self.assertIs(ret_v_mean, regrouped_mean)
      self.assertIsNot(components_sum[0], components_sum[1])
      self.assertIsNot(components_mean[0], components_mean[1])

      # Apply updates
      self.evaluate(variables.global_variables_initializer())
      self.evaluate([
          y for x in ret_ops  # pylint: disable=g-complex-comprehension
          for y in distribution.experimental_local_results(x)
      ])
      expected_sum = 0.0
      expected_mean = 0.0
      for i, d in enumerate(distribution.extended.worker_devices):
        # Should see different values on different devices.
        v_sum_value = self.evaluate(ret_v_sum.get(d).read_value())
        v_mean_value = self.evaluate(ret_v_mean.get(d).read_value())
        expected = i + 3.0
        self.assertEqual(expected, v_sum_value)
        expected_sum += expected
        expected = i * 6.0
        self.assertEqual(expected, v_mean_value)
        expected_mean += expected
      expected_mean /= len(distribution.extended.worker_devices)

      # Without get(device), should return the value you get by
      # applying the reduction across all replicas (whether you use
      # read_var(), get(), or nothing).
      self.assertEqual(expected_sum, self.evaluate(
          distribution.extended.read_var(ret_v_sum)))
      self.assertEqual(expected_mean, self.evaluate(
          distribution.extended.read_var(ret_v_mean)))
      self.assertEqual(expected_sum, self.evaluate(ret_v_sum.get()))
      self.assertEqual(expected_mean, self.evaluate(ret_v_mean.get()))
      self.assertEqual(expected_sum, self.evaluate(ret_v_sum))
      self.assertEqual(expected_mean, self.evaluate(ret_v_mean))

  # TODO(priyag): Update this test to work in eager mode as well.
  def testDynamicRnnVariables(self, distribution):

    def model_fn():
      inputs = constant_op.constant(2 * [2 * [[0.0, 1.0, 2.0, 3.0, 4.0]]])
      cell_fw = rnn_cell_impl.LSTMCell(300)
      cell_bw = rnn_cell_impl.LSTMCell(300)
      (outputs, _) = rnn.bidirectional_dynamic_rnn(
          cell_fw, cell_bw, inputs, dtype=dtypes.float32)
      return outputs

    with context.graph_mode(), distribution.scope():
      result = distribution.extended.call_for_each_replica(model_fn)
      # Two variables are created by the RNN layer.
      self.assertEqual(2, len(result))
      for v in result:
        self.assertIsInstance(v, values.DistributedValues)
        _, v1 = distribution.experimental_local_results(v)
        self.assertStartsWith(v1._op.name, "replica_1/")

  def testSyncOnReadVariableUpdate(self, distribution):
    if context.executing_eagerly():
      self.skipTest("Skip the test due to b/137400477.")

    def model_fn():
      v_sum = variable_scope.variable(
          1.0,
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.SUM)
      self.assertIsInstance(v_sum, values.SyncOnReadVariable)
      return v_sum

    def update(var, value):
      return var.assign(value)

    with distribution.scope():
      ret_v_sum = distribution.extended.call_for_each_replica(model_fn)

      # Initialize variables.
      self.evaluate(variables.global_variables_initializer())
      # Assert that the aggregated value of the sync on read var is the sum
      # of the individual values before running the update ops.
      self.assertEqual(
          1.0,
          self.evaluate(
              ret_v_sum.get(
                  distribution.extended.worker_devices[0]).read_value()))
      self.assertEqual(2.0, self.evaluate(ret_v_sum))

      # Apply updates.
      update_ops = distribution.extended.update(
          ret_v_sum, update, args=(5.0,), group=False)
      self.evaluate(update_ops)
      # Assert that the aggregated value of the sync on read vars is the sum
      # of the individual values after running the update ops.
      self.assertEqual(
          5.0,
          self.evaluate(
              ret_v_sum.get(
                  distribution.extended.worker_devices[0]).read_value()))
      self.assertEqual(10.0, self.evaluate(ret_v_sum))

  def testVarDistributeStrategy(self, distribution):
    with distribution.scope():
      mirrored = variable_scope.variable(1.0)
      sync_on_read = variable_scope.variable(
          1.0, synchronization=variable_scope.VariableSynchronization.ON_READ)
      self.assertIs(distribution, mirrored.distribute_strategy)
      self.assertIs(distribution, sync_on_read.distribute_strategy)


if __name__ == "__main__":
  test.main()
