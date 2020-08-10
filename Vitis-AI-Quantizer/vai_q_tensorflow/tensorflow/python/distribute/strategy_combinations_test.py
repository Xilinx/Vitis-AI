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
"""Tests for a little bit of strategy_combinations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test


class StrategyCombinationsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    # Need to call set_virtual_cpus_to_at_least() in setUp with the maximum
    # value needed in any test.
    strategy_combinations.set_virtual_cpus_to_at_least(3)
    super(StrategyCombinationsTest, self).setUp()

  def test3VirtualCPUs(self):
    cpu_device = config.list_physical_devices("CPU")[0]
    self.assertLen(config.get_virtual_device_configuration(cpu_device), 3)

  def testSetVirtualCPUsAgain(self):
    strategy_combinations.set_virtual_cpus_to_at_least(2)
    cpu_device = config.list_physical_devices("CPU")[0]
    self.assertLen(config.get_virtual_device_configuration(cpu_device), 3)

  def testSetVirtualCPUsErrors(self):
    with self.assertRaises(ValueError):
      strategy_combinations.set_virtual_cpus_to_at_least(0)
    with self.assertRaisesRegexp(RuntimeError, "with 3 < 5 virtual CPUs"):
      strategy_combinations.set_virtual_cpus_to_at_least(5)

  @combinations.generate(combinations.combine(
      distribution=[strategy_combinations.mirrored_strategy_with_cpu_1_and_2],
      mode=["graph", "eager"]))
  def testMirrored2CPUs(self, distribution):
    with distribution.scope():
      one_per_replica = distribution.experimental_run_v2(
          lambda: constant_op.constant(1))
      num_replicas = distribution.reduce(
          reduce_util.ReduceOp.SUM, one_per_replica, axis=None)
      self.assertEqual(2, self.evaluate(num_replicas))


if __name__ == "__main__":
  test.main()
