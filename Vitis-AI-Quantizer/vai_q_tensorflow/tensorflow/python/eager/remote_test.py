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
"""Tests for remote execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import server_lib


class SingleWorkerTest(test.TestCase):

  def setUp(self):
    super(SingleWorkerTest, self).setUp()

    workers, _ = test_util.create_local_cluster(1, 0)
    remote.connect_to_remote_host(workers[0].target)

  def testMultiDeviceFunctionBasic(self):

    @def_function.function
    def basic(i):
      with ops.device('/job:localhost/replica:0/task:0/cpu:0'):
        a = constant_op.constant([2]) + i
      with ops.device('/job:worker/replica:0/task:0/cpu:0'):
        b = constant_op.constant([1])

      return a + b

    self.assertAllEqual(basic(constant_op.constant([2])).numpy(), [5])
    self.assertAllEqual(basic(constant_op.constant([1])).numpy(), [4])

  def testMultiDeviceFunctionVariable(self):
    with ops.device('/job:worker/replica:0/task:0/cpu:0'):
      variable_b = variables.Variable(1)

    @def_function.function
    def with_variable(i):
      return i + variable_b

    self.assertAllEqual(with_variable(constant_op.constant([2])).numpy(), [3])

  def testMultiDeviceFunctionRemoteOutput(self):
    with ops.device('/job:worker/replica:0/task:0/cpu:0'):
      variable_b = variables.Variable(1)

    @def_function.function
    def remote_output(i):
      return variable_b, i + variable_b

    with self.assertRaises(errors.UnimplementedError) as cm:
      remote_output(constant_op.constant([1]))

    self.assertIn(
        'Currently, outputting tensors on remote devices is not supported.',
        cm.exception.message)

  def testMultiDeviceFunctionAmbiguousDevice(self):

    @def_function.function
    def ambiguous_device(i):
      with ops.device('cpu:0'):
        return i + constant_op.constant([2])

    with self.assertRaises(errors.InvalidArgumentError) as cm:
      with ops.device('/job:worker/replica:0/task:0/cpu:0'):
        ambiguous_device(constant_op.constant([2])).numpy()

    self.assertIn('the output node must match exactly one device',
                  cm.exception.message)

  def testStreaming(self):
    """A mini stress test for streaming - issuing many RPCs back to back."""
    with ops.device('job:worker/replica:0/task:0/device:CPU:0'):
      x = array_ops.ones([2, 2])
      y = array_ops.zeros([2, 2])
      num_iters = 200
      for _ in range(num_iters):
        y = x + y
        # Ask for y's shape after every 10 additions on average.
        # This exercises waiting for remote shape logic in TensorHandle.
        if random.randint(1, 10) == 1:
          _ = y.shape
    np.testing.assert_array_equal(
        [[num_iters, num_iters], [num_iters, num_iters]], y.numpy())


class MultiWorkersTest(test.TestCase):

  def setUp(self):
    super(MultiWorkersTest, self).setUp()

    workers, _ = test_util.create_local_cluster(3, 0)
    remote.connect_to_remote_host(
        [workers[0].target, workers[1].target, workers[2].target])

  def testMultiDeviceFunctionOnLocalDevice(self):
    with ops.device('/job:worker/replica:0/task:1'):
      variable_b = variables.Variable(1.0)

    @def_function.function
    def remote_function(i):
      with ops.device('/job:worker/replica:0/task:0'):
        a = i + variable_b
      c = a + 1.0
      return c

    self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

  def testMultiDeviceFunctionOnRemoteDevice(self):
    with ops.device('/job:worker/replica:0/task:1'):
      variable_b = variables.Variable(1.0)

    @def_function.function
    def remote_function(i):
      with ops.device('/job:worker/replica:0/task:0'):
        a = i + variable_b
      c = a + 1.0
      return c

    context.context().mirroring_policy = context.MIRRORING_NONE

    with ops.device('/job:worker/replica:0/task:0'):
      self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

    if test_util.is_gpu_available():
      with ops.device('/job:worker/replica:0/task:0/device:GPU:0'):
        self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

    context.context().mirroring_policy = context.MIRRORING_ALL

    with ops.device('/job:worker/replica:0/task:0'):
      self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

    if test_util.is_gpu_available():
      with ops.device('/job:worker/replica:0/task:0/device:GPU:0'):
        self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

  def testMultiDeviceWhileLoopOnRemoteDevice(self):
    with ops.device('/job:worker/replica:0/task:1'):
      variable_b = variables.Variable(1.0)

    @def_function.function
    def remote_function(i):

      def body(i, _):
        with ops.device('/job:worker/replica:0/task:0'):
          a = i + variable_b
        return a + 1.0, 1

      return control_flow_ops.while_loop_v2(lambda _, d: d < 1, body, [i, 0])[0]

    context.context().mirroring_policy = context.MIRRORING_NONE

    with ops.device('/job:worker/replica:0/task:0'):
      self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

    if test_util.is_gpu_available():
      with ops.device('/job:worker/replica:0/task:0/device:GPU:0'):
        self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

    context.context().mirroring_policy = context.MIRRORING_ALL

    with ops.device('/job:worker/replica:0/task:0'):
      self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

    if test_util.is_gpu_available():
      with ops.device('/job:worker/replica:0/task:0/device:GPU:0'):
        self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])

  def testSimpleParameterServer(self):

    with ops.device('/job:worker/task:2/device:CPU:0'):
      v1 = variables.Variable(initial_value=0)
      v2 = variables.Variable(initial_value=10)

    @def_function.function
    def worker_fn():
      v1.assign_add(1)
      v2.assign_sub(2)
      return v1.read_value() + v2.read_value()

    with ops.device('/job:worker/task:0/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 9)

    with ops.device('/job:worker/task:1/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 8)


_GRPC_PREFIX = 'grpc://'


class MultiJobsTest(test.TestCase):

  def setUp(self):
    super(MultiJobsTest, self).setUp()

    workers, ps = test_util.create_local_cluster(2, 1)

    cluster = {
        'my_worker': [
            _strip_prefix(workers[0].target, _GRPC_PREFIX),
            _strip_prefix(workers[1].target, _GRPC_PREFIX),
        ],
        'my_ps': [_strip_prefix(ps[0].target, _GRPC_PREFIX)],
    }

    remote.connect_to_cluster(server_lib.ClusterSpec(cluster))

  def testSimpleParameterServer(self):

    with ops.device('/job:my_ps/task:0/device:CPU:0'):
      v1 = variables.Variable(initial_value=0)
      v2 = variables.Variable(initial_value=10)

    @def_function.function
    def worker_fn():
      v1.assign_add(1)
      v2.assign_sub(2)
      return v1.read_value() + v2.read_value()

    with ops.device('/job:my_worker/task:0/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 9)

    with ops.device('/job:my_worker/task:1/device:CPU:0'):
      self.assertAllEqual(worker_fn(), 8)


def _strip_prefix(s, prefix):
  return s[len(prefix):] if s.startswith(prefix) else s


if __name__ == '__main__':
  test.main()
