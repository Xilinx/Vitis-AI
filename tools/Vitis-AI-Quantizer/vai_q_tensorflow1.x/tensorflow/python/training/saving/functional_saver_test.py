# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Tests for the functional saver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import test
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import gfile
from tensorflow.python.training.saving import functional_saver
from tensorflow.python.training.saving import saveable_object_util


class SaverTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_resource_variable(self):
    v1 = resource_variable_ops.ResourceVariable(2.)
    self.evaluate(v1.initializer)
    saver = functional_saver._SingleDeviceSaver(
        saveable_object_util.saveable_objects_for_op(v1, "x"))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    self.evaluate(saver.save(constant_op.constant(prefix)))
    self.assertEqual(2, len(gfile.Glob(prefix + "*")))
    self.evaluate(v1.assign(1.))
    self.evaluate(saver.restore(prefix))
    self.assertEqual(2., self.evaluate(v1))

    v2 = resource_variable_ops.ResourceVariable(3.)
    self.evaluate(v2.initializer)
    second_saver = functional_saver._SingleDeviceSaver(
        saveable_object_util.saveable_objects_for_op(v2, "x"))
    self.evaluate(second_saver.restore(prefix))
    self.assertEqual(2., self.evaluate(v2))

  def test_to_proto(self):
    v1 = resource_variable_ops.ResourceVariable(2.)
    saver = functional_saver.MultiDeviceSaver(
        saveable_object_util.saveable_objects_for_op(v1, "x"))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")

    proto_accumulator = []
    wrapped = wrap_function.wrap_function(
        lambda: proto_accumulator.append(saver.to_proto()), signature=())
    self.assertEqual(1, len(proto_accumulator))
    proto = proto_accumulator[0]
    save = wrapped.prune(
        feeds=wrapped.graph.get_tensor_by_name(proto.filename_tensor_name),
        fetches=wrapped.graph.get_tensor_by_name(proto.save_tensor_name))
    restore = wrapped.prune(
        feeds=wrapped.graph.get_tensor_by_name(proto.filename_tensor_name),
        fetches=wrapped.graph.get_operation_by_name(proto.restore_op_name))
    save_path = save(constant_op.constant(prefix))
    v1.assign(1.)
    restore(constant_op.constant(save_path))
    self.assertEqual(2., self.evaluate(v1))

    v2 = resource_variable_ops.ResourceVariable(3.)
    second_saver = functional_saver.MultiDeviceSaver(
        saveable_object_util.saveable_objects_for_op(v2, "x"))
    second_saver.restore(save_path)
    self.assertEqual(2., self.evaluate(v2))

  @test_util.run_v1_only(
      "Needs an API to setup multiple devices, b/124805129")
  # Set up multiple devices when graph building. Before test.main() we configure
  # the devices for eager execution.
  @test_util.run_in_graph_and_eager_modes(
      config=config_pb2.ConfigProto(device_count={"CPU": 3}))
  def test_checkpoint_is_sharded_by_device(self):
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.)
    with ops.device("cpu:1"):
      v1 = resource_variable_ops.ResourceVariable(1.)
    with ops.device("cpu:2"):
      v2 = resource_variable_ops.ResourceVariable(2.)

    self.evaluate([v0.initializer, v1.initializer, v2.initializer])
    saver = functional_saver.MultiDeviceSaver(
        list(saveable_object_util.saveable_objects_for_op(v0, "v0"))
        + list(saveable_object_util.saveable_objects_for_op(v1, "v1"))
        + list(saveable_object_util.saveable_objects_for_op(v2, "v2")))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    self.evaluate(saver.save(constant_op.constant(prefix)))
    self.assertEqual(4, len(gfile.Glob(prefix + "*")))
    self.evaluate(v0.assign(-1.))
    self.evaluate(v1.assign(-1.))
    self.evaluate(v2.assign(-1.))
    self.evaluate(saver.restore(constant_op.constant(prefix)))
    self.assertEqual(0., self.evaluate(v0))
    self.assertEqual(1., self.evaluate(v1))
    self.assertEqual(2., self.evaluate(v2))


if __name__ == "__main__":
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(device_count={"CPU": 3}))
  test.main()
