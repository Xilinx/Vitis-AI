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
"""Benchmarks for checkpoint-related APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.platform import test
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import util


class _TrivialSaveable(saveable_object.SaveableObject):

  def __init__(self, name):
    op = lambda: array_ops.ones([])
    super(_TrivialSaveable, self).__init__(
        op=op,
        specs=[saveable_object.SaveSpec(
            op, "", name, dtype=dtypes.float32, device="CPU:0")],
        name=name)

  def restore(self, restored_tensors, restored_shapes):
    return control_flow_ops.no_op()


class _TrivialRestore(base.Trackable):

  def _gather_saveables_for_checkpoint(self):
    return {base.VARIABLE_VALUE_KEY: _TrivialSaveable}


class _LazyTrivialObjects(module.Module):

  def __init__(self):
    self.existing = [_TrivialRestore() for _ in range(5)]
    self.lazy = []

  def __call__(self):
    if not self.lazy:
      self.lazy.extend(_TrivialRestore() for _ in range(5))
    return


def _save_checkpoint():
  original_checkpoint = util.Checkpoint(m=_LazyTrivialObjects())
  original_checkpoint.m()
  return original_checkpoint.write(os.path.join(test.get_temp_dir(), "ckpt"))


class SavingBenchmarks(test.Benchmark):

  def _run(self, func, num_iters, execution_mode=None):
    func()
    start = time.time()
    for _ in xrange(num_iters):
      func()
    end = time.time()
    mean_us = (end - start) * 1e6 / num_iters
    self.report_benchmark(
        iters=num_iters,
        wall_time=mean_us,
        extras={"examples_per_sec": num_iters / (end - start)})

  def benchmark_baseline_no_restore(self):

    def _create_and_call():
      checkpoint = util.Checkpoint(m=_LazyTrivialObjects())
      checkpoint.m()

    self._run(_create_and_call, 3)

  def benchmark_batch_restore(self):
    checkpoint_path = _save_checkpoint()

    def _create_and_call():
      checkpoint = util.Checkpoint(m=_LazyTrivialObjects())
      checkpoint.m()
      checkpoint.restore(checkpoint_path)

    self._run(_create_and_call, 3)

  def benchmark_restore_on_create(self):
    checkpoint_path = _save_checkpoint()

    def _create_and_call():
      checkpoint = util.Checkpoint(m=_LazyTrivialObjects())
      checkpoint.restore(checkpoint_path)
      checkpoint.m()

    self._run(_create_and_call, 3)

  def benchmark_raw_restore(self):
    checkpoint_path = _save_checkpoint()
    all_names, all_dtypes = zip(*pywrap_tensorflow.NewCheckpointReader(
        checkpoint_path).get_variable_to_dtype_map().items())

    def _call_restore_v2():
      gen_io_ops.restore_v2(checkpoint_path, all_names, [""] * len(all_names),
                            all_dtypes)

    self._run(_call_restore_v2, 3)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
