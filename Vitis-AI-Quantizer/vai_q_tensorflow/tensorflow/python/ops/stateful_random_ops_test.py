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
"""Tests for stateful_random_ops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from absl.testing import parameterized
import numpy as np

from tensorflow.python.distribute import values as dist_values
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.random import util as \
random_test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import stateful_random_ops as \
random
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


g_seeded = None
g_unseeded = None


GPU_FLOATS = [dtypes.float16, dtypes.float32, dtypes.float64]
CPU_FLOATS = GPU_FLOATS + [dtypes.bfloat16]
FLOATS = GPU_FLOATS
INTS = [dtypes.int32, dtypes.int64]


class StatefulRandomOpsTest(test.TestCase, parameterized.TestCase):

  def testCreateRNGStateIntSeed(self):
    """Tests `create_rng_state` when `seed` is int."""
    # using leading 'F' to test overflow tolerance
    state = random.create_rng_state(0xFFFF222233334444FFAA666677778888,
                                    random.RNG_ALG_PHILOX)
    self.assertAllEqual(
        list(map(random._uint_to_int,
                 [0xFFAA666677778888, 0xFFFF222233334444] +
                 [0] * (random.PHILOX_STATE_SIZE - 2))),
        state)

  def assertAllDifferent(self, tensors):
    """Checks that there are no duplicate elements anywhere among the tensors.

    Args:
      tensors: a list of tensors. They can have different shapes.
    """
    tensors = [array_ops.reshape(t, shape=[-1]) for t in tensors]
    ls = array_ops.concat(tensors, axis=0).numpy().tolist()
    self.assertAllEqual(len(ls), len(set(ls)))

  @test_util.run_v2_only
  def testNonDeterministicInts(self):
    """Tests that non_deterministic_ints returns different results every time.

    This test is flaky, but with very low probability of failing.
    """
    shape = [2, 3]
    dtype = dtypes.int64
    a = random.non_deterministic_ints(shape=shape, dtype=dtype)
    self.assertAllEqual(shape, a.shape)
    self.assertEqual(dtype, a.dtype)
    b = random.non_deterministic_ints(shape, dtype=dtype)
    self.assertAllDifferent([a, b])

  @test_util.run_v2_only
  def testBatchSeeds(self):
    """Test for batch seeds.
    """
    shape = [2, 3]
    count = 6
    gen = random.Generator.from_seed(1234)
    keys1 = gen._make_int64_keys(shape=shape)
    keys2 = gen._make_int64_keys(shape=shape)
    self.assertAllDifferent([keys1, keys2])
    seeds1 = gen.make_seeds(count=count)
    seeds2 = gen.make_seeds(count=count)
    self.assertAllDifferent([seeds1[0, :], seeds2[0, :]])
    gens = gen.split(count=count)
    self.assertAllEqual(count, len(gens))
    randoms = [g.uniform_full_int(shape=shape, dtype=dtypes.int32)
               for g in gens]
    self.assertAllDifferent(randoms)
    # Tests graph mode.
    @def_function.function
    def f():
      return gen.make_seeds(count=count)
    for _ in range(3):
      f()

  def assertRegex(self, pattern, text):
    self.assertTrue(
        re.search(pattern, text),
        "Can't find pattern '%s' in text '%s'" % (pattern, text))

  @test_util.run_v2_only
  @test_util.run_cuda_only
  def testCrossDeviceSplit(self):
    """Tests that a CPU RNG can split into RNGs on GPU.
    """
    with ops.device("/device:CPU:0"):
      gen = random.Generator.from_seed(1234)  # gen is on CPU
      self.assertRegex("CPU", gen.state.device)
    with ops.device(test_util.gpu_device_name()):
      gens = gen.split(count=10)  # gens are on GPU
      self.assertRegex("GPU", gens[0].state.device)

  @test_util.run_v2_only
  def testReset(self):
    shape = [2, 3]
    gen = random.Generator.from_seed(0)
    for resetter in [
        lambda g: g.reset(state=[1, 2, 3]),
        lambda g: g.reset_from_seed(1234),
        lambda g: g.reset_from_key_counter(key=1, counter=[2, 3]),
    ]:
      resetter(gen)
      expected_normal = gen.normal(shape)
      @def_function.function
      def f(resetter):
        resetter(gen)
        return gen.normal(shape)
      def check_results(expected_normal, v):
        self.assertAllEqual(expected_normal, v)
      check_results(expected_normal, f(resetter))
      check_results(expected_normal, f(resetter))

  @test_util.run_v2_only
  def testGeneratorCreation(self):
    """Tests generator creation, in both eager and tf.function.

    The interaction between Generator creation and defun should be the same as
    tf.Variable.
    """
    shape = [2, 3]
    alg = random.RNG_ALG_PHILOX
    for constructor in [
        lambda: random.Generator(state=[1, 2, 3], alg=alg),
        lambda: random.Generator.from_seed(1234),
        lambda: random.Generator.from_key_counter(  # pylint: disable=g-long-lambda
            key=1, counter=[2, 3], alg=alg),
    ]:
      gen = constructor()
      # Tests tf.function
      expected_normal1 = gen.normal(shape)
      expected_normal2 = gen.normal(shape)
      global g_seeded
      g_seeded = None
      @def_function.function
      def f(constructor):
        global g_seeded
        # defun'ed function should only create variables once
        if g_seeded is None:
          g_seeded = constructor()
        return g_seeded.normal(shape)
      def check_results(expected_normal, v):
        self.assertAllEqual(expected_normal, v)
      check_results(expected_normal1, f(constructor))
      check_results(expected_normal2, f(constructor))

  @test_util.run_v2_only
  def testGeneratorCreationUnseeded(self):
    """Tests generator creation, the unseeded case."""
    shape = [2, 3]
    global g_unseeded
    g_unseeded = None
    @def_function.function
    def f():
      global g_unseeded
      # defun'ed function should only create variables once
      if g_unseeded is None:
        g_unseeded = random.Generator.from_non_deterministic_state()
      return g_unseeded.normal(shape)
    self.assertAllEqual(shape, f().shape)

  @test_util.run_v2_only
  def testGeneratorCopy(self):
    """Tests copying a generator."""
    g = random.Generator.from_seed(0)
    g_copy = random.Generator(g)
    self.assertAllEqual(g.algorithm, g_copy.algorithm)
    self.assertAllEqual(g.state.read_value(), g_copy.state.read_value())
    # Tests tf.function
    global g_seeded
    g_seeded = None
    # Do the same in tf.function
    @def_function.function
    def f():
      global g_seeded
      # defun'ed function should only create variables once
      if g_seeded is None:
        g_seeded = random.Generator(g)
      self.assertAllEqual(g.algorithm, g_seeded.algorithm)
      self.assertAllEqual(g.state.read_value(), g_seeded.state.read_value())
    f()

  @test_util.run_v1_only(
      ("This test is specifically for checking TF1 compatibility. "
       "It cannot run under TF2."))
  def testTF1(self):
    seed = 1234
    shape = [2, 3]
    expected_normal1 = constant_op.constant(
        [[0.9356609, 1.0854305, -0.93788373],
         [-0.50615472, 1.31697023, 0.71375787]], dtype=dtypes.float32)
    expected_normal2 = constant_op.constant(
        [[-0.3964749, 0.8369565, -0.30946946],
         [1.1206646, 1.00852597, -0.10185789]], dtype=dtypes.float32)
    with self.cached_session() as sess:
      gen1 = random.Generator.from_seed(seed)
      gen2 = random.Generator.from_non_deterministic_state()
      sess.run((gen1._state_var.initializer, gen2._state_var.initializer))
      r1 = gen1.normal(shape, dtype=dtypes.float32)
      r2 = gen2.normal(shape, dtype=dtypes.float32)
      def f():
        return sess.run((r1, r2))
      def check_results(expected_normal, v1, v2):
        self.assertAllClose(expected_normal, v1, rtol=1e-5, atol=1e-5)
        self.assertAllEqual(shape, v2.shape)
      check_results(expected_normal1, *f())
      check_results(expected_normal2, *f())

  @test_util.run_v2_only
  @test_util.also_run_as_tf_function
  def testEagerAndDefun(self):
    """A simple test to make sure the op works in eager and defunned mode."""
    random.get_global_generator().normal((3,))

  @test_util.run_v2_only
  def testOpSeedSelectionAfterSetSeed(self):
    """Tests that op-seed selection is reset after reseting global generator.

    Fixing GitHub issue 9171:
    https://github.com/tensorflow/tensorflow/issues/9171
    """
    shape = (3,)
    random.get_global_generator().reset_from_seed(1)
    a = random.get_global_generator().normal(shape)
    random.get_global_generator().reset_from_seed(1)
    b = random.get_global_generator().normal(shape)
    self.assertAllEqual(a, b)

    # Now do the above again using accelerated ('defun'ed) computation
    @def_function.function
    def f():
      return random.get_global_generator().normal(shape)

    random.get_global_generator().reset_from_seed(1)
    c = f()
    random.get_global_generator().reset_from_seed(1)
    d = f()
    self.assertAllEqual(c, d)
    self.assertAllEqual(a, c)

  @test_util.run_v2_only
  def testOpSeedSelectionNotSensitive(self):
    """Test that op-seed selection is not sensitive to trivial changes.

    Test that op-seed selection is not sensitive to trivial computation
    (i.e. graph) changes.

    Fixing b/32087099
    """
    def f(include_print):
      shape = constant_op.constant([5])
      if include_print:
        shape = logging_ops.Print(shape, [shape])
      return random.get_global_generator().normal(shape)

    def compare(fst_includes_print, snd_includes_print):
      random.get_global_generator().reset_from_seed(50)
      fst = f(fst_includes_print)
      random.get_global_generator().reset_from_seed(50)
      snd = f(snd_includes_print)
      self.assertAllEqual(fst, snd)
      # Now do the above again using accelerated (defunned) 'f'.
      # Running 'f' with two different Boolean arguments should cause
      # two different graphs to be generated, hence demonstrating the
      # insensitivity to graph changes.
      f_acc = def_function.function(f)
      random.get_global_generator().reset_from_seed(50)
      fst = f_acc(fst_includes_print)
      random.get_global_generator().reset_from_seed(50)
      snd = f_acc(snd_includes_print)
      self.assertAllEqual(fst, snd)

    compare(False, False)
    compare(True, True)
    compare(True, False)

  @test_util.run_v2_only
  def testKey(self):
    key = 1234
    gen = random.Generator(state=[0, 0, key], alg=random.RNG_ALG_PHILOX)
    got = gen.key
    self.assertAllEqual(key, got)
    @def_function.function
    def f():
      return gen.key
    got = f()
    self.assertAllEqual(key, got)

  @test_util.run_v2_only
  def testSkip(self):
    key = 1234
    counter = 5678
    gen = random.Generator(state=[counter, 0, key], alg=random.RNG_ALG_PHILOX)
    delta = 432
    gen.skip(delta)
    new_counter = gen._state_var[0]
    self.assertAllEqual(counter + delta * 256, new_counter)

  def _sameAsOldRandomOps(self, device, floats):
    def compare(dtype, old, new):
      seed1, seed2 = 79, 25
      # note how the two seeds for the old op correspond to the seed for the new
      # op
      with ops.device(device):
        gen = random.Generator(state=[0, seed2, seed1],
                               alg=random.RNG_ALG_PHILOX)

      # create a graph for the old op in order to call it many times
      @def_function.function
      def run_old():
        with ops.device(device):
          return old(dtype, seed1, seed2)

      def run_new():
        with ops.device(device):
          return new(dtype, gen)

      for _ in range(100):
        self.assertAllEqual(run_old(), run_new())

    shape = constant_op.constant([4, 7])
    minval = 128
    maxval = 256

    # passing `dtype` around to compress go/gpylint-faq#cell-var-from-loop and
    # go/gpylint-faq#undefined-loop-variable
    def old_normal(dtype, seed1, seed2):
      return gen_random_ops.random_standard_normal(
          shape, dtype=dtype, seed=seed1, seed2=seed2)
    def new_normal(dtype, gen):
      return gen._standard_normal(shape, dtype=dtype)
    def old_truncated_normal(dtype, seed1, seed2):
      return gen_random_ops.truncated_normal(
          shape, dtype=dtype, seed=seed1, seed2=seed2)
    def new_truncated_normal(dtype, gen):
      return gen._truncated_normal(shape, dtype=dtype)
    def old_uniform_int(dtype, seed1, seed2):
      minval2 = constant_op.constant(minval, dtype=dtype)
      maxval2 = constant_op.constant(maxval, dtype=dtype)
      return gen_random_ops.random_uniform_int(
          shape, minval=minval2, maxval=maxval2, seed=seed1, seed2=seed2)
    def new_uniform_int(dtype, gen):
      return gen.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    def old_uniform(dtype, seed1, seed2):
      return gen_random_ops.random_uniform(
          shape, dtype=dtype, seed=seed1, seed2=seed2)
    def new_uniform(dtype, gen):
      return gen._uniform(shape, dtype=dtype)

    for dtype in floats:
      compare(dtype, old_normal, new_normal)
      compare(dtype, old_truncated_normal, new_truncated_normal)
      compare(dtype, old_uniform, new_uniform)
    for dtype in INTS:
      compare(dtype, old_uniform_int, new_uniform_int)

  @test_util.run_v2_only
  def testSameAsOldRandomOpsCPU(self):
    """Tests that the generated numbers are the same as the old random_ops.py.

    The CPU version.
    """
    self._sameAsOldRandomOps("/device:CPU:0", CPU_FLOATS)

  @test_util.run_v2_only
  @test_util.run_cuda_only
  def testSameAsOldRandomOpsGPU(self):
    """Tests that the generated numbers are the same as the old random_ops.py.

    The GPU version.
    """
    self._sameAsOldRandomOps(test_util.gpu_device_name(), GPU_FLOATS)

  @parameterized.parameters(INTS + [dtypes.uint32, dtypes.uint64])
  @test_util.run_v2_only
  @test_util.run_cuda_only
  def testGPUEqualsCPU(self, dtype):
    """Tests that GPU and CPU generate the same integer outputs."""
    seed = 1234
    shape = [315, 49]
    with ops.device("/device:CPU:0"):
      cpu = random.Generator.from_seed(seed).uniform_full_int(
          shape=shape, dtype=dtype)
    with ops.device(test_util.gpu_device_name()):
      gpu = random.Generator.from_seed(seed).uniform_full_int(
          shape=shape, dtype=dtype)
    self.assertAllEqual(cpu, gpu)

  @parameterized.parameters(FLOATS + INTS)
  @test_util.run_v2_only
  def testUniformIsInRange(self, dtype):
    minval = 2
    maxval = 33
    size = 1000
    gen = random.Generator.from_seed(1234)
    x = gen.uniform(
        shape=[size], dtype=dtype, minval=minval, maxval=maxval).numpy()
    self.assertTrue(np.all(x >= minval))
    self.assertTrue(np.all(x < maxval))

  @parameterized.parameters(FLOATS)
  @test_util.run_v2_only
  def testNormalIsFinite(self, dtype):
    gen = random.Generator.from_seed(1234)
    x = gen.normal(shape=[10000], dtype=dtype).numpy()
    self.assertTrue(np.all(np.isfinite(x)))

  @parameterized.parameters(FLOATS + INTS)
  @test_util.run_v2_only
  def testDistributionOfUniform(self, dtype):
    """Use Pearson's Chi-squared test to test for uniformity."""
    n = 1000
    seed = 12
    gen = random.Generator.from_seed(seed)
    maxval = 1
    if dtype.is_integer:
      maxval = 100
    x = gen.uniform(shape=[n], maxval=maxval, dtype=dtype).numpy()
    if maxval > 1:
      # Normalize y to range [0, 1).
      x = x.astype(float) / maxval
    # Tests that the values are distributed amongst 10 bins with equal
    # probability. 16.92 is the Chi^2 value for 9 degrees of freedom with
    # p=0.05. This test is probabilistic and would be flaky if the random
    # seed were not fixed.
    val = random_test_util.chi_squared(x, 10)
    self.assertLess(val, 16.92)

  @parameterized.parameters(FLOATS)
  @test_util.run_v2_only
  def testDistributionOfNormal(self, dtype):
    """Use Anderson-Darling test to test distribution appears normal."""
    n = 1000
    gen = random.Generator.from_seed(1234)
    x = gen.normal(shape=[n], dtype=dtype).numpy()
    # The constant 2.492 is the 5% critical value for the Anderson-Darling
    # test where the mean and variance are known. This test is probabilistic
    # so to avoid flakiness the seed is fixed.
    self.assertLess(
        random_test_util.anderson_darling(x.astype(float)), 2.492)

  @test_util.run_v2_only
  def testErrors(self):
    """Tests that proper errors are raised.
    """
    shape = [2, 3]
    gen = random.Generator.from_seed(1234)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        r"must have shape \[\], not"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          gen.state.handle, [0, 0], shape)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        r"must have shape \[\], not"):
      gen_stateful_random_ops.rng_skip(
          gen.state.handle, gen.algorithm, [0, 0])
    with self.assertRaisesWithPredicateMatch(
        TypeError, "EagerTensor of dtype int64"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          gen.state.handle, 1.1, shape)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        "Unsupported algorithm id"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          gen.state.handle, 123, shape)
    var = variables.Variable([0, 0], dtype=dtypes.int32)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        "dtype of RNG state variable must be int64, not"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          var.handle, random.RNG_ALG_PHILOX, shape)
    var = variables.Variable([[0]], dtype=dtypes.int64)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        "RNG state must have one and only one dimension, not"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          var.handle, random.RNG_ALG_PHILOX, shape)
    var = variables.Variable([0], dtype=dtypes.int64)
    with self.assertRaisesWithPredicateMatch(
        errors.InvalidArgumentError,
        "For the Philox algorithm, the size of state must be at least"):
      gen_stateful_random_ops.stateful_standard_normal_v2(
          var.handle, random.RNG_ALG_PHILOX, shape)

  @test_util.run_v2_only
  def testSetGlobalGeneratorBadWithDefun(self):
    """Demonstrates that set_global_generator don't work properly with defun.
    """
    shape = (3,)

    @def_function.function
    def f():
      return random.get_global_generator().normal(shape)

    random.set_global_generator(random.Generator.from_seed(50))
    with self.assertRaisesWithPredicateMatch(
        errors.NotFoundError, "Resource .+ does not exist"):
      _ = f()
      random.set_global_generator(random.Generator.from_seed(50))
      _ = f()

  @test_util.run_v2_only
  @test_util.run_cuda_only
  def testMirroredStratSeq(self):
    """Tests RNG/MirrorStrategy interaction #1.

    If an RNG is created outside strategy.scope(), all replicas will access the
    same RNG object, and accesses are serialized.
    """
    shape = [3, 4]
    dtype = dtypes.int32
    gen = random.Generator.from_seed(1234)
    strat = MirroredStrategy(devices=["/cpu:0", test_util.gpu_device_name()])
    with strat.scope():
      def f():
        t1 = gen.uniform_full_int(shape=shape, dtype=dtype)
        t2 = gen.uniform_full_int(shape=shape, dtype=dtype)
        t = array_ops.stack([t1, t2])
        return t
      results = strat.extended.call_for_each_replica(
          fn=f)
      values = results.values
      self.assertAllEqual(2, len(values))
      self.assertAllDifferent(values)

  @test_util.run_v2_only
  @test_util.run_cuda_only
  def testMirroredStratParaSync(self):
    """Tests RNG/MirrorStrategy interaction #2.

    If an RNG is created inside strategy.scope(), each replica gets an
    mirror of this RNG. If they access their RNGs in the same
    manner, their random-number streams are the same.
    """
    shape = [3, 4]
    dtype = dtypes.int32
    strat = MirroredStrategy(devices=["/cpu:0", test_util.gpu_device_name()])
    with strat.scope():
      gen = random.Generator.from_seed(1234)
      def f():
        t1 = gen.uniform_full_int(shape=shape, dtype=dtype)
        t2 = gen.uniform_full_int(shape=shape, dtype=dtype)
        t = array_ops.stack([t1, t2])
        return t
      results = strat.extended.call_for_each_replica(fn=f)
      values = results.values
      self.assertAllEqual(2, len(values))
      self.assertAllEqual(values[0], values[1])

  @test_util.run_v2_only
  @test_util.run_cuda_only
  def testMirroredStratParaSyncWithinFun(self):
    """Tests RNG/MirrorStrategy interaction #2b.

    If the RNG creation is within `f` in situation #2, the replicas'
    random-number streams are still the same. Note that whether the RNG creation
    is within strategy.scope() or not doesn't affect the result in this case
    (putting in inside strategy.scope() will cause unnecessary mirror creation
    and waste memory though).
    """
    shape = [3, 4]
    dtype = dtypes.int32
    strat = MirroredStrategy(devices=["/cpu:0", test_util.gpu_device_name()])
    def f():
      gen = random.Generator.from_seed(1234)
      t1 = gen.uniform_full_int(shape=shape, dtype=dtype)
      t2 = gen.uniform_full_int(shape=shape, dtype=dtype)
      t = array_ops.stack([t1, t2])
      return t
    results = strat.extended.call_for_each_replica(fn=f)
    values = results.values
    self.assertAllEqual(2, len(values))
    self.assertAllEqual(values[0], values[1])

  @test_util.run_v2_only
  @test_util.run_cuda_only
  def testMirroredStratUnseedSync(self):
    """Tests RNG/MirrorStrategy interaction #2c.

    If the RNG created in situation #2 is unseeded, the replicas' random-number
    streams are still the same.

    If the RNG created in situation #2b is unseeded, the replicas' random-number
    streams will be different. We can't test this for now because the op
    'NonDeterministicInts' is not implemented on GPU yet.
    """
    shape = [3, 4]
    dtype = dtypes.int32
    strat = MirroredStrategy(devices=["/cpu:0", test_util.gpu_device_name()])
    # TODO(wangpeng): support calling `random.Generator()` inside `f` (i.e.
    #   inside `call_for_each_replica` so that each replica can get a
    #   different random-number stream. The only obstacle is that op
    #   'NonDeterministicInts' is not implemented on GPU.)
    with strat.scope():
      gen = random.Generator.from_non_deterministic_state()
      def f():
        t1 = gen.uniform_full_int(shape=shape, dtype=dtype)
        t2 = gen.uniform_full_int(shape=shape, dtype=dtype)
        t = array_ops.stack([t1, t2])
        return t
      results = strat.extended.call_for_each_replica(fn=f)
      values = results.values
      self.assertAllEqual(2, len(values))
      self.assertAllEqual(values[0], values[1])

  @test_util.run_v2_only
  @test_util.run_cuda_only
  def testMirroredStratParaAsync(self):
    """Tests RNG/MirrorStrategy interaction #3.

    The user can create n independent RNGs outside strategy.scope(), where n
    is the number of replicas, and give one to each replica. The replicas can
    thus get different random-number streams.
    """
    shape = [3, 4]
    dtype = dtypes.int32
    gens = random.get_global_generator().split(count=2)
    devices = ["/cpu:0", test_util.gpu_device_name()]
    strat = MirroredStrategy(devices=devices)
    # Use `PerReplica` to specify which `gen` is sent to which replica
    gens = dist_values.PerReplica(
        device_map=dist_values.ReplicaDeviceMap(devices),
        values=[[g] for g in gens])
    with strat.scope():
      def f(gen):
        t1 = gen.uniform_full_int(shape=shape, dtype=dtype)
        t2 = gen.uniform_full_int(shape=shape, dtype=dtype)
        t = array_ops.stack([t1, t2])
        return t
      results = strat.extended.call_for_each_replica(
          fn=f, args=gens)
      values = results.values
      self.assertAllEqual(2, len(values))
      self.assertAllDifferent(values)


if __name__ == "__main__":
  test.main()
