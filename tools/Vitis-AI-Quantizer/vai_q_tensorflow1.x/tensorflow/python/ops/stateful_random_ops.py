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
"""Operations for generating random numbers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util.tf_export import tf_export

# A seed for random ops (stateful and stateless) will always be 1024
# bits, all of which will be sent to the C++ code. The actual C++
# implementation of some algorithms may only use a lower part of the bits.

MAX_INT64 = 2**63 - 1
MIN_INT64 = -(2**63)
UINT64_SPAN = 2**64
# 'Variable' doesn't support uint32 or uint64 yet (due to reasons explained in
# b/111604096 and cl/171681867), so I use signed int here. I choose int64
# instead of int32 here because `VarHandleOp` doesn't support int32 on GPU.
SEED_TYPE = "int64"
SEED_MIN = MIN_INT64
SEED_MAX = MAX_INT64
SEED_UINT_SPAN = UINT64_SPAN
SEED_TYPE_BITS = 64
SEED_BIT_MASK = 0xFFFFFFFFFFFFFFFF
SEED_SIZE = 16  # in units of SEED_TYPE


STATE_TYPE = SEED_TYPE
ALGORITHM_TYPE = STATE_TYPE
RNG_ALG_PHILOX = 1
RNG_ALG_THREEFRY = 2
DEFAULT_ALGORITHM = RNG_ALG_PHILOX


PHILOX_STATE_SIZE = 3
THREEFRY_STATE_SIZE = 2


def non_deterministic_ints(shape, dtype=dtypes.int64):
  """Non-deterministically generates some integers.

  This op may use some OS-provided source of non-determinism (e.g. an RNG), so
  each execution will give different results.

  Args:
    shape: the shape of the result.
    dtype: (optional) the dtype of the result.

  Returns:
    a tensor whose element values are non-deterministically chosen.
  """
  return gen_stateful_random_ops.non_deterministic_ints(
      shape=shape, dtype=dtype)


def _uint_to_int(n):
  if n > SEED_MAX:
    n = n - SEED_UINT_SPAN
  return n


def _make_1d_state(state_size, seed):
  """Makes a 1-D RNG state.

  Args:
    state_size: an integer.
    seed: an integer or 1-D tensor.

  Returns:
    a 1-D tensor of shape [state_size] and dtype STATE_TYPE.
  """
  int_types = (int,) if sys.version_info >= (3, 0) else (int, long)
  if isinstance(seed, int_types):
    # chop the Python integer (infinite precision) into chunks of SEED_TYPE
    ls = []
    for _ in range(state_size):
      ls.append(seed & SEED_BIT_MASK)
      seed >>= SEED_TYPE_BITS
    seed = ls
  # to avoid overflow error from np.asarray
  seed = list(map(_uint_to_int, seed))
  seed = np.asarray(seed, dtype=STATE_TYPE)
  if len(seed.shape) != 1:
    raise ValueError(
        "seed should only have one dimension; got shape: %s" % seed.shape)
  seed = seed[0:state_size]
  # Padding with zeros on the *left* if too short. Padding on the right would
  # cause a small seed to be used as the "counter" while the "key" is always
  # zero (for counter-based RNG algorithms), because in the current memory
  # layout counter is stored before key. In such a situation two RNGs with
  # two different small seeds may generate overlapping outputs.
  seed_size = seed.shape[0]
  if seed_size < state_size:
    seed = np.pad(
        seed, [(state_size - seed_size, 0)],
        mode="constant",
        constant_values=0)
  assert seed.shape == (state_size,), "Wrong seed.shape: %s" % seed.shape
  return seed


def _get_state_size(alg):
  if alg == RNG_ALG_PHILOX:
    return PHILOX_STATE_SIZE
  elif alg == RNG_ALG_THREEFRY:
    return THREEFRY_STATE_SIZE
  else:
    raise ValueError("Unsupported algorithm id: %s" % alg)


def _make_state_from_seed(seed, alg):
  return _make_1d_state(_get_state_size(alg), seed)


@tf_export("random.experimental.create_rng_state")
def create_rng_state(seed, algorithm):
  """Creates a RNG state.

  Args:
    seed: an integer or 1-D tensor.
    algorithm: an integer representing the RNG algorithm.

  Returns:
    a 1-D tensor whose size depends on the algorithm.
  """
  return _make_state_from_seed(seed, algorithm)


def _shape_tensor(shape):
  """Convert to an int32 or int64 tensor, defaulting to int64 if empty."""
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = dtypes.int64
  else:
    dtype = None
  return ops.convert_to_tensor(shape, dtype=dtype, name="shape")


def _convert_to_state_tensor(t):
  if isinstance(t, list):
    # to avoid out-of-range error from ops.convert_to_tensor
    t = list(map(_uint_to_int, t))
  return ops.convert_to_tensor(t, dtype=STATE_TYPE)


@tf_export("random.experimental.Generator")
class Generator(tracking.AutoTrackable):
  """Random-number generator.

  It uses Variable to manage its internal state, and allows choosing an
  Random-Number-Generation (RNG) algorithm.

  CPU, GPU and TPU with the same algorithm and seed will generate the same
  integer random numbers. Float-point results (such as the output of `normal`)
  may have small numerical discrepancies between CPU and GPU.
  """

  def __init__(self, copy_from=None, state=None, alg=None):
    """Creates a generator.

    The new generator will be initialized by one of the following ways, with
    decreasing precedence:
    (1) If `copy_from` is not None, the new generator is initialized by copying
        information from another generator.
    (3) If `state` and `alg` are not None (they must be set together), the new
        generator is initialized by a state.

    Args:
      copy_from: a generator to be copied from.
      state: a vector of dtype STATE_TYPE representing the initial state of the
        RNG, whose length and semantics are algorithm-specific.
      alg: the RNG algorithm. Possible values are `RNG_ALG_PHILOX` for the
        Philox algorithm and `RNG_ALG_THREEFRY` for the ThreeFry
        algorithm (see paper 'Parallel Random Numbers: As Easy as 1, 2, 3'
        [https://www.thesalmons.org/john/random123/papers/random123sc11.pdf]).
        Note `RNG_ALG_PHILOX` guarantees the same numbers are produced (given
        the same random state) across all architextures (CPU, GPU, XLA etc).
    """
    if copy_from is not None:
      # All other arguments should be None
      assert (alg or state) is None
      self._state_var = variables.Variable(copy_from.state, dtype=STATE_TYPE,
                                           trainable=False)
      self._alg_var = copy_from.algorithm

    else:
      assert alg is not None and state is not None
      state = _convert_to_state_tensor(state)
      state.shape.assert_is_compatible_with([_get_state_size(alg)])
      self._state_var = variables.Variable(state, dtype=STATE_TYPE,
                                           trainable=False)
      self._alg_var = alg

  @classmethod
  def from_state(cls, state, alg):
    """Creates a generator from a state.

    See `__init__` for description of `state` and `alg`.

    Args:
      state: the new state.
      alg: the RNG algorithm.

    Returns:
      The new generator.
    """
    return cls(alg=alg, state=state)

  @classmethod
  def from_seed(cls, seed, alg=None):
    """Creates a generator from a seed.

    A seed is a 1024-bit unsigned integer represented either as a Python
    integer or a vector of integers. Seeds shorter than 1024-bit will be
    padded. The padding, the internal structure of a seed and the way a seed
    is converted to a state are all opaque (unspecified). The only semantics
    specification of seeds is that two different seeds are likely to produce
    two independent generators (but no guarantee).

    Args:
      seed: the seed for the RNG.
      alg: (optional) the RNG algorithm. If None, it will be auto-selected. See
        `__init__` for its possible values.

    Returns:
      The new generator.
    """
    if alg is None:
      # TODO(wangpeng): more sophisticated algorithm selection
      alg = DEFAULT_ALGORITHM
    state = create_rng_state(seed, alg)
    return cls(state=state, alg=alg)

  @classmethod
  def from_non_deterministic_state(cls, alg=None):
    """Creates a generator by non-deterministically initializing its state.

    The source of the non-determinism will be platform- and time-dependent.

    Args:
      alg: (optional) the RNG algorithm. If None, it will be auto-selected. See
        `__init__` for its possible values.

    Returns:
      The new generator.
    """
    if alg is None:
      # TODO(wangpeng): more sophisticated algorithm selection
      alg = DEFAULT_ALGORITHM
    state = non_deterministic_ints(shape=[_get_state_size(alg)],
                                   dtype=SEED_TYPE)
    return cls(state=state, alg=alg)

  @classmethod
  def from_key_counter(cls, key, counter, alg):
    """Creates a generator from a key and a counter.

    This constructor only applies if the algorithm is a counter-based algorithm.
    See method `key` for the meaning of "key" and "counter".

    Args:
      key: the key for the RNG, a scalar of type STATE_TYPE.
      counter: a vector of dtype STATE_TYPE representing the initial counter for
        the RNG, whose length is algorithm-specific.,
      alg: the RNG algorithm. If None, it will be auto-selected. See
        `__init__` for its possible values.

    Returns:
      The new generator.
    """
    counter = _convert_to_state_tensor(counter)
    key = _convert_to_state_tensor(key)
    counter.shape.assert_is_compatible_with([_get_state_size(alg) - 1])
    key.shape.assert_is_compatible_with([])
    key = array_ops.reshape(key, [1])
    state = array_ops.concat([counter, key], 0)
    return cls(state=state, alg=alg)

  def reset(self, state):
    """Resets the generator by a new state.

    See `__init__` for the meaning of "state".

    Args:
      state: the new state.
    """
    state = _convert_to_state_tensor(state)
    state.shape.assert_is_compatible_with([_get_state_size(self.algorithm)])
    self._state_var.assign(state)

  def reset_from_seed(self, seed):
    """Resets the generator by a new seed.

    See `from_seed` for the meaning of "seed".

    Args:
      seed: the new seed.
    """
    state = create_rng_state(seed, self.algorithm)
    self._state_var.assign(state)

  def reset_from_key_counter(self, key, counter):
    """Resets the generator by a new key-counter pair.

    See `from_key_counter` for the meaning of "key" and "counter".

    Args:
      key: the new key.
      counter: the new counter.
    """
    counter = _convert_to_state_tensor(counter)
    key = _convert_to_state_tensor(key)
    counter.shape.assert_is_compatible_with(
        [_get_state_size(self.algorithm) - 1])
    key.shape.assert_is_compatible_with([])
    key = array_ops.reshape(key, [1])
    state = array_ops.concat([counter, key], 0)
    self._state_var.assign(state)

  @property
  def state(self):
    """The internal state of the RNG."""
    return self._state_var

  @property
  def algorithm(self):
    """The RNG algorithm."""
    return self._alg_var

  def _standard_normal(self, shape, dtype):
    return gen_stateful_random_ops.stateful_standard_normal_v2(
        self.state.handle, self.algorithm, shape, dtype=dtype)

  @property
  def key(self):
    """The 'key' part of the state of a counter-based RNG.

    For a counter-base RNG algorithm such as Philox and ThreeFry (as
    described in paper 'Parallel Random Numbers: As Easy as 1, 2, 3'
    [https://www.thesalmons.org/john/random123/papers/random123sc11.pdf]),
    the RNG state consists of two parts: counter and key. The output is
    generated via the formula: output=hash(key, counter), i.e. a hashing of
    the counter parametrized by the key. Two RNGs with two different keys can
    be thought as generating two independent random-number streams (a stream
    is formed by increasing the counter).

    Returns:
      A scalar which is the 'key' part of the state, if the RNG algorithm is
        counter-based; otherwise it raises a ValueError.
    """
    alg = self.algorithm
    if alg == RNG_ALG_PHILOX or alg == RNG_ALG_THREEFRY:
      return self._state_var[-1]
    else:
      raise ValueError("Unsupported algorithm id: %s" % alg)

  def skip(self, delta):
    """Advance the counter of a counter-based RNG.

    Args:
      delta: the amount of advancement. The state of the RNG after
        `skip(n)` will be the same as that after `normal([n])`
        (or any other distribution). The actual increment added to the
        counter is an unspecified implementation detail.
    """
    gen_stateful_random_ops.rng_skip(self.state.handle, self.algorithm, delta)

  # The following functions return a tensor and as a side effect update
  # self._state_var.
  def normal(self, shape, mean=0.0, stddev=1.0, dtype=dtypes.float32,
             name=None):
    """Outputs random values from a normal distribution.

    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output
        tensor.
      mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal
        distribution.
      stddev: A 0-D Tensor or Python value of type `dtype`. The standard
        deviation of the normal distribution.
      dtype: The type of the output.
      name: A name for the operation (optional).

    Returns:
      A tensor of the specified shape filled with random normal values.
    """
    with ops.name_scope(name, "stateful_normal", [shape, mean, stddev]) as name:
      shape = _shape_tensor(shape)
      mean = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
      stddev = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
      rnd = self._standard_normal(shape, dtype=dtype)
      return math_ops.add(rnd * stddev, mean, name=name)

  def _truncated_normal(self, shape, dtype):
    return gen_stateful_random_ops.stateful_truncated_normal(
        self.state.handle, self.algorithm, shape, dtype=dtype)

  def truncated_normal(self, shape,
                       mean=0.0,
                       stddev=1.0,
                       dtype=dtypes.float32,
                       name=None):
    """Outputs random values from a truncated normal distribution.

    The generated values follow a normal distribution with specified mean and
    standard deviation, except that values whose magnitude is more than
    2 standard deviations from the mean are dropped and re-picked.

    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output
        tensor.
      mean: A 0-D Tensor or Python value of type `dtype`. The mean of the
        truncated normal distribution.
      stddev: A 0-D Tensor or Python value of type `dtype`. The standard
        deviation of the normal distribution, before truncation.
      dtype: The type of the output.
      name: A name for the operation (optional).

    Returns:
      A tensor of the specified shape filled with random truncated normal
        values.
    """
    with ops.name_scope(
        name, "truncated_normal", [shape, mean, stddev]) as name:
      shape_tensor = _shape_tensor(shape)
      mean_tensor = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
      stddev_tensor = ops.convert_to_tensor(stddev, dtype=dtype, name="stddev")
      rnd = self._truncated_normal(shape_tensor, dtype=dtype)
      mul = rnd * stddev_tensor
      return math_ops.add(mul, mean_tensor, name=name)

  def _uniform(self, shape, dtype):
    return gen_stateful_random_ops.stateful_uniform(
        self.state.handle, self.algorithm, shape=shape, dtype=dtype)

  def uniform(self, shape, minval=0, maxval=None,
              dtype=dtypes.float32, name=None):
    """Outputs random values from a uniform distribution.

    The generated values follow a uniform distribution in the range
    `[minval, maxval)`. The lower bound `minval` is included in the range, while
    the upper bound `maxval` is excluded. (For float numbers especially
    low-precision types like bfloat16, because of
    rounding, the result may sometimes include `maxval`.)

    For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
    be specified explicitly.

    In the integer case, the random integers are slightly biased unless
    `maxval - minval` is an exact power of two.  The bias is small for values of
    `maxval - minval` significantly smaller than the range of the output (either
    `2**32` or `2**64`).

    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output
        tensor.
      minval: A 0-D Tensor or Python value of type `dtype`. The lower bound on
        the range of random values to generate.  Defaults to 0.
      maxval: A 0-D Tensor or Python value of type `dtype`. The upper bound on
        the range of random values to generate.  Defaults to 1 if `dtype` is
        floating point.
      dtype: The type of the output.
      name: A name for the operation (optional).

    Returns:
      A tensor of the specified shape filled with random uniform values.

    Raises:
      ValueError: If `dtype` is integral and `maxval` is not specified.
    """
    dtype = dtypes.as_dtype(dtype)
    if maxval is None:
      if dtype.is_integer:
        raise ValueError("Must specify maxval for integer dtype %r" % dtype)
      maxval = 1
    with ops.name_scope(name, "stateful_uniform",
                        [shape, minval, maxval]) as name:
      shape = _shape_tensor(shape)
      minval = ops.convert_to_tensor(minval, dtype=dtype, name="min")
      maxval = ops.convert_to_tensor(maxval, dtype=dtype, name="max")
      if dtype.is_integer:
        return gen_stateful_random_ops.stateful_uniform_int(
            self.state.handle, self.algorithm, shape=shape,
            minval=minval, maxval=maxval, name=name)
      else:
        rnd = self._uniform(shape=shape, dtype=dtype)
        return math_ops.add(rnd * (maxval - minval), minval, name=name)

  def uniform_full_int(self, shape, dtype=dtypes.uint64, name=None):
    """Uniform distribution on an integer type's entire range.

    The other method `uniform` only covers the range [minval, maxval), which
    cannot be `dtype`'s full range because `maxval` is of type `dtype`.

    Args:
      shape: the shape of the output.
      dtype: (optional) the integer type, default to uint64.
      name: (optional) the name of the node.

    Returns:
      A tensor of random numbers of the required shape.
    """
    dtype = dtypes.as_dtype(dtype)
    with ops.name_scope(name, "stateful_uniform_full_int",
                        [shape]) as name:
      shape = _shape_tensor(shape)
      return gen_stateful_random_ops.stateful_uniform_full_int(
          self.state.handle, self.algorithm, shape=shape,
          dtype=dtype, name=name)

  def binomial(self, shape, counts, probs, dtype=dtypes.int32, name=None):
    """Outputs random values from a binomial distribution.

    The generated values follow a binomial distribution with specified count and
    probability of success parameters.

    Example:

    ```python
    counts = [10., 20.]
    # Probability of success.
    probs = [0.8, 0.9]

    rng = tf.random.experimental.Generator.from_seed(seed=234)
    binomial_samples = rng.binomial(shape=[2], counts=counts, probs=probs)
    ```


    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output
        tensor.
      counts: A 0/1-D Tensor or Python value. The counts of the binomial
        distribution.  Must be broadcastable with the leftmost dimension
        defined by `shape`.
      probs: A 0/1-D Tensor or Python value. The probability of success for the
        binomial distribution.  Must be broadcastable with the leftmost
        dimension defined by `shape`.
      dtype: The type of the output. Default: tf.int32
      name: A name for the operation (optional).

    Returns:
      samples: A Tensor of the specified shape filled with random binomial
        values.  For each i, each samples[i, ...] is an independent draw from
        the binomial distribution on counts[i] trials with probability of
        success probs[i].
    """
    dtype = dtypes.as_dtype(dtype)
    with ops.name_scope(name, "binomial", [shape, counts, probs]) as name:
      counts = ops.convert_to_tensor(counts, name="counts")
      probs = ops.convert_to_tensor(probs, name="probs")
      shape_tensor = _shape_tensor(shape)
      return gen_stateful_random_ops.stateful_random_binomial(
          self.state.handle,
          self.algorithm,
          shape=shape_tensor,
          counts=counts,
          probs=probs,
          dtype=dtype,
          name=name)

  # TODO(wangpeng): implement other distributions

  def _make_int64_keys(self, shape=()):
    # New independent keys are generated via
    # `new_key[i] = hash(old_key, counter+i)`, which is exactly what
    # `uniform_full_int(dtype=int64)` does for PhiloxRandom_64_128_128 and
    # ThreeFry_64_64_64.
    return self.uniform_full_int(shape=shape, dtype=dtypes.int64)

  def make_seeds(self, count=1):
    """Generates seeds for stateless random ops.

    For example:

    ```python
    seeds = get_global_generator().make_seeds(count=10)
    for i in range(10):
      seed = seeds[:, i]
      numbers = stateless_random_normal(shape=[2, 3], seed=seed)
      ...
    ```

    Args:
      count: the number of seed pairs (note that stateless random ops need a
        pair of seeds to invoke).

    Returns:
      A tensor of shape [2, count] and dtype int64.
    """
    alg = self.algorithm
    if alg == RNG_ALG_PHILOX or alg == RNG_ALG_THREEFRY:
      keys = self._make_int64_keys(shape=[count])
      # The two seeds for stateless random ops don't have individual semantics
      # and are scrambled together, so setting one to zero is fine.
      zeros = array_ops.zeros_like(keys)
      return array_ops.stack([keys, zeros])
    else:
      raise ValueError("Unsupported algorithm id: %s" % alg)

  def split(self, count=1):
    """Returns a list of independent `Generator` objects.

    Two generators are independent of each other in the sense that the
    random-number streams they generate don't have statistically detectable
    correlations. The new generators are also independent of the old one.
    The old generator's state will be changed (like other random-number
    generating methods), so two calls of `split` will return different
    new generators.

    For example:

    ```python
    gens = get_global_generator().split(count=10)
    for gen in gens:
      numbers = gen.normal(shape=[2, 3])
      # ...
    gens2 = get_global_generator().split(count=10)
    # gens2 will be different from gens
    ```

    The new generators will be put on the current device (possible different
    from the old generator's), for example:

    ```python
    with tf.device("/device:CPU:0"):
      gen = Generator(seed=1234)  # gen is on CPU
    with tf.device("/device:GPU:0"):
      gens = gen.split(count=10)  # gens are on GPU
    ```

    Args:
      count: the number of generators to return.

    Returns:
      A list (length `count`) of `Generator` objects independent of each other.
      The new generators have the same RNG algorithm as the old one.
    """
    def _key_to_state(alg, key):
      # Padding with zeros on the left. The zeros will be the counter.
      return [0] * (_get_state_size(alg) - 1) + [key]

    alg = self.algorithm
    if alg == RNG_ALG_PHILOX or alg == RNG_ALG_THREEFRY:
      keys = self._make_int64_keys(shape=[count])
      return [Generator(state=_key_to_state(alg, key), alg=alg)
              for key in keys.numpy()]
    else:
      raise ValueError("Unsupported algorithm id: %s" % alg)


# It's not safe to create TF ops before `init_google` is called, so this is
# initialized to None and get a value the first time `get_global_generator` is
# called.
global_generator = None


@tf_export("random.experimental.get_global_generator")
def get_global_generator():
  global global_generator
  if global_generator is None:
    global_generator = Generator.from_non_deterministic_state()
  return global_generator


@tf_export("random.experimental.set_global_generator")
def set_global_generator(generator):
  """Replaces the global generator with another `Generator` object.

  This function creates a new Generator object (and the Variable object within),
  which does not work well with tf.function because (1) tf.function puts
  restrictions on Variable creation thus reset_global_generator can't be freely
  used inside tf.function; (2) redirecting a global variable to
  a new object is problematic with tf.function because the old object may be
  captured by a 'tf.function'ed function and still be used by it.
  A 'tf.function'ed function only keeps weak references to variables,
  so deleting a variable and then calling that function again may raise an
  error, as demonstrated by
  random_test.py/RandomTest.testResetGlobalGeneratorBadWithDefun .

  Args:
    generator: the new `Generator` object.
  """
  global global_generator
  global_generator = generator
