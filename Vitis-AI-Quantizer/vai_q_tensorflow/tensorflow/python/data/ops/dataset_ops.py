# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Python wrappers for Datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import enum
import functools
import threading
import warnings
import weakref

import numpy as np
import six
from six.moves import queue as Queue  # pylint: disable=redefined-builtin


from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.compat import compat
from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.experimental.ops import optimization_options
from tensorflow.python.data.experimental.ops import stats_options
from tensorflow.python.data.experimental.ops import threading_options
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import options as options_lib
from tensorflow.python.data.util import random_seed
from tensorflow.python.data.util import sparse
from tensorflow.python.data.util import structure
from tensorflow.python.data.util import traverse
from tensorflow.python.eager import context
from tensorflow.python.eager import function as eager_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as core_random_seed
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as tracking_base
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest as tf_nest
from tensorflow.python.util.tf_export import tf_export

# Loaded lazily due to a circular dependency (roughly
# tf.function->wrap_function->dataset->autograph->tf.function).
# TODO(b/133251390): Use a regular import.
wrap_function = lazy_loader.LazyLoader(
    "wrap_function", globals(),
    "tensorflow.python.eager.wrap_function")
# TODO(mdan): Create a public API for this.
autograph_ctx = lazy_loader.LazyLoader(
    "autograph_ctx", globals(),
    "tensorflow.python.autograph.core.ag_ctx")
autograph = lazy_loader.LazyLoader(
    "autograph", globals(),
    "tensorflow.python.autograph.impl.api")

ops.NotDifferentiable("ReduceDataset")


# A constant that can be used to enable auto-tuning.
AUTOTUNE = -1
tf_export("data.experimental.AUTOTUNE").export_constant(__name__, "AUTOTUNE")


class AutotuneAlgorithm(enum.Enum):
  HILL_CLIMB = 0
  GRADIENT_DESCENT = 1


@tf_export("data.Dataset", v1=[])
@six.add_metaclass(abc.ABCMeta)
class DatasetV2(tracking_base.Trackable, composite_tensor.CompositeTensor):
  """Represents a potentially large set of elements.

  A `Dataset` can be used to represent an input pipeline as a
  collection of elements and a "logical plan" of transformations that act on
  those elements.

  A dataset contains elements that each have the same (nested) structure and the
  individual components of the structure can be of any type representable by
  `tf.TypeSpec`, including `tf.Tensor`, `tf.data.Dataset`, `tf.SparseTensor`,
  `tf.RaggedTensor`, or `tf.TensorArray`.

  Example elements:
  ```python
  # Integer element
  a = 1
  # Float element
  b = 2.0
  # Tuple element with 2 components
  c = (1, 2)
  # Dict element with 3 components
  d = {"a": (2, 2), "b": 3}
  # Element containing a dataset
  e = tf.data.Dataset.from_element(10)
  ```
  """

  def __init__(self, variant_tensor):
    """Creates a DatasetV2 object.

    This is a difference between DatasetV1 and DatasetV2. DatasetV1 does not
    take anything in its constructor whereas in the DatasetV2, we expect
    subclasses to create a variant_tensor and pass it in to the super() call.

    Args:
      variant_tensor: A DT_VARIANT tensor that represents the dataset.
    """
    self._variant_tensor_attr = variant_tensor
    weak_self = weakref.proxy(self)
    self._variant_tracker = self._track_trackable(
        _VariantTracker(
            self._variant_tensor,
            # _trace_variant_creation only works when executing eagerly, so we
            # don't want to run it immediately. We also want the _VariantTracker
            # to have a weak reference to the Dataset to avoid creating
            # reference cycles and making work for the garbage collector.
            lambda: weak_self._trace_variant_creation()()),  # pylint: disable=unnecessary-lambda,protected-access
        name="_variant_tracker")
    self._graph_attr = ops.get_default_graph()

  @property
  def _variant_tensor(self):
    return self._variant_tensor_attr

  @_variant_tensor.setter
  def _variant_tensor(self, _):
    raise ValueError("The _variant_tensor property is read-only")

  def _as_serialized_graph(self, stateful_whitelist=None):
    """Produces serialized graph representation of the dataset.

    Args:
      stateful_whitelist: Comma separated list of ops whose stateful attribute
        should be ignored during serialization.

    Returns:
      A scalar `tf.Tensor` of `tf.string` type, representing this dataset as a
      serialized graph.
    """
    if compat.forward_compatible(2019, 9, 10) or stateful_whitelist:
      return gen_dataset_ops.dataset_to_graph(self._variant_tensor,
                                              stateful_whitelist)
    else:
      return gen_dataset_ops.dataset_to_graph(self._variant_tensor)

  def _trace_variant_creation(self):
    """Traces a function which outputs a variant `tf.Tensor` for this dataset.

    Note that creating this function involves evaluating an op, and is currently
    only supported when executing eagerly.

    Returns:
      A zero-argument `ConcreteFunction` which outputs a variant `tf.Tensor`.
    """
    variant = self._variant_tensor
    if not isinstance(variant, ops.EagerTensor):
      raise NotImplementedError(
          "Can only export Datasets which were created executing eagerly. "
          "Please file a feature request if this is important to you.")
    with context.eager_mode(), ops.device("CPU"):
      graph_def = graph_pb2.GraphDef().FromString(
          self._as_serialized_graph().numpy())  # pylint: disable=protected-access
    output_node_name = None
    for node in graph_def.node:
      if node.op == "_Retval":
        if output_node_name is not None:
          raise AssertionError(
              "Found multiple return values from the dataset's graph, expected "
              "only one.")
        output_node_name, = node.input
    if output_node_name is None:
      raise AssertionError("Could not find the dataset's output node.")
    # Add functions used in this Dataset to the function's graph, since they
    # need to follow it around (and for example be added to a SavedModel which
    # references the dataset).
    variant_function = wrap_function.function_from_graph_def(
        graph_def, inputs=[], outputs=output_node_name + ":0")
    for used_function in self._functions():
      used_function.function.add_to_graph(variant_function.graph)
    return variant_function

  @abc.abstractmethod
  def _inputs(self):
    """Returns a list of the input datasets of the dataset."""

    raise NotImplementedError("Dataset._inputs")

  @property
  def _graph(self):
    return self._graph_attr

  @_graph.setter
  def _graph(self, _):
    raise ValueError("The _graph property is read-only")

  def _has_captured_ref(self):
    """Whether this dataset uses a function that captures ref variables.

    Returns:
      A boolean, which if true indicates that the dataset or one of its inputs
      uses a function that captures ref variables.
    """
    if context.executing_eagerly():
      # RefVariables are not supported in eager mode
      return False

    def is_tensor_or_parent_ref(tensor):
      if tensor.dtype._is_ref_dtype:  # pylint: disable=protected-access
        return True
      return any([is_tensor_or_parent_ref(x) for x in tensor.op.inputs])

    for fn in self._functions():
      if any([is_tensor_or_parent_ref(t) for t in fn.function.captured_inputs]):
        return True

    return any(
        [input_dataset._has_captured_ref() for input_dataset in self._inputs()])  # pylint: disable=protected-access

  # TODO(jsimsa): Change this to be the transitive closure of functions used
  # by this dataset and its inputs.
  def _functions(self):
    """Returns a list of functions associated with this dataset.

    Returns:
      A list of `StructuredFunctionWrapper` objects.
    """
    return []

  def options(self):
    """Returns the options for this dataset and its inputs.

    Returns:
      A `tf.data.Options` object representing the dataset options.
    """
    options = Options()
    for input_dataset in self._inputs():
      input_options = input_dataset.options()
      if input_options is not None:
        options = options.merge(input_options)
    return options

  def _apply_options(self):
    """Apply options, such as optimization configuration, to the dataset."""

    dataset = self
    options = self.options()
    if options.experimental_threading is not None:
      t_options = options.experimental_threading
      if t_options.max_intra_op_parallelism is not None:
        dataset = _MaxIntraOpParallelismDataset(
            dataset, t_options.max_intra_op_parallelism)
      if t_options.private_threadpool_size is not None:
        dataset = _PrivateThreadPoolDataset(dataset,
                                            t_options.private_threadpool_size)
    # pylint: disable=protected-access
    static_optimizations = options._static_optimizations()
    static_optimization_configs = options._static_optimization_configs()
    # pylint: enable=protected-access
    if static_optimizations:
      if self._has_captured_ref():
        warnings.warn(
            "tf.data static optimizations are not compatible with tf.Variable. "
            "The following optimizations will be disabled: %s. To enable "
            "optimizations, use resource variables instead by calling "
            "`tf.enable_resource_variables()` at the start of the program." %
            ", ".join(static_optimizations))
      else:
        dataset = _OptimizeDataset(dataset, static_optimizations,
                                   static_optimization_configs)

    autotune = True
    algorithm = AutotuneAlgorithm.HILL_CLIMB
    cpu_budget = 0  # Indicates that all CPU cores should be used.
    if options.experimental_optimization is not None:
      if options.experimental_optimization.autotune is False:  # pylint: disable=g-bool-id-comparison
        autotune = False
      if options.experimental_optimization.autotune_algorithm is not None:
        algorithm = options.experimental_optimization.autotune_algorithm
      if options.experimental_optimization.autotune_cpu_budget is not None:
        cpu_budget = options.experimental_optimization.autotune_cpu_budget

    if autotune:
      dataset = _ModelDataset(dataset, algorithm, cpu_budget)

    if options.experimental_stats and options.experimental_stats.aggregator:  # pylint: disable=line-too-long
      dataset = _SetStatsAggregatorDataset(  # pylint: disable=protected-access
          dataset, options.experimental_stats.aggregator,
          options.experimental_stats.prefix,
          options.experimental_stats.counter_prefix)
    return dataset

  def __iter__(self):
    """Creates an `Iterator` for enumerating the elements of this dataset.

    The returned iterator implements the Python iterator protocol and therefore
    can only be used in eager mode.

    Returns:
      An `Iterator` over the elements of this dataset.

    Raises:
      RuntimeError: If not inside of tf.function and not executing eagerly.
    """
    if (context.executing_eagerly()
        or ops.get_default_graph()._building_function):  # pylint: disable=protected-access
      return iterator_ops.IteratorV2(self)
    else:
      raise RuntimeError("__iter__() is only supported inside of tf.function "
                         "or when eager execution is enabled.")

  @abc.abstractproperty
  def element_spec(self):
    """The type specification of an element of this dataset.

    Returns:
      A nested structure of `tf.TypeSpec` objects matching the structure of an
      element of this dataset and specifying the type of individual components.
    """
    raise NotImplementedError("Dataset.element_spec")

  def __repr__(self):
    output_shapes = nest.map_structure(str, get_legacy_output_shapes(self))
    output_shapes = str(output_shapes).replace("'", "")
    output_types = nest.map_structure(repr, get_legacy_output_types(self))
    output_types = str(output_types).replace("'", "")
    return ("<%s shapes: %s, types: %s>" % (type(self).__name__, output_shapes,
                                            output_types))

  @property
  def _flat_shapes(self):
    """Returns a list `tf.TensorShapes`s for the element tensor representation.

    Returns:
      A list `tf.TensorShapes`s for the element tensor representation.
    """
    return structure.get_flat_tensor_shapes(self.element_spec)

  @property
  def _flat_types(self):
    """Returns a list `tf.DType`s for the element tensor representation.

    Returns:
      A list `tf.DType`s for the element tensor representation.
    """
    return structure.get_flat_tensor_types(self.element_spec)

  @property
  def _flat_structure(self):
    """Helper for setting `output_shapes` and `output_types` attrs of an op.

    Most dataset op constructors expect `output_shapes` and `output_types`
    arguments that represent the flattened structure of an element. This helper
    function generates these attrs as a keyword argument dictionary, allowing
    `Dataset._variant_tensor` implementations to pass `**self._flat_structure`
    to the op constructor.

    Returns:
      A dictionary of keyword arguments that can be passed to a dataset op
      constructor.
    """
    return {
        "output_shapes": self._flat_shapes,
        "output_types": self._flat_types,
    }

  @property
  def _type_spec(self):
    return DatasetSpec(self.element_spec)

  @staticmethod
  def from_tensors(tensors):
    """Creates a `Dataset` with a single element, comprising the given tensors.

    Note that if `tensors` contains a NumPy array, and eager execution is not
    enabled, the values will be embedded in the graph as one or more
    `tf.constant` operations. For large datasets (> 1 GB), this can waste
    memory and run into byte limits of graph serialization. If `tensors`
    contains one or more large NumPy arrays, consider the alternative described
    in [this
    guide](https://tensorflow.org/guide/datasets#consuming_numpy_arrays).

    Args:
      tensors: A dataset element.

    Returns:
      Dataset: A `Dataset`.
    """
    return TensorDataset(tensors)

  @staticmethod
  def from_tensor_slices(tensors):
    """Creates a `Dataset` whose elements are slices of the given tensors.

    Note that if `tensors` contains a NumPy array, and eager execution is not
    enabled, the values will be embedded in the graph as one or more
    `tf.constant` operations. For large datasets (> 1 GB), this can waste
    memory and run into byte limits of graph serialization. If `tensors`
    contains one or more large NumPy arrays, consider the alternative described
    in [this guide](
    https://tensorflow.org/guide/datasets#consuming_numpy_arrays).

    Args:
      tensors: A dataset element, with each component having the same size in
        the 0th dimension.

    Returns:
      Dataset: A `Dataset`.
    """
    return TensorSliceDataset(tensors)

  class _GeneratorState(object):
    """Stores outstanding iterators created from a Python generator.

    This class keeps track of potentially multiple iterators that may have
    been created from a generator, e.g. in the case that the dataset is
    repeated, or nested within a parallel computation.
    """

    def __init__(self, generator):
      self._generator = generator
      self._lock = threading.Lock()
      self._next_id = 0  # GUARDED_BY(self._lock)
      self._args = {}
      self._iterators = {}

    def get_next_id(self, *args):
      with self._lock:
        ret = self._next_id
        self._next_id += 1
      self._args[ret] = args
      # NOTE(mrry): Explicitly create an array of `np.int64` because implicit
      # casting in `py_func()` will create an array of `np.int32` on Windows,
      # leading to a runtime error.
      return np.array(ret, dtype=np.int64)

    def get_iterator(self, iterator_id):
      try:
        return self._iterators[iterator_id]
      except KeyError:
        iterator = iter(self._generator(*self._args.pop(iterator_id)))
        self._iterators[iterator_id] = iterator
        return iterator

    def iterator_completed(self, iterator_id):
      del self._iterators[iterator_id]

  @staticmethod
  def from_generator(generator, output_types, output_shapes=None, args=None):
    """Creates a `Dataset` whose elements are generated by `generator`.

    The `generator` argument must be a callable object that returns
    an object that supports the `iter()` protocol (e.g. a generator function).
    The elements generated by `generator` must be compatible with the given
    `output_types` and (optional) `output_shapes` arguments.

    For example:

    ```python
    import itertools
    tf.compat.v1.enable_eager_execution()

    def gen():
      for i in itertools.count(1):
        yield (i, [1] * i)

    ds = tf.data.Dataset.from_generator(
        gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))

    for value in ds.take(2):
      print value
    # (1, array([1]))
    # (2, array([1, 1]))
    ```

    NOTE: The current implementation of `Dataset.from_generator()` uses
    `tf.numpy_function` and inherits the same constraints. In particular, it
    requires the `Dataset`- and `Iterator`-related operations to be placed
    on a device in the same process as the Python program that called
    `Dataset.from_generator()`. The body of `generator` will not be
    serialized in a `GraphDef`, and you should not use this method if you
    need to serialize your model and restore it in a different environment.

    NOTE: If `generator` depends on mutable global variables or other external
    state, be aware that the runtime may invoke `generator` multiple times
    (in order to support repeating the `Dataset`) and at any time
    between the call to `Dataset.from_generator()` and the production of the
    first element from the generator. Mutating global variables or external
    state can cause undefined behavior, and we recommend that you explicitly
    cache any external state in `generator` before calling
    `Dataset.from_generator()`.

    Args:
      generator: A callable object that returns an object that supports the
        `iter()` protocol. If `args` is not specified, `generator` must take no
        arguments; otherwise it must take as many arguments as there are values
        in `args`.
      output_types: A nested structure of `tf.DType` objects corresponding to
        each component of an element yielded by `generator`.
      output_shapes: (Optional.) A nested structure of `tf.TensorShape` objects
        corresponding to each component of an element yielded by `generator`.
      args: (Optional.) A tuple of `tf.Tensor` objects that will be evaluated
        and passed to `generator` as NumPy-array arguments.

    Returns:
      Dataset: A `Dataset`.
    """
    if not callable(generator):
      raise TypeError("`generator` must be callable.")
    if output_shapes is None:
      output_shapes = nest.map_structure(
          lambda _: tensor_shape.TensorShape(None), output_types)
    else:
      output_shapes = nest.map_structure_up_to(
          output_types, tensor_shape.as_shape, output_shapes)
    if args is None:
      args = ()
    else:
      args = tuple(ops.convert_n_to_tensor(args, name="args"))

    flattened_types = [dtypes.as_dtype(dt) for dt in nest.flatten(output_types)]
    flattened_shapes = nest.flatten(output_shapes)

    generator_state = DatasetV2._GeneratorState(generator)

    def get_iterator_id_fn(unused_dummy):
      """Creates a unique `iterator_id` for each pass over the dataset.

      The returned `iterator_id` disambiguates between multiple concurrently
      existing iterators.

      Args:
        unused_dummy: Ignored value.

      Returns:
        A `tf.int64` tensor whose value uniquely identifies an iterator in
        `generator_state`.
      """
      return script_ops.numpy_function(generator_state.get_next_id, args,
                                       dtypes.int64)

    def generator_next_fn(iterator_id_t):
      """Generates the next element from iterator with ID `iterator_id_t`.

      We map this function across an infinite repetition of the
      `iterator_id_t`, and raise `StopIteration` to terminate the iteration.

      Args:
        iterator_id_t: A `tf.int64` tensor whose value uniquely identifies the
          iterator in `generator_state` from which to generate an element.

      Returns:
        The next element to generate from the iterator.
      """

      def generator_py_func(iterator_id):
        """A `py_func` that will be called to invoke the iterator."""
        # `next()` raises `StopIteration` when there are no more
        # elements remaining to be generated.
        values = next(generator_state.get_iterator(iterator_id))

        # Use the same _convert function from the py_func() implementation to
        # convert the returned values to arrays early, so that we can inspect
        # their values.
        try:
          flattened_values = nest.flatten_up_to(output_types, values)
        except (TypeError, ValueError):
          raise TypeError(
              "`generator` yielded an element that did not match the expected "
              "structure. The expected structure was %s, but the yielded "
              "element was %s." % (output_types, values))
        ret_arrays = []
        for ret, dtype in zip(flattened_values, flattened_types):
          try:
            ret_arrays.append(script_ops.FuncRegistry._convert(  # pylint: disable=protected-access
                ret, dtype=dtype.as_numpy_dtype))
          except (TypeError, ValueError):
            raise TypeError(
                "`generator` yielded an element that could not be converted to "
                "the expected type. The expected type was %s, but the yielded "
                "element was %s." % (dtype.name, ret))

        # Additional type and shape checking to ensure that the components
        # of the generated element match the `output_types` and `output_shapes`
        # arguments.
        for (ret_array, expected_dtype, expected_shape) in zip(
            ret_arrays, flattened_types, flattened_shapes):
          if ret_array.dtype != expected_dtype.as_numpy_dtype:
            raise TypeError(
                "`generator` yielded an element of type %s where an element "
                "of type %s was expected." % (ret_array.dtype,
                                              expected_dtype.as_numpy_dtype))
          if not expected_shape.is_compatible_with(ret_array.shape):
            raise ValueError(
                "`generator` yielded an element of shape %s where an element "
                "of shape %s was expected." % (ret_array.shape, expected_shape))

        return ret_arrays

      flat_values = script_ops.numpy_function(generator_py_func,
                                              [iterator_id_t], flattened_types)

      # The `py_func()` op drops the inferred shapes, so we add them back in
      # here.
      if output_shapes is not None:
        for ret_t, shape in zip(flat_values, flattened_shapes):
          ret_t.set_shape(shape)

      return nest.pack_sequence_as(output_types, flat_values)

    def finalize_fn(iterator_id_t):
      """Releases host-side state for the iterator with ID `iterator_id_t`."""

      def finalize_py_func(iterator_id):
        generator_state.iterator_completed(iterator_id)
        # We return a dummy value so that the `finalize_fn` has a valid
        # signature.
        # NOTE(mrry): Explicitly create an array of `np.int64` because implicit
        # casting in `py_func()` will create an array of `np.int32` on Windows,
        # leading to a runtime error.
        return np.array(0, dtype=np.int64)

      return script_ops.numpy_function(finalize_py_func, [iterator_id_t],
                                       dtypes.int64)

    # This function associates each traversal of `generator` with a unique
    # iterator ID.
    def flat_map_fn(dummy_arg):
      # The `get_iterator_id_fn` gets a unique ID for the current instance of
      # of the generator.
      # The `generator_next_fn` gets the next element from the iterator with the
      # given ID, and raises StopIteration when that iterator contains no
      # more elements.
      return _GeneratorDataset(dummy_arg, get_iterator_id_fn, generator_next_fn,
                               finalize_fn)

    # A single-element dataset that, each time it is evaluated, contains a
    # freshly-generated and unique (for the returned dataset) int64
    # ID that will be used to identify the appropriate Python state, which
    # is encapsulated in `generator_state`, and captured in
    # `get_iterator_id_map_fn`.
    dummy = 0
    id_dataset = Dataset.from_tensors(dummy)

    # A dataset that contains all of the elements generated by a
    # single iterator created from `generator`, identified by the
    # iterator ID contained in `id_dataset`. Lifting the iteration
    # into a flat_map here enables multiple repetitions and/or nested
    # versions of the returned dataset to be created, because it forces
    # the generation of a new ID for each version.
    return id_dataset.flat_map(flat_map_fn)

  @staticmethod
  def range(*args):
    """Creates a `Dataset` of a step-separated range of values.

    For example:

    ```python
    Dataset.range(5) == [0, 1, 2, 3, 4]
    Dataset.range(2, 5) == [2, 3, 4]
    Dataset.range(1, 5, 2) == [1, 3]
    Dataset.range(1, 5, -2) == []
    Dataset.range(5, 1) == []
    Dataset.range(5, 1, -2) == [5, 3]
    ```

    Args:
      *args: follows the same semantics as python's xrange.
        len(args) == 1 -> start = 0, stop = args[0], step = 1
        len(args) == 2 -> start = args[0], stop = args[1], step = 1
        len(args) == 3 -> start = args[0], stop = args[1, stop = args[2]

    Returns:
      Dataset: A `RangeDataset`.

    Raises:
      ValueError: if len(args) == 0.
    """
    return RangeDataset(*args)

  @staticmethod
  def zip(datasets):
    """Creates a `Dataset` by zipping together the given datasets.

    This method has similar semantics to the built-in `zip()` function
    in Python, with the main difference being that the `datasets`
    argument can be an arbitrary nested structure of `Dataset` objects.
    For example:

    ```python
    a = Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
    b = Dataset.range(4, 7)  # ==> [ 4, 5, 6 ]
    c = Dataset.range(7, 13).batch(2)  # ==> [ [7, 8], [9, 10], [11, 12] ]
    d = Dataset.range(13, 15)  # ==> [ 13, 14 ]

    # The nested structure of the `datasets` argument determines the
    # structure of elements in the resulting dataset.
    Dataset.zip((a, b))  # ==> [ (1, 4), (2, 5), (3, 6) ]
    Dataset.zip((b, a))  # ==> [ (4, 1), (5, 2), (6, 3) ]

    # The `datasets` argument may contain an arbitrary number of
    # datasets.
    Dataset.zip((a, b, c))  # ==> [ (1, 4, [7, 8]),
                            #       (2, 5, [9, 10]),
                            #       (3, 6, [11, 12]) ]

    # The number of elements in the resulting dataset is the same as
    # the size of the smallest dataset in `datasets`.
    Dataset.zip((a, d))  # ==> [ (1, 13), (2, 14) ]
    ```

    Args:
      datasets: A nested structure of datasets.

    Returns:
      Dataset: A `Dataset`.
    """
    return ZipDataset(datasets)

  def concatenate(self, dataset):
    """Creates a `Dataset` by concatenating the given dataset with this dataset.

    ```python
    a = Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
    b = Dataset.range(4, 8)  # ==> [ 4, 5, 6, 7 ]

    # The input dataset and dataset to be concatenated should have the same
    # nested structures and output types.
    # c = Dataset.range(8, 14).batch(2)  # ==> [ [8, 9], [10, 11], [12, 13] ]
    # d = Dataset.from_tensor_slices([14.0, 15.0, 16.0])
    # a.concatenate(c) and a.concatenate(d) would result in error.

    a.concatenate(b)  # ==> [ 1, 2, 3, 4, 5, 6, 7 ]
    ```

    Args:
      dataset: `Dataset` to be concatenated.

    Returns:
      Dataset: A `Dataset`.
    """
    return ConcatenateDataset(self, dataset)

  def prefetch(self, buffer_size):
    """Creates a `Dataset` that prefetches elements from this dataset.

    Note: Like other `Dataset` methods, prefetch operates on the
    elements of the input dataset. It has no concept of examples vs. batches.
    `examples.prefetch(2)` will prefetch two elements (2 examples),
    while `examples.batch(20).prefetch(2)` will prefetch 2 elements
    (2 batches, of 20 examples each).

    Args:
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the maximum
        number of elements that will be buffered when prefetching.

    Returns:
      Dataset: A `Dataset`.
    """
    return PrefetchDataset(self, buffer_size)

  @staticmethod
  def list_files(file_pattern, shuffle=None, seed=None):
    """A dataset of all files matching one or more glob patterns.

    NOTE: The default behavior of this method is to return filenames in
    a non-deterministic random shuffled order. Pass a `seed` or `shuffle=False`
    to get results in a deterministic order.

    Example:
      If we had the following files on our filesystem:
        - /path/to/dir/a.txt
        - /path/to/dir/b.py
        - /path/to/dir/c.py
      If we pass "/path/to/dir/*.py" as the directory, the dataset
      would produce:
        - /path/to/dir/b.py
        - /path/to/dir/c.py

    Args:
      file_pattern: A string, a list of strings, or a `tf.Tensor` of string type
        (scalar or vector), representing the filename glob (i.e. shell wildcard)
        pattern(s) that will be matched.
      shuffle: (Optional.) If `True`, the file names will be shuffled randomly.
        Defaults to `True`.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
        seed that will be used to create the distribution. See
        `tf.compat.v1.set_random_seed` for behavior.

    Returns:
     Dataset: A `Dataset` of strings corresponding to file names.
    """
    with ops.name_scope("list_files"):
      if shuffle is None:
        shuffle = True
      file_pattern = ops.convert_to_tensor(
          file_pattern, dtype=dtypes.string, name="file_pattern")
      matching_files = gen_io_ops.matching_files(file_pattern)

      # Raise an exception if `file_pattern` does not match any files.
      condition = math_ops.greater(array_ops.shape(matching_files)[0], 0,
                                   name="match_not_empty")

      message = math_ops.add(
          "No files matched pattern: ",
          string_ops.reduce_join(file_pattern, separator=", "), name="message")

      assert_not_empty = control_flow_ops.Assert(
          condition, [message], summarize=1, name="assert_not_empty")
      with ops.control_dependencies([assert_not_empty]):
        matching_files = array_ops.identity(matching_files)

      dataset = Dataset.from_tensor_slices(matching_files)
      if shuffle:
        # NOTE(mrry): The shuffle buffer size must be greater than zero, but the
        # list of files might be empty.
        buffer_size = math_ops.maximum(
            array_ops.shape(matching_files, out_type=dtypes.int64)[0], 1)
        dataset = dataset.shuffle(buffer_size, seed=seed)
      return dataset

  def repeat(self, count=None):
    """Repeats this dataset `count` times.

    NOTE: If this dataset is a function of global state (e.g. a random number
    generator), then different repetitions may produce different elements.

    Args:
      count: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        number of times the dataset should be repeated. The default behavior (if
        `count` is `None` or `-1`) is for the dataset be repeated indefinitely.

    Returns:
      Dataset: A `Dataset`.
    """
    return RepeatDataset(self, count)

  def enumerate(self, start=0):
    """Enumerates the elements of this dataset.

    It is similar to python's `enumerate`.

    For example:

    ```python
    # NOTE: The following examples use `{ ... }` to represent the
    # contents of a dataset.
    a = { 1, 2, 3 }
    b = { (7, 8), (9, 10) }

    # The nested structure of the `datasets` argument determines the
    # structure of elements in the resulting dataset.
    a.enumerate(start=5)) == { (5, 1), (6, 2), (7, 3) }
    b.enumerate() == { (0, (7, 8)), (1, (9, 10)) }
    ```

    Args:
      start: A `tf.int64` scalar `tf.Tensor`, representing the start value for
        enumeration.

    Returns:
      Dataset: A `Dataset`.
    """

    max_value = np.iinfo(dtypes.int64.as_numpy_dtype).max
    return Dataset.zip((Dataset.range(start, max_value), self))

  def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None):
    """Randomly shuffles the elements of this dataset.

    This dataset fills a buffer with `buffer_size` elements, then randomly
    samples elements from this buffer, replacing the selected elements with new
    elements. For perfect shuffling, a buffer size greater than or equal to the
    full size of the dataset is required.

    For instance, if your dataset contains 10,000 elements but `buffer_size` is
    set to 1,000, then `shuffle` will initially select a random element from
    only the first 1,000 elements in the buffer. Once an element is selected,
    its space in the buffer is replaced by the next (i.e. 1,001-st) element,
    maintaining the 1,000 element buffer.

    Args:
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements from this dataset from which the new dataset will sample.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
        seed that will be used to create the distribution. See
        `tf.compat.v1.set_random_seed` for behavior.
      reshuffle_each_iteration: (Optional.) A boolean, which if true indicates
        that the dataset should be pseudorandomly reshuffled each time it is
        iterated over. (Defaults to `True`.)

    Returns:
      Dataset: A `Dataset`.
    """
    return ShuffleDataset(self, buffer_size, seed, reshuffle_each_iteration)

  def cache(self, filename=""):
    """Caches the elements in this dataset.

    Args:
      filename: A `tf.string` scalar `tf.Tensor`, representing the name of a
        directory on the filesystem to use for caching elements in this Dataset.
        If a filename is not provided, the dataset will be cached in memory.

    Returns:
      Dataset: A `Dataset`.
    """
    return CacheDataset(self, filename)

  def take(self, count):
    """Creates a `Dataset` with at most `count` elements from this dataset.

    Args:
      count: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements of this dataset that should be taken to form the new dataset.
        If `count` is -1, or if `count` is greater than the size of this
        dataset, the new dataset will contain all elements of this dataset.

    Returns:
      Dataset: A `Dataset`.
    """
    return TakeDataset(self, count)

  def skip(self, count):
    """Creates a `Dataset` that skips `count` elements from this dataset.

    Args:
      count: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements of this dataset that should be skipped to form the new dataset.
        If `count` is greater than the size of this dataset, the new dataset
        will contain no elements.  If `count` is -1, skips the entire dataset.

    Returns:
      Dataset: A `Dataset`.
    """
    return SkipDataset(self, count)

  def shard(self, num_shards, index):
    """Creates a `Dataset` that includes only 1/`num_shards` of this dataset.

    This dataset operator is very useful when running distributed training, as
    it allows each worker to read a unique subset.

    When reading a single input file, you can skip elements as follows:

    ```python
    d = tf.data.TFRecordDataset(input_file)
    d = d.shard(num_workers, worker_index)
    d = d.repeat(num_epochs)
    d = d.shuffle(shuffle_buffer_size)
    d = d.map(parser_fn, num_parallel_calls=num_map_threads)
    ```

    Important caveats:

    - Be sure to shard before you use any randomizing operator (such as
      shuffle).
    - Generally it is best if the shard operator is used early in the dataset
      pipeline. For example, when reading from a set of TFRecord files, shard
      before converting the dataset to input samples. This avoids reading every
      file on every worker. The following is an example of an efficient
      sharding strategy within a complete pipeline:

    ```python
    d = Dataset.list_files(pattern)
    d = d.shard(num_workers, worker_index)
    d = d.repeat(num_epochs)
    d = d.shuffle(shuffle_buffer_size)
    d = d.interleave(tf.data.TFRecordDataset,
                     cycle_length=num_readers, block_length=1)
    d = d.map(parser_fn, num_parallel_calls=num_map_threads)
    ```

    Args:
      num_shards: A `tf.int64` scalar `tf.Tensor`, representing the number of
        shards operating in parallel.
      index: A `tf.int64` scalar `tf.Tensor`, representing the worker index.

    Returns:
      Dataset: A `Dataset`.

    Raises:
      InvalidArgumentError: if `num_shards` or `index` are illegal values.
        Note: error checking is done on a best-effort basis, and errors aren't
        guaranteed to be caught upon dataset creation. (e.g. providing in a
        placeholder tensor bypasses the early checking, and will instead result
        in an error during a session.run call.)
    """
    return ShardDataset(self, num_shards, index)

  def batch(self, batch_size, drop_remainder=False):
    """Combines consecutive elements of this dataset into batches.

    The components of the resulting element will have an additional outer
    dimension, which will be `batch_size` (or `N % batch_size` for the last
    element if `batch_size` does not divide the number of input elements `N`
    evenly and `drop_remainder` is `False`). If your program depends on the
    batches having the same outer dimension, you should set the `drop_remainder`
    argument to `True` to prevent the smaller batch from being produced.

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.

    Returns:
      Dataset: A `Dataset`.
    """
    return BatchDataset(self, batch_size, drop_remainder)

  def padded_batch(self,
                   batch_size,
                   padded_shapes,
                   padding_values=None,
                   drop_remainder=False):
    """Combines consecutive elements of this dataset into padded batches.

    This transformation combines multiple consecutive elements of the input
    dataset into a single element.

    Like `tf.data.Dataset.batch`, the components of the resulting element will
    have an additional outer dimension, which will be `batch_size` (or
    `N % batch_size` for the last element if `batch_size` does not divide the
    number of input elements `N` evenly and `drop_remainder` is `False`). If
    your program depends on the batches having the same outer dimension, you
    should set the `drop_remainder` argument to `True` to prevent the smaller
    batch from being produced.

    Unlike `tf.data.Dataset.batch`, the input elements to be batched may have
    different shapes, and this transformation will pad each component to the
    respective shape in `padding_shapes`. The `padding_shapes` argument
    determines the resulting shape for each dimension of each component in an
    output element:

    * If the dimension is a constant (e.g. `tf.compat.v1.Dimension(37)`), the
    component
      will be padded out to that length in that dimension.
    * If the dimension is unknown (e.g. `tf.compat.v1.Dimension(None)`), the
    component
      will be padded out to the maximum length of all elements in that
      dimension.

    See also `tf.data.experimental.dense_to_sparse_batch`, which combines
    elements that may have different shapes into a `tf.SparseTensor`.

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      padded_shapes: A nested structure of `tf.TensorShape` or `tf.int64` vector
        tensor-like objects representing the shape to which the respective
        component of each input element should be padded prior to batching. Any
        unknown dimensions (e.g. `tf.compat.v1.Dimension(None)` in a
        `tf.TensorShape` or `-1` in a tensor-like object) will be padded to the
        maximum size of that dimension in each batch.
      padding_values: (Optional.) A nested structure of scalar-shaped
        `tf.Tensor`, representing the padding values to use for the respective
        components.  Defaults are `0` for numeric types and the empty string for
        string types.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.

    Returns:
      Dataset: A `Dataset`.
    """
    return PaddedBatchDataset(self, batch_size, padded_shapes, padding_values,
                              drop_remainder)

  def map(self, map_func, num_parallel_calls=None):
    """Maps `map_func` across the elements of this dataset.

    This transformation applies `map_func` to each element of this dataset, and
    returns a new dataset containing the transformed elements, in the same
    order as they appeared in the input.

    For example:

    ```python
    a = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]

    a.map(lambda x: x + 1)  # ==> [ 2, 3, 4, 5, 6 ]
    ```

    The input signature of `map_func` is determined by the structure of each
    element in this dataset. For example:

    ```python
    # NOTE: The following examples use `{ ... }` to represent the
    # contents of a dataset.
    # Each element is a `tf.Tensor` object.
    a = { 1, 2, 3, 4, 5 }
    # `map_func` takes a single argument of type `tf.Tensor` with the same
    # shape and dtype.
    result = a.map(lambda x: ...)

    # Each element is a tuple containing two `tf.Tensor` objects.
    b = { (1, "foo"), (2, "bar"), (3, "baz") }
    # `map_func` takes two arguments of type `tf.Tensor`.
    result = b.map(lambda x_int, y_str: ...)

    # Each element is a dictionary mapping strings to `tf.Tensor` objects.
    c = { {"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}, {"a": 3, "b": "baz"} }
    # `map_func` takes a single argument of type `dict` with the same keys as
    # the elements.
    result = c.map(lambda d: ...)
    ```

    The value or values returned by `map_func` determine the structure of each
    element in the returned dataset.

    ```python
    # `map_func` returns a scalar `tf.Tensor` of type `tf.float32`.
    def f(...):
      return tf.constant(37.0)
    result = dataset.map(f)
    result.output_classes == tf.Tensor
    result.output_types == tf.float32
    result.output_shapes == []  # scalar

    # `map_func` returns two `tf.Tensor` objects.
    def g(...):
      return tf.constant(37.0), tf.constant(["Foo", "Bar", "Baz"])
    result = dataset.map(g)
    result.output_classes == (tf.Tensor, tf.Tensor)
    result.output_types == (tf.float32, tf.string)
    result.output_shapes == ([], [3])

    # Python primitives, lists, and NumPy arrays are implicitly converted to
    # `tf.Tensor`.
    def h(...):
      return 37.0, ["Foo", "Bar", "Baz"], np.array([1.0, 2.0] dtype=np.float64)
    result = dataset.map(h)
    result.output_classes == (tf.Tensor, tf.Tensor, tf.Tensor)
    result.output_types == (tf.float32, tf.string, tf.float64)
    result.output_shapes == ([], [3], [2])

    # `map_func` can return nested structures.
    def i(...):
      return {"a": 37.0, "b": [42, 16]}, "foo"
    result.output_classes == ({"a": tf.Tensor, "b": tf.Tensor}, tf.Tensor)
    result.output_types == ({"a": tf.float32, "b": tf.int32}, tf.string)
    result.output_shapes == ({"a": [], "b": [2]}, [])
    ```

    `map_func` can accept as arguments and return any type of dataset element.

    Note that irrespective of the context in which `map_func` is defined (eager
    vs. graph), tf.data traces the function and executes it as a graph. To use
    Python code inside of the function you have two options:

    1) Rely on AutoGraph to convert Python code into an equivalent graph
    computation. The downside of this approach is that AutoGraph can convert
    some but not all Python code.

    2) Use `tf.py_function`, which allows you to write arbitrary Python code but
    will generally result in worse performance than 1). For example:

    ```python
    d = tf.data.Dataset.from_tensor_slices(['hello', 'world'])

    # transform a string tensor to upper case string using a Python function
    def upper_case_fn(t: tf.Tensor) -> str:
        return t.numpy().decode('utf-8').upper()

    d.map(lambda x: tf.py_function(func=upper_case_fn,
          inp=[x], Tout=tf.string))  # ==> [ "HELLO", "WORLD" ]
    ```

    Args:
      map_func: A function mapping a dataset element to another dataset element.
      num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
        representing the number elements to process asynchronously in parallel.
        If not specified, elements will be processed sequentially. If the value
        `tf.data.experimental.AUTOTUNE` is used, then the number of parallel
        calls is set dynamically based on available CPU.

    Returns:
      Dataset: A `Dataset`.
    """
    if num_parallel_calls is None:
      return MapDataset(self, map_func, preserve_cardinality=True)
    else:
      return ParallelMapDataset(
          self, map_func, num_parallel_calls, preserve_cardinality=True)

  def flat_map(self, map_func):
    """Maps `map_func` across this dataset and flattens the result.

    Use `flat_map` if you want to make sure that the order of your dataset
    stays the same. For example, to flatten a dataset of batches into a
    dataset of their elements:

    ```python
    a = Dataset.from_tensor_slices([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ])

    a.flat_map(lambda x: Dataset.from_tensor_slices(x + 1)) # ==>
    #  [ 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
    ```

    `tf.data.Dataset.interleave()` is a generalization of `flat_map`, since
    `flat_map` produces the same output as
    `tf.data.Dataset.interleave(cycle_length=1)`

    Args:
      map_func: A function mapping a dataset element to a dataset.

    Returns:
      Dataset: A `Dataset`.
    """
    return FlatMapDataset(self, map_func)

  def interleave(self,
                 map_func,
                 cycle_length=AUTOTUNE,
                 block_length=1,
                 num_parallel_calls=None):
    """Maps `map_func` across this dataset, and interleaves the results.

    For example, you can use `Dataset.interleave()` to process many input files
    concurrently:

    ```python
    # Preprocess 4 files concurrently, and interleave blocks of 16 records from
    # each file.
    filenames = ["/var/data/file1.txt", "/var/data/file2.txt", ...]
    dataset = (Dataset.from_tensor_slices(filenames)
               .interleave(lambda x:
                   TextLineDataset(x).map(parse_fn, num_parallel_calls=1),
                   cycle_length=4, block_length=16))
    ```

    The `cycle_length` and `block_length` arguments control the order in which
    elements are produced. `cycle_length` controls the number of input elements
    that are processed concurrently. If you set `cycle_length` to 1, this
    transformation will handle one input element at a time, and will produce
    identical results to `tf.data.Dataset.flat_map`. In general,
    this transformation will apply `map_func` to `cycle_length` input elements,
    open iterators on the returned `Dataset` objects, and cycle through them
    producing `block_length` consecutive elements from each iterator, and
    consuming the next input element each time it reaches the end of an
    iterator.

    For example:

    ```python
    a = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]

    # NOTE: New lines indicate "block" boundaries.
    a.interleave(lambda x: Dataset.from_tensors(x).repeat(6),
                cycle_length=2, block_length=4)  # ==> [1, 1, 1, 1,
                                                 #      2, 2, 2, 2,
                                                 #      1, 1,
                                                 #      2, 2,
                                                 #      3, 3, 3, 3,
                                                 #      4, 4, 4, 4,
                                                 #      3, 3,
                                                 #      4, 4,
                                                 #      5, 5, 5, 5,
                                                 #      5, 5]
    ```

    NOTE: The order of elements yielded by this transformation is
    deterministic, as long as `map_func` is a pure function. If
    `map_func` contains any stateful operations, the order in which
    that state is accessed is undefined.

    Args:
      map_func: A function mapping a dataset element to a dataset.
      cycle_length: (Optional.) The number of input elements that will be
        processed concurrently. If not specified, the value will be derived from
        the number of available CPU cores. If the `num_parallel_calls` argument
        is set to `tf.data.experimental.AUTOTUNE`, the `cycle_length` argument
        also identifies the maximum degree of parallelism.
      block_length: (Optional.) The number of consecutive elements to produce
        from each input element before cycling to another input element.
      num_parallel_calls: (Optional.) If specified, the implementation creates a
        threadpool, which is used to fetch inputs from cycle elements
        asynchronously and in parallel. The default behavior is to fetch inputs
        from cycle elements synchronously with no parallelism. If the value
        `tf.data.experimental.AUTOTUNE` is used, then the number of parallel
        calls is set dynamically based on available CPU.

    Returns:
      Dataset: A `Dataset`.
    """
    if num_parallel_calls is None:
      return InterleaveDataset(self, map_func, cycle_length, block_length)
    else:
      return ParallelInterleaveDataset(self, map_func, cycle_length,
                                       block_length, num_parallel_calls)

  def filter(self, predicate):
    """Filters this dataset according to `predicate`.

    ```python
    d = tf.data.Dataset.from_tensor_slices([1, 2, 3])

    d = d.filter(lambda x: x < 3)  # ==> [1, 2]

    # `tf.math.equal(x, y)` is required for equality comparison
    def filter_fn(x):
      return tf.math.equal(x, 1)

    d = d.filter(filter_fn)  # ==> [1]
    ```

    Args:
      predicate: A function mapping a dataset element to a boolean.

    Returns:
      Dataset: The `Dataset` containing the elements of this dataset for which
          `predicate` is `True`.
    """
    return FilterDataset(self, predicate)

  def apply(self, transformation_func):
    """Applies a transformation function to this dataset.

    `apply` enables chaining of custom `Dataset` transformations, which are
    represented as functions that take one `Dataset` argument and return a
    transformed `Dataset`.

    For example:

    ```
    dataset = (dataset.map(lambda x: x ** 2)
               .apply(group_by_window(key_func, reduce_func, window_size))
               .map(lambda x: x ** 3))
    ```

    Args:
      transformation_func: A function that takes one `Dataset` argument and
        returns a `Dataset`.

    Returns:
      Dataset: The `Dataset` returned by applying `transformation_func` to this
          dataset.
    """
    dataset = transformation_func(self)
    if not isinstance(dataset, DatasetV2):
      raise TypeError(
          "`transformation_func` must return a Dataset. Got {}.".format(
              dataset))
    dataset._input_datasets = [self]  # pylint: disable=protected-access
    return dataset

  def window(self, size, shift=None, stride=1, drop_remainder=False):
    """Combines (nests of) input elements into a dataset of (nests of) windows.

    A "window" is a finite dataset of flat elements of size `size` (or possibly
    fewer if there are not enough input elements to fill the window and
    `drop_remainder` evaluates to false).

    The `stride` argument determines the stride of the input elements, and the
    `shift` argument determines the shift of the window.

    For example, letting {...} to represent a Dataset:

    - `tf.data.Dataset.range(7).window(2)` produces
      `{{0, 1}, {2, 3}, {4, 5}, {6}}`
    - `tf.data.Dataset.range(7).window(3, 2, 1, True)` produces
      `{{0, 1, 2}, {2, 3, 4}, {4, 5, 6}}`
    - `tf.data.Dataset.range(7).window(3, 1, 2, True)` produces
      `{{0, 2, 4}, {1, 3, 5}, {2, 4, 6}}`

    Note that when the `window` transformation is applied to a dataset of
    nested elements, it produces a dataset of nested windows.

    For example:

    - `tf.data.Dataset.from_tensor_slices((range(4), range(4))).window(2)`
      produces `{({0, 1}, {0, 1}), ({2, 3}, {2, 3})}`
    - `tf.data.Dataset.from_tensor_slices({"a": range(4)}).window(2)`
      produces `{{"a": {0, 1}}, {"a": {2, 3}}}`

    Args:
      size: A `tf.int64` scalar `tf.Tensor`, representing the number of elements
        of the input dataset to combine into a window.
      shift: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        forward shift of the sliding window in each iteration. Defaults to
        `size`.
      stride: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        stride of the input elements in the sliding window.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether a window should be dropped in case its size is smaller than
        `window_size`.

    Returns:
      Dataset: A `Dataset` of (nests of) windows -- a finite datasets of flat
        elements created from the (nests of) input elements.

    """
    if shift is None:
      shift = size
    return WindowDataset(self, size, shift, stride, drop_remainder)

  def reduce(self, initial_state, reduce_func):
    """Reduces the input dataset to a single element.

    The transformation calls `reduce_func` successively on every element of
    the input dataset until the dataset is exhausted, aggregating information in
    its internal state. The `initial_state` argument is used for the initial
    state and the final state is returned as the result.

    For example:
    - `tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, _: x + 1)`
      produces `5`
    - `tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, y: x + y)`
      produces `10`

    Args:
      initial_state: An element representing the initial state of the
        transformation.
      reduce_func: A function that maps `(old_state, input_element)` to
        `new_state`. It must take two arguments and return a new element
        The structure of `new_state` must match the structure of
        `initial_state`.

    Returns:
      A dataset element corresponding to the final state of the transformation.

    """

    with ops.name_scope("initial_state"):
      initial_state = structure.normalize_element(initial_state)
    state_structure = structure.type_spec_from_value(initial_state)

    # Iteratively rerun the reduce function until reaching a fixed point on
    # `state_structure`.
    need_to_rerun = True
    while need_to_rerun:

      wrapped_func = StructuredFunctionWrapper(
          reduce_func,
          "reduce()",
          input_structure=(state_structure, self.element_spec),
          add_to_graph=False)

      # Extract and validate class information from the returned values.
      output_classes = wrapped_func.output_classes
      state_classes = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
          state_structure)
      for new_state_class, state_class in zip(
          nest.flatten(output_classes), nest.flatten(state_classes)):
        if not issubclass(new_state_class, state_class):
          raise TypeError(
              "The element classes for the new state must match the initial "
              "state. Expected %s; got %s." %
              (state_classes, wrapped_func.output_classes))

      # Extract and validate type information from the returned values.
      output_types = wrapped_func.output_types
      state_types = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
          state_structure)
      for new_state_type, state_type in zip(
          nest.flatten(output_types), nest.flatten(state_types)):
        if new_state_type != state_type:
          raise TypeError(
              "The element types for the new state must match the initial "
              "state. Expected %s; got %s." %
              (state_types, wrapped_func.output_types))

      # Extract shape information from the returned values.
      output_shapes = wrapped_func.output_shapes
      state_shapes = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
          state_structure)
      flat_state_shapes = nest.flatten(state_shapes)
      flat_new_state_shapes = nest.flatten(output_shapes)
      weakened_state_shapes = [
          original.most_specific_compatible_shape(new)
          for original, new in zip(flat_state_shapes, flat_new_state_shapes)
      ]

      need_to_rerun = False
      for original_shape, weakened_shape in zip(flat_state_shapes,
                                                weakened_state_shapes):
        if original_shape.ndims is not None and (
            weakened_shape.ndims is None or
            original_shape.as_list() != weakened_shape.as_list()):
          need_to_rerun = True
          break

      if need_to_rerun:
        # TODO(b/110122868): Support a "most specific compatible structure"
        # method for combining structures, to avoid using legacy structures
        # here.
        state_structure = structure.convert_legacy_structure(
            state_types,
            nest.pack_sequence_as(state_shapes, weakened_state_shapes),
            state_classes)

    reduce_func = wrapped_func.function
    reduce_func.add_to_graph(ops.get_default_graph())

    # pylint: disable=protected-access
    return structure.from_compatible_tensor_list(
        state_structure,
        gen_dataset_ops.reduce_dataset(
            self._variant_tensor,
            structure.to_tensor_list(state_structure, initial_state),
            reduce_func.captured_inputs,
            f=reduce_func,
            output_shapes=structure.get_flat_tensor_shapes(state_structure),
            output_types=structure.get_flat_tensor_types(state_structure)))

  def unbatch(self):
    """Splits elements of a dataset into multiple elements.

    For example, if elements of the dataset are shaped `[B, a0, a1, ...]`,
    where `B` may vary for each input element, then for each element in the
    dataset, the unbatched dataset will contain `B` consecutive elements
    of shape `[a0, a1, ...]`.

    ```python
    # NOTE: The following example uses `{ ... }` to represent the contents
    # of a dataset.
    ds = { ['a', 'b', 'c'], ['a', 'b'], ['a', 'b', 'c', 'd'] }

    ds.unbatch() == {'a', 'b', 'c', 'a', 'b', 'a', 'b', 'c', 'd'}
    ```

    Returns:
      A `Dataset` transformation function, which can be passed to
      `tf.data.Dataset.apply`.
    """

    # NOTE(mrry): We must ensure that any non-tensor components in `dataset`
    # are normalized to their dense tensor representation, so that the
    # non-tensor oblivious unbatching logic will slice them appropriately.
    # This leads to a somewhat inefficient re-encoding step for all non-tensor
    # components.
    #
    # TODO(mrry): Consider optimizing this if it turns out to be a bottleneck.
    def normalize(arg, *rest):
      # pylint: disable=protected-access
      if rest:
        return structure.to_batched_tensor_list(self.element_spec,
                                                (arg,) + rest)
      else:
        return structure.to_batched_tensor_list(self.element_spec, arg)

    normalized_dataset = self.map(normalize)

    # NOTE(mrry): Our `map()` has lost information about the structure of
    # non-tensor components, so re-apply the structure of the original dataset.
    restructured_dataset = _RestructuredDataset(normalized_dataset,
                                                self.element_spec)
    return _UnbatchDataset(restructured_dataset)

  def with_options(self, options):
    """Returns a new `tf.data.Dataset` with the given options set.

    The options are "global" in the sense they apply to the entire dataset.
    If options are set multiple times, they are merged as long as different
    options do not use different non-default values.

    Args:
      options: A `tf.data.Options` that identifies the options the use.

    Returns:
      Dataset: A `Dataset` with the given options.

    Raises:
      ValueError: when an option is set more than once to a non-default value
    """
    return _OptionsDataset(self, options)


@tf_export(v1=["data.Dataset"])
class DatasetV1(DatasetV2):
  """Represents a potentially large set of elements.

  A `Dataset` can be used to represent an input pipeline as a
  collection of elements and a "logical plan" of transformations that act on
  those elements.
  """

  def __init__(self):
    try:
      variant_tensor = self._as_variant_tensor()
    except AttributeError as e:
      if "_as_variant_tensor" in str(e):
        raise AttributeError("Please use _variant_tensor instead of "
                             "_as_variant_tensor() to obtain the variant "
                             "associated with a dataset")
      raise AttributeError("{}: A likely cause of this error is that the super "
                           "call for this dataset is not the last line of the "
                           "__init__ method. The base class causes the "
                           "_as_variant_tensor call in its constructor and "
                           "if that uses attributes defined in the __init__ "
                           "method, those attrs need to be defined before the "
                           "super call.".format(e))
    super(DatasetV1, self).__init__(variant_tensor)

  @abc.abstractmethod
  def _as_variant_tensor(self):
    """Creates a scalar `tf.Tensor` of `tf.variant` representing this dataset.

    Returns:
      A scalar `tf.Tensor` of `tf.variant` type, which represents this dataset.
    """
    raise NotImplementedError("Dataset._as_variant_tensor")

  @deprecation.deprecated(
      None, "Use `for ... in dataset:` to iterate over a dataset. If using "
      "`tf.estimator`, return the `Dataset` object directly from your input "
      "function. As a last resort, you can use "
      "`tf.compat.v1.data.make_one_shot_iterator(dataset)`.")
  def make_one_shot_iterator(self):
    """Creates an `Iterator` for enumerating the elements of this dataset.

    Note: The returned iterator will be initialized automatically.
    A "one-shot" iterator does not currently support re-initialization.

    Returns:
      An `Iterator` over the elements of this dataset.
    """
    return self._make_one_shot_iterator()

  def _make_one_shot_iterator(self):  # pylint: disable=missing-docstring
    if context.executing_eagerly():
      return iterator_ops.IteratorV2(self)

    _ensure_same_dataset_graph(self)
    # Now that we create datasets at python object creation time, the capture
    # by value _make_dataset() function would try to capture these variant
    # tensor dataset inputs, which are marked as stateful ops and would throw
    # an error if we try and capture them. We therefore traverse the graph
    # to find all these ops and whitelist them so that the capturing
    # logic instead of throwing an error recreates these ops which is what was
    # happening before.
    all_ds_ops = traverse.obtain_all_variant_tensor_ops(self)
    graph_level_seed, op_level_seed = core_random_seed.get_seed(None)

    # NOTE(mrry): We capture by value here to ensure that `_make_dataset()` is
    # a 0-argument function.
    @function.Defun(capture_by_value=True, whitelisted_stateful_ops=all_ds_ops)
    def _make_dataset():
      """Factory function for a dataset."""
      # NOTE(mrry): `Defun` does not capture the graph-level seed from the
      # enclosing graph, so if a graph-level seed is present we set the local
      # graph seed based on a combination of the graph- and op-level seeds.
      if graph_level_seed is not None:
        assert op_level_seed is not None
        core_random_seed.set_random_seed(
            (graph_level_seed + 87654321 * op_level_seed) % (2 ** 63 - 1))

      dataset = self._apply_options()
      return dataset._variant_tensor  # pylint: disable=protected-access

    try:
      _make_dataset.add_to_graph(ops.get_default_graph())
    except ValueError as err:
      if "Cannot capture a stateful node" in str(err):
        raise ValueError(
            "Failed to create a one-shot iterator for a dataset. "
            "`Dataset.make_one_shot_iterator()` does not support datasets that "
            "capture stateful objects, such as a `Variable` or `LookupTable`. "
            "In these cases, use `Dataset.make_initializable_iterator()`. "
            "(Original error: %s)" % err)
      else:
        six.reraise(ValueError, err)

    # pylint: disable=protected-access
    return iterator_ops.Iterator(
        gen_dataset_ops.one_shot_iterator(
            dataset_factory=_make_dataset, **self._flat_structure), None,
        get_legacy_output_types(self), get_legacy_output_shapes(self),
        get_legacy_output_classes(self))

  @deprecation.deprecated(
      None, "Use `for ... in dataset:` to iterate over a dataset. If using "
      "`tf.estimator`, return the `Dataset` object directly from your input "
      "function. As a last resort, you can use "
      "`tf.compat.v1.data.make_initializable_iterator(dataset)`.")
  def make_initializable_iterator(self, shared_name=None):
    """Creates an `Iterator` for enumerating the elements of this dataset.

    Note: The returned iterator will be in an uninitialized state,
    and you must run the `iterator.initializer` operation before using it:

    ```python
    dataset = ...
    iterator = dataset.make_initializable_iterator()
    # ...
    sess.run(iterator.initializer)
    ```

    Args:
      shared_name: (Optional.) If non-empty, the returned iterator will be
        shared under the given name across multiple sessions that share the same
        devices (e.g. when using a remote server).

    Returns:
      An `Iterator` over the elements of this dataset.

    Raises:
      RuntimeError: If eager execution is enabled.
    """

    return self._make_initializable_iterator(shared_name)

  def _make_initializable_iterator(self, shared_name=None):  # pylint: disable=missing-docstring
    if context.executing_eagerly():
      raise RuntimeError(
          "dataset.make_initializable_iterator is not supported when eager "
          "execution is enabled.")
    _ensure_same_dataset_graph(self)
    dataset = self._apply_options()
    if shared_name is None:
      shared_name = ""
    iterator_resource = gen_dataset_ops.iterator_v2(
        container="", shared_name=shared_name, **self._flat_structure)
    with ops.colocate_with(iterator_resource):
      initializer = gen_dataset_ops.make_iterator(
          dataset._variant_tensor,  # pylint: disable=protected-access
          iterator_resource)
    # pylint: disable=protected-access
    return iterator_ops.Iterator(
        iterator_resource, initializer, get_legacy_output_types(dataset),
        get_legacy_output_shapes(dataset), get_legacy_output_classes(dataset))

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_classes(dataset)`.")
  def output_classes(self):
    """Returns the class of each component of an element of this dataset.

    Returns:
      A nested structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
        self.element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_shapes(dataset)`.")
  def output_shapes(self):
    """Returns the shape of each component of an element of this dataset.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
        self.element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_types(dataset)`.")
  def output_types(self):
    """Returns the type of each component of an element of this dataset.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
        self.element_spec)

  @property
  def element_spec(self):
    # TODO(b/110122868): Remove this override once all `Dataset` instances
    # implement `element_structure`.
    return structure.convert_legacy_structure(
        self.output_types, self.output_shapes, self.output_classes)

  @staticmethod
  @functools.wraps(DatasetV2.from_tensors)
  def from_tensors(tensors):
    return DatasetV1Adapter(DatasetV2.from_tensors(tensors))

  @staticmethod
  @functools.wraps(DatasetV2.from_tensor_slices)
  def from_tensor_slices(tensors):
    return DatasetV1Adapter(DatasetV2.from_tensor_slices(tensors))

  @staticmethod
  @deprecation.deprecated(None, "Use `tf.data.Dataset.from_tensor_slices()`.")
  def from_sparse_tensor_slices(sparse_tensor):
    """Splits each rank-N `tf.SparseTensor` in this dataset row-wise.

    Args:
      sparse_tensor: A `tf.SparseTensor`.

    Returns:
      Dataset: A `Dataset` of rank-(N-1) sparse tensors.
    """
    return DatasetV1Adapter(SparseTensorSliceDataset(sparse_tensor))

  @staticmethod
  @functools.wraps(DatasetV2.from_generator)
  def from_generator(generator, output_types, output_shapes=None, args=None):
    return DatasetV1Adapter(DatasetV2.from_generator(
        generator, output_types, output_shapes, args))

  @staticmethod
  @functools.wraps(DatasetV2.range)
  def range(*args):
    return DatasetV1Adapter(DatasetV2.range(*args))

  @staticmethod
  @functools.wraps(DatasetV2.zip)
  def zip(datasets):
    return DatasetV1Adapter(DatasetV2.zip(datasets))

  @functools.wraps(DatasetV2.concatenate)
  def concatenate(self, dataset):
    return DatasetV1Adapter(super(DatasetV1, self).concatenate(dataset))

  @functools.wraps(DatasetV2.prefetch)
  def prefetch(self, buffer_size):
    return DatasetV1Adapter(super(DatasetV1, self).prefetch(buffer_size))

  @staticmethod
  @functools.wraps(DatasetV2.list_files)
  def list_files(file_pattern, shuffle=None, seed=None):
    return DatasetV1Adapter(DatasetV2.list_files(file_pattern, shuffle, seed))

  @functools.wraps(DatasetV2.repeat)
  def repeat(self, count=None):
    return DatasetV1Adapter(super(DatasetV1, self).repeat(count))

  @functools.wraps(DatasetV2.shuffle)
  def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=None):
    return DatasetV1Adapter(super(DatasetV1, self).shuffle(
        buffer_size, seed, reshuffle_each_iteration))

  @functools.wraps(DatasetV2.cache)
  def cache(self, filename=""):
    return DatasetV1Adapter(super(DatasetV1, self).cache(filename))

  @functools.wraps(DatasetV2.take)
  def take(self, count):
    return DatasetV1Adapter(super(DatasetV1, self).take(count))

  @functools.wraps(DatasetV2.skip)
  def skip(self, count):
    return DatasetV1Adapter(super(DatasetV1, self).skip(count))

  @functools.wraps(DatasetV2.shard)
  def shard(self, num_shards, index):
    return DatasetV1Adapter(super(DatasetV1, self).shard(num_shards, index))

  @functools.wraps(DatasetV2.batch)
  def batch(self, batch_size, drop_remainder=False):
    return DatasetV1Adapter(super(DatasetV1, self).batch(
        batch_size, drop_remainder))

  @functools.wraps(DatasetV2.padded_batch)
  def padded_batch(self,
                   batch_size,
                   padded_shapes,
                   padding_values=None,
                   drop_remainder=False):
    return DatasetV1Adapter(super(DatasetV1, self).padded_batch(
        batch_size, padded_shapes, padding_values, drop_remainder))

  @functools.wraps(DatasetV2.map)
  def map(self, map_func, num_parallel_calls=None):
    if num_parallel_calls is None:
      return DatasetV1Adapter(
          MapDataset(self, map_func, preserve_cardinality=False))
    else:
      return DatasetV1Adapter(
          ParallelMapDataset(
              self, map_func, num_parallel_calls, preserve_cardinality=False))

  @deprecation.deprecated(None, "Use `tf.data.Dataset.map()")
  def map_with_legacy_function(self, map_func, num_parallel_calls=None):
    """Maps `map_func` across the elements of this dataset.

    NOTE: This is an escape hatch for existing uses of `map` that do not work
    with V2 functions. New uses are strongly discouraged and existing uses
    should migrate to `map` as this method will be removed in V2.

    Args:
      map_func: A function mapping a nested structure of tensors (having shapes
        and types defined by `self.output_shapes` and `self.output_types`) to
        another nested structure of tensors.
      num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
        representing the number elements to process asynchronously in parallel.
        If not specified, elements will be processed sequentially. If the value
        `tf.data.experimental.AUTOTUNE` is used, then the number of parallel
        calls is set dynamically based on available CPU.

    Returns:
      Dataset: A `Dataset`.
    """
    if num_parallel_calls is None:
      return DatasetV1Adapter(
          MapDataset(
              self,
              map_func,
              preserve_cardinality=False,
              use_legacy_function=True))
    else:
      return DatasetV1Adapter(
          ParallelMapDataset(
              self,
              map_func,
              num_parallel_calls,
              preserve_cardinality=False,
              use_legacy_function=True))

  @functools.wraps(DatasetV2.flat_map)
  def flat_map(self, map_func):
    return DatasetV1Adapter(super(DatasetV1, self).flat_map(map_func))

  @functools.wraps(DatasetV2.interleave)
  def interleave(self,
                 map_func,
                 cycle_length=AUTOTUNE,
                 block_length=1,
                 num_parallel_calls=None):
    return DatasetV1Adapter(super(DatasetV1, self).interleave(
        map_func, cycle_length, block_length, num_parallel_calls))

  @functools.wraps(DatasetV2.filter)
  def filter(self, predicate):
    return DatasetV1Adapter(super(DatasetV1, self).filter(predicate))

  @deprecation.deprecated(None, "Use `tf.data.Dataset.filter()")
  def filter_with_legacy_function(self, predicate):
    """Filters this dataset according to `predicate`.

    NOTE: This is an escape hatch for existing uses of `filter` that do not work
    with V2 functions. New uses are strongly discouraged and existing uses
    should migrate to `filter` as this method will be removed in V2.

    Args:
      predicate: A function mapping a nested structure of tensors (having shapes
        and types defined by `self.output_shapes` and `self.output_types`) to a
        scalar `tf.bool` tensor.

    Returns:
      Dataset: The `Dataset` containing the elements of this dataset for which
          `predicate` is `True`.
    """
    return FilterDataset(self, predicate, use_legacy_function=True)

  @functools.wraps(DatasetV2.apply)
  def apply(self, transformation_func):
    return DatasetV1Adapter(super(DatasetV1, self).apply(transformation_func))

  @functools.wraps(DatasetV2.window)
  def window(self, size, shift=None, stride=1, drop_remainder=False):
    return DatasetV1Adapter(super(DatasetV1, self).window(
        size, shift, stride, drop_remainder))

  @functools.wraps(DatasetV2.with_options)
  def with_options(self, options):
    return DatasetV1Adapter(super(DatasetV1, self).with_options(options))


# TODO(b/119044825): Until all `tf.data` unit tests are converted to V2, keep
# this alias in place.
Dataset = DatasetV1


class DatasetV1Adapter(DatasetV1):
  """Wraps a V2 `Dataset` object in the `tf.compat.v1.data.Dataset` API."""

  def __init__(self, dataset):
    self._dataset = dataset
    super(DatasetV1Adapter, self).__init__()

  def _as_variant_tensor(self):
    return self._dataset._variant_tensor  # pylint: disable=protected-access

  def _has_captured_ref(self):
    return self._dataset._has_captured_ref()  # pylint: disable=protected-access

  def _inputs(self):
    return self._dataset._inputs()  # pylint: disable=protected-access

  def _functions(self):
    return self._dataset._functions()  # pylint: disable=protected-access

  def options(self):
    return self._dataset.options()

  @property
  def element_spec(self):
    return self._dataset.element_spec  # pylint: disable=protected-access

  def __iter__(self):
    return iter(self._dataset)


def _ensure_same_dataset_graph(dataset):
  """Walks the dataset graph to ensure all datasets come from the same graph."""
  current_graph = ops.get_default_graph()
  bfs_q = Queue.Queue()
  bfs_q.put(dataset)  # pylint: disable=protected-access
  visited = []
  while not bfs_q.empty():
    ds = bfs_q.get()
    visited.append(ds)
    ds_graph = ds._graph  # pylint: disable=protected-access
    if current_graph != ds_graph:
      logging.warning("The graph (" + str(current_graph) + ") of the iterator "
                      "is different from the graph (" + str(ds_graph) + ") "
                      "the dataset: " + str(ds._variant_tensor) + " was "  # pylint: disable=protected-access
                      "created in. If you are using the Estimator API, "
                      "make sure that no part of the dataset returned by the "
                      "`input_fn` function is defined outside the `input_fn` "
                      "function. Please ensure that all datasets in the "
                      "pipeline are created in the same graph as the iterator. "
                      "NOTE: This warning will become an error in future "
                      "versions of TensorFlow.")
    for input_ds in ds._inputs():  # pylint: disable=protected-access
      if input_ds not in visited:
        bfs_q.put(input_ds)


@tf_export(v1=["data.make_one_shot_iterator"])
def make_one_shot_iterator(dataset):
  """Creates a `tf.compat.v1.data.Iterator` for enumerating the elements of a dataset.

  Note: The returned iterator will be initialized automatically.
  A "one-shot" iterator does not support re-initialization.

  Args:
    dataset: A `tf.data.Dataset`.

  Returns:
    A `tf.compat.v1.data.Iterator` over the elements of this dataset.
  """
  try:
    # Call the defined `_make_one_shot_iterator()` if there is one, because some
    # datasets (e.g. for prefetching) override its behavior.
    return dataset._make_one_shot_iterator()  # pylint: disable=protected-access
  except AttributeError:
    return DatasetV1Adapter(dataset)._make_one_shot_iterator()  # pylint: disable=protected-access


@tf_export(v1=["data.make_initializable_iterator"])
def make_initializable_iterator(dataset, shared_name=None):
  """Creates a `tf.compat.v1.data.Iterator` for enumerating the elements of a dataset.

  Note: The returned iterator will be in an uninitialized state,
  and you must run the `iterator.initializer` operation before using it:

  ```python
  dataset = ...
  iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
  # ...
  sess.run(iterator.initializer)
  ```

  Args:
    dataset: A `tf.data.Dataset`.
    shared_name: (Optional.) If non-empty, the returned iterator will be shared
      under the given name across multiple sessions that share the same devices
      (e.g. when using a remote server).

  Returns:
    A `tf.compat.v1.data.Iterator` over the elements of `dataset`.

  Raises:
    RuntimeError: If eager execution is enabled.
  """
  try:
    # Call the defined `_make_initializable_iterator()` if there is one, because
    # some datasets (e.g. for prefetching) override its behavior.
    return dataset._make_initializable_iterator(shared_name)  # pylint: disable=protected-access
  except AttributeError:
    return DatasetV1Adapter(dataset)._make_initializable_iterator(shared_name)  # pylint: disable=protected-access


@tf_export("data.experimental.get_structure")
def get_structure(dataset_or_iterator):
  """Returns the type specification of an element of a `Dataset` or `Iterator`.

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.Iterator`.

  Returns:
    A nested structure of `tf.TypeSpec` objects matching the structure of an
    element of `dataset_or_iterator` and spacifying the type of individal
    components.

  Raises:
    TypeError: If `dataset_or_iterator` is not a `Dataset` or `Iterator` object.
  """
  try:
    return dataset_or_iterator.element_spec  # pylint: disable=protected-access
  except AttributeError:
    raise TypeError("`dataset_or_iterator` must be a Dataset or Iterator "
                    "object, but got %s." % type(dataset_or_iterator))


@tf_export(v1=["data.get_output_classes"])
def get_legacy_output_classes(dataset_or_iterator):
  """Returns the output classes of a `Dataset` or `Iterator` elements.

  This utility method replaces the deprecated-in-V2
  `tf.compat.v1.Dataset.output_classes` property.

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.IteratorV2`.

  Returns:
    A nested structure of Python `type` objects matching the structure of the
    dataset / iterator elements and specifying the class of the individual
    components.
  """
  return nest.map_structure(
      lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
      get_structure(dataset_or_iterator))


@tf_export(v1=["data.get_output_shapes"])
def get_legacy_output_shapes(dataset_or_iterator):
  """Returns the output shapes of a `Dataset` or `Iterator` elements.

  This utility method replaces the deprecated-in-V2
  `tf.compat.v1.Dataset.output_shapes` property.

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.Iterator`.

  Returns:
    A nested structure of `tf.TensorShape` objects matching the structure of
    the dataset / iterator elements and specifying the shape of the individual
    components.
  """
  return nest.map_structure(
      lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
      get_structure(dataset_or_iterator))


@tf_export(v1=["data.get_output_types"])
def get_legacy_output_types(dataset_or_iterator):
  """Returns the output shapes of a `Dataset` or `Iterator` elements.

  This utility method replaces the deprecated-in-V2
  `tf.compat.v1.Dataset.output_types` property.

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.Iterator`.

  Returns:
    A nested structure of `tf.DType` objects objects matching the structure of
    dataset / iterator elements and specifying the shape of the individual
    components.
  """
  return nest.map_structure(
      lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
      get_structure(dataset_or_iterator))


@tf_export("data.Options")
class Options(options_lib.OptionsBase):
  """Represents options for tf.data.Dataset.

  An `Options` object can be, for instance, used to control which static
  optimizations to apply or whether to use performance modeling to dynamically
  tune the parallelism of operations such as `tf.data.Dataset.map` or
  `tf.data.Dataset.interleave`.
  """

  experimental_deterministic = options_lib.create_option(
      name="experimental_deterministic",
      ty=bool,
      docstring=
      "Whether the outputs need to be produced in deterministic order. If None,"
      " defaults to True.")

  experimental_distribute = options_lib.create_option(
      name="experimental_distribute",
      ty=distribute_options.DistributeOptions,
      docstring=
      "The distribution strategy options associated with the dataset. See "
      "`tf.data.experimental.DistributeOptions` for more details.",
      default_factory=distribute_options.DistributeOptions)

  experimental_optimization = options_lib.create_option(
      name="experimental_optimization",
      ty=optimization_options.OptimizationOptions,
      docstring=
      "The optimization options associated with the dataset. See "
      "`tf.data.experimental.OptimizationOptions` for more details.",
      default_factory=optimization_options.OptimizationOptions)

  experimental_slack = options_lib.create_option(
      name="experimental_slack",
      ty=bool,
      docstring="Whether to introduce 'slack' in the last `prefetch` of the "
      "input pipeline, if it exists. This may reduce CPU contention with "
      "accelerator host-side activity at the start of a step. The slack "
      "frequency is determined by the number of devices attached to this "
      "input pipeline. If None, defaults to False.")

  experimental_stats = options_lib.create_option(
      name="experimental_stats",
      ty=stats_options.StatsOptions,
      docstring=
      "The statistics options associated with the dataset. See "
      "`tf.data.experimental.StatsOptions` for more details.",
      default_factory=stats_options.StatsOptions)

  experimental_threading = options_lib.create_option(
      name="experimental_threading",
      ty=threading_options.ThreadingOptions,
      docstring=
      "The threading options associated with the dataset. See "
      "`tf.data.experimental.ThreadingOptions` for more details.",
      default_factory=threading_options.ThreadingOptions)

  experimental_stateful_whitelist = options_lib.create_option(
      name="experimental_stateful_whitelist",
      ty=list,
      docstring="By default, tf.data will refuse to serialize a dataset or "
      "checkpoint its iterator if the dataset contains a stateful op as the "
      "serialization / checkpointing won't be able to capture its state. "
      "Users can -- at their own risk -- override this restriction by "
      "explicitly whitelisting stateful ops by specifying them in this list.")

  def _static_optimizations(self):
    """Produces the list of enabled static optimizations."""

    result = []
    result.extend(self.experimental_optimization._static_optimizations())  # pylint: disable=protected-access

    if self.experimental_deterministic is False:
      result.append("make_sloppy")
    if self.experimental_stats and self.experimental_stats.latency_all_edges:
      result.append("latency_all_edges")
    if self.experimental_slack:
      result.append("slack")
    if (self.experimental_distribute and
        self.experimental_distribute._make_stateless):  # pylint: disable=protected-access
      result.append("make_stateless")
    return result

  def _static_optimization_configs(self):
    """Produces the list of configurations for enabled static optimizations."""
    result = []
    if self.experimental_optimization:
      result.extend(
          self.experimental_optimization._static_optimization_configs())  # pylint: disable=protected-access

    if self.experimental_slack:
      num_devices = self.experimental_distribute.num_devices
      if num_devices is None:
        num_devices = 1
      result.append("slack:slack_period:%d" % num_devices)
    return result

  def merge(self, options):
    """Merges itself with the given `tf.data.Options`.

    The given `tf.data.Options` can be merged as long as there does not exist an
    attribute that is set to different values in `self` and `options`.

    Args:
      options: a `tf.data.Options` to merge with

    Raises:
      ValueError: if the given `tf.data.Options` cannot be merged

    Returns:
      New `tf.data.Options()` object which is the result of merging self with
      the input `tf.data.Options`.
    """
    return options_lib.merge_options(self, options)


class DatasetSource(DatasetV2):
  """Abstract class representing a dataset with no inputs."""

  def _inputs(self):
    return []


class UnaryDataset(DatasetV2):
  """Abstract class representing a dataset with one input."""

  def __init__(self, input_dataset, variant_tensor):
    self._input_dataset = input_dataset
    super(UnaryDataset, self).__init__(variant_tensor)

  def _inputs(self):
    return [self._input_dataset]


class UnaryUnchangedStructureDataset(UnaryDataset):
  """Represents a unary dataset with the same input and output structure."""

  def __init__(self, input_dataset, variant_tensor):
    self._input_dataset = input_dataset
    super(UnaryUnchangedStructureDataset, self).__init__(
        input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._input_dataset.element_spec


class TensorDataset(DatasetSource):
  """A `Dataset` with a single element."""

  def __init__(self, element):
    """See `Dataset.from_tensors()` for details."""
    element = structure.normalize_element(element)
    self._structure = structure.type_spec_from_value(element)
    self._tensors = structure.to_tensor_list(self._structure, element)

    variant_tensor = gen_dataset_ops.tensor_dataset(
        self._tensors,
        output_shapes=structure.get_flat_tensor_shapes(self._structure))
    super(TensorDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._structure


class TensorSliceDataset(DatasetSource):
  """A `Dataset` of slices from a dataset element."""

  def __init__(self, element):
    """See `Dataset.from_tensor_slices()` for details."""
    element = structure.normalize_element(element)
    batched_spec = structure.type_spec_from_value(element)
    self._tensors = structure.to_batched_tensor_list(batched_spec, element)
    self._structure = nest.map_structure(
        lambda component_spec: component_spec._unbatch(), batched_spec)  # pylint: disable=protected-access

    batch_dim = tensor_shape.Dimension(tensor_shape.dimension_value(
        self._tensors[0].get_shape()[0]))
    for t in self._tensors[1:]:
      batch_dim.assert_is_compatible_with(tensor_shape.Dimension(
          tensor_shape.dimension_value(t.get_shape()[0])))

    variant_tensor = gen_dataset_ops.tensor_slice_dataset(
        self._tensors,
        output_shapes=structure.get_flat_tensor_shapes(self._structure))
    super(TensorSliceDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._structure


class SparseTensorSliceDataset(DatasetSource):
  """A `Dataset` that splits a rank-N `tf.SparseTensor` into its rows."""

  def __init__(self, sparse_tensor):
    """See `Dataset.from_sparse_tensor_slices()` for details."""
    if not isinstance(sparse_tensor, sparse_tensor_lib.SparseTensor):
      raise TypeError(
          "`sparse_tensor` must be a `tf.SparseTensor` object. Was {}.".format(
              sparse_tensor))
    self._sparse_tensor = sparse_tensor

    indices_shape = self._sparse_tensor.indices.get_shape()
    shape_shape = self._sparse_tensor.dense_shape.get_shape()
    rank = (indices_shape.dims[1] - 1).merge_with(shape_shape.dims[0] - 1)
    self._structure = (tensor_spec.TensorSpec([None, rank], dtypes.int64),
                       tensor_spec.TensorSpec([None],
                                              self._sparse_tensor.dtype),
                       tensor_spec.TensorSpec([rank], dtypes.int64))

    variant_tensor = gen_dataset_ops.sparse_tensor_slice_dataset(
        self._sparse_tensor.indices, self._sparse_tensor.values,
        self._sparse_tensor.dense_shape)
    super(SparseTensorSliceDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._structure


class _VariantDataset(DatasetV2):
  """A Dataset wrapper around a `tf.variant`-typed function argument."""

  def __init__(self, dataset_variant, structure):
    self._structure = structure
    super(_VariantDataset, self).__init__(dataset_variant)

  def _inputs(self):
    return []

  @property
  def element_spec(self):
    return self._structure


class _NestedVariant(composite_tensor.CompositeTensor):

  def __init__(self, variant_tensor, element_spec, dataset_shape):
    self._variant_tensor = variant_tensor
    self._element_spec = element_spec
    self._dataset_shape = dataset_shape

  @property
  def _type_spec(self):
    return DatasetSpec(self._element_spec, self._dataset_shape)


@tf_export("data.experimental.from_variant")
def from_variant(variant, structure):
  """Constructs a dataset from the given variant and structure.

  Args:
    variant: A scalar `tf.variant` tensor representing a dataset.
    structure: A `tf.data.experimental.Structure` object representing the
      structure of each element in the dataset.

  Returns:
    A `tf.data.Dataset` instance.
  """
  return _VariantDataset(variant, structure)  # pylint: disable=protected-access


@tf_export("data.experimental.to_variant")
def to_variant(dataset):
  """Returns a variant representing the given dataset.

  Args:
    dataset: A `tf.data.Dataset`.

  Returns:
    A scalar `tf.variant` tensor representing the given dataset.
  """
  return dataset._variant_tensor  # pylint: disable=protected-access


@tf_export(
    "data.DatasetSpec",
    v1=["data.DatasetSpec", "data.experimental.DatasetStructure"])
class DatasetSpec(type_spec.BatchableTypeSpec):
  """Type specification for `tf.data.Dataset`."""

  __slots__ = ["_element_spec", "_dataset_shape"]

  def __init__(self, element_spec, dataset_shape=()):
    self._element_spec = element_spec
    self._dataset_shape = tensor_shape.as_shape(dataset_shape)

  @property
  def value_type(self):
    return _VariantDataset

  def _serialize(self):
    return (self._element_spec, self._dataset_shape)

  @property
  def _component_specs(self):
    return tensor_spec.TensorSpec(self._dataset_shape, dtypes.variant)

  def _to_components(self, value):
    return value._variant_tensor  # pylint: disable=protected-access

  def _from_components(self, components):
    # pylint: disable=protected-access
    if self._dataset_shape.ndims == 0:
      return _VariantDataset(components, self._element_spec)
    else:
      return _NestedVariant(components, self._element_spec, self._dataset_shape)

  def _to_tensor_list(self, value):
    return [
        ops.convert_to_tensor(
            tf_nest.map_structure(lambda x: x._variant_tensor, value))  # pylint: disable=protected-access
    ]

  @staticmethod
  def from_value(value):
    return DatasetSpec(value.element_spec)  # pylint: disable=protected-access

  def _batch(self, batch_size):
    return DatasetSpec(
        self._element_spec,
        tensor_shape.TensorShape([batch_size]).concatenate(self._dataset_shape))

  def _unbatch(self):
    if self._dataset_shape.ndims == 0:
      raise ValueError("Unbatching a dataset is only supported for rank >= 1")
    return DatasetSpec(self._element_spec, self._dataset_shape[1:])

  def _to_batched_tensor_list(self, value):
    if self._dataset_shape.ndims == 0:
      raise ValueError("Unbatching a dataset is only supported for rank >= 1")
    return self._to_tensor_list(value)

  def _to_legacy_output_types(self):
    return self

  def _to_legacy_output_shapes(self):
    return self

  def _to_legacy_output_classes(self):
    return self


class StructuredFunctionWrapper(object):
  """A function wrapper that supports structured arguments and return values."""

  # pylint: disable=protected-access
  def __init__(self,
               func,
               transformation_name,
               dataset=None,
               input_classes=None,
               input_shapes=None,
               input_types=None,
               input_structure=None,
               add_to_graph=True,
               use_legacy_function=False,
               defun_kwargs=None):
    """Creates a new `StructuredFunctionWrapper` for the given function.

    Args:
      func: A function from a nested structure to another nested structure.
      transformation_name: Human-readable name of the transformation in which
        this function is being instantiated, for error messages.
      dataset: (Optional.) A `tf.data.Dataset`. If given, the structure of this
        dataset will be assumed as the structure for `func` arguments; otherwise
        `input_classes`, `input_shapes`, and `input_types` must be defined.
      input_classes: (Optional.) A nested structure of `type`. If given, this
        argument defines the Python types for `func` arguments.
      input_shapes: (Optional.) A nested structure of `tf.TensorShape`. If
        given, this argument defines the shapes and structure for `func`
        arguments.
      input_types: (Optional.) A nested structure of `tf.DType`. If given, this
        argument defines the element types and structure for `func` arguments.
      input_structure: (Optional.) A `Structure` object. If given, this argument
        defines the element types and structure for `func` arguments.
      add_to_graph: (Optional.) If `True`, the function will be added to the
        default graph.
      use_legacy_function: (Optional.) A boolean that determines whether the
        function be created using `tensorflow.python.eager.function.defun`
        (default behavior) or `tensorflow.python.framework.function.Defun`
        (legacy beheavior).
      defun_kwargs: (Optional.) A dictionary mapping string argument names to
        values. If supplied, will be passed to `function` as keyword arguments.

    Raises:
      ValueError: If an invalid combination of `dataset`, `input_classes`,
        `input_shapes`, and `input_types` is passed.
    """
    if input_structure is None:
      if dataset is None:
        if input_classes is None or input_shapes is None or input_types is None:
          raise ValueError("Either `dataset`, `input_structure` or all of "
                           "`input_classes`, `input_shapes`, and `input_types` "
                           "must be specified.")
        self._input_structure = structure.convert_legacy_structure(
            input_types, input_shapes, input_classes)
      else:
        if not (input_classes is None and input_shapes is None and
                input_types is None):
          raise ValueError("Either `dataset`, `input_structure` or all of "
                           "`input_classes`, `input_shapes`, and `input_types` "
                           "must be specified.")
        self._input_structure = dataset.element_spec
    else:
      if not (dataset is None and input_classes is None and input_shapes is None
              and input_types is None):
        raise ValueError("Either `dataset`, `input_structure`, or all of "
                         "`input_classes`, `input_shapes`, and `input_types` "
                         "must be specified.")
      self._input_structure = input_structure

    self._func = func

    if defun_kwargs is None:
      defun_kwargs = {}

    readable_transformation_name = transformation_name.replace(
        ".", "_")[:-2] if len(transformation_name) > 2 else ""

    func_name = "_".join(
        [readable_transformation_name,
         function_utils.get_func_name(func)])

    ag_ctx = autograph_ctx.control_status_ctx()

    def _warn_if_collections(transformation_name):
      """Prints a warning if the given graph uses common graph collections.

      NOTE(mrry): Currently a warning is only generated for resources. Any
      variables created will be automatically hoisted out to the outermost scope
      using `init_scope()`. Some collections (such as for control-flow contexts)
      are benign and should not generate a warning.

      Args:
        transformation_name: A human-readable name for the transformation.
      """
      warnings.warn("Creating resources inside a function passed to %s "
                    "is not supported. Create each resource outside the "
                    "function, and capture it inside the function to use it." %
                    transformation_name, stacklevel=5)

    def _wrapper_helper(*args):
      """Wrapper for passing nested structures to and from tf.data functions."""
      nested_args = structure.from_compatible_tensor_list(
          self._input_structure, args)
      if not _should_unpack_args(nested_args):
        nested_args = (nested_args,)

      ret = autograph.tf_convert(func, ag_ctx)(*nested_args)
      # If `func` returns a list of tensors, `nest.flatten()` and
      # `ops.convert_to_tensor()` would conspire to attempt to stack
      # those tensors into a single tensor, because the customized
      # version of `nest.flatten()` does not recurse into lists. Since
      # it is more likely that the list arose from returning the
      # result of an operation (such as `tf.numpy_function()`) that returns a
      # list of not-necessarily-stackable tensors, we treat the
      # returned value is a `tuple` instead. A user wishing to pack
      # the return value into a single tensor can use an explicit
      # `tf.stack()` before returning.
      if isinstance(ret, list):
        ret = tuple(ret)

      try:
        self._output_structure = structure.type_spec_from_value(ret)
      except (ValueError, TypeError):
        raise TypeError("Unsupported return value from function passed to "
                        "%s: %s." % (transformation_name, ret))
      return ret

    if use_legacy_function:
      func_name = func_name + "_" + str(ops.uid())

      @function.Defun(
          *structure.get_flat_tensor_types(self._input_structure),
          func_name=func_name,
          **defun_kwargs)
      def wrapper_fn(*args):
        ret = _wrapper_helper(*args)
        # _warn_if_collections(transformation_name, ops.get_default_graph(), 0)
        return structure.to_tensor_list(self._output_structure, ret)

      self._function = wrapper_fn
      resource_tracker = tracking.ResourceTracker()
      with tracking.resource_tracker_scope(resource_tracker):
        if add_to_graph:
          self._function.add_to_graph(ops.get_default_graph())
        else:
          # Use the private method that will execute `wrapper_fn` but delay
          # adding it to the graph in case (e.g.) we need to rerun the function.
          self._function._create_definition_if_needed()
      if resource_tracker.resources:
        _warn_if_collections(transformation_name)

    else:
      defun_kwargs.update({"func_name": func_name})

      # Note: _wrapper_helper will apply autograph based on context.
      @eager_function.defun_with_attributes(
          input_signature=structure.get_flat_tensor_specs(
              self._input_structure),
          autograph=False,
          attributes=defun_kwargs)
      def wrapper_fn(*args):  # pylint: disable=missing-docstring
        ret = _wrapper_helper(*args)
        ret = structure.to_tensor_list(self._output_structure, ret)
        return [ops.convert_to_tensor(t) for t in ret]

      resource_tracker = tracking.ResourceTracker()
      with tracking.resource_tracker_scope(resource_tracker):
        self._function = wrapper_fn._get_concrete_function_internal()
        if add_to_graph:
          self._function.add_to_graph(ops.get_default_graph())
      if resource_tracker.resources:
        _warn_if_collections(transformation_name)

      outer_graph_seed = ops.get_default_graph().seed
      if outer_graph_seed and self._function.graph.seed == outer_graph_seed:
        if self._function.graph._seed_used:
          warnings.warn(
              "Seed %s from outer graph might be getting used by function %s, "
              "if the random op has not been provided any seed. Explicitly set "
              "the seed in the function if this is not the intended behavior."
              %(outer_graph_seed, func_name), stacklevel=4)
  # pylint: enable=protected-access

  @property
  def output_structure(self):
    return self._output_structure

  @property
  def output_classes(self):
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
        self._output_structure)

  @property
  def output_shapes(self):
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
        self._output_structure)

  @property
  def output_types(self):
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
        self._output_structure)

  @property
  def function(self):
    return self._function


class _GeneratorDataset(DatasetSource):
  """A `Dataset` that generates elements by invoking a function."""

  def __init__(self, init_args, init_func, next_func, finalize_func):
    """Constructs a `_GeneratorDataset`.

    Args:
      init_args: A nested structure representing the arguments to `init_func`.
      init_func: A TensorFlow function that will be called on `init_args` each
        time a C++ iterator over this dataset is constructed. Returns a nested
        structure representing the "state" of the dataset.
      next_func: A TensorFlow function that will be called on the result of
        `init_func` to produce each element, and that raises `OutOfRangeError`
        to terminate iteration.
      finalize_func: A TensorFlow function that will be called on the result of
        `init_func` immediately before a C++ iterator over this dataset is
        destroyed. The return value is ignored.
    """
    self._init_args = init_args

    self._init_structure = structure.type_spec_from_value(init_args)

    self._init_func = StructuredFunctionWrapper(
        init_func,
        self._transformation_name(),
        input_structure=self._init_structure)

    self._next_func = StructuredFunctionWrapper(
        next_func,
        self._transformation_name(),
        input_structure=self._init_func.output_structure)

    self._finalize_func = StructuredFunctionWrapper(
        finalize_func,
        self._transformation_name(),
        input_structure=self._init_func.output_structure)
    variant_tensor = gen_dataset_ops.generator_dataset(
        structure.to_tensor_list(self._init_structure, self._init_args) +
        self._init_func.function.captured_inputs,
        self._next_func.function.captured_inputs,
        self._finalize_func.function.captured_inputs,
        init_func=self._init_func.function,
        next_func=self._next_func.function,
        finalize_func=self._finalize_func.function,
        **self._flat_structure)
    super(_GeneratorDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._next_func.output_structure

  def _transformation_name(self):
    return "Dataset.from_generator()"


class ZipDataset(DatasetV2):
  """A `Dataset` that zips its inputs together."""

  def __init__(self, datasets):
    """See `Dataset.zip()` for details."""
    for ds in nest.flatten(datasets):
      if not isinstance(ds, DatasetV2):
        if isinstance(ds, list):
          message = ("The argument to `Dataset.zip()` must be a nested "
                     "structure of `Dataset` objects. Nested structures do not "
                     "support Python lists; please use a tuple instead.")
        else:
          message = ("The argument to `Dataset.zip()` must be a nested "
                     "structure of `Dataset` objects.")
        raise TypeError(message)
    self._datasets = datasets
    self._structure = nest.pack_sequence_as(
        self._datasets,
        [ds.element_spec for ds in nest.flatten(self._datasets)])
    variant_tensor = gen_dataset_ops.zip_dataset(
        [ds._variant_tensor for ds in nest.flatten(self._datasets)],
        **self._flat_structure)
    super(ZipDataset, self).__init__(variant_tensor)

  def _inputs(self):
    return nest.flatten(self._datasets)

  @property
  def element_spec(self):
    return self._structure


class ConcatenateDataset(DatasetV2):
  """A `Dataset` that concatenates its input with given dataset."""

  def __init__(self, input_dataset, dataset_to_concatenate):
    """See `Dataset.concatenate()` for details."""
    self._input_dataset = input_dataset
    self._dataset_to_concatenate = dataset_to_concatenate

    output_types = get_legacy_output_types(input_dataset)
    if output_types != get_legacy_output_types(dataset_to_concatenate):
      raise TypeError(
          "Two datasets to concatenate have different types %s and %s" %
          (output_types, get_legacy_output_types(dataset_to_concatenate)))

    output_classes = get_legacy_output_classes(input_dataset)
    if output_classes != get_legacy_output_classes(dataset_to_concatenate):
      raise TypeError(
          "Two datasets to concatenate have different classes %s and %s" %
          (output_classes, get_legacy_output_classes(dataset_to_concatenate)))

    input_shapes = get_legacy_output_shapes(self._input_dataset)
    output_shapes = nest.pack_sequence_as(input_shapes, [
        ts1.most_specific_compatible_shape(ts2)
        for (ts1, ts2) in zip(
            nest.flatten(input_shapes),
            nest.flatten(get_legacy_output_shapes(
                self._dataset_to_concatenate)))
    ])

    self._structure = structure.convert_legacy_structure(
        output_types, output_shapes, output_classes)

    self._input_datasets = [input_dataset, dataset_to_concatenate]
    # pylint: disable=protected-access
    variant_tensor = gen_dataset_ops.concatenate_dataset(
        input_dataset._variant_tensor, dataset_to_concatenate._variant_tensor,
        **self._flat_structure)
    # pylint: enable=protected-access
    super(ConcatenateDataset, self).__init__(variant_tensor)

  def _inputs(self):
    return self._input_datasets

  @property
  def element_spec(self):
    return self._structure


class RepeatDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that repeats its input several times."""

  def __init__(self, input_dataset, count):
    """See `Dataset.repeat()` for details."""
    self._input_dataset = input_dataset
    if count is None:
      self._count = constant_op.constant(-1, dtype=dtypes.int64, name="count")
    else:
      self._count = ops.convert_to_tensor(
          count, dtype=dtypes.int64, name="count")
    variant_tensor = gen_dataset_ops.repeat_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        count=self._count,
        **self._flat_structure)
    super(RepeatDataset, self).__init__(input_dataset, variant_tensor)


class RangeDataset(DatasetSource):
  """A `Dataset` of a step separated range of values."""

  def __init__(self, *args):
    """See `Dataset.range()` for details."""
    self._parse_args(*args)
    self._structure = tensor_spec.TensorSpec([], dtypes.int64)
    variant_tensor = gen_dataset_ops.range_dataset(
        start=self._start,
        stop=self._stop,
        step=self._step,
        **self._flat_structure)
    super(RangeDataset, self).__init__(variant_tensor)

  def _parse_args(self, *args):
    """Parse arguments according to the same rules as the `range()` builtin."""
    if len(args) == 1:
      self._start = self._build_tensor(0, "start")
      self._stop = self._build_tensor(args[0], "stop")
      self._step = self._build_tensor(1, "step")
    elif len(args) == 2:
      self._start = self._build_tensor(args[0], "start")
      self._stop = self._build_tensor(args[1], "stop")
      self._step = self._build_tensor(1, "step")
    elif len(args) == 3:
      self._start = self._build_tensor(args[0], "start")
      self._stop = self._build_tensor(args[1], "stop")
      self._step = self._build_tensor(args[2], "step")
    else:
      raise ValueError("Invalid arguments to RangeDataset: %s" % str(args))

  def _build_tensor(self, int64_value, name):
    return ops.convert_to_tensor(int64_value, dtype=dtypes.int64, name=name)

  @property
  def element_spec(self):
    return self._structure


class _MemoryCacheDeleter(object):
  """An object which cleans up an anonymous memory cache resource.

  An alternative to defining a __del__ method on an object. Even if the parent
  object is part of a reference cycle, the cycle will be collectable.
  """

  def __init__(self, handle, device, deleter):
    self._deleter = deleter
    self._handle = handle
    self._device = device
    self._eager_mode = context.executing_eagerly()

  def __del__(self):
    with ops.device(self._device):
      # Make sure the resource is deleted in the same mode as it was created in.
      if self._eager_mode:
        with context.eager_mode():
          gen_dataset_ops.delete_memory_cache(
              handle=self._handle, deleter=self._deleter)
      else:
        with context.graph_mode():
          gen_dataset_ops.delete_memory_cache(
              handle=self._handle, deleter=self._deleter)


class _MemoryCache(object):
  """Represents a memory cache resource."""

  def __init__(self):
    super(_MemoryCache, self).__init__()
    self._device = context.context().device_name
    self._handle, self._deleter = (gen_dataset_ops.anonymous_memory_cache())
    self._resource_deleter = _MemoryCacheDeleter(
        handle=self._handle, device=self._device, deleter=self._deleter)

  @property
  def handle(self):
    return self._handle


class CacheDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that caches elements of its input."""

  def __init__(self, input_dataset, filename):
    """See `Dataset.cache()` for details."""
    self._input_dataset = input_dataset
    self._filename = ops.convert_to_tensor(
        filename, dtype=dtypes.string, name="filename")
    if tf2.enabled() and (context.executing_eagerly() or
                          ops.get_default_graph()._building_function):  # pylint: disable=protected-access
      self._cache = _MemoryCache()
      variant_tensor = gen_dataset_ops.cache_dataset_v2(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          filename=self._filename,
          cache=self._cache.handle,
          **self._flat_structure)
    else:
      variant_tensor = gen_dataset_ops.cache_dataset(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          filename=self._filename,
          **self._flat_structure)
    super(CacheDataset, self).__init__(input_dataset, variant_tensor)


class _RandomSeedGeneratorDeleter(object):
  """An object which cleans up an anonymous random seed generator resource.

  An alternative to defining a __del__ method on an object. Even if the parent
  object is part of a reference cycle, the cycle will be collectable.
  """

  def __init__(self, handle, device, deleter):
    self._deleter = deleter
    self._handle = handle
    self._device = device
    self._eager_mode = context.executing_eagerly()

  def __del__(self):
    with ops.device(self._device):
      # Make sure the resource is deleted in the same mode as it was created in.
      if self._eager_mode:
        with context.eager_mode():
          gen_dataset_ops.delete_random_seed_generator(
              handle=self._handle, deleter=self._deleter)
      else:
        with context.graph_mode():
          gen_dataset_ops.delete_random_seed_generator(
              handle=self._handle, deleter=self._deleter)


class _RandomSeedGenerator(object):
  """Represents a random seed generator resource."""

  def __init__(self, seed, seed2):
    super(_RandomSeedGenerator, self).__init__()
    self._device = context.context().device_name
    self._handle, self._deleter = (
        gen_dataset_ops.anonymous_random_seed_generator(seed=seed, seed2=seed2))
    self._resource_deleter = _RandomSeedGeneratorDeleter(
        handle=self._handle, device=self._device, deleter=self._deleter)

  @property
  def handle(self):
    return self._handle


class ShuffleDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that randomly shuffles the elements of its input."""

  def __init__(self,
               input_dataset,
               buffer_size,
               seed=None,
               reshuffle_each_iteration=None):
    """Randomly shuffles the elements of this dataset.

    Args:
      input_dataset: The input dataset.
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements from this dataset from which the new dataset will sample.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
        seed that will be used to create the distribution. See
        `tf.compat.v1.set_random_seed` for behavior.
      reshuffle_each_iteration: (Optional.) A boolean, which if true indicates
        that the dataset should be pseudorandomly reshuffled each time it is
        iterated over. (Defaults to `True`.)

    Returns:
      A `Dataset`.

    Raises:
      ValueError: if invalid arguments are provided.
    """
    self._input_dataset = input_dataset
    self._buffer_size = ops.convert_to_tensor(
        buffer_size, dtype=dtypes.int64, name="buffer_size")
    self._seed, self._seed2 = random_seed.get_seed(seed)

    if reshuffle_each_iteration is None:
      self._reshuffle_each_iteration = True
    else:
      self._reshuffle_each_iteration = reshuffle_each_iteration

    if tf2.enabled() and self._reshuffle_each_iteration and (
        context.executing_eagerly() or
        ops.get_default_graph()._building_function):  # pylint: disable=protected-access
      self._seed_generator = _RandomSeedGenerator(self._seed, self._seed2)
      variant_tensor = gen_dataset_ops.shuffle_dataset_v2(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          buffer_size=self._buffer_size,
          seed_generator=self._seed_generator.handle,
          **self._flat_structure)
    else:
      variant_tensor = gen_dataset_ops.shuffle_dataset(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          buffer_size=self._buffer_size,
          seed=self._seed,
          seed2=self._seed2,
          reshuffle_each_iteration=self._reshuffle_each_iteration,
          **self._flat_structure)
    super(ShuffleDataset, self).__init__(input_dataset, variant_tensor)


class TakeDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` containing the first `count` elements from its input."""

  def __init__(self, input_dataset, count):
    """See `Dataset.take()` for details."""
    self._input_dataset = input_dataset
    self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name="count")
    variant_tensor = gen_dataset_ops.take_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        count=self._count,
        **self._flat_structure)
    super(TakeDataset, self).__init__(input_dataset, variant_tensor)


class SkipDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` skipping the first `count` elements from its input."""

  def __init__(self, input_dataset, count):
    """See `Dataset.skip()` for details."""
    self._input_dataset = input_dataset
    self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name="count")
    variant_tensor = gen_dataset_ops.skip_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        count=self._count,
        **self._flat_structure)
    super(SkipDataset, self).__init__(input_dataset, variant_tensor)


class ShardDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` for sharding its input."""

  def __init__(self, input_dataset, num_shards, index):
    """See `Dataset.shard()` for details."""
    self._input_dataset = input_dataset
    self._num_shards = ops.convert_to_tensor(
        num_shards, dtype=dtypes.int64, name="num_shards")
    self._index = ops.convert_to_tensor(index, dtype=dtypes.int64, name="index")
    variant_tensor = gen_dataset_ops.shard_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        num_shards=self._num_shards,
        index=self._index,
        **self._flat_structure)
    super(ShardDataset, self).__init__(input_dataset, variant_tensor)


class BatchDataset(UnaryDataset):
  """A `Dataset` that batches contiguous elements from its input."""

  def __init__(self, input_dataset, batch_size, drop_remainder):
    """See `Dataset.batch()` for details."""
    self._input_dataset = input_dataset
    self._batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")

    constant_drop_remainder = tensor_util.constant_value(self._drop_remainder)
    # pylint: disable=protected-access
    if constant_drop_remainder:
      # NOTE(mrry): `constant_drop_remainder` may be `None` (unknown statically)
      # or `False` (explicitly retaining the remainder).
      # pylint: disable=g-long-lambda
      self._structure = nest.map_structure(
          lambda component_spec: component_spec._batch(
              tensor_util.constant_value(self._batch_size)),
          input_dataset.element_spec)
    else:
      self._structure = nest.map_structure(
          lambda component_spec: component_spec._batch(None),
          input_dataset.element_spec)
    variant_tensor = gen_dataset_ops.batch_dataset_v2(
        input_dataset._variant_tensor,
        batch_size=self._batch_size,
        drop_remainder=self._drop_remainder,
        **self._flat_structure)
    super(BatchDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure


class _VariantTracker(tracking.CapturableResource):
  """Allows export of functions capturing a Dataset in SavedModels.

  When saving a SavedModel, `tf.saved_model.save` traverses the object
  graph. Since Datasets reference _VariantTracker objects, that traversal will
  find a _VariantTracker for each Dataset and so know how to save and restore
  functions which reference the Dataset's variant Tensor.
  """

  def __init__(self, variant_tensor, resource_creator):
    """Record that `variant_tensor` is associated with `resource_creator`.

    Args:
      variant_tensor: The variant-dtype Tensor associated with the Dataset. This
        Tensor will be a captured input to functions which use the Dataset, and
        is used by saving code to identify the corresponding _VariantTracker.
      resource_creator: A zero-argument function which creates a new
        variant-dtype Tensor. This function will be included in SavedModels and
        run to re-create the Dataset's variant Tensor on restore.
    """
    super(_VariantTracker, self).__init__(device="CPU")
    self._resource_handle = variant_tensor
    self._create_resource = resource_creator


def _is_padded_shape_compatible_with(padded_shape, input_component_shape):
  """Returns `True` if `input_component_shape` can be padded to `padded_shape`.

  Args:
    padded_shape: A `tf.TensorShape`.
    input_component_shape: A `tf.TensorShape`.

  Returns:
    `True` if `input_component_shape` can be padded to `padded_shape`, otherwise
    `False`.
  """

  if padded_shape.dims is None or input_component_shape.dims is None:
    return True
  if len(padded_shape.dims) != len(input_component_shape.dims):
    return False
  for padded_dim, input_dim in zip(
      padded_shape.dims, input_component_shape.dims):
    if (padded_dim.value is not None and input_dim.value is not None
        and padded_dim.value < input_dim.value):
      return False
  return True


def _padded_shape_to_tensor(padded_shape, input_component_shape):
  """Converts `padded_shape` to a `tf.Tensor` representing that shape.

  Args:
    padded_shape: A shape-like object, which may be a `tf.TensorShape`, a Python
      sequence, or a 1-D `tf.Tensor` of `tf.int64` elements.
    input_component_shape: A `tf.TensorShape`, with which `padded_shape` must
      be compatible.

  Returns:
    A 1-D `tf.Tensor` of `tf.int64` elements, representing `padded_shape`.

  Raises:
    ValueError: If `padded_shape` is not a shape or not compatible with
      `input_component_shape`.
    TypeError: If `padded_shape` is not convertible to a `tf.int64` tensor.
  """
  try:
    # Try to convert the `padded_shape` to a `tf.TensorShape`
    padded_shape_as_shape = tensor_shape.as_shape(padded_shape)
    # We will return the "canonical" tensor representation, which uses
    # `-1` in place of `None`.
    ret = ops.convert_to_tensor(
        [dim if dim is not None else -1
         for dim in padded_shape_as_shape.as_list()], dtype=dtypes.int64)
  except (TypeError, ValueError):
    # The argument was not trivially convertible to a
    # `tf.TensorShape`, so fall back on the conversion to tensor
    # machinery.
    ret = ops.convert_to_tensor(padded_shape, preferred_dtype=dtypes.int64)
    if ret.shape.dims is not None and len(ret.shape.dims) != 1:
      raise ValueError(
          "Padded shape %s must be a 1-D tensor of tf.int64 values, but its "
          "shape was %s." % (padded_shape, ret.shape))
    if ret.dtype != dtypes.int64:
      raise TypeError(
          "Padded shape %s must be a 1-D tensor of tf.int64 values, but its "
          "element type was %s." % (padded_shape, ret.dtype.name))
    padded_shape_as_shape = tensor_util.constant_value_as_shape(ret)

  if not _is_padded_shape_compatible_with(padded_shape_as_shape,
                                          input_component_shape):
    raise ValueError("The padded shape %s is not compatible with the "
                     "corresponding input component shape %s."
                     % (padded_shape_as_shape, input_component_shape))

  return ret


def _padding_value_to_tensor(value, output_type):
  """Converts the padding value to a tensor.

  Args:
    value: The padding value.
    output_type: Its expected dtype.

  Returns:
    A scalar `Tensor`.

  Raises:
    ValueError: if the padding value is not a scalar.
    TypeError: if the padding value's type does not match `output_type`.
  """
  value = ops.convert_to_tensor(value, name="padding_value")
  if not value.shape.is_compatible_with(tensor_shape.TensorShape([])):
    raise ValueError("Padding value should be a scalar, but is not: %s" % value)
  if value.dtype != output_type:
    raise TypeError("Padding value tensor (%s) does not match output type: %s" %
                    (value, output_type))
  return value


def _default_padding(input_dataset):
  """Returns default padding tensors in a structure matching `input_dataset`."""
  def make_zero(t):
    if t.base_dtype == dtypes.string:
      return ""
    elif t.base_dtype == dtypes.variant:
      error_msg = ("Unable to create padding for field of type 'variant' "
                   "because t.base_type == dtypes.variant == "
                   "{}.".format(
                       t.base_dtype))
      raise TypeError(error_msg)
    else:
      return np.zeros_like(t.as_numpy_dtype())

  return nest.map_structure(
      make_zero, get_legacy_output_types(input_dataset))


class PaddedBatchDataset(UnaryDataset):
  """A `Dataset` that batches and pads contiguous elements from its input."""

  def __init__(self, input_dataset, batch_size, padded_shapes, padding_values,
               drop_remainder):
    """See `Dataset.batch()` for details."""
    self._input_dataset = input_dataset
    if sparse.any_sparse(get_legacy_output_classes(input_dataset)):
      # TODO(b/63669786): support batching of sparse tensors
      raise TypeError(
          "Batching of padded sparse tensors is not currently supported")
    self._input_dataset = input_dataset
    self._batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")
    padding_values = (
        padding_values
        if padding_values is not None else _default_padding(input_dataset))

    input_shapes = get_legacy_output_shapes(input_dataset)
    flat_padded_shapes = nest.flatten_up_to(input_shapes, padded_shapes)

    flat_padded_shapes_as_tensors = []

    for input_component_shape, padded_shape in zip(
        nest.flatten(input_shapes), flat_padded_shapes):
      flat_padded_shapes_as_tensors.append(
          _padded_shape_to_tensor(padded_shape, input_component_shape))

    self._padded_shapes = nest.pack_sequence_as(input_shapes,
                                                flat_padded_shapes_as_tensors)

    self._padding_values = nest.map_structure_up_to(
        input_shapes, _padding_value_to_tensor, padding_values,
        get_legacy_output_types(input_dataset))
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")

    def _padded_shape_to_batch_shape(s):
      return tensor_shape.TensorShape([
          tensor_util.constant_value(self._batch_size)
          if smart_cond.smart_constant_value(self._drop_remainder) else None
      ]).concatenate(tensor_util.constant_value_as_shape(s))

    output_shapes = nest.map_structure(
        _padded_shape_to_batch_shape, self._padded_shapes)
    self._structure = structure.convert_legacy_structure(
        get_legacy_output_types(self._input_dataset), output_shapes,
        get_legacy_output_classes(self._input_dataset))

    # pylint: disable=protected-access
    # TODO(jsimsa): Switch to using v2 only any time after 6/30/2018.
    if smart_cond.smart_constant_value(self._drop_remainder) is False:
      variant_tensor = gen_dataset_ops.padded_batch_dataset(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          batch_size=self._batch_size,
          padded_shapes=[
              ops.convert_to_tensor(s, dtype=dtypes.int64)
              for s in nest.flatten(self._padded_shapes)
          ],
          padding_values=nest.flatten(self._padding_values),
          output_shapes=structure.get_flat_tensor_shapes(self._structure))
    else:
      variant_tensor = gen_dataset_ops.padded_batch_dataset_v2(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          batch_size=self._batch_size,
          padded_shapes=[
              ops.convert_to_tensor(s, dtype=dtypes.int64)
              for s in nest.flatten(self._padded_shapes)
          ],
          padding_values=nest.flatten(self._padding_values),
          drop_remainder=self._drop_remainder,
          output_shapes=structure.get_flat_tensor_shapes(self._structure))
    super(PaddedBatchDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure


def _should_unpack_args(args):
  """Returns `True` if `args` should be `*args` when passed to a callable."""
  return type(args) is tuple  # pylint: disable=unidiomatic-typecheck


class MapDataset(UnaryDataset):
  """A `Dataset` that maps a function over elements in its input."""

  def __init__(self,
               input_dataset,
               map_func,
               use_inter_op_parallelism=True,
               preserve_cardinality=False,
               use_legacy_function=False):
    """See `Dataset.map()` for details."""
    self._input_dataset = input_dataset
    self._use_inter_op_parallelism = use_inter_op_parallelism
    self._preserve_cardinality = preserve_cardinality
    self._map_func = StructuredFunctionWrapper(
        map_func,
        self._transformation_name(),
        dataset=input_dataset,
        use_legacy_function=use_legacy_function)
    variant_tensor = gen_dataset_ops.map_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        use_inter_op_parallelism=self._use_inter_op_parallelism,
        preserve_cardinality=self._preserve_cardinality,
        **self._flat_structure)
    super(MapDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._map_func.output_structure

  def _transformation_name(self):
    return "Dataset.map()"


class ParallelMapDataset(UnaryDataset):
  """A `Dataset` that maps a function over elements in its input in parallel."""

  def __init__(self,
               input_dataset,
               map_func,
               num_parallel_calls,
               use_inter_op_parallelism=True,
               preserve_cardinality=False,
               use_legacy_function=False):
    """See `Dataset.map()` for details."""
    self._input_dataset = input_dataset
    self._use_inter_op_parallelism = use_inter_op_parallelism
    self._map_func = StructuredFunctionWrapper(
        map_func,
        self._transformation_name(),
        dataset=input_dataset,
        use_legacy_function=use_legacy_function)
    self._num_parallel_calls = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int32, name="num_parallel_calls")
    self._preserve_cardinality = preserve_cardinality
    variant_tensor = gen_dataset_ops.parallel_map_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        num_parallel_calls=self._num_parallel_calls,
        use_inter_op_parallelism=self._use_inter_op_parallelism,
        preserve_cardinality=self._preserve_cardinality,
        **self._flat_structure)
    super(ParallelMapDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._map_func.output_structure

  def _transformation_name(self):
    return "Dataset.map()"


class FlatMapDataset(UnaryDataset):
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(self, input_dataset, map_func):
    """See `Dataset.flat_map()` for details."""
    self._input_dataset = input_dataset
    self._map_func = StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)
    if not isinstance(self._map_func.output_structure, DatasetSpec):
      raise TypeError(
          "`map_func` must return a `Dataset` object. Got {}".format(
              type(self._map_func.output_structure)))
    self._structure = self._map_func.output_structure._element_spec  # pylint: disable=protected-access
    variant_tensor = gen_dataset_ops.flat_map_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        **self._flat_structure)
    super(FlatMapDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._structure

  def _transformation_name(self):
    return "Dataset.flat_map()"


class InterleaveDataset(UnaryDataset):
  """A `Dataset` that interleaves the result of transformed inputs."""

  def __init__(self, input_dataset, map_func, cycle_length, block_length):
    """See `Dataset.interleave()` for details."""
    self._input_dataset = input_dataset
    self._map_func = StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)
    if not isinstance(self._map_func.output_structure, DatasetSpec):
      raise TypeError(
          "`map_func` must return a `Dataset` object. Got {}".format(
              type(self._map_func.output_structure)))
    self._structure = self._map_func.output_structure._element_spec  # pylint: disable=protected-access
    self._cycle_length = ops.convert_to_tensor(
        cycle_length, dtype=dtypes.int64, name="cycle_length")
    self._block_length = ops.convert_to_tensor(
        block_length, dtype=dtypes.int64, name="block_length")

    variant_tensor = gen_dataset_ops.interleave_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,  # pylint: disable=protected-access
        self._cycle_length,
        self._block_length,
        f=self._map_func.function,
        **self._flat_structure)
    super(InterleaveDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._structure

  def _transformation_name(self):
    return "Dataset.interleave()"


class ParallelInterleaveDataset(UnaryDataset):
  """A `Dataset` that maps a function over its input and interleaves the result."""

  def __init__(self, input_dataset, map_func, cycle_length, block_length,
               num_parallel_calls):
    """See `Dataset.interleave()` for details."""
    self._input_dataset = input_dataset
    self._map_func = StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)
    if not isinstance(self._map_func.output_structure, DatasetSpec):
      raise TypeError(
          "`map_func` must return a `Dataset` object. Got {}".format(
              type(self._map_func.output_structure)))
    self._structure = self._map_func.output_structure._element_spec  # pylint: disable=protected-access
    self._cycle_length = ops.convert_to_tensor(
        cycle_length, dtype=dtypes.int64, name="cycle_length")
    self._block_length = ops.convert_to_tensor(
        block_length, dtype=dtypes.int64, name="block_length")
    self._num_parallel_calls = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int64, name="num_parallel_calls")
    variant_tensor = gen_dataset_ops.parallel_interleave_dataset_v2(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,  # pylint: disable=protected-access
        self._cycle_length,
        self._block_length,
        self._num_parallel_calls,
        f=self._map_func.function,
        **self._flat_structure)
    super(ParallelInterleaveDataset, self).__init__(input_dataset,
                                                    variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._structure

  def _transformation_name(self):
    return "Dataset.interleave()"


class FilterDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that filters its input according to a predicate function."""

  def __init__(self, input_dataset, predicate, use_legacy_function=False):
    """See `Dataset.filter()` for details."""
    self._input_dataset = input_dataset
    wrapped_func = StructuredFunctionWrapper(
        predicate,
        self._transformation_name(),
        dataset=input_dataset,
        use_legacy_function=use_legacy_function)
    if not wrapped_func.output_structure.is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.bool)):
      error_msg = ("`predicate` return type must be convertible to a scalar "
                   "boolean tensor. Was {}.").format(
                       wrapped_func.output_structure)
      raise ValueError(error_msg)
    self._predicate = wrapped_func
    variant_tensor = gen_dataset_ops.filter_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        other_arguments=self._predicate.function.captured_inputs,
        predicate=self._predicate.function,
        **self._flat_structure)
    super(FilterDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._predicate]

  def _transformation_name(self):
    return "Dataset.filter()"


class PrefetchDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that asynchronously prefetches its input."""

  def __init__(self, input_dataset, buffer_size, slack_period=None):
    """See `Dataset.prefetch()` for details.

    Args:
      input_dataset: The input dataset.
      buffer_size: See `Dataset.prefetch()` for details.
      slack_period: (Optional.) An integer. If non-zero, determines the number
        of GetNext calls before injecting slack into the execution. This may
        reduce CPU contention at the start of a step. Note that a tensorflow
        user should not have to set this manually; enable this behavior
        automatically via `tf.data.Options.experimental_slack` instead. Defaults
        to None.
    """
    self._input_dataset = input_dataset
    if buffer_size is None:
      buffer_size = -1  # This is the sentinel for auto-tuning.
    self._buffer_size = ops.convert_to_tensor(
        buffer_size, dtype=dtypes.int64, name="buffer_size")
    variant_tensor = gen_dataset_ops.prefetch_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        buffer_size=self._buffer_size,
        slack_period=slack_period,
        **self._flat_structure)
    super(PrefetchDataset, self).__init__(input_dataset, variant_tensor)


class WindowDataset(UnaryDataset):
  """A dataset that creates window datasets from the input elements."""

  def __init__(self, input_dataset, size, shift, stride, drop_remainder):
    """See `window_dataset()` for more details."""
    self._input_dataset = input_dataset
    self._size = ops.convert_to_tensor(size, dtype=dtypes.int64, name="size")
    self._shift = ops.convert_to_tensor(shift, dtype=dtypes.int64, name="shift")
    self._stride = ops.convert_to_tensor(
        stride, dtype=dtypes.int64, name="stride")
    self._drop_remainder = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")
    self._structure = nest.pack_sequence_as(
        get_legacy_output_classes(input_dataset), [
            DatasetSpec(  # pylint: disable=g-complex-comprehension
                structure.convert_legacy_structure(
                    output_type, output_shape, output_class))
            for output_class, output_shape, output_type in zip(
                nest.flatten(get_legacy_output_classes(input_dataset)),
                nest.flatten(get_legacy_output_shapes(input_dataset)),
                nest.flatten(get_legacy_output_types(input_dataset)))
        ])
    variant_tensor = gen_dataset_ops.window_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._size,
        self._shift,
        self._stride,
        self._drop_remainder,
        **self._flat_structure)
    super(WindowDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure


class _OptionsDataset(UnaryUnchangedStructureDataset):
  """An identity `Dataset` that stores options."""

  def __init__(self, input_dataset, options):
    self._input_dataset = input_dataset
    self._options = input_dataset.options()
    if self._options:
      self._options = self._options.merge(options)
    else:
      self._options = options
    variant_tensor = input_dataset._variant_tensor  # pylint: disable=protected-access
    super(_OptionsDataset, self).__init__(input_dataset, variant_tensor)

  def options(self):
    return self._options


class _ModelDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that acts as an identity, and models performance."""

  def __init__(self, input_dataset, algorithm, cpu_budget):
    self._input_dataset = input_dataset
    # TODO(jsimsa): This check is introduced for forward compatibility and can
    # be removed after 7/24/2019. At that point, all servers are expected to
    # recognize the `algorithm` attribute.
    if algorithm != AutotuneAlgorithm.HILL_CLIMB:
      variant_tensor = gen_dataset_ops.model_dataset(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          algorithm=algorithm,
          cpu_budget=cpu_budget,
          **self._flat_structure)
    else:
      variant_tensor = gen_dataset_ops.model_dataset(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          cpu_budget=cpu_budget,
          **self._flat_structure)
    super(_ModelDataset, self).__init__(input_dataset, variant_tensor)


class _OptimizeDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that acts as an identity, and applies optimizations."""

  def __init__(self, input_dataset, optimizations, optimization_configs=None):
    self._input_dataset = input_dataset
    if optimizations is None:
      optimizations = []
    if optimization_configs is None:
      optimization_configs = []
    self._optimizations = ops.convert_to_tensor(
        optimizations, dtype=dtypes.string, name="optimizations")
    variant_tensor = gen_dataset_ops.optimize_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._optimizations,
        optimization_configs=optimization_configs,
        **self._flat_structure)
    super(_OptimizeDataset, self).__init__(input_dataset, variant_tensor)


class _SetStatsAggregatorDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that acts as an identity, and sets a stats aggregator."""

  def __init__(self, input_dataset, aggregator, prefix, counter_prefix):
    self._input_dataset = input_dataset
    self._stats_aggregator = aggregator
    self._prefix = prefix
    self._counter_prefix = counter_prefix
    variant_tensor = ged_ops.set_stats_aggregator_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._stats_aggregator._resource,  # pylint: disable=protected-access
        self._prefix,
        self._counter_prefix,
        **self._flat_structure)
    super(_SetStatsAggregatorDataset, self).__init__(input_dataset,
                                                     variant_tensor)


class _MaxIntraOpParallelismDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that acts as an identity, overriding intra-op parallelism."""

  def __init__(self, input_dataset, max_intra_op_parallelism):
    self._input_dataset = input_dataset
    self._max_intra_op_parallelism = ops.convert_to_tensor(
        max_intra_op_parallelism,
        dtype=dtypes.int64,
        name="max_intra_op_parallelism")
    variant_tensor = ged_ops.max_intra_op_parallelism_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._max_intra_op_parallelism,
        **self._flat_structure)
    super(_MaxIntraOpParallelismDataset, self).__init__(input_dataset,
                                                        variant_tensor)


class _PrivateThreadPoolDataset(UnaryUnchangedStructureDataset):
  """A `Dataset` that acts as an identity, setting a private threadpool."""

  def __init__(self, input_dataset, num_threads):
    self._input_dataset = input_dataset
    self._num_threads = ops.convert_to_tensor(
        num_threads, dtype=dtypes.int64, name="num_threads")
    variant_tensor = ged_ops.private_thread_pool_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._num_threads,
        **self._flat_structure)
    super(_PrivateThreadPoolDataset, self).__init__(input_dataset,
                                                    variant_tensor)


class _RestructuredDataset(UnaryDataset):
  """An internal helper for changing the structure and shape of a dataset."""

  def __init__(self, dataset, structure):
    self._input_dataset = dataset
    self._structure = structure

    variant_tensor = self._input_dataset._variant_tensor  # pylint: disable=protected-access
    super(_RestructuredDataset, self).__init__(dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure


class _UnbatchDataset(UnaryDataset):
  """A dataset that splits the elements of its input into multiple elements."""

  def __init__(self, input_dataset):
    """See `unbatch()` for more details."""
    flat_shapes = input_dataset._flat_shapes  # pylint: disable=protected-access
    if any(s.ndims == 0 for s in flat_shapes):
      raise ValueError("Cannot unbatch an input with scalar components.")
    known_batch_dim = tensor_shape.Dimension(None)
    for s in flat_shapes:
      try:
        known_batch_dim = known_batch_dim.merge_with(s[0])
      except ValueError:
        raise ValueError("Cannot unbatch an input whose components have "
                         "different batch sizes.")
    self._input_dataset = input_dataset
    self._structure = nest.map_structure(
        lambda component_spec: component_spec._unbatch(),  # pylint: disable=protected-access
        get_structure(input_dataset))
    variant_tensor = ged_ops.unbatch_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        **self._flat_structure)
    super(_UnbatchDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def element_spec(self):
    return self._structure
