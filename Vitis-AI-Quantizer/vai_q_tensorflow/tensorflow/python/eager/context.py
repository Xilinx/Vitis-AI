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
"""State management for eager execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import copy
import random
import threading
import numpy as np
import six

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import tf2
from tensorflow.python.eager import executor
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.util import compat
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export

GRAPH_MODE = 0
EAGER_MODE = 1

default_execution_mode = EAGER_MODE if tf2.enabled() else GRAPH_MODE

# Cache from (old_device_name, partial_new_device_name) -> (new_device_name,
# new_device_spec).
# Note that we do not protect this with a lock and instead rely on python's GIL
# and the idempotent nature of writes to provide thread safety.
_device_parsing_cache = {}
_starting_device_spec = pydev.DeviceSpec.from_string("")

_MAXINT32 = 2**31 - 1

DEVICE_PLACEMENT_EXPLICIT = pywrap_tensorflow.TFE_DEVICE_PLACEMENT_EXPLICIT
DEVICE_PLACEMENT_WARN = pywrap_tensorflow.TFE_DEVICE_PLACEMENT_WARN
DEVICE_PLACEMENT_SILENT = pywrap_tensorflow.TFE_DEVICE_PLACEMENT_SILENT
DEVICE_PLACEMENT_SILENT_FOR_INT32 = (
    pywrap_tensorflow.TFE_DEVICE_PLACEMENT_SILENT_FOR_INT32)

SYNC = 0
ASYNC = 1

MIRRORING_NONE = pywrap_tensorflow.TFE_MIRRORING_NONE
MIRRORING_ALL = pywrap_tensorflow.TFE_MIRRORING_ALL

_python_eager_context_create_counter = monitoring.Counter(
    "/tensorflow/api/python/eager_context_create_counter",
    "Counter for number of eager contexts created in Python.")


class _EagerTensorCache(object):
  """Simple cache which evicts items based on length in a FIFO manner."""

  def __init__(self, max_items=256, max_tensor_size=10000):
    self._data = collections.OrderedDict()
    self._max_items = max_items
    self._max_tensor_size = max_tensor_size

  def put(self, key, value):
    if value._num_elements() > self._max_tensor_size:  # pylint: disable=protected-access
      return

    self._data[key] = value

    if len(self._data) > self._max_items:
      self._data.popitem(last=False)

  def get(self, key):
    return self._data.get(key, None)

  def flush(self):
    self._data = {}


class FunctionCallOptions(object):
  """Options applied at call sites of eager functions.

  Eager functions are functions decorated with tf.contrib.eager.defun.
  """

  def __init__(self, executor_type=None, config_proto=None):
    """Constructor.

    Args:
      executor_type: (optional) name of the executor to be used to execute the
        eager function. If None or an empty string, the default Tensorflow
        executor will be used.
      config_proto: (optional) a `config_pb2.ConfigProto` proto or
        a serialized string of that proto.
        The config used by Grappler when optimizing the function graph.
        Each concrete function is optimized the first time is called. Changing
        config_proto after the first call has no effect.
        If config_proto is None, an empty RewriterConfig will be used.
    """
    self.config_proto_serialized = config_proto
    self.executor_type = executor_type

  @property
  def executor_type(self):
    return self._executor_type

  @executor_type.setter
  def executor_type(self, executor_type):
    self._executor_type = executor_type

  @property
  def config_proto_serialized(self):
    return self._config_proto_serialized

  @config_proto_serialized.setter
  def config_proto_serialized(self, config):
    if isinstance(config, config_pb2.ConfigProto):
      self._config_proto_serialized = config.SerializeToString()
    elif isinstance(config, str):
      self._config_proto_serialized = config
    elif config is None:
      self._config_proto_serialized = (
          config_pb2.ConfigProto().SerializeToString())
    else:
      raise ValueError("the rewriter config must be either a "
                       "config_pb2.ConfigProto, or a serialized string of that "
                       "proto or None. got: {}".format(type(config)))


# Map from context_id (an int) to _TensorCaches.
# Dicts are thread safe in CPython.
# TODO(iga): Remove this once TensorCaches are moved to C++.
_tensor_caches_map = {}


class _TensorCaches(threading.local):
  """Thread local tensor caches."""

  def __init__(self):
    super(_TensorCaches, self).__init__()
    self._ones_rank_cache = None
    self._zeros_cache = None

  @property
  def ones_rank_cache(self):
    if not self._ones_rank_cache:
      self._ones_rank_cache = _EagerTensorCache()
    return self._ones_rank_cache

  @property
  def zeros_cache(self):
    if not self._zeros_cache:
      self._zeros_cache = _EagerTensorCache()
    return self._zeros_cache


class _ThreadLocalData(threading.local):
  """Thread local storage for the eager context."""

  def __init__(self):
    super(_ThreadLocalData, self).__init__()
    self.device_spec = _starting_device_spec
    self.device_name = ""
    self.mode = default_execution_mode
    self.is_eager = default_execution_mode == EAGER_MODE
    self.scope_name = ""
    self.summary_writer = None
    self.summary_recording = None
    self.summary_recording_distribution_strategy = True
    self.summary_step = None
    self.function_call_options = None
    self.executor = None


ContextSwitch = collections.namedtuple(
    "ContextSwitch", ["is_building_function", "enter_context_fn",
                      "device_stack"])


# `_ContextSwitchStack` is a `threading.local` to match the semantics of
# ``DefaultGraphStack`, which is also a `threading.local`.
class _ContextSwitchStack(threading.local):
  """A thread-local stack of context switches."""

  def __init__(self, eager):
    super(_ContextSwitchStack, self).__init__()
    self.stack = []
    if eager:
      # Initialize the stack with a pointer to enter the eager context; this
      # ensures that the fact that eager execution was enabled is propagated
      # across threads, since (1) `enable_eager_execution` modifies a
      # process-level flag (`default_execution_mode`) and (2) `__init__` is
      # called each time a threading.local object is used in a separate thread.
      self.push(is_building_function=False, enter_context_fn=eager_mode,
                device_stack=None)

  def push(self, is_building_function, enter_context_fn, device_stack):
    """Push metadata about a context switch onto the stack.

    A context switch can take any one of the two forms: installing a graph as
    the default graph, or entering the eager context. For each context switch,
    we record whether or not the entered context is building a function.

    Args:
      is_building_function: (bool.) Whether the context is building a function.
      enter_context_fn: (function.) A callable that executes the context switch.
        For example, `graph.as_default` or `eager_mode`.
      device_stack: If applicable, the device function stack for this
        graph. When breaking out of graphs in init_scope, the innermost nonempty
        device stack is used. Eager contexts put `None` here and the value is
        never used.
    """

    self.stack.append(
        ContextSwitch(is_building_function, enter_context_fn, device_stack))

  def pop(self):
    """Pop the stack."""

    self.stack.pop()


class LogicalDevice(
    collections.namedtuple("LogicalDevice", ["name", "device_type"])):
  """Abstraction for a device initialized by the runtime.

  A LogicalDevice corresponds to a initialized instance on a PhysicalDevice or a
  remote device available in the cluster. Tensors and operations can be placed
  on a specific LogicalDevice by calling `tf.device()` with the `name` of the
  LogicalDevice.

  Fields:
    name: The fully qualified name of the device. Can be used for Op or function
      placement.
    device_type: String declaring the type of device such as "CPU" or "GPU".
  """
  pass


@tf_export("config.experimental.VirtualDeviceConfiguration")
class VirtualDeviceConfiguration(
    collections.namedtuple("VirtualDeviceConfiguration", ["memory_limit"])):
  """Configuration class for virtual devices for a PhysicalDevice.

  Fields:
    memory_limit: (optional) Maximum memory (in MB) to allocate on the virtual
      device. Currently only supported for GPUs.
  """

  def __new__(cls, memory_limit=None):
    return super(VirtualDeviceConfiguration, cls).__new__(cls, memory_limit)


class PhysicalDevice(
    collections.namedtuple("PhysicalDevice", ["name", "device_type"])):
  """Abstraction for a locally visible physical device.

  TensorFlow can utilize various devices such as the CPU or multiple GPUs
  for computation. Before initializing a local device for use, the user can
  customize certain properties of the device such as it's visibility or memory
  configuration.

  Once a PhysicalDevice is initialized one or many LogicalDevice objects are
  created. Use tf.config.set_virtual_device_configuration() to create multiple
  LogicalDevice objects for a PhysicalDevice. This is useful when separation
  between models is needed.

  Fields:
    name: Unique identifier for device.
    device_type: String declaring the type of device such as "CPU" or "GPU".
  """
  pass


class _AtomicCounter(object):
  """A simple atomic counter."""

  def __init__(self):
    self._value = 0
    self._lock = threading.Lock()

  def increment_and_get(self):
    with self._lock:
      self._value += 1
      return self._value


_context_id_counter = _AtomicCounter()


class _TensorCacheDeleter(object):
  """Deletes tensor caches for a given context."""

  def __init__(self, context_id):
    self._context_id = context_id

  def __del__(self):
    if _tensor_caches_map is None:
      return
    if self._context_id in _tensor_caches_map:
      del _tensor_caches_map[self._context_id]


# Thread-local stack of execution callbacks.
_post_execution_callbacks = threading.local()


# TODO(agarwal): rename to EagerContext / EagerRuntime ?
# TODO(agarwal): consider keeping the corresponding Graph here.
class Context(object):
  """Environment in which eager operations execute."""

  # TODO(agarwal): create and link in some documentation for `execution_mode`.
  # pylint: disable=redefined-outer-name
  def __init__(self,
               config=None,
               device_policy=None,
               execution_mode=None,
               server_def=None):
    """Creates a new Context.

    Args:
      config: (Optional.) A `ConfigProto` protocol buffer with configuration
        options for the Context. Note that a lot of these options may be
        currently unimplemented or irrelevant when eager execution is enabled.
      device_policy: (Optional.) What policy to use when trying to run an
        operation on a device with inputs which are not on that device.
        When set to None, an appropriate value will be picked automatically.
        The value picked may change between TensorFlow releases.

        Defaults to DEVICE_PLACEMENT_SILENT.
        Valid values:
        - DEVICE_PLACEMENT_EXPLICIT: raises an error if the placement is
          not correct.
        - DEVICE_PLACEMENT_WARN: copies the tensors which are not on the
          right device but raises a warning.
        - DEVICE_PLACEMENT_SILENT: silently copies the tensors. This might
          hide performance problems.
        - DEVICE_PLACEMENT_SILENT_FOR_INT32: silently copies int32 tensors,
          raising errors on the other ones.
      execution_mode: (Optional.) Policy controlling how operations dispatched
        are actually executed. When set to None, an appropriate value will be
        picked automatically. The value picked may change between TensorFlow
        releases.
        Valid values:
        - SYNC: executes each operation synchronously.
        - ASYNC: executes each operation asynchronously. These
          operations may return "non-ready" handles.
      server_def: (Optional.) A tensorflow::ServerDef proto.
        Enables execution on remote devices. GrpcServers need to be started by
        creating an identical server_def to this, and setting the appropriate
        task_indexes, so that the servers can communicate. It will then be
        possible to execute operations on remote devices.

    Raises:
     ValueError: If execution_mode is not valid.
    """
    # This _id is used only to index the tensor caches.
    # TODO(iga): Remove this when tensor caches are moved to C++.
    self._id = _context_id_counter.increment_and_get()
    self._tensor_cache_deleter = _TensorCacheDeleter(self._id)
    _tensor_caches_map[self._id] = _TensorCaches()

    self._config = config
    self._thread_local_data = _ThreadLocalData()
    self._context_switches = _ContextSwitchStack(self.executing_eagerly())
    self._context_handle = None
    self._context_devices = None
    self._seed = None
    self._initialize_lock = threading.Lock()
    self._initialized = False
    if device_policy is None:
      device_policy = DEVICE_PLACEMENT_SILENT
    self._device_policy = device_policy
    self._mirroring_policy = None
    if execution_mode not in (None, SYNC, ASYNC):
      raise ValueError(
          "execution_mode should be None/SYNC/ASYNC. Got %s" % execution_mode)
    if execution_mode is None:
      execution_mode = SYNC
    self._default_is_async = execution_mode == ASYNC
    self._server_def = server_def
    self._collective_ops_server_def = None
    self._collective_leader = None
    self._collective_scoped_allocator_enabled_ops = None
    self._collective_use_nccl_communication = None
    self._collective_device_filters = None

    self._device_lock = threading.Lock()
    self._physical_devices = None
    self._visible_device_list = []
    self._memory_growth_map = None
    self._virtual_device_map = {}

    # Values set after construction
    self._optimizer_jit = None
    self._intra_op_parallelism_threads = None
    self._inter_op_parallelism_threads = None
    self._soft_device_placement = None
    self._log_device_placement = None
    self._optimizer_experimental_options = {}

    _python_eager_context_create_counter.get_cell().increase_by(1)
  # pylint: enable=redefined-outer-name

  def _set_global_seed(self, seed):
    """Set a global eager mode seed for random ops."""
    self._seed = seed
    # `random.Random(seed)` needs `seed` to be hashable, while values of type
    # e.g. `np.int64` or `np.ndarray` are not. We use `int(...)` to convert them
    # to int.
    try:
      hash(seed)
    except TypeError:
      seed = int(np.array(seed))
    self._rng = random.Random(seed)
    # Also clear the kernel cache, to reset any existing seeds
    if self._context_handle is not None:
      pywrap_tensorflow.TFE_ContextClearCaches(self._context_handle)

  def _internal_operation_seed(self):
    """Returns a fake operation seed.

      In eager mode, user shouldn't set or depend on operation seed.
      Here, we generate a random seed based on global seed to make
      operation's randomness different and depend on the global seed.

    Returns:
      A fake operation seed based on global seed.
    """
    return self._rng.randint(0, _MAXINT32)

  def _initialize_logical_devices(self):
    """Helper to initialize devices."""
    # Store list of devices
    self._logical_devices = []
    self._context_devices = []
    device_list = pywrap_tensorflow.TFE_ContextListDevices(
        self._context_handle)
    try:
      self._num_gpus = 0
      for i in range(pywrap_tensorflow.TF_DeviceListCount(device_list)):
        dev_name = pywrap_tensorflow.TF_DeviceListName(device_list, i)
        self._context_devices.append(pydev.canonical_name(dev_name))
        spec = pydev.DeviceSpec.from_string(dev_name)
        self._logical_devices.append(
            LogicalDevice(name=dev_name, device_type=spec.device_type))
        dev_type = pywrap_tensorflow.TF_DeviceListType(device_list, i)
        if dev_type == "GPU":
          self._num_gpus += 1

    finally:
      pywrap_tensorflow.TF_DeleteDeviceList(device_list)

  def ensure_initialized(self):
    """Initialize handle and devices if not already done so."""
    if self._initialized:
      return
    with self._initialize_lock:
      if self._initialized:
        return
      assert self._context_devices is None
      opts = pywrap_tensorflow.TFE_NewContextOptions()
      try:
        config_str = self.config.SerializeToString()
        pywrap_tensorflow.TFE_ContextOptionsSetConfig(opts, config_str)
        if self._device_policy is not None:
          pywrap_tensorflow.TFE_ContextOptionsSetDevicePlacementPolicy(
              opts, self._device_policy)
        if self._mirroring_policy is not None:
          pywrap_tensorflow.TFE_ContextOptionsSetMirroringPolicy(
              opts, self._mirroring_policy)
        if self._default_is_async == ASYNC:
          pywrap_tensorflow.TFE_ContextOptionsSetAsync(opts, True)
        context_handle = pywrap_tensorflow.TFE_NewContext(opts)
      finally:
        pywrap_tensorflow.TFE_DeleteContextOptions(opts)
      assert not (self._server_def and self._collective_ops_server_def), (
          "Cannot enable remote execution as well as collective ops at the "
          "moment. If this is important to you, please file an issue.")
      if self._server_def is not None:
        server_def_str = self._server_def.SerializeToString()
        pywrap_tensorflow.TFE_ContextSetServerDef(context_handle, 600,
                                                  server_def_str)
      elif self._collective_ops_server_def is not None:
        server_def_str = self._collective_ops_server_def.SerializeToString()
        pywrap_tensorflow.TFE_EnableCollectiveOps(context_handle,
                                                  server_def_str)

      self._context_handle = context_handle
      self._initialize_logical_devices()
      self._initialized = True

  def _clear_caches(self):
    self.ones_rank_cache().flush()
    self.zeros_cache().flush()
    pywrap_tensorflow.TFE_ClearScalarCache()

  def set_server_def(self, server_def, keep_alive_secs=600):
    """Allow setting a server_def on the context.

    When a server def is replaced, it effectively clears a bunch of caches
    within the context. If you attempt to use a tensor object that was pointing
    to a tensor on the remote device, it will raise an error.

    Args:
      server_def: A tensorflow::ServerDef proto.
        Enables execution on remote devices.
      keep_alive_secs: Num. seconds after which the remote end will hang up.
        As long as the client is still alive, the server state for the context
        will be kept alive. If the client is killed (or there is some failure),
        the server will clean up its context keep_alive_secs after the final RPC
        it receives.

    Raises:
      ValueError: if server_def is None.
    """
    if not server_def:
      raise ValueError("server_def is None.")

    self._server_def = server_def

    if self._context_handle:
      server_def_str = server_def.SerializeToString()
      pywrap_tensorflow.TFE_ContextSetServerDef(self._context_handle,
                                                keep_alive_secs, server_def_str)
      self._initialize_logical_devices()

    # Clear all the caches in case there are remote tensors in them.
    self._clear_caches()

  def enable_collective_ops(self, server_def):
    """Enable distributed collective ops with an appropriate server_def.

    Args:
      server_def: A tensorflow::ServerDef proto. Enables execution on remote
        devices.

    Raises:
      ValueError: if server_def is None.
      RuntimeError: if this method is not called at program startup.
    """
    if not server_def:
      raise ValueError("server_def is None.")

    if self._context_handle is not None:
      raise RuntimeError("Collective ops must be enabled at program startup")

    self._collective_ops_server_def = server_def

  def configure_collective_ops(
      self,
      collective_leader="",
      scoped_allocator_enabled_ops=("CollectiveReduce",),
      use_nccl_communication=False,
      device_filters=None):
    """Configure collective ops.

      Collective group leader is necessary for collective ops to run, other
      configurations are mainly for the purpose of performance.

    Args:
      collective_leader: a device string for collective leader, e.g.
        "/job:worker/replica:0/task:"; empty string means local execution of
          collective ops.
      scoped_allocator_enabled_ops: a tuple or a list of op names for scoped
        allocator to run with.
      use_nccl_communication: whether to use nccl communication for collective
        ops.
      device_filters: a tuple or a list of device strings. If set, corresponding
        task can only see the devices filtered by these device filters.

    Raises:
      RuntimeError: if this method is not called at program startup.
    """
    if self._collective_leader is not None:
      if (self._collective_leader != collective_leader or
          self._collective_scoped_allocator_enabled_ops !=
          scoped_allocator_enabled_ops or
          self._collective_use_nccl_communication != use_nccl_communication or
          self._collective_device_filters != device_filters):
        raise ValueError("Collective ops are already configured.")
      else:
        return

    if self._context_handle is not None:
      raise RuntimeError("Collective ops must be configured at program startup")

    self._collective_leader = collective_leader
    self._collective_scoped_allocator_enabled_ops = scoped_allocator_enabled_ops
    self._collective_use_nccl_communication = use_nccl_communication
    self._collective_device_filters = device_filters

  @property
  def _handle(self):
    if self._context_handle is None:
      raise AssertionError("Context must be initialized first.")

    return self._context_handle

  @property
  def _devices(self):
    if self._context_devices is None:
      raise AssertionError("Context must be initialized first.")

    return self._context_devices

  def __str__(self):
    if self._context_handle is None:
      return "Eager TensorFlow Context. Devices currently uninitialized."
    else:
      devices = self._devices
      lines = ["Eager TensorFlow Context with %d devices" % (len(devices))]
      for i, d in enumerate(devices):
        lines.append("   Device %d: %s" % (i, d))
      return "\n".join(lines)

  @tf_contextlib.contextmanager
  def _mode(self, mode):
    """A context manager to allow setting the mode to EAGER/GRAPH."""
    ctx = self._thread_local_data
    old_mode = ctx.mode
    old_is_eager = ctx.is_eager
    ctx.mode = mode
    ctx.is_eager = mode == EAGER_MODE
    if mode == EAGER_MODE:
      # Entering graph mode does not provide us with sufficient information to
      # record a context switch; graph-based context switches are only logged
      # when a graph is registered as the default graph.
      self.context_switches.push(False, eager_mode, None)
    try:
      yield
    finally:
      ctx.is_eager = old_is_eager
      ctx.mode = old_mode
      if mode == EAGER_MODE:
        self.context_switches.pop()

  def executing_eagerly(self):
    """Returns True if current thread has eager executing enabled."""
    return self._thread_local_data.is_eager

  def ones_rank_cache(self):
    """Per-device cache for scalars."""
    return _tensor_caches_map[self._id].ones_rank_cache

  def zeros_cache(self):
    """Per-device cache for scalars."""
    return _tensor_caches_map[self._id].zeros_cache

  @property
  def scope_name(self):
    """Returns scope name for the current thread."""
    return self._thread_local_data.scope_name

  @scope_name.setter
  def scope_name(self, s):
    """Sets scope name for the current thread."""
    self._thread_local_data.scope_name = s

  @property
  def summary_writer(self):
    """Returns default summary writer for the current thread."""
    return self._thread_local_data.summary_writer

  @summary_writer.setter
  def summary_writer(self, writer):
    """Sets default summary writer for the current thread."""
    self._thread_local_data.summary_writer = writer

  @property
  def summary_recording(self):
    """Returns summary recording condition."""
    return self._thread_local_data.summary_recording

  @summary_recording.setter
  def summary_recording(self, condition):
    """Sets summary recording condition."""
    self._thread_local_data.summary_recording = condition

  @property
  def summary_recording_distribution_strategy(self):
    """Returns summary recording condition for distribution strategy."""
    return self._thread_local_data.summary_recording_distribution_strategy

  @summary_recording_distribution_strategy.setter
  def summary_recording_distribution_strategy(self, condition):
    """Sets summary recording condition for distribution strategy."""
    self._thread_local_data.summary_recording_distribution_strategy = condition

  @property
  def summary_step(self):
    """Returns summary step variable."""
    return self._thread_local_data.summary_step

  @summary_step.setter
  def summary_step(self, step):
    """Sets summary step variable."""
    self._thread_local_data.summary_step = step

  @property
  def device_name(self):
    """Returns the device name for the current thread."""
    return self._thread_local_data.device_name

  @property
  def device_spec(self):
    """Returns the device spec for the current thread."""
    return self._thread_local_data.device_spec

  def _set_device(self, device_name, device_spec):
    self._thread_local_data.device_name = device_name
    self._thread_local_data.device_spec = device_spec

  def device(self, name):
    """Context-manager to force placement of operations and Tensors on a device.

    Args:
      name: Name of the device or None to get default placement.

    Returns:
      Context manager that forces device placement.

    Raises:
      ValueError: If name is not a string or is an invalid device name.
      RuntimeError: If device scopes are not properly nested.
    """
    return _EagerDeviceContext(self, name)

  def devices(self):
    """List of the names of devices available to execute operations."""
    return self._devices

  # TODO(fishx): remove this property.
  @property
  def execution_mode(self):
    """Gets execution mode for current thread."""
    return ASYNC if self.is_async() else SYNC

  @execution_mode.setter
  def execution_mode(self, mode):
    """Sets execution mode for current thread."""
    if mode not in (None, SYNC, ASYNC):
      raise ValueError(
          "Execution mode should be None/SYNC/ASYNC. Got %s" % mode)

    if mode is None:
      mode = SYNC

    enable_async = (mode == ASYNC)
    if self.is_async() != enable_async:
      # Only set the execution mode if the context has already been initialized
      if self._context_handle is not None:
        self.executor.wait()
        executor_new = executor.new_executor(enable_async)
        self._thread_local_data.executor = executor_new
        pywrap_tensorflow.TFE_ContextSetExecutorForThread(
            self._context_handle, executor_new.handle())
      else:
        self._default_is_async = enable_async

  def is_async(self):
    if self._context_handle is not None:
      return self.executor.is_async()
    else:
      return self._default_is_async

  @property
  def executor(self):
    ensure_initialized()
    return executor.Executor(
        pywrap_tensorflow.TFE_ContextGetExecutorForThread(self._context_handle))

  @executor.setter
  def executor(self, e):
    ensure_initialized()
    pywrap_tensorflow.TFE_ContextSetExecutorForThread(self._context_handle,
                                                      e.handle())

  @property
  def config(self):
    """Return the ConfigProto with all runtime deltas applied."""
    # Ensure physical devices have been discovered and config has been imported
    self._initialize_physical_devices()

    config = config_pb2.ConfigProto()
    if self._config is not None:
      config.CopyFrom(self._config)

    if self._optimizer_jit is not None:
      config.graph_options.optimizer_options.global_jit_level = (
          config_pb2.OptimizerOptions.ON_1
          if self._optimizer_jit else config_pb2.OptimizerOptions.OFF)
    if self._intra_op_parallelism_threads is not None:
      config.intra_op_parallelism_threads = self._intra_op_parallelism_threads
    if self._inter_op_parallelism_threads is not None:
      config.inter_op_parallelism_threads = self._inter_op_parallelism_threads

    if self._soft_device_placement is not None:
      config.allow_soft_placement = self._soft_device_placement
    else:
      config.allow_soft_placement = self.executing_eagerly()

    if self._log_device_placement is not None:
      config.log_device_placement = self._log_device_placement

    def rewriter_toggle(option):
      toggle = self._optimizer_experimental_options.get(option, None)
      if toggle is None:
        return

      setattr(config.graph_options.rewrite_options,
              option,
              (rewriter_config_pb2.RewriterConfig.ON
               if toggle else rewriter_config_pb2.RewriterConfig.OFF))

    def rewriter_bool(option):
      toggle = self._optimizer_experimental_options.get(option, None)
      if toggle is None:
        return

      setattr(config.graph_options.rewrite_options,
              option,
              toggle)

    rewriter_toggle("layout_optimizer")
    rewriter_toggle("constant_folding")
    rewriter_toggle("shape_optimization")
    rewriter_toggle("remapping")
    rewriter_toggle("arithmetic_optimization")
    rewriter_toggle("dependency_optimization")
    rewriter_toggle("loop_optimization")
    rewriter_toggle("function_optimization")
    rewriter_toggle("debug_stripper")
    rewriter_bool("disable_model_pruning")
    rewriter_toggle("scoped_allocator_optimization")
    rewriter_toggle("pin_to_host_optimization")
    rewriter_toggle("implementation_selector")
    rewriter_toggle("auto_mixed_precision")
    rewriter_bool("disable_meta_optimizer")
    nodes = self._optimizer_experimental_options.get("min_graph_nodes", None)
    if nodes is not None:
      config.graph_options.rewrite_options.min_graph_nodes = nodes

    # Compute device counts
    config.device_count["CPU"] = 0
    config.device_count["GPU"] = 0
    for dev in self._physical_devices:
      if dev not in self._visible_device_list:
        continue

      virtual_devices = self._virtual_device_map.get(dev)
      if virtual_devices is None:
        config.device_count[dev.device_type] += 1
      else:
        config.device_count[dev.device_type] += len(virtual_devices)

    # Configure gpu_options
    gpu_options = self._compute_gpu_options()
    config.gpu_options.MergeFrom(gpu_options)

    # Configure collective ops
    if self._collective_leader:
      config.experimental.collective_group_leader = self._collective_leader
    if self._collective_scoped_allocator_enabled_ops:
      rewrite_options = config.graph_options.rewrite_options
      rewrite_options.scoped_allocator_optimization = (
          rewriter_config_pb2.RewriterConfig.ON)
      del rewrite_options.scoped_allocator_opts.enable_op[:]
      for op in self._collective_scoped_allocator_enabled_ops:
        rewrite_options.scoped_allocator_opts.enable_op.append(op)
    if self._collective_use_nccl_communication:
      config.experimental.collective_nccl = True
    if self._collective_device_filters:
      del config.device_filters[:]
      for f in self._collective_device_filters:
        config.device_filters.append(f)

    return config

  def _compute_gpu_options(self):
    """Build the GPUOptions proto."""
    visible_device_list = []
    virtual_devices = []
    gpu_index = -1
    memory_growths = set()
    for dev in self.list_physical_devices("GPU"):
      gpu_index += 1

      if dev not in self._visible_device_list:
        continue

      growth = self._memory_growth_map[dev]
      memory_growths.add(growth)
      visible_device_list.append(str(gpu_index))

      if self._virtual_device_map:
        vdevs = self._virtual_device_map.get(dev, [])
        device_limits = []
        for virt_dev in vdevs:
          device_limits.append(virt_dev.memory_limit)

        virtual_devices.append(
            config_pb2.GPUOptions.Experimental.VirtualDevices(
                memory_limit_mb=device_limits))

    # Only compute growth if virtual devices have not been configured and we
    # have GPUs
    if not virtual_devices and memory_growths:
      if len(memory_growths) > 1:
        raise ValueError("Memory growth cannot differ between GPU devices")
      allow_growth = memory_growths.pop()
    else:
      allow_growth = None

    return config_pb2.GPUOptions(
        allow_growth=allow_growth,
        visible_device_list=",".join(visible_device_list),
        experimental=config_pb2.GPUOptions.Experimental(
            virtual_devices=virtual_devices))

  @property
  def function_call_options(self):
    """Returns function call options for current thread.

    Note that the returned object is still referenced by the eager context.

    Returns: the FunctionCallOptions for current thread.
    """
    if self._thread_local_data.function_call_options is None:
      config = self.config

      # Default to soft placement for functions unless specified
      if self._soft_device_placement is None:
        config.allow_soft_placement = True
      self._thread_local_data.function_call_options = FunctionCallOptions(
          config_proto=config)

    return self._thread_local_data.function_call_options

  @function_call_options.setter
  def function_call_options(self, options):
    """Returns function call options for current thread."""
    self._thread_local_data.function_call_options = options

  def num_gpus(self):
    """The number of GPUs available to execute operations."""
    self.ensure_initialized()
    return self._num_gpus

  def add_function(self, fn):
    """Add a function definition to the context.

    Once added, the function (identified by its name) can be executed like any
    other operation.

    Args:
      fn: A wrapped TF_Function (returned from TF_GraphToFunction_wrapper).
    """
    self.ensure_initialized()
    pywrap_tensorflow.TFE_ContextAddFunction(self._handle, fn)

  def add_function_def(self, fdef):
    """Add a function definition to the context.

    Once added, the function (identified by its name) can be executed like any
    other operation.

    Args:
      fdef: A FunctionDef protocol buffer message.
    """
    self.ensure_initialized()
    fdef_string = fdef.SerializeToString()
    pywrap_tensorflow.TFE_ContextAddFunctionDef(
        self._handle, fdef_string, len(fdef_string))

  def remove_function(self, name):
    """Remove a function from the context.

    Once removed, the function cannot be executed anymore.

    Args:
      name: function signature name.
    """
    self.ensure_initialized()
    pywrap_tensorflow.TFE_ContextRemoveFunction(self._handle, name)

  def has_function(self, name):
    """Check if a function `name` is registered."""
    self.ensure_initialized()
    return bool(pywrap_tensorflow.TFE_ContextHasFunction(self._handle, name))

  def add_post_execution_callback(self, callback):
    """Add a post-execution callback to the context.

    A post-execution callback is invoked immediately after an eager operation or
    function has finished execution, providing access to the op's type, name
    input and output tensors. Multiple execution callbacks can be added, in
    which case the callbacks will be invoked in the order in which they are
    added.

    Args:
      callback: a callable of the signature
      `f(op_type, op_name, attrs, inputs, outputs)`.
      `op_type` is the type of the operation that was just executed (e.g.,
        `MatMul`).
      `op_name` is the name of the operation that has was just executed. This
        name is set by the client who created the operation and can be `None` if
        it is unset.
      `attrs` contains the attributes of the operation as a `tuple` of
        alternating attribute names and attribute values.
      `inputs` is the `list` of input `Tensor`(s) to the op.
      `outputs` is the `list` of output `Tensor`(s) from the op.
       Return value(s) from the callback are ignored.
    """
    self.post_execution_callbacks.append(callback)

  def clear_post_execution_callbacks(self):
    """Clear all post-execution callbacks added to the context."""
    del self.post_execution_callbacks[:]

  @property
  def post_execution_callbacks(self):
    """Get the list of post-execution callbacks added to the context."""
    if not hasattr(_post_execution_callbacks, "callbacks"):
      _post_execution_callbacks.callbacks = []
    return _post_execution_callbacks.callbacks

  def _initialize_physical_devices(self):
    """Get local devices visible to the system."""
    # We lazy initialize self._physical_devices since we do not want to do this
    # the constructor since the backend may not be initialized yet.
    with self._device_lock:
      if self._physical_devices is not None:
        return

      devs = pywrap_tensorflow.TF_ListPhysicalDevices()
      self._physical_devices = [
          PhysicalDevice(name=d.decode(),
                         device_type=d.decode().split(":")[1]) for d in devs]
      # Construct the visible device list from all physical devices but ignore
      # XLA devices
      self._visible_device_list = [
          d for d in self._physical_devices
          if not d.device_type.startswith("XLA")
      ]
      self._memory_growth_map = {
          d: None for d in self._physical_devices if d.device_type == "GPU"
      }

    # Import device settings that may have been passed into the constructor
    self._import_config()

  def list_physical_devices(self, device_type=None):
    """List local devices visible to the system.

    This API allows a client to query the devices before they have been
    initialized by the eager runtime. Additionally a user can filter by device
    type, to get only CPUs or GPUs.

    Args:
      device_type: Optional device type to limit results to

    Returns:
      List of PhysicalDevice objects.
    """
    self._initialize_physical_devices()

    if device_type is not None:
      return [
          d for d in self._physical_devices
          if device_type is None or device_type == d.device_type
      ]

    return self._physical_devices

  def _import_config(self):
    """Import config if passed in during construction.

    If Context was created with a ConfigProto such as when calling
    tf.compat.v1.enable_eager_execution(), then we need to pull out the
    various pieces we might be replacing and import then into our internal
    class representation.
    """
    if self._config is None:
      return

    num_cpus = self._config.device_count.get("CPU", 1)
    if num_cpus != 1:
      cpus = [d for d in self._physical_devices if d.device_type == "CPU"]
      if num_cpus == 0:
        self.set_visible_devices([], "CPU")
      elif num_cpus > 1:
        self.set_virtual_device_configuration(
            cpus[0], [VirtualDeviceConfiguration() for _ in range(num_cpus)])

    # Parse GPU options
    gpus = [d for d in self._physical_devices if d.device_type == "GPU"]

    # If there are no GPUs detected, simply ignore all the GPU options passed in
    # rather than doing any validation checks.
    if not gpus:
      return

    gpu_count = self._config.device_count.get("GPU", None)

    visible_gpus = []
    # TODO(gjn): Handle importing existing virtual GPU configuration
    visible_indices = self._config.gpu_options.visible_device_list
    if visible_indices:
      for index in visible_indices.split(","):
        if int(index) >= len(gpus):
          raise ValueError("Invalid visible device index: %s" % index)
        visible_gpus.append(gpus[int(index)])
    else:
      visible_gpus = gpus

    if gpu_count is not None:
      visible_gpus = visible_gpus[:gpu_count]

    self.set_visible_devices(visible_gpus, "GPU")

  def list_logical_devices(self, device_type=None):
    """Return logical devices."""
    self.ensure_initialized()

    devices = []
    for dev in self._logical_devices:
      if device_type is not None and device_type != dev.device_type:
        continue

      devices.append(dev)

    return devices

  def get_visible_devices(self, device_type=None):
    """Get the list of visible devices."""
    self._initialize_physical_devices()

    if device_type is None:
      return self._visible_device_list
    else:
      return [
          d for d in self._visible_device_list if d.device_type == device_type
      ]

  def set_visible_devices(self, devices, device_type=None):
    """Set the list of visible devices."""
    self._initialize_physical_devices()

    if not isinstance(devices, list):
      devices = [devices]

    for d in devices:
      if d not in self._physical_devices:
        raise ValueError("Unrecognized device: %s" % repr(d))
      if device_type is not None and d.device_type != device_type:
        raise ValueError("Unrecognized device: %s" % repr(d))

    visible_device_list = []
    if device_type is not None:
      visible_device_list = [
          d for d in self._visible_device_list if d.device_type != device_type
      ]

    visible_device_list += devices

    if self._visible_device_list == visible_device_list:
      return

    if self._context_handle is not None:
      raise RuntimeError(
          "Visible devices cannot be modified after being initialized")

    self._visible_device_list = visible_device_list

  def get_memory_growth(self, dev):
    """Get if memory growth is enabled for a PhysicalDevice."""
    self._initialize_physical_devices()

    if dev not in self._physical_devices:
      raise ValueError("Unrecognized device: %s" % repr(dev))

    return self._memory_growth_map[dev]

  def set_memory_growth(self, dev, enable):
    """Set if memory growth should be enabled for a PhysicalDevice."""
    self._initialize_physical_devices()

    if dev not in self._physical_devices:
      raise ValueError("Unrecognized device: %s" % repr(dev))

    if dev in self._virtual_device_map:
      raise ValueError(
          "Cannot set memory growth on device when virtual devices configured")

    if dev.device_type != "GPU":
      raise ValueError("Cannot set memory growth on non-GPU devices")

    if self._memory_growth_map.get(dev) == enable:
      return

    if self._context_handle is not None:
      raise RuntimeError(
          "Physical devices cannot be modified after being initialized")

    self._memory_growth_map[dev] = enable

  def get_virtual_device_configuration(self, dev):
    """Get the virtual device configuration for a PhysicalDevice."""
    self._initialize_physical_devices()

    if dev not in self._physical_devices:
      raise ValueError("Unrecognized device: %s" % repr(dev))

    return self._virtual_device_map.get(dev)

  def set_virtual_device_configuration(self, dev, virtual_devices):
    """Set the virtual device configuration for a PhysicalDevice."""
    self._initialize_physical_devices()

    if dev not in self._physical_devices:
      raise ValueError("Unrecognized device: %s" % repr(dev))

    if dev.device_type == "CPU":
      for vdev in virtual_devices:
        if vdev.memory_limit is not None:
          raise ValueError("Setting memory limit on CPU virtual devices is "
                           "currently not supported")
    elif dev.device_type == "GPU":
      for vdev in virtual_devices:
        if vdev.memory_limit is None:
          raise ValueError(
              "Setting memory limit is required for GPU virtual devices is")
    else:
      raise ValueError("Virtual devices are not supported for %s" %
                       dev.device_type())

    if self._virtual_device_map.get(dev) == virtual_devices:
      return

    if self._context_handle is not None:
      raise RuntimeError(
          "Virtual devices cannot be modified after being initialized")

    self._virtual_device_map[dev] = virtual_devices

  @property
  def optimizer_jit(self):
    level = self.config.graph_options.optimizer_options.global_jit_level
    return (level == config_pb2.OptimizerOptions.ON_1 or
            level == config_pb2.OptimizerOptions.ON_2)

  @optimizer_jit.setter
  def optimizer_jit(self, enabled):
    self._optimizer_jit = enabled

    self._thread_local_data.function_call_options = None

  def get_optimizer_experimental_options(self):
    """Get experimental options for the optimizer.

    Returns:
      Dictionary of current option values
    """
    rewrite_options = self.config.graph_options.rewrite_options
    options = {}

    def rewriter_toggle(option):
      attr = getattr(rewrite_options, option)
      if attr != 0:
        options[option] = (attr == rewriter_config_pb2.RewriterConfig.ON)

    def rewriter_bool(option):
      options[option] = getattr(rewrite_options, option)

    rewriter_toggle("layout_optimizer")
    rewriter_toggle("constant_folding")
    rewriter_toggle("shape_optimization")
    rewriter_toggle("remapping")
    rewriter_toggle("arithmetic_optimization")
    rewriter_toggle("dependency_optimization")
    rewriter_toggle("loop_optimization")
    rewriter_toggle("function_optimization")
    rewriter_toggle("debug_stripper")
    rewriter_bool("disable_model_pruning")
    rewriter_toggle("scoped_allocator_optimization")
    rewriter_toggle("pin_to_host_optimization")
    rewriter_toggle("implementation_selector")
    rewriter_toggle("auto_mixed_precision")
    rewriter_bool("disable_meta_optimizer")

    if rewrite_options.min_graph_nodes != 0:
      options["min_graph_nodes"] = rewrite_options.min_graph_nodes

    return options

  def set_optimizer_experimental_options(self, options):
    """Set experimental options for the optimizer.

    Args:
      options: Dictionary of options to modify
    """
    self._optimizer_experimental_options.update(options)

    self._thread_local_data.function_call_options = None

  @property
  def intra_op_parallelism_threads(self):
    return self.config.intra_op_parallelism_threads

  @intra_op_parallelism_threads.setter
  def intra_op_parallelism_threads(self, num_threads):
    if self._intra_op_parallelism_threads == num_threads:
      return

    if self._context_handle is not None:
      raise RuntimeError(
          "Intra op parallelism cannot be modified after initialization.")

    self._intra_op_parallelism_threads = num_threads

  @property
  def inter_op_parallelism_threads(self):
    return self.config.inter_op_parallelism_threads

  @inter_op_parallelism_threads.setter
  def inter_op_parallelism_threads(self, num_threads):
    if self._inter_op_parallelism_threads == num_threads:
      return

    if self._context_handle is not None:
      raise RuntimeError(
          "Inter op parallelism cannot be modified after initialization.")

    self._inter_op_parallelism_threads = num_threads

  @property
  def soft_device_placement(self):
    return self.config.allow_soft_placement

  @soft_device_placement.setter
  def soft_device_placement(self, enabled):
    self._soft_device_placement = enabled

    self._thread_local_data.function_call_options = None

  @property
  def log_device_placement(self):
    return self.config.log_device_placement

  @log_device_placement.setter
  def log_device_placement(self, enabled):
    if self._log_device_placement == enabled:
      return

    if self._context_handle is not None:
      raise RuntimeError(
          "Device placement logging must be set at program startup")

    self._log_device_placement = enabled
    self._thread_local_data.function_call_options = None

  @property
  def device_policy(self):
    # Only get the policy from the context if it has already been initialized
    if self._context_handle is not None:
      return pywrap_tensorflow.TFE_ContextGetDevicePlacementPolicy(self._handle)

    return self._device_policy

  @device_policy.setter
  def device_policy(self, policy):
    if policy is None:
      policy = DEVICE_PLACEMENT_SILENT

    if self._device_policy != policy:
      self._device_policy = policy

      # Only set the policy if the context has already been initialized
      if self._context_handle is not None:
        pywrap_tensorflow.TFE_ContextSetThreadLocalDevicePlacementPolicy(
            self._handle, self._device_policy)

  @property
  def mirroring_policy(self):
    # Only get the policy from the context if it has already been initialized
    if self._context_handle is not None:
      return pywrap_tensorflow.TFE_ContextGetMirroringPolicy(self._handle)

    return self._mirroring_policy

  @mirroring_policy.setter
  def mirroring_policy(self, policy):
    if policy is None:
      policy = MIRRORING_NONE

    if self._mirroring_policy != policy:
      self._mirroring_policy = policy

      # Only set the policy if the context has already been initialized
      if self._context_handle is not None:
        pywrap_tensorflow.TFE_ContextSetThreadLocalMirroringPolicy(
            self._handle, self._mirroring_policy)

  def enable_run_metadata(self):
    """Enables tracing of op execution via RunMetadata.

    To retrieve the accumulated metadata call context.export_run_metadata()
    and to stop tracing call context.disable_run_metadata().
    """
    self.ensure_initialized()
    pywrap_tensorflow.TFE_ContextEnableRunMetadata(self._handle)

  def disable_run_metadata(self):
    """Disables tracing of op execution via RunMetadata."""
    if not self._context_handle:
      return
    pywrap_tensorflow.TFE_ContextDisableRunMetadata(self._context_handle)

  def enable_graph_collection(self):
    """Enables graph collection of executed functions.

    To retrieve the accumulated graphs call context.export_run_metadata()
    and to stop collecting graphs call context.disable_graph_collection().
    """
    self.ensure_initialized()
    pywrap_tensorflow.TFE_ContextEnableGraphCollection(self._handle)

  def disable_graph_collection(self):
    """Disables graph collection of executed functions."""
    if not self._context_handle:
      return
    pywrap_tensorflow.TFE_ContextDisableGraphCollection(self._context_handle)

  def export_run_metadata(self):
    """Returns a RunMetadata proto with accumulated information.

    The returned protocol buffer contains information since the most recent call
    to either enable_run_metadata or export_run_metadata.

    Returns:
      A RunMetadata protocol buffer. Or None if not enabled.
    """
    if not self._context_handle:
      return None
    with c_api_util.tf_buffer() as buffer_:
      pywrap_tensorflow.TFE_ContextExportRunMetadata(
          self._context_handle, buffer_)
      proto_data = pywrap_tensorflow.TF_GetBuffer(buffer_)
    run_metadata = config_pb2.RunMetadata()
    run_metadata.ParseFromString(compat.as_bytes(proto_data))
    return run_metadata

  @property
  def context_switches(self):
    """Returns a stack of context switches."""
    return self._context_switches

  def start_step(self):
    pywrap_tensorflow.TFE_ContextStartStep(self._handle)

  def end_step(self):
    pywrap_tensorflow.TFE_ContextEndStep(self._handle)


class _EagerDeviceContext(object):
  """Context-manager forcing placement of ops and Tensors on a device."""

  def __init__(self, ctx, device_name):
    self._device_name = device_name
    self._ctx = ctx
    self._stack = []

  def __enter__(self):
    ctx = self._ctx
    old_device_name = ctx.device_name
    old_device_spec = ctx.device_spec
    new_device_name = self._device_name
    cache_key = (old_device_name, new_device_name)
    try:
      new_device_name, new_device_spec = _device_parsing_cache[cache_key]
    except TypeError:
      # Error while trying to compute the cache key.
      raise ValueError("Expecting a string device name. Got %s(%s)" %
                       (type(new_device_name), new_device_name))
    except KeyError:
      # Handle a cache miss.
      if new_device_name is not None:
        if not isinstance(new_device_name, six.string_types):
          raise ValueError("Expecting a string device name. Got %s(%s)" %
                           (type(new_device_name), new_device_name))
        device_spec = pydev.DeviceSpec.from_string(new_device_name)
        if old_device_name:
          new_device_spec = copy.copy(old_device_spec)
        else:
          ctx.ensure_initialized()
          new_device_spec = pydev.DeviceSpec.from_string(
              ctx._context_devices[0])  # pylint: disable=protected-access
        new_device_spec = new_device_spec.make_merged_spec(device_spec)
      else:
        new_device_spec = pydev.DeviceSpec.from_string("")
      new_device_name = new_device_spec.to_string()
      _device_parsing_cache[cache_key] = (new_device_name, new_device_spec)

    ctx._set_device(new_device_name, new_device_spec)  # pylint: disable=protected-access
    self._stack.append((old_device_name, old_device_spec, new_device_spec))

  def __exit__(self, *ex_info):
    ctx = self._ctx
    old_device_name, old_device_spec, new_device_spec = self._stack[-1]
    if ctx.device_spec is not new_device_spec:
      raise RuntimeError(
          "Exiting device scope without proper scope nesting")
    del self._stack[-1]
    ctx._set_device(old_device_name, old_device_spec)  # pylint: disable=protected-access


# Do not set directly. Use _set_context.
_context = None
_context_lock = threading.Lock()


def _set_context_locked(ctx):
  global _context
  pywrap_tensorflow.TFE_Py_SetEagerContext(ctx)
  _context = ctx


def _set_context(ctx):
  with _context_lock:
    _set_context_locked(ctx)


def _create_context():
  with _context_lock:
    if _context is None:
      ctx = Context()
      _set_context_locked(ctx)


def context():
  """Returns a singleton context object."""
  if _context is None:
    _create_context()
  return _context


def context_safe():
  """Returns current context (or None if one hasn't been initialized)."""
  return _context


def ensure_initialized():
  """Initialize the context."""
  context().ensure_initialized()


def set_global_seed(seed):
  """Sets the eager mode seed."""
  context()._set_global_seed(seed)  # pylint: disable=protected-access


def global_seed():
  """Returns the eager mode seed."""
  return context()._seed  # pylint: disable=protected-access


def internal_operation_seed():
  """Returns the operation seed generated based on global seed."""
  return context()._internal_operation_seed()  # pylint: disable=protected-access


@tf_export("executing_eagerly")
def executing_eagerly():
  """Returns True if the current thread has eager execution enabled.

  Eager execution is typically enabled via
  `tf.compat.v1.enable_eager_execution`, but may also be enabled within the
  context of a Python function via tf.contrib.eager.py_func.
  """
  if context_safe() is None:
    return default_execution_mode == EAGER_MODE

  return context().executing_eagerly()


def in_eager_mode():
  """Use executing_eagerly() instead. This function will be removed."""
  return executing_eagerly()


def shared_name(name=None):
  """Returns the anonymous shared name GUID if no shared name is specified.

  In eager mode we need to use a unique shared name to avoid spurious sharing
  issues. The runtime generates a unique name on our behalf when the reserved
  GUID is used as a shared name.

  Args:
    name: Optional shared name

  Returns:
    Eager compatible shared name.
  """
  if name or not executing_eagerly():
    return name

  # Ensure a unique name when eager execution is enabled to avoid spurious
  # sharing issues.
  return "cd2c89b7-88b7-44c8-ad83-06c2a9158347"


def graph_mode():
  """Context-manager to disable eager execution for the current thread."""
  return context()._mode(GRAPH_MODE)  # pylint: disable=protected-access


def eager_mode():
  """Context-manager to enable eager execution for the current thread."""
  return context()._mode(EAGER_MODE)  # pylint: disable=protected-access


# TODO(agarwal): get rid of this and use ops.name_scope instead.
@contextlib.contextmanager
def namescope(name):
  """ContextManager for creating hierarchical name scopes."""
  ctx = context()
  old_name = ctx.scope_name
  ctx.scope_name = "%s/%s" % (old_name, name) if old_name else name
  try:
    yield
  finally:
    ctx.scope_name = old_name


def scope_name():
  """Name of the current scope."""
  return context().scope_name


def device(name):
  """Context-manager to force placement of operations and Tensors on a device.

  Example:
  ```python
  with tf.device('gpu:0'):
    with tf.device('cpu:0'):
      shape = tf.constant([], dtype=tf.int32)
    x = tf.random.truncated_normal(shape, tf.float32)
  ```
  will ensure that the `shape` Tensor is on CPU but the `truncated_normal`
  operation runs on GPU 0.

  Args:
    name: Name of the device (see context().devices()), or None to
      perform automatic placement.

  Returns:
    Context manager for setting the device.
  """
  ensure_initialized()
  return context().device(name)


@tf_export("config.experimental_list_devices")
def list_devices():
  """List the names of the available devices.

  Returns:
    Names of the available devices, as a `list`.
  """
  ensure_initialized()
  return context().devices()


@tf_export("debugging.get_log_device_placement")
def get_log_device_placement():
  """Get if device placements are logged.

  Returns:
    If device placements are logged.
  """
  return context().log_device_placement


@tf_export("debugging.set_log_device_placement")
def set_log_device_placement(enabled):
  """Set if device placements should be logged.

  Args:
    enabled: Whether to enabled device placement logging.
  """
  context().log_device_placement = enabled


@tf_contextlib.contextmanager
def device_policy(policy):
  """Context manager for setting device placement policy for current thread."""
  ctx = context()
  old_policy = ctx.device_policy
  try:
    ctx.device_policy = policy
    yield
  finally:
    ctx.device_policy = old_policy


@tf_contextlib.contextmanager
def mirroring_policy(policy):
  """Context manager for setting mirroring policy for current thread."""
  ctx = context()
  old_policy = ctx.mirroring_policy
  try:
    ctx.mirroring_policy = policy
    yield
  finally:
    ctx.mirroring_policy = old_policy


def set_execution_mode(mode):
  """Sets execution mode for the current thread."""
  context().execution_mode = mode


# TODO(fishx): remove this method.
@tf_contextlib.contextmanager
def execution_mode(mode):
  """Context manager for setting execution mode for current thread."""
  ctx = context()
  executor_new = executor.new_executor(mode == ASYNC)
  executor_old = ctx.executor
  try:
    executor_old.wait()
    ctx.executor = executor_new
    yield
  finally:
    ctx.executor = executor_old
    executor_new.wait()


@tf_contextlib.contextmanager
def executor_scope(e):
  """Context manager for changing executor for current thread.

  Args:
    e: A Executor to execute eager ops under this scope. Setting it to None will
      switch back to use the default executor for the context.

  Yields:
    Context manager for setting the executor for current thread.
  """
  ctx = context()
  executor_old = ctx.executor
  try:
    ctx.executor = e
    yield
  finally:
    ctx.executor = executor_old


@tf_export("experimental.function_executor_type")
@tf_contextlib.contextmanager
def function_executor_type(executor_type):
  """Context manager for setting the executor of eager defined functions.

  Eager defined functions are functions decorated by tf.contrib.eager.defun.

  Args:
    executor_type: a string for the name of the executor to be used to execute
      functions defined by tf.contrib.eager.defun.

  Yields:
    Context manager for setting the executor of eager defined functions.
  """
  current_options = context().function_call_options
  old_options = copy.copy(current_options)
  try:
    current_options.executor_type = executor_type
    yield
  finally:
    context().function_call_options = old_options


def is_async():
  """Returns true if current thread is in async mode."""
  return context().is_async()


def async_wait():
  """Waits for ops dispatched in ASYNC mode to finish."""
  return context().executor.wait()


def async_clear_error():
  """Clears errors raised during ASYNC execution mode."""
  return context().executor.clear_error()


def num_gpus():
  """Get the number of available GPU devices.

  Returns:
    The number of available GPU devices.
  """
  return context().num_gpus()


def enable_run_metadata():
  """Enables tracing of op execution via RunMetadata.

  To retrieve the accumulated metadata call context.export_run_metadata()
  and to stop tracing call context.disable_run_metadata().
  """
  context().enable_run_metadata()


def disable_run_metadata():
  """Disables tracing of op execution via RunMetadata."""
  context().disable_run_metadata()


def enable_graph_collection():
  """Enables graph collection of executed functions.

  To retrieve the accumulated graphs call context.export_run_metadata()
  and to stop collecting graphs call context.disable_graph_collection().
  """
  context().enable_graph_collection()


def disable_graph_collection():
  """Disables graph collection of executed functions."""
  context().disable_graph_collection()


def export_run_metadata():
  """Returns a RunMetadata proto with accumulated information.

  The returned protocol buffer contains information since the most recent call
  to either enable_run_metadata or export_run_metadata.

  Returns:
    A RunMetadata protocol buffer.
  """
  return context().export_run_metadata()


def set_server_def(server_def):
  context().set_server_def(server_def)


def add_function(fdef):
  """Add a function definition to the context."""
  context().add_function(fdef)


def remove_function(name):
  """Remove a function from the context."""
  context().remove_function(name)


# Not every user creates a Context via context.context()
# (for example, enable_eager_execution in python/framework/ops.py),
# but they do all import this file.  Note that IS_IN_GRAPH_MODE and
# in_graph_mode are both parameterless functions.
def _tmp_in_graph_mode():
  if context_safe() is None:
    # Context not yet initialized. Assume graph mode following the
    # default implementation in `is_in_graph_mode`.
    return True
  return not executing_eagerly()


is_in_graph_mode.IS_IN_GRAPH_MODE = _tmp_in_graph_mode
