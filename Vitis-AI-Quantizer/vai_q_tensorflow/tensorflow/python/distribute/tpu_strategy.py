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
# ==============================================================================
"""TPU Strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import copy
import weakref

import numpy as np

from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


def get_tpu_system_metadata(tpu_cluster_resolver):
  """Retrieves TPU system metadata given a TPUClusterResolver."""
  master = tpu_cluster_resolver.master()

  # pylint: disable=protected-access
  cluster_spec = tpu_cluster_resolver.cluster_spec()
  cluster_def = cluster_spec.as_cluster_def() if cluster_spec else None
  tpu_system_metadata = (
      tpu_system_metadata_lib._query_tpu_system_metadata(
          master,
          cluster_def=cluster_def,
          query_topology=False))

  return tpu_system_metadata


@contextlib.contextmanager
def maybe_init_scope():
  if ops.executing_eagerly_outside_functions():
    yield
  else:
    with ops.init_scope():
      yield


@tf_export("distribute.experimental.TPUStrategy", v1=[])
class TPUStrategy(distribute_lib.Strategy):
  """TPU distribution strategy implementation."""

  def __init__(self,
               tpu_cluster_resolver=None,
               device_assignment=None):
    """Initializes the TPUStrategy object.

    Args:
      tpu_cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
          which provides information about the TPU cluster.
      device_assignment: Optional `tf.tpu.experimental.DeviceAssignment` to
          specify the placement of replicas on the TPU cluster. Currently only
          supports the usecase of using a single core within a TPU cluster.
    """
    super(TPUStrategy, self).__init__(TPUExtended(
        self, tpu_cluster_resolver, device_assignment=device_assignment))

  # TODO(cjfj): Modify `_call_for_each_replica` in `TPUExtended` such that this
  # can use the default implementation.
  # This implementation runs a single step. It does not use infeed or outfeed.
  def experimental_run_v2(self, fn, args=(), kwargs=None):
    """See base class."""
    # Note: the target function is converted to graph even when in Eager mode,
    # so autograph is on by default here.
    fn = autograph.tf_convert(fn, ag_ctx.control_status_ctx())
    return self.extended.tpu_run(fn, args, kwargs)


@tf_export(v1=["distribute.experimental.TPUStrategy"])
class TPUStrategyV1(distribute_lib.StrategyV1):
  """TPU distribution strategy implementation."""

  def __init__(self,
               tpu_cluster_resolver=None,
               steps_per_run=None,
               device_assignment=None):
    """Initializes the TPUStrategy object.

    Args:
      tpu_cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
          which provides information about the TPU cluster.
      steps_per_run: Number of steps to run on device before returning to the
          host. Note that this can have side-effects on performance, hooks,
          metrics, summaries etc.
          This parameter is only used when Distribution Strategy is used with
          estimator or keras.
      device_assignment: Optional `tf.tpu.experimental.DeviceAssignment` to
          specify the placement of replicas on the TPU cluster. Currently only
          supports the usecase of using a single core within a TPU cluster.
    """
    super(TPUStrategyV1, self).__init__(TPUExtended(
        self, tpu_cluster_resolver, steps_per_run, device_assignment))

  @property
  def steps_per_run(self):
    """DEPRECATED: use .extended.steps_per_run instead."""
    return self._extended.steps_per_run

  # TODO(cjfj): Modify `_call_for_each_replica` in `TPUExtended` such that this
  # can use the default implementation.
  # This implementation runs a single step. It does not use infeed or outfeed.
  def experimental_run_v2(self, fn, args=(), kwargs=None):
    """See base class."""
    fn = autograph.tf_convert(fn, ag_ctx.control_status_ctx())
    return self.extended.tpu_run(fn, args, kwargs)


# TODO(josh11b): Switch to V2 when we no longer need to support tf.compat.v1.
class TPUExtended(distribute_lib.StrategyExtendedV1):
  """Implementation of TPUStrategy."""

  def __init__(self,
               container_strategy,
               tpu_cluster_resolver=None,
               steps_per_run=None,
               device_assignment=None):
    super(TPUExtended, self).__init__(container_strategy)

    if tpu_cluster_resolver is None:
      tpu_cluster_resolver = TPUClusterResolver("")

    if steps_per_run is None:
      # TODO(frankchn): Warn when we are being used by DS/Keras and this is
      # not specified.
      steps_per_run = 1

    self._tpu_function_cache = weakref.WeakKeyDictionary()
    self._tpu_cluster_resolver = tpu_cluster_resolver
    self._tpu_metadata = get_tpu_system_metadata(self._tpu_cluster_resolver)
    self._device_assignment = device_assignment

    # Device assignment is currently only supported for 1 core case.
    if self._device_assignment:
      assert isinstance(self._device_assignment,
                        device_assignment_lib.DeviceAssignment)
      if self._device_assignment.num_replicas != 1:
        raise ValueError("Device assignment is only supported for a single "
                         "core single replica case currently.")
      if self._device_assignment.num_cores_per_replica != 1:
        raise ValueError("Device assignment is only supported for a single "
                         "core single replica case currently.")
      if not all(self._device_assignment.core_assignment[0][0] == [0, 0, 0]):
        raise ValueError("Device assignment is only supported for a single "
                         "core single replica case currently.")

    # TODO(jhseu): Switch to DeviceAssignment to support pods and model
    # parallelism.
    self._tpu_devices = [d.name for d in self._tpu_metadata.devices
                         if "device:TPU:" in d.name]

    self._host_device = device_util.get_host_for_device(self._tpu_devices[0])

    # Only create variables for the number of replicas we're running.
    self._tpu_devices = self._tpu_devices[:self._num_replicas_in_sync]
    self._device_map = values.ReplicaDeviceMap(self._tpu_devices)

    # Preload the data onto the TPUs.
    input_worker_devices = collections.OrderedDict()
    for tpu_device in self._tpu_devices:
      host_device = device_util.get_host_for_device(tpu_device)
      input_worker_devices.setdefault(host_device, [])
      input_worker_devices[host_device].append(tpu_device)
    self._input_workers = input_lib.InputWorkers(
        self._device_map, tuple(input_worker_devices.items()))

    # TODO(sourabhbajaj): Remove this once performance of running one step
    # at a time is comparable to multiple steps.
    self.steps_per_run = steps_per_run
    self._require_static_shapes = True

    self.experimental_enable_get_next_as_optional = True

  def _validate_colocate_with_variable(self, colocate_with_variable):
    values.validate_colocate(colocate_with_variable, self)

  def _make_dataset_iterator(self, dataset):
    """Make iterators for each of the TPU hosts."""
    return input_lib.DatasetIterator(
        dataset,
        self._input_workers,
        self._container_strategy(),
        split_batch_by=self._num_replicas_in_sync)

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    input_contexts = []
    num_workers = self._input_workers.num_workers
    for i in range(num_workers):
      input_contexts.append(distribute_lib.InputContext(
          num_input_pipelines=num_workers,
          input_pipeline_id=i,
          num_replicas_in_sync=self._num_replicas_in_sync))
    return input_lib.InputFunctionIterator(
        input_fn,
        self._input_workers,
        input_contexts,
        self._container_strategy())

  def _experimental_make_numpy_dataset(self, numpy_input, session):
    return numpy_dataset.one_host_numpy_dataset(
        numpy_input, numpy_dataset.SingleDevice(self._host_device),
        session)

  def _experimental_distribute_dataset(self, dataset):
    return input_lib.get_distributed_dataset(
        dataset,
        self._input_workers,
        self._container_strategy(),
        split_batch_by=self._num_replicas_in_sync)

  def _experimental_distribute_datasets_from_function(self, dataset_fn):
    input_contexts = []
    num_workers = self._input_workers.num_workers
    for i in range(num_workers):
      input_contexts.append(distribute_lib.InputContext(
          num_input_pipelines=num_workers,
          input_pipeline_id=i,
          num_replicas_in_sync=self._num_replicas_in_sync))

    return input_lib.get_distributed_datasets_from_function(
        dataset_fn,
        self._input_workers,
        input_contexts,
        self._container_strategy())

  # TODO(priyag): Deal with OutOfRange errors once b/111349762 is fixed.
  # TODO(sourabhbajaj): Remove the initial_loop_values parameter when we have
  # a mechanism to infer the outputs of `fn`. Pending b/110550782.
  def _experimental_run_steps_on_iterator(
      self, fn, multi_worker_iterator, iterations, initial_loop_values=None):
    # Wrap `fn` for repeat.
    if initial_loop_values is None:
      initial_loop_values = {}
    initial_loop_values = nest.flatten(initial_loop_values)
    ctx = input_lib.MultiStepContext()

    def run_fn(inputs):
      """Single step on the TPU device."""
      fn_result = fn(ctx, inputs)
      flat_last_step_outputs = nest.flatten(ctx.last_step_outputs)
      if flat_last_step_outputs:
        with ops.control_dependencies([fn_result]):
          return [array_ops.identity(f) for f in flat_last_step_outputs]
      else:
        return fn_result

    # We capture the control_flow_context at this point, before we run `fn`
    # inside a while_loop and TPU replicate context. This is useful in cases
    # where we might need to exit these contexts and get back to the outer
    # context to do some things, for e.g. create an op which should be
    # evaluated only once at the end of the loop on the host. One such usage
    # is in creating metrics' value op.
    self._outer_control_flow_context = (
        ops.get_default_graph()._get_control_flow_context())  # pylint: disable=protected-access

    def rewrite_fn(*args):
      """The rewritten step fn running on TPU."""
      del args

      per_replica_inputs = multi_worker_iterator.get_next()
      replicate_inputs = []
      for replica_id in range(self._num_replicas_in_sync):
        select_replica = lambda x: values.select_replica(replica_id, x)  # pylint: disable=cell-var-from-loop
        replicate_inputs.append((nest.map_structure(
            select_replica, per_replica_inputs),))

      replicate_outputs = tpu.replicate(
          run_fn, replicate_inputs, device_assignment=self._device_assignment)

      # If run_fn has tensor outputs, tpu.replicate returns a list of list. We
      # will flatten it in this case. If run_fn has no tensor outputs,
      # tpu.replicate returns a list of no_ops, we will keep the output as it
      # is.
      if isinstance(replicate_outputs[0], list):
        replicate_outputs = nest.flatten(replicate_outputs)

      return replicate_outputs

    # TODO(sourabhbajaj): The input to while loop should be based on the
    # output type of the step_fn
    assert isinstance(initial_loop_values, list)
    initial_loop_values = initial_loop_values * self._num_replicas_in_sync

    # Put the while loop op on TPU host 0.
    with ops.device(self._host_device):
      if self.steps_per_run == 1:
        replicate_outputs = rewrite_fn()
      else:
        replicate_outputs = training_loop.repeat(iterations, rewrite_fn,
                                                 initial_loop_values)

    del self._outer_control_flow_context
    ctx.run_op = control_flow_ops.group(replicate_outputs)

    if isinstance(replicate_outputs, list):
      # Filter out any ops from the outputs, typically this would be the case
      # when there were no tensor outputs.
      last_step_tensor_outputs = [
          x for x in replicate_outputs if not isinstance(x, ops.Operation)
      ]

      # Outputs are currently of the structure (flattened)
      # [output0_device0, output1_device0, output2_device0,
      #  output0_device1, output1_device1, output2_device1,
      #  ...]
      # Convert this to the following structure instead: (grouped by output)
      # [[output0_device0, output0_device1],
      #  [output1_device0, output1_device1],
      #  [output2_device0, output2_device1]]
      output_num = len(last_step_tensor_outputs) // self._num_replicas_in_sync
      last_step_tensor_outputs = [
          last_step_tensor_outputs[i::output_num] for i in range(output_num)
      ]
    else:
      # no tensors returned.
      last_step_tensor_outputs = []

    _set_last_step_outputs(ctx, last_step_tensor_outputs)
    return ctx

  def _call_for_each_replica(self, fn, args, kwargs):
    # TODO(jhseu): Consider making it so call_for_each_replica implies that
    # we're in a tpu.rewrite(), and update TPUMirroredVariable accordingly.
    with _TPUReplicaContext(self._container_strategy()):
      return fn(*args, **kwargs)

  def _experimental_initialize_system(self):
    """Experimental method added to be used by Estimator.

    This is a private method only to be used by Estimator. Other frameworks
    should directly be calling `tf.tpu.experimental.initialize_tpu_system`
    """
    tpu_strategy_util.initialize_tpu_system(self._tpu_cluster_resolver)

  def _create_variable(self, next_creator, *args, **kwargs):
    """Create a TPUMirroredVariable. See `DistributionStrategy.scope`."""
    colocate_with = kwargs.pop("colocate_with", None)
    if colocate_with is None:
      device_map = self._device_map
      logical_device = 0  # TODO(josh11b): Get logical device from scope here.
    elif isinstance(colocate_with, numpy_dataset.SingleDevice):
      with ops.device(colocate_with.device):
        return next_creator(*args, **kwargs)
    else:
      device_map = colocate_with.device_map
      logical_device = colocate_with.logical_device

    def _real_mirrored_creator(devices, *args, **kwargs):  # pylint: disable=g-missing-docstring
      initial_value = None
      value_list = []
      for i, d in enumerate(devices):
        with ops.device(d):
          if i == 0:
            initial_value = kwargs["initial_value"]
            # Note: some v1 code expects variable initializer creation to happen
            # inside a init_scope.
            with maybe_init_scope():
              initial_value = initial_value() if callable(
                  initial_value) else initial_value

          if i > 0:
            # Give replicas meaningful distinct names:
            var0name = value_list[0].name.split(":")[0]
            # We append a / to variable names created on replicas with id > 0 to
            # ensure that we ignore the name scope and instead use the given
            # name as the absolute name of the variable.
            kwargs["name"] = "%s/replica_%d/" % (var0name, i)
          kwargs["initial_value"] = initial_value

          with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
            v = next_creator(*args, **kwargs)

          assert not isinstance(v, values.TPUMirroredVariable)
          value_list.append(v)
      return value_list

    return distribute_lib.create_mirrored_variable(
        self._container_strategy(), device_map, logical_device,
        _real_mirrored_creator, values.TPUMirroredVariable,
        values.TPUSyncOnReadVariable, *args, **kwargs)

  def _reduce_to(self, reduce_op, value, destinations):
    if values._enclosing_tpu_context() is not None:  # pylint: disable=protected-access
      if reduce_op == reduce_util.ReduceOp.MEAN:
        # TODO(jhseu):  Revisit once we support model-parallelism.
        value *= (1. / self._num_replicas_in_sync)
      elif reduce_op != reduce_util.ReduceOp.SUM:
        raise NotImplementedError(
            "Currently only support sum & mean in TPUStrategy.")
      return tpu_ops.cross_replica_sum(value)

    if not isinstance(value, values.DistributedValues):
      # This function handles reducing values that are not PerReplica or
      # Mirrored values. For example, the same value could be present on all
      # replicas in which case `value` would be a single value or value could
      # be 0.
      return cross_device_ops_lib.reduce_non_distributed_value(
          reduce_op, self._device_map, value, destinations)

    # TODO(cjfj): Detect when it is possible to use `cross_replica_sum`.
    # Always performs the reduction on the TPU host.
    with ops.device(self._host_device):
      output = math_ops.add_n(value.values)
      if reduce_op == reduce_util.ReduceOp.MEAN:
        output *= (1. / len(value.values))

    devices = cross_device_ops_lib.get_devices_from(destinations)

    if len(devices) == 1:
      # If necessary, copy to requested destination.
      dest_canonical = device_util.canonicalize(devices[0])
      host_canonical = device_util.canonicalize(self._host_device)

      if dest_canonical != host_canonical:
        with ops.device(dest_canonical):
          output = array_ops.identity(output)
    else:
      output = cross_device_ops_lib.simple_broadcast(output, destinations)

    return output

  def _update(self, var, fn, args, kwargs, group):
    assert isinstance(var, values.TPUVariableMixin) or isinstance(
        var, resource_variable_ops.BaseResourceVariable)
    if values._enclosing_tpu_context() is not None:  # pylint: disable=protected-access
      if group:
        return fn(var, *args, **kwargs)
      else:
        return (fn(var, *args, **kwargs),)

    # Otherwise, we revert to MirroredStrategy behavior and update each variable
    # directly.
    updates = []
    for i, (d, v) in enumerate(zip(var.devices, var.values)):
      name = "update_%d" % i
      with ops.device(d), distribute_lib.UpdateContext(d), ops.name_scope(name):
        # If args and kwargs are not mirrored, the value is returned as is.
        updates.append(fn(v,
                          *values.select_device_mirrored(d, args),
                          **values.select_device_mirrored(d, kwargs)))
    return values.update_regroup(self, self._device_map, updates, group)

  def read_var(self, var):
    assert isinstance(var, values.TPUVariableMixin) or isinstance(
        var, resource_variable_ops.BaseResourceVariable)
    return var.read_value()

  def _local_results(self, val):
    if isinstance(val, values.DistributedValues):
      # Return in a deterministic order.
      return tuple(val.get(device=d) for d in sorted(val.devices))
    elif isinstance(val, list):
      # TODO(josh11b): We need to remove this case; per device values should
      # be represented using a PerReplica wrapper instead of a list with
      # one entry per device.
      return tuple(val)
    elif isinstance(val, values.TPUMirroredVariable):
      # pylint: disable=protected-access
      if values._enclosing_tpu_context() is not None:
        return (val,)
      return val.values
    return (val,)

  def value_container(self, value):
    return value

  def _broadcast_to(self, tensor, destinations):
    del destinations
    return tensor

  @property
  def num_hosts(self):
    if self._device_assignment is None:
      return self._tpu_metadata.num_hosts

    return len(set([self._device_assignment.host_device(r)
                    for r in range(self._device_assignment.num_replicas)]))

  @property
  def num_replicas_per_host(self):
    if self._device_assignment is None:
      return self._tpu_metadata.num_of_cores_per_host

    # TODO(sourabhbajaj): Remove this method we use inputs and remove infeed
    # as the computation of num_replicas_per_host is not a constant
    # when using device_assignment. This is a temporary workaround to support
    # StatefulRNN as everything is 1 in that case.
    # This method needs to take host_id as input for correct computation.
    max_models_per_host = (self._tpu_metadata.num_of_cores_per_host //
                           self._device_assignment.num_cores_per_replica)
    models_per_host = min(self._device_assignment.num_replicas,
                          max_models_per_host)
    return models_per_host * self._device_assignment.num_cores_per_replica

  @property
  def _num_replicas_in_sync(self):
    if self._device_assignment is None:
      return self._tpu_metadata.num_cores
    return (self._device_assignment.num_replicas *
            self._device_assignment.num_cores_per_replica)

  @property
  def experimental_between_graph(self):
    return False

  @property
  def experimental_should_init(self):
    return True

  @property
  def should_checkpoint(self):
    return True

  @property
  def should_save_summary(self):
    return True

  @property
  def worker_devices(self):
    return self._tpu_devices

  @property
  def parameter_devices(self):
    return self._tpu_devices

  def non_slot_devices(self, var_list):
    return self._host_device

  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    del colocate_with
    with ops.device(self._host_device), distribute_lib.UpdateContext(
        self._host_device):
      result = fn(*args, **kwargs)
      if group:
        return result
      else:
        return nest.map_structure(self._local_results, result)

  def _configure(self,
                 session_config=None,
                 cluster_spec=None,
                 task_type=None,
                 task_id=None):
    del cluster_spec, task_type, task_id
    if session_config:
      session_config.CopyFrom(self._update_config_proto(session_config))

  def _update_config_proto(self, config_proto):
    updated_config = copy.deepcopy(config_proto)
    updated_config.isolate_session_state = True
    cluster_spec = self._tpu_cluster_resolver.cluster_spec()
    if cluster_spec:
      updated_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    return updated_config

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    """`make_dataset_iterator` and `make_numpy_iterator` use global batch size.

    `make_input_fn_iterator` assumes per-replica batching.

    Returns:
      Boolean.
    """
    return True

  def tpu_run(self, fn, args, kwargs):
    func = self._tpu_function_creator(fn)
    return func(args, kwargs)

  def _tpu_function_creator(self, fn):
    if fn in self._tpu_function_cache:
      return self._tpu_function_cache[fn]

    strategy = self._container_strategy()

    def tpu_function(args, kwargs):
      """TF Function used to replicate the user computation."""
      if kwargs is None:
        kwargs = {}

      # Remove None at the end of args as they are not replicatable
      # If there are None in the middle we can't do anything about it
      # so let those cases fail.
      # For example when Keras model predict is used they pass the targets as
      # None. We want to handle it here so all client libraries don't have to
      # do this as other strategies can handle None values better.
      while args and args[-1] is None:
        args = args[:-1]

      # Used to re-structure flattened output tensors from `tpu.replicate()`
      # into a structured format.
      result = [[]]

      def replicated_fn(replica_id, replica_args, replica_kwargs):
        """Wraps user function to provide replica ID and `Tensor` inputs."""
        with _TPUReplicaContext(strategy, replica_id_in_sync_group=replica_id):
          result[0] = fn(*replica_args, **replica_kwargs)
        return result[0]

      replicate_inputs = []  # By replica.
      for i in range(strategy.num_replicas_in_sync):
        replicate_inputs.append(
            [constant_op.constant(i, dtype=dtypes.int32),
             values.select_replica(i, args),
             values.select_replica(i, kwargs)])

      # Construct and pass `maximum_shapes` so that we could support dynamic
      # shapes using dynamic padder.
      if replicate_inputs:
        maximum_shapes = []
        flattened_list = nest.flatten(replicate_inputs[0])
        for input_tensor in flattened_list:
          if tensor_util.is_tensor(input_tensor):
            maximum_shape = input_tensor.get_shape()
          else:
            maximum_shape = tensor_shape.TensorShape(np.shape(input_tensor))
          maximum_shapes.append(maximum_shape)
        maximum_shapes = nest.pack_sequence_as(replicate_inputs[0],
                                               maximum_shapes)
      else:
        maximum_shapes = None

      with strategy.scope():
        replicate_outputs = tpu.replicate(
            replicated_fn,
            replicate_inputs,
            device_assignment=self._device_assignment,
            maximum_shapes=maximum_shapes)

      # Remove all no ops that may have been added during 'tpu.replicate()'
      if isinstance(result[0], list):
        result[0] = [
            output for output in result[0] if tensor_util.is_tensor(output)
        ]

      # Workaround for `tpu.replicate` behaviour when single `Tensor` returned.
      if result[0] is None:
        replicate_outputs = [None] * len(replicate_outputs)
      else:
        replicate_outputs = [
            nest.pack_sequence_as(result[0], nest.flatten(replica_output))
            for replica_output in replicate_outputs
        ]
      device_map = self._device_map  # pylint: disable=protected-access
      return values.regroup(device_map, replicate_outputs)

    if context.executing_eagerly():
      tpu_function = def_function.function(tpu_function)

    self._tpu_function_cache[fn] = tpu_function
    return tpu_function


class _TPUReplicaContext(distribute_lib.ReplicaContext):
  """Replication Context class for TPU Strategy."""

  # TODO(sourabhbajaj): Call for each replica should be updating this.
  # TODO(b/118385803): Always properly initialize replica_id.
  def __init__(self, strategy, replica_id_in_sync_group=None):
    if replica_id_in_sync_group is None:
      replica_id_in_sync_group = constant_op.constant(0, dtypes.int32)
    distribute_lib.ReplicaContext.__init__(
        self, strategy, replica_id_in_sync_group=replica_id_in_sync_group)

  @property
  def devices(self):
    distribute_lib.require_replica_context(self)
    ds = self._strategy
    replica_id = tensor_util.constant_value(self._replica_id_in_sync_group)

    if replica_id is None:  # Non-constant `Tensor` inside `tpu.replicate`.
      # TODO(cjfj): Return other devices when model parallelism is supported.
      return (tpu.core(0),)
    else:
      return (ds.extended.worker_devices[replica_id],)


def _set_last_step_outputs(ctx, last_step_tensor_outputs):
  """Sets the last step outputs on the given context."""
  # Convert replicate_outputs to the original dict structure of
  # last_step_outputs.
  last_step_tensor_outputs_dict = nest.pack_sequence_as(
      ctx.last_step_outputs, last_step_tensor_outputs)

  for name, reduce_op in ctx._last_step_outputs_reduce_ops.items():  # pylint: disable=protected-access
    output = last_step_tensor_outputs_dict[name]
    # For outputs that have already been reduced, take the first value
    # from the list as each value should be the same. Else return the full
    # list of values.
    # TODO(josh11b): If reduce_op is NONE, we should return a PerReplica
    # value.
    if reduce_op is not None:
      # TODO(priyag): Should this return the element or a list with 1 element
      last_step_tensor_outputs_dict[name] = output[0]
  ctx._set_last_step_outputs(last_step_tensor_outputs_dict)  # pylint: disable=protected-access
