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
"""Class implementing a multi-worker parameter server tf.distribute strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy


from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import SimpleClusterResolver
from tensorflow.python.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import device_setter
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

_LOCAL_CPU = "/device:CPU:0"


# TODO(yuefengz): maybe cache variables on local CPU.
@tf_export("distribute.experimental.ParameterServerStrategy", v1=[])
class ParameterServerStrategy(distribute_lib.Strategy):
  """An asynchronous multi-worker parameter server tf.distribute strategy.

  This strategy requires two jobs: workers and parameter servers. Variables and
  updates to those variables will be assigned to parameter servers and other
  operations are assigned to workers.

  When each worker has more than one GPU, operations will be replicated on all
  GPUs. Even though operations may be replicated, variables are not and each
  worker shares a common view for which parameter server a variable is assigned
  to.

  By default it uses `TFConfigClusterResolver` to detect configurations for
  multi-worker training. This requires a 'TF_CONFIG' environment variable and
  the 'TF_CONFIG' must have a cluster spec.

  This class assumes each worker is running the same code independently, but
  parameter servers are running a standard server. This means that while each
  worker will synchronously compute a single gradient update across all GPUs,
  updates between workers proceed asynchronously. Operations that occur only on
  the first replica (such as incrementing the global step), will occur on the
  first replica *of every worker*.

  It is expected to call `call_for_each_replica(fn, ...)` for any
  operations which potentially can be replicated across replicas (i.e. multiple
  GPUs) even if there is only CPU or one GPU. When defining the `fn`, extra
  caution needs to be taken:

  1) It is generally not recommended to open a device scope under the strategy's
  scope. A device scope (i.e. calling `tf.device`) will be merged with or
  override the device for operations but will not change the device for
  variables.

  2) It is also not recommended to open a colocation scope (i.e. calling
  `tf.compat.v1.colocate_with`) under the strategy's scope. For colocating
  variables, use `strategy.extended.colocate_vars_with` instead. Colocation of
  ops will possibly create device assignment conflicts.

  Note: This strategy only works with the Estimator API. Pass an instance of
  this strategy to the `experimental_distribute` argument when you create the
  `RunConfig`. This instance of `RunConfig` should then be passed to the
  `Estimator` instance on which `train_and_evaluate` is called.

  For Example:
  ```
  strategy = tf.distribute.experimental.ParameterServerStrategy()
  run_config = tf.estimator.RunConfig(
      experimental_distribute.train_distribute=strategy)
  estimator = tf.estimator.Estimator(config=run_config)
  tf.estimator.train_and_evaluate(estimator,...)
  """

  def __init__(self, cluster_resolver=None):
    """Initializes this strategy with an optional `cluster_resolver`.

    Args:
      cluster_resolver: Optional
        `tf.distribute.cluster_resolver.ClusterResolver` object. Defaults to a
        `tf.distribute.cluster_resolver.TFConfigClusterResolver`.
    """
    if cluster_resolver is None:
      cluster_resolver = TFConfigClusterResolver()
    if not cluster_resolver.cluster_spec():
      raise ValueError("Cluster spec must be non-empty in `cluster_resolver`.")
    extended = ParameterServerStrategyExtended(
        self, cluster_resolver=cluster_resolver)
    super(ParameterServerStrategy, self).__init__(extended)


@tf_export(v1=["distribute.experimental.ParameterServerStrategy"])  # pylint: disable=missing-docstring
class ParameterServerStrategyV1(distribute_lib.StrategyV1):

  __doc__ = ParameterServerStrategy.__doc__

  def __init__(self, cluster_resolver=None):
    """Initializes this strategy."""
    super(ParameterServerStrategyV1, self).__init__(
        ParameterServerStrategyExtended(
            self, cluster_resolver=cluster_resolver))
  __init__.__doc__ = ParameterServerStrategy.__init__.__doc__


# TODO(josh11b): Switch to V2 when we no longer need to support tf.compat.v1.
class ParameterServerStrategyExtended(distribute_lib.StrategyExtendedV1):
  """Implementation of ParameterServerStrategy and CentralStorageStrategy."""

  def __init__(self,
               container_strategy,
               cluster_resolver=None,
               compute_devices=None,
               parameter_device=None):
    super(ParameterServerStrategyExtended, self).__init__(container_strategy)
    self._initialize_strategy(
        cluster_resolver=cluster_resolver,
        compute_devices=compute_devices,
        parameter_device=parameter_device)

    # We typically don't need to do all-reduce in this strategy.
    self._cross_device_ops = (
        cross_device_ops_lib.ReductionToOneDevice(reduce_to_device=_LOCAL_CPU))

  def _initialize_strategy(self,
                           cluster_resolver=None,
                           compute_devices=None,
                           parameter_device=None):
    if cluster_resolver and cluster_resolver.cluster_spec():
      self._initialize_multi_worker(cluster_resolver)
    else:
      self._initialize_local(
          compute_devices, parameter_device, cluster_resolver=cluster_resolver)

  def _initialize_multi_worker(self, cluster_resolver):
    """Initialize devices for multiple workers.

    It creates variable devices and compute devices. Variables and operations
    will be assigned to them respectively. We have one compute device per
    replica. The variable device is a device function or device string. The
    default variable device assigns variables to parameter servers in a
    round-robin fashion.

    Args:
      cluster_resolver: a descendant of `ClusterResolver` object.

    Raises:
      ValueError: if the cluster doesn't have ps jobs.
    """
    # TODO(b/126786766): TFConfigClusterResolver returns wrong number of GPUs in
    # some cases.
    if isinstance(cluster_resolver, TFConfigClusterResolver):
      num_gpus = context.num_gpus()
    else:
      num_gpus = cluster_resolver.num_accelerators().get("GPU", 0)

    # Save the num_gpus_per_worker for configure method.
    self._num_gpus_per_worker = num_gpus

    cluster_spec = cluster_resolver.cluster_spec()
    task_type = cluster_resolver.task_type
    task_id = cluster_resolver.task_id
    if not task_type or task_id is None:
      raise ValueError("When `cluster_spec` is given, you must also specify "
                       "`task_type` and `task_id`")
    cluster_spec = multi_worker_util.normalize_cluster_spec(cluster_spec)
    assert cluster_spec.as_dict()

    worker_device = "/job:%s/task:%d" % (task_type, task_id)
    self._input_host_device = numpy_dataset.SingleDevice(worker_device)

    # Define compute devices which is a list of device strings and one for each
    # replica. When there are GPUs, replicate operations on these GPUs.
    # Otherwise, place operations on CPU.
    if num_gpus > 0:
      compute_devices = tuple(
          "%s/device:GPU:%d" % (worker_device, i) for i in range(num_gpus))
    else:
      compute_devices = (worker_device,)

    self._device_map = values.ReplicaDeviceMap(compute_devices)
    self._input_workers = input_lib.InputWorkers(
        self._device_map, [(worker_device, compute_devices)])

    # In distributed mode, place variables on ps jobs in a round-robin fashion.
    # Note that devices returned from `replica_device_setter` are not
    # canonical and therefore we don't canonicalize all variable devices to
    # make them consistent.
    # TODO(yuefengz): support passing a strategy object to control variable
    # assignment.
    # TODO(yuefengz): merge the logic of replica_device_setter into this
    # class.
    num_ps_replicas = len(cluster_spec.as_dict().get("ps", []))
    if num_ps_replicas == 0:
      raise ValueError("The cluster spec needs to have `ps` jobs.")
    self._variable_device = device_setter.replica_device_setter(
        ps_tasks=num_ps_replicas,
        worker_device=worker_device,
        merge_devices=True,
        cluster=cluster_spec)

    # The `_parameter_devices` is needed for the `parameter_devices` property
    # and is a list of all variable devices. Here parameter devices are all
    # tasks of the "ps" job.
    self._parameter_devices = tuple(map("/job:ps/task:{}".format,
                                        range(num_ps_replicas)))

    # Add a default device so that ops without specified devices will not end up
    # on other workers.
    self._default_device = worker_device

    self._is_chief = multi_worker_util.is_chief(cluster_spec, task_type,
                                                task_id)
    self._cluster_spec = cluster_spec
    self._task_type = task_type
    self._task_id = task_id

    logging.info(
        "Multi-worker ParameterServerStrategy with "
        "cluster_spec = %r, task_type = %r, task_id = %r, "
        "num_ps_replicas = %r, is_chief = %r, device_map = %r, "
        "variable_device = %r", cluster_spec.as_dict(), task_type, task_id,
        num_ps_replicas, self._is_chief, self._device_map,
        self._variable_device)

  # TODO(yuefengz): get rid of cluster_resolver argument when contrib's
  # version no longer depends on this class.
  def _initialize_local(self,
                        compute_devices,
                        parameter_device,
                        cluster_resolver=None):
    """Initialize local devices for training."""
    worker_device = device_util.canonicalize("/device:CPU:0")
    self._input_host_device = numpy_dataset.SingleDevice(worker_device)

    if compute_devices is None:
      if not cluster_resolver:
        num_gpus = context.num_gpus()
      else:
        num_gpus = cluster_resolver.num_accelerators().get("GPU", 0)
      # Save the num_gpus_per_worker for configure method which is used by the
      # contrib version.
      self._num_gpus_per_worker = num_gpus

      compute_devices = device_util.local_devices_from_num_gpus(num_gpus)

    if parameter_device is None:
      # If there is only one GPU, put everything on that GPU. Otherwise, place
      # variables on CPU.
      if len(compute_devices) == 1:
        parameter_device = compute_devices[0]
      else:
        parameter_device = _LOCAL_CPU

    self._device_map = values.ReplicaDeviceMap(compute_devices)
    self._input_workers = input_lib.InputWorkers(
        self._device_map, [(worker_device, compute_devices)])

    self._variable_device = parameter_device
    self._parameter_devices = (parameter_device,)
    self._is_chief = True
    self._cluster_spec = None
    self._task_type = None
    self._task_id = None

    logging.info(
        "ParameterServerStrategy with compute_devices = %r, "
        "variable_device = %r", compute_devices, self._variable_device)

  def _validate_colocate_with_variable(self, colocate_with_variable):
    values.validate_colocate(colocate_with_variable, self)

  def _experimental_distribute_dataset(self, dataset):
    return input_lib.get_distributed_dataset(
        dataset,
        self._input_workers,
        self._container_strategy(),
        split_batch_by=self._num_replicas_in_sync)

  def _make_dataset_iterator(self, dataset):
    return input_lib.DatasetIterator(
        dataset,
        self._input_workers,
        self._container_strategy(),
        split_batch_by=self._num_replicas_in_sync)

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    """Distributes the dataset to each local GPU."""
    if self._cluster_spec:
      input_pipeline_id = multi_worker_util.id_in_cluster(
          self._cluster_spec, self._task_type, self._task_id)
      num_input_pipelines = multi_worker_util.worker_count(
          self._cluster_spec, self._task_type)
    else:
      input_pipeline_id = 0
      num_input_pipelines = 1
    input_context = distribute_lib.InputContext(
        num_input_pipelines=num_input_pipelines,
        input_pipeline_id=input_pipeline_id,
        num_replicas_in_sync=self._num_replicas_in_sync)
    return input_lib.InputFunctionIterator(input_fn, self._input_workers,
                                           [input_context],
                                           self._container_strategy())

  def _experimental_make_numpy_dataset(self, numpy_input, session):
    return numpy_dataset.one_host_numpy_dataset(
        numpy_input, self._input_host_device, session)

  def _experimental_distribute_datasets_from_function(self, dataset_fn):
    if self._cluster_spec:
      input_pipeline_id = multi_worker_util.id_in_cluster(
          self._cluster_spec, self._task_type, self._task_id)
      num_input_pipelines = multi_worker_util.worker_count(
          self._cluster_spec, self._task_type)
    else:
      input_pipeline_id = 0
      num_input_pipelines = 1

    input_context = distribute_lib.InputContext(
        num_input_pipelines=num_input_pipelines,
        input_pipeline_id=input_pipeline_id,
        num_replicas_in_sync=self._num_replicas_in_sync)

    return input_lib.get_distributed_datasets_from_function(
        dataset_fn,
        self._input_workers,
        [input_context],
        self._container_strategy())

  def _broadcast_to(self, tensor, destinations):
    # This is both a fast path for Python constants, and a way to delay
    # converting Python values to a tensor until we know what type it
    # should be converted to. Otherwise we have trouble with:
    #   global_step.assign_add(1)
    # since the `1` gets broadcast as an int32 but global_step is int64.
    if isinstance(tensor, (float, int)):
      return tensor
    if not cross_device_ops_lib.check_destinations(destinations):
      # TODO(josh11b): Use current logical device instead of 0 here.
      destinations = values.LogicalDeviceSpec(
          device_map=self._device_map, logical_device=0)
    return self._cross_device_ops.broadcast(tensor, destinations)

  def _allow_variable_partition(self):
    return not context.executing_eagerly()

  # TODO(yuefengz): Not all ops in device_setter.STANDARD_PS_OPS will go through
  # this creator, such as "MutableHashTable".
  def _create_variable(self, next_creator, *args, **kwargs):
    if self._num_replicas_in_sync > 1:
      aggregation = kwargs.pop("aggregation", vs.VariableAggregation.NONE)
      if aggregation not in (
          vs.VariableAggregation.NONE,
          vs.VariableAggregation.SUM,
          vs.VariableAggregation.MEAN,
          vs.VariableAggregation.ONLY_FIRST_REPLICA
      ):
        raise ValueError("Invalid variable aggregation mode: " + aggregation +
                         " for variable: " + kwargs["name"])

      def var_creator(*args, **kwargs):
        """Create an AggregatingVariable and fix up collections."""
        # Record what collections this variable should be added to.
        collections = kwargs.pop("collections", None)
        if collections is None:
          collections = [ops.GraphKeys.GLOBAL_VARIABLES]
        kwargs["collections"] = []

        # Create and wrap the variable.
        v = next_creator(*args, **kwargs)
        wrapped = values.AggregatingVariable(
            self._container_strategy(), v, aggregation)

        # Add the wrapped variable to the requested collections.
        # The handling of eager mode and the global step matches
        # ResourceVariable._init_from_args().
        if not context.executing_eagerly():
          g = ops.get_default_graph()
          # If "trainable" is True, next_creator() will add the contained
          # variable to the TRAINABLE_VARIABLES collection, so we manually
          # remove it and replace with the wrapper. We can't set "trainable"
          # to False for next_creator() since that causes functions like
          # implicit_gradients to skip those variables.
          if kwargs.get("trainable", True):
            collections.append(ops.GraphKeys.TRAINABLE_VARIABLES)
            l = g.get_collection_ref(ops.GraphKeys.TRAINABLE_VARIABLES)
            if v in l:
              l.remove(v)
          g.add_to_collections(collections, wrapped)
        elif ops.GraphKeys.GLOBAL_STEP in collections:
          ops.add_to_collections(ops.GraphKeys.GLOBAL_STEP, wrapped)

        return wrapped
    else:
      var_creator = next_creator

    if "colocate_with" in kwargs:
      colocate_with = kwargs["colocate_with"]
      if isinstance(colocate_with, numpy_dataset.SingleDevice):
        with ops.device(colocate_with.device):
          return var_creator(*args, **kwargs)
      with ops.device(None):
        with ops.colocate_with(colocate_with):
          return var_creator(*args, **kwargs)

    with ops.colocate_with(None, ignore_existing=True):
      with ops.device(self._variable_device):
        return var_creator(*args, **kwargs)

  def _call_for_each_replica(self, fn, args, kwargs):
    # pylint: disable=protected-access
    return mirrored_strategy._call_for_each_replica(
        self._container_strategy(), self._device_map, fn, args, kwargs)

  def _verify_destinations_not_different_worker(self, destinations):
    if not self._cluster_spec:
      return
    if destinations is None:
      return
    for d in cross_device_ops_lib.get_devices_from(destinations):
      d_spec = tf_device.DeviceSpec.from_string(d)
      if d_spec.job == self._task_type and d_spec.task != self._task_id:
        raise ValueError(
            "Cannot reduce to another worker: %r, current worker is %r" %
            (d, self._input_workers.worker_devices[0]))

  def _reduce_to(self, reduce_op, value, destinations):
    self._verify_destinations_not_different_worker(destinations)
    if not isinstance(value, values.DistributedValues):
      # pylint: disable=protected-access
      return cross_device_ops_lib.reduce_non_distributed_value(
          reduce_op, self._device_map, value, destinations)
    return self._cross_device_ops.reduce(
        reduce_op, value, destinations=destinations)

  def _batch_reduce_to(self, reduce_op, value_destination_pairs):
    for _, destinations in value_destination_pairs:
      self._verify_destinations_not_different_worker(destinations)
    return self._cross_device_ops.batch_reduce(reduce_op,
                                               value_destination_pairs)

  def _select_single_value(self, structured):
    """Select any single value in `structured`."""

    def _select_fn(x):  # pylint: disable=g-missing-docstring
      if isinstance(x, values.Mirrored):
        if len(x.devices) == 1:
          return x.primary
        else:
          raise ValueError(
              "You cannot update variable with a Mirrored object with multiple "
              "components %r when using ParameterServerStrategy. You must "
              "specify a single value or a Mirrored with a single value." % x)
      elif isinstance(x, values.PerReplica):
        raise ValueError(
            "You cannot update variable with a PerReplica object %r when using "
            "ParameterServerStrategy. You must specify a single value or a "
            "Mirrored with a single value" % x)
      else:
        return x

    return nest.map_structure(_select_fn, structured)

  def _update(self, var, fn, args, kwargs, group):
    if isinstance(var, values.AggregatingVariable):
      var = var.get()
    if not isinstance(var, resource_variable_ops.BaseResourceVariable):
      raise ValueError(
          "You can not update `var` %r. It must be a Variable." % var)
    with ops.colocate_with(var), distribute_lib.UpdateContext(var.device):
      result = fn(var, *self._select_single_value(args),
                  **self._select_single_value(kwargs))
      if group:
        return result
      else:
        return nest.map_structure(self._local_results, result)

  # TODO(yuefengz): does it need to call _select_single_value?
  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    with ops.device(
        colocate_with.device), distribute_lib.UpdateContext(colocate_with):
      result = fn(*args, **kwargs)
      if group:
        return result
      else:
        return nest.map_structure(self._local_results, result)

  def _local_results(self, val):
    if isinstance(val, values.DistributedValues):
      return val.values
    return (val,)

  def value_container(self, val):
    if (hasattr(val, "_aggregating_container") and
        not isinstance(val, values.AggregatingVariable)):
      wrapper = val._aggregating_container()  # pylint: disable=protected-access
      if wrapper is not None:
        return wrapper
    return val

  def read_var(self, var):
    # No need to distinguish between normal variables and replica-local
    # variables.
    return array_ops.identity(var)

  def _configure(self,
                 session_config=None,
                 cluster_spec=None,
                 task_type=None,
                 task_id=None):
    """Configures the strategy class with `cluser_spec`.

    The strategy object will be re-initialized if `cluster_spec` is passed to
    `configure` but was not passed when instantiating the strategy.

    Args:
      session_config: Session config object.
      cluster_spec: a dict, ClusterDef or ClusterSpec object specifying the
        cluster configurations.
      task_type: the current task type.
      task_id: the current task id.

    Raises:
      ValueError: if `cluster_spec` is given but `task_type` or `task_id` is
        not.
    """
    if cluster_spec:
      # Use the num_gpus_per_worker recorded in constructor since _configure
      # doesn't take num_gpus.
      cluster_resolver = SimpleClusterResolver(
          cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
          task_type=task_type,
          task_id=task_id,
          num_accelerators={"GPU": self._num_gpus_per_worker})
      self._initialize_multi_worker(cluster_resolver)

    if session_config:
      session_config.CopyFrom(self._update_config_proto(session_config))

  def _update_config_proto(self, config_proto):
    updated_config = copy.deepcopy(config_proto)
    if not self._cluster_spec:
      updated_config.isolate_session_state = True
      return updated_config

    updated_config.isolate_session_state = False

    assert self._task_type
    assert self._task_id is not None

    # The device filters prevent communication between workers.
    del updated_config.device_filters[:]
    if self._task_type in ["chief", "worker"]:
      updated_config.device_filters.extend(
          ["/job:%s/task:%d" % (self._task_type, self._task_id), "/job:ps"])
    elif self._task_type == "evaluator":
      updated_config.device_filters.append(
          "/job:%s/task:%d" % (self._task_type, self._task_id))
    return updated_config

  @property
  def _num_replicas_in_sync(self):
    return self._device_map.num_replicas_in_graph

  @property
  def worker_devices(self):
    return self._device_map.all_devices

  @property
  def worker_devices_by_replica(self):
    return self._device_map.devices_by_replica

  @property
  def parameter_devices(self):
    return self._parameter_devices

  def non_slot_devices(self, var_list):
    return min(var_list, key=lambda x: x.name)

  @property
  def experimental_between_graph(self):
    # TODO(yuefengz): Should this return False in the local case?
    return True

  @property
  def experimental_should_init(self):
    return self._is_chief

  @property
  def should_checkpoint(self):
    return self._is_chief

  @property
  def should_save_summary(self):
    return self._is_chief

  # TODO(priyag): Delete this once all strategies use global batch size.
  @property
  def _global_batch_size(self):
    """`make_dataset_iterator` and `make_numpy_iterator` use global batch size.

    `make_input_fn_iterator` assumes per-replica batching.

    Returns:
      Boolean.
    """
    return True
