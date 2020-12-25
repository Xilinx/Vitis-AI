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
"""Import a trackable object from a SavedModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import function_deserialization
from tensorflow.python.saved_model import load_v1_in_v2
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import graph_view
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


def _unused_handle():
  """Returns a placeholder as handle that is not supposed to be accessed."""
  error_message = ("Trying to access a placeholder that is not supposed to be "
                   "executed. This means you are executing a graph generated "
                   "from cross-replica context in an in-replica context.")

  assert_op = control_flow_ops.Assert(
      array_ops.placeholder_with_default(False, shape=()),
      [error_message])

  with ops.control_dependencies([assert_op]):
    return array_ops.placeholder(dtype=dtypes.resource)


class _WrapperFunction(function.ConcreteFunction):
  """A class wraps a concrete function to handle different distributed contexts.

  The reason for wrapping a concrete function is because the _captured_inputs
  fields used for in-replica context and cross-replica context are different.
  When `load()` is called from within a tf.distribute.strategy scope, the
  captured inputs are distributed variables. When using these distributed
  variables during calling the function, we need different approaches when it is
  in-replica and when it is not in-replica. When it is in replica, naturally we
  should use the corresponding component of the distributed variable; when it is
  not in-replica, calling the function should mean that it is constructing a
  graph that is not actually going to be used. A typical use case is when
  constructing a functional model. In this case, return a placeholder with a
  control dependency to ensure that is is never accessed.
  """

  def __init__(self, concrete_function):
    # Shallow copy the concrete_function
    self.__dict__.update(vars(concrete_function))

  def _call_flat(self, args, captured_inputs, cancellation_manager=None):

    def get_in_replica_handle(x):
      return x.handle if ds_values.is_distributed_variable(x) else x

    def get_cross_replica_handle(x):
      return _unused_handle() if ds_values.is_distributed_variable(x) else x

    if ds_context.get_replica_context() is not None:  # in-replica context
      captured_inputs = list(map(get_in_replica_handle, captured_inputs))
    else:  # cross-replica context
      captured_inputs = list(
          map(get_cross_replica_handle, captured_inputs))
    return super(_WrapperFunction, self)._call_flat(args, captured_inputs,
                                                    cancellation_manager)


class Loader(object):
  """Helper class to load an object-based SavedModel."""

  def __init__(self, object_graph_proto, saved_model_proto, export_dir):
    meta_graph = saved_model_proto.meta_graphs[0]
    self._asset_file_def = meta_graph.asset_file_def
    self._operation_attributes = {
        node.name: node.attr for node in meta_graph.graph_def.node}
    self._proto = object_graph_proto
    self._export_dir = export_dir
    self._concrete_functions = (
        function_deserialization.load_function_def_library(
            meta_graph.graph_def.library))

    for name, concrete_function in self._concrete_functions.items():
      # Wrap all the concrete function so that they are capable of dealing with
      # both in replica and cross replica cases.
      self._concrete_functions[name] = _WrapperFunction(concrete_function)

    self._load_all()
    # TODO(b/124045874): There are limitations with functions whose captures
    # trigger other functions to be executed. For now it is only guaranteed to
    # work if the captures of a function only trigger functions without
    # captures.
    self._setup_functions_structures()
    self._setup_functions_captures()
    self._restore_checkpoint()

    for node in self._nodes:
      if isinstance(node, tracking.CapturableResource):
        init_op = node._initialize()  # pylint: disable=protected-access
        if not context.executing_eagerly():
          ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)

  def _setup_functions_structures(self):
    """Setup structure for inputs and outputs of restored functions."""
    coder = nested_structure_coder.StructureCoder()
    for name, proto in sorted(self._proto.concrete_functions.items()):
      concrete_function = self._concrete_functions[name]
      # By setting the structured_outputs directly, we can rely on this
      # function_lib.ConcreteFunction object to perform the output repacking
      # logic. The only limitation of that logic is that it only works
      # with output that is convertible to Tensors and the conversion
      # always happens. For example tf.TensorShape([2, 3]) will be
      # converted to Tensor representing [2, 3].
      original_outputs = coder.decode_proto(proto.output_signature)
      # The original_outputs here had Tensors converted to TensorSpecs, so
      # the restored function's structured_outputs field will not be
      # exactly the same. Fortunately the repacking logic cares only about
      # the structure.
      # TODO(vbardiovsky): Should we just replicate the structures, with
      # Nones instead of real objects?
      concrete_function._func_graph.structured_outputs = original_outputs  # pylint: disable=protected-access
      concrete_function._func_graph.structured_input_signature = (  # pylint: disable=protected-access
          coder.decode_proto(proto.canonicalized_input_signature))

  def _setup_functions_captures(self):
    """Setup captures and variables in restored functions."""
    concrete_functions = sorted(self._proto.concrete_functions.items())
    for name, proto in concrete_functions:
      concrete_function = self._concrete_functions[name]
      bound_inputs = [
          self._get_tensor_from_node(node_id)
          for node_id in proto.bound_inputs]
      bound_variables = [
          self._nodes[node_id]
          for node_id in proto.bound_inputs
          if self._proto.nodes[node_id].WhichOneof("kind") == "variable"
      ]
      # TODO(andresp): This is only injecting the captured inputs into the
      # concrete function, note that we did not modify the FuncGraph
      # itself.
      concrete_function._captured_inputs = bound_inputs  # pylint: disable=protected-access
      concrete_function._func_graph.variables = bound_variables  # pylint: disable=protected-access
      if bound_inputs:
        for bound_input, internal_capture in zip(
            bound_inputs, concrete_function.inputs[-len(bound_inputs):]):
          if ds_values.is_distributed_variable(bound_input):
            concrete_function.graph.capture_distributed_variable(
                bound_input, internal_capture)
          else:
            concrete_function.graph._captures[ops.tensor_id(bound_input)] = (  # pylint: disable=protected-access
                bound_input, internal_capture)
            if internal_capture.dtype == dtypes.resource:
              if resource_variable_ops.is_resource_variable(bound_input):
                try:
                  handle = bound_input.handle
                except ValueError:
                  # For mirrored variables we'll copy handle data for components
                  # as they get captured.
                  pass
                else:
                  custom_gradient.copy_handle_data(handle, internal_capture)
              else:
                custom_gradient.copy_handle_data(bound_input, internal_capture)
            # Setting "captures" first means "capture" won't create a new
            # placeholder for this input.
            concrete_function.graph.capture(bound_input)

  def _get_tensor_from_node(self, node_id):
    """Resolves a node id into a tensor to be captured for a function."""
    with ops.init_scope():
      obj = self._nodes[node_id]
      if ds_values.is_distributed_variable(obj):
        return obj
      elif resource_variable_ops.is_resource_variable(obj):
        return obj.handle
      elif isinstance(obj, tracking.TrackableAsset):
        return obj.asset_path
      elif tensor_util.is_tensor(obj):
        return obj
      elif isinstance(obj, tracking.CapturableResource):
        # Note: this executes restored functions in the CapturableResource.
        return obj.resource_handle
      raise ValueError("Can't convert node %s to tensor" % (type(obj)))

  def _load_all(self):
    """Load all saved objects and wire their properties."""
    # Maps from node ids to recreated objects
    nodes = {}
    # Maps from node ids to setter functions (same signature as setattr) for
    # setting dependencies.
    node_setters = {}

    # Figure out which objects are slot variables. These objects are created
    # with Optimizer.add_slot rather than _recreate_variable.
    slot_variable_node_ids = set()
    for proto in self._proto.nodes:
      for slot_variable_proto in proto.slot_variables:
        slot_variable_node_ids.add(slot_variable_proto.slot_variable_node_id)

    # Re-create everything except slot variables.
    for node_id, proto in enumerate(self._proto.nodes):
      if node_id in slot_variable_node_ids:
        # Defer recreating slot variables so we can use the public Optimizer
        # interface.
        continue
      node, setter = self._recreate(proto)
      nodes[node_id] = node
      node_setters[node_id] = setter

    # Now that we have created the variables being optimized, we have enough
    # information to re-create slot variables for them.
    for node_id, proto in enumerate(self._proto.nodes):
      optimizer_object = nodes[node_id]
      for slot_variable_proto in proto.slot_variables:
        optimized_variable = nodes[
            slot_variable_proto.original_variable_node_id]
        slot_variable = optimizer_object.add_slot(
            var=optimized_variable,
            slot_name=slot_variable_proto.slot_name)
        nodes[slot_variable_proto.slot_variable_node_id] = slot_variable
        node_setters[slot_variable_proto.slot_variable_node_id] = setattr

    self._nodes = []

    # After creating the objects, construct the edges between the objects.
    for node_id, object_proto in enumerate(self._proto.nodes):
      obj = nodes[node_id]
      setter = node_setters[node_id]
      self._nodes.append(obj)

      for reference in object_proto.children:
        setter(obj, reference.local_name, nodes[reference.node_id])
        # Note: if an object has an attribute `__call__` add a class method
        # that allows `obj()` syntax to work. This is done per-instance to
        # allow `callable` to be used to find out if an object is callable.
        if reference.local_name == "__call__" and not callable(obj):
          setattr(type(obj), "__call__", _call_attribute)

  def _restore_checkpoint(self):
    """Load state from checkpoint into the deserialized objects."""
    variables_path = saved_model_utils.get_variables_path(self._export_dir)
    # TODO(andresp): Clean use of private methods of TrackableSaver.
    # pylint: disable=protected-access
    saver = util.TrackableSaver(graph_view.ObjectGraphView(self.get(0)))
    with ops.device("CPU"):
      saver._file_prefix_placeholder = constant_op.constant(variables_path)
    load_status = saver.restore(variables_path)
    load_status.assert_existing_objects_matched()
    checkpoint = load_status._checkpoint

    # When running in eager mode, the `restore` call above has already run and
    # restored the state of trackables, call `position.restore_ops()` will
    # return an empty list as there is nothing left to do. In graph mode, that
    # will return the list of ops that must run to restore the object on that
    # position. We have to wire them in the initializers of the objects so that
    # they get initialized properly when using common practices (e.g. the ones
    # used by ManagedSession) without further user action.
    for object_id, obj in dict(checkpoint.object_by_proto_id).items():
      position = base.CheckpointPosition(checkpoint=checkpoint,
                                         proto_id=object_id)
      restore_ops = position.restore_ops()
      if restore_ops:
        if resource_variable_ops.is_resource_variable(obj):
          obj._initializer_op = restore_ops
        else:
          raise NotImplementedError(
              ("Missing functionality to restore state of object "
               "%r from the checkpoint." % obj))

  def get(self, node_id):
    return self._nodes[node_id]

  def _recreate(self, proto):
    """Creates a Python object from a SavedObject protocol buffer."""
    factory = {
        "user_object": lambda: self._recreate_user_object(proto.user_object),
        "asset": lambda: self._recreate_asset(proto.asset),
        "function": lambda: self._recreate_function(proto.function),
        "bare_concrete_function": functools.partial(
            self._recreate_bare_concrete_function,
            proto.bare_concrete_function),
        "variable": lambda: self._recreate_variable(proto.variable),
        "constant": lambda: self._recreate_constant(proto.constant),
        "resource": lambda: self._recreate_resource(proto.resource),
    }
    kind = proto.WhichOneof("kind")
    if kind not in factory:
      raise ValueError("Unknown SavedObject type: %r" % kind)
    return factory[kind]()

  def _recreate_user_object(self, proto):
    """Instantiates a SavedUserObject."""
    looked_up = revived_types.deserialize(proto)
    if looked_up is None:
      return self._recreate_base_user_object(proto)
    return looked_up

  def _recreate_base_user_object(self, proto):
    del proto
    # Note: each user object has its own class. This allows to make each one
    # individually callable by adding a `__call__` method to the classes of
    # the objects instances that have a `__call__` property.

    class _UserObject(tracking.AutoTrackable):
      pass

    return _UserObject(), setattr

  def _recreate_asset(self, proto):
    filename = os.path.join(
        saved_model_utils.get_assets_dir(self._export_dir),
        self._asset_file_def[proto.asset_file_def_index].filename)
    return tracking.TrackableAsset(filename), setattr

  def _recreate_function(self, proto):
    return function_deserialization.recreate_function(
        proto, self._concrete_functions), setattr

  def _recreate_bare_concrete_function(self, proto):
    return function_deserialization.setup_bare_concrete_function(
        proto, self._concrete_functions), setattr

  def _recreate_variable(self, proto):
    name = proto.name if proto.name else None
    if name is not None:
      dbg_name = name
    else:
      dbg_name = "<variable loaded from saved model>"
    synchronization, aggregation, trainable = (
        variables.validate_synchronization_aggregation_trainable(
            proto.synchronization, proto.aggregation, proto.trainable,
            name=dbg_name))

    def uninitialized_variable_creator(next_creator, **kwargs):
      """A variable creator that creates uninitialized variables."""
      del next_creator
      return resource_variable_ops.UninitializedVariable(**kwargs)

    # Create a variable_creator_scope that creates uninitialized variables with
    # a lower priority such that a potential distributed variable_creator_scope
    # can take precedence.
    with ops.get_default_graph()._variable_creator_scope(  # pylint: disable=protected-access
        uninitialized_variable_creator,
        priority=50):
      return variables.Variable(
          shape=proto.shape,
          dtype=proto.dtype,
          name=name,
          trainable=trainable,
          synchronization=synchronization,
          aggregation=aggregation), setattr

  def _recreate_constant(self, proto):
    tensor_proto = self._operation_attributes[proto.operation]["value"].tensor
    ndarray = tensor_util.MakeNdarray(tensor_proto)
    if dtypes.as_dtype(tensor_proto.dtype) == dtypes.string:
      with ops.device("CPU"):
        imported_constant = constant_op.constant(ndarray)
    else:
      imported_constant = constant_op.constant(ndarray)
    return imported_constant, setattr

  def _recreate_resource(self, proto):
    return _RestoredResource(device=proto.device), setattr


# TODO(b/124205571,b/124092991): Solve destruction of resources.
class _RestoredResource(tracking.TrackableResource):
  """Restored SavedResource."""

  def __init__(self, device=""):
    super(_RestoredResource, self).__init__(device=device)
    self._destroy_resource_fn = None

  def _create_resource(self):
    raise RuntimeError()

  def _initialize(self):
    raise RuntimeError()

  @property
  def _destroy_resource(self):
    return self._destroy_resource_fn

  @_destroy_resource.setter
  def _destroy_resource(self, destroy_resource_fn):
    self._resource_deleter = tracking.CapturableResourceDeleter(
        destroy_resource_fn)
    self._destroy_resource_fn = destroy_resource_fn

  def _list_functions_for_serialization(self, unused_serialization_cache):
    # Overwrite this method to avoid the implementation of
    # base class to re-wrap the polymorphic functions into
    # another layer of `tf.function`.
    functions = {
        "_create_resource": self._create_resource,
        "_initialize": self._initialize,
    }
    if self._destroy_resource:
      functions.update(_destroy_resource=self._destroy_resource)
    return functions


def _call_attribute(instance, *args, **kwargs):
  return instance.__call__(*args, **kwargs)


@tf_export("saved_model.load", v1=["saved_model.load_v2"])
def load(export_dir, tags=None):
  """Load a SavedModel from `export_dir`.

  Signatures associated with the SavedModel are available as functions:

  ```python
  imported = tf.saved_model.load(path)
  f = imported.signatures["serving_default"]
  print(f(x=tf.constant([[1.]])))
  ```

  Objects exported with `tf.saved_model.save` additionally have trackable
  objects and functions assigned to attributes:

  ```python
  exported = tf.train.Checkpoint(v=tf.Variable(3.))
  exported.f = tf.function(
      lambda x: exported.v * x,
      input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  tf.saved_model.save(exported, path)
  imported = tf.saved_model.load(path)
  assert 3. == imported.v.numpy()
  assert 6. == imported.f(x=tf.constant(2.)).numpy()
  ```

  _Loading Keras models_

  Keras models are trackable, so they can be saved to SavedModel. The object
  returned by `tf.saved_model.load` is not a Keras object (i.e. doesn't have
  `.fit`, `.predict`, etc. methods). A few attributes and functions are still
  available: `.variables`, `.trainable_variables` and `.__call__`.

  ```python
  model = tf.keras.Model(...)
  tf.saved_model.save(model, path)
  imported = tf.saved_model.load(path)
  outputs = imported(inputs)
  ```

  Use `tf.keras.models.load_model` to restore the Keras model.

  _Importing SavedModels from TensorFlow 1.x_

  SavedModels from `tf.estimator.Estimator` or 1.x SavedModel APIs have a flat
  graph instead of `tf.function` objects. These SavedModels will have functions
  corresponding to their signatures in the `.signatures` attribute, but also
  have a `.prune` method which allows you to extract functions for new
  subgraphs. This is equivalent to importing the SavedModel and naming feeds and
  fetches in a Session from TensorFlow 1.x.

  ```python
  imported = tf.saved_model.load(path_to_v1_saved_model)
  pruned = imported.prune("x:0", "out:0")
  pruned(tf.ones([]))
  ```

  See `tf.compat.v1.wrap_function` for details. These SavedModels also have a
  `.variables` attribute containing imported variables, and a `.graph` attribute
  representing the whole imported graph. For SavedModels exported from
  `tf.saved_model.save`, variables are instead assigned to whichever attributes
  they were assigned before export.

  Args:
    export_dir: The SavedModel directory to load from.
    tags: A tag or sequence of tags identifying the MetaGraph to load. Optional
      if the SavedModel contains a single MetaGraph, as for those exported from
      `tf.saved_model.load`.

  Returns:
    A trackable object with a `signatures` attribute mapping from signature
    keys to functions. If the SavedModel was exported by `tf.saved_model.load`,
    it also points to trackable objects and functions which were attached
    to the exported object.

  Raises:
    ValueError: If `tags` don't match a MetaGraph in the SavedModel.
  """
  return load_internal(export_dir, tags)


def load_internal(export_dir, tags=None, loader_cls=Loader):
  """Loader implementation."""
  if tags is not None and not isinstance(tags, set):
    # Supports e.g. tags=SERVING and tags=[SERVING]. Sets aren't considered
    # sequences for nest.flatten, so we put those through as-is.
    tags = nest.flatten(tags)
  saved_model_proto = loader_impl.parse_saved_model(export_dir)
  if (len(saved_model_proto.meta_graphs) == 1
      and saved_model_proto.meta_graphs[0].HasField("object_graph_def")):
    meta_graph_def = saved_model_proto.meta_graphs[0]
    if (tags is not None
        and set(tags) != set(meta_graph_def.meta_info_def.tags)):
      raise ValueError(
          ("The SavedModel at {} has one MetaGraph with tags {}, but got an "
           "incompatible argument tags={} to tf.saved_model.load. You may omit "
           "it, pass 'None', or pass matching tags.")
          .format(export_dir, meta_graph_def.meta_info_def.tags, tags))
    object_graph_proto = meta_graph_def.object_graph_def
    with ops.init_scope():
      loader = loader_cls(object_graph_proto,
                          saved_model_proto,
                          export_dir)
      root = loader.get(0)
    root.tensorflow_version = meta_graph_def.meta_info_def.tensorflow_version
    root.tensorflow_git_version = (
        meta_graph_def.meta_info_def.tensorflow_git_version)
  else:
    with ops.init_scope():
      root = load_v1_in_v2.load(export_dir, tags)
  return root
