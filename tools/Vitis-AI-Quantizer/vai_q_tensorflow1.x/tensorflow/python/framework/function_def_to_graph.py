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
"""Utlity to convert FunctionDef to GraphDef and Graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import versions
from tensorflow.python.framework.func_graph import FuncGraph


def function_def_to_graph(fdef, input_shapes=None, copy_functions=True):
  """Converts a FunctionDef to a FuncGraph (sub-class Graph).

  The returned FuncGraph's `name`, `inputs` and `outputs` fields will be set.
  The input tensors are represented as placeholders.

  Note: `FuncGraph.inputs` and `FuncGraph.captures` are not set and may be set
  by the caller.

  Args:
    fdef: FunctionDef.
    input_shapes: Optional. A list of TensorShape objects of the shapes of
      function inputs. Defaults to the function's "_input_shapes" attribute. If
      specified, its length must match length of `fdef.signature.input_arg`. If
      a shape is None, the corresponding input placeholder will have unknown
      shape.
    copy_functions: Whether to copy all functions that exists in default graph
      (independently of being used or not) to the created FuncGraph. Functions
      required for graph import will be copied regardless.

  Returns:
    A FuncGraph.
  """
  func_graph = FuncGraph(fdef.signature.name)
  if input_shapes is None:
    input_shapes_attr = fdef.attr.get("_input_shapes", None)
    if input_shapes_attr is not None:
      input_shapes = input_shapes_attr.list.shape
  graph_def, nested_to_flat_tensor_name = function_def_to_graph_def(
      fdef, input_shapes, copy_functions)

  with func_graph.as_default():
    # Add all function nodes to the graph.
    importer.import_graph_def_for_function(graph_def, name="")

    # Initialize fields specific to FuncGraph.

    # inputs
    input_tensor_names = [
        nested_to_flat_tensor_name[arg.name] for arg in fdef.signature.input_arg
    ]
    func_graph.inputs = [
        func_graph.get_tensor_by_name(name) for name in input_tensor_names
    ]

    # outputs
    output_tensor_names = [
        nested_to_flat_tensor_name[fdef.ret[arg.name]]
        for arg in fdef.signature.output_arg
    ]
    func_graph.outputs = [
        func_graph.get_tensor_by_name(name) for name in output_tensor_names
    ]
    func_graph.control_outputs = [
        func_graph.get_operation_by_name(fdef.control_ret[ret_name])
        for ret_name in fdef.signature.control_output
    ]
    for node in graph_def.node:
      output_shapes = node.attr.get("_output_shapes", None)
      if output_shapes is not None:
        op = func_graph.get_operation_by_name(node.name)
        # _output_shapes for functions can sometimes be too long because the
        # output-intermediates-for-gradients version of the function was
        # substituted before saving. We'll accept that here. (See b/133666530).
        for output_index, shape in enumerate(
            output_shapes.list.shape[:len(op.outputs)]):
          op.outputs[output_index].set_shape(shape)
    output_names = {}
    for ret_arg_def, tensor_name in zip(
        fdef.signature.output_arg, output_tensor_names):
      output_names[ops.tensor_id(
          func_graph.get_tensor_by_name(tensor_name))] = (
              ret_arg_def.name)
    func_graph._output_names = output_names  # pylint: disable=protected-access
  return func_graph


def is_function(fname):
  """Checks for a function definition with `fname` in the current context."""
  if context.executing_eagerly():
    return context.context().has_function(fname)
  else:
    return ops.get_default_graph()._is_function(fname)  # pylint: disable=protected-access


def function_def_to_graph_def(fdef, input_shapes=None, copy_functions=True):
  """Convert a FunctionDef to a GraphDef.

  Steps:
  1. Creates placeholder nodes corresponding to inputs in
     `FunctionDef.signature.input_arg`.
  2. Adds NodeDefs in `FunctionDef.node_def` to `GraphDef.node`.
  3. Renames inputs of all nodes to use the convention of GraphDef instead of
     FunctionDef. See comment on `FunctionDef.node_def` on how the tensor naming
     in FunctionDefs is different from GraphDefs.

  Args:
    fdef: FunctionDef.
    input_shapes: Optional. A list of TensorShape objects of the shapes of
      function inputs. If specified, its length must match length of
      `fdef.signature.input_arg`. If a shape is None, the corresponding input
      placeholder will have unknown shape.
    copy_functions: Whether to copy all functions that exists in default graph
      (independently of being used or not) to the created GraphDef. Directly
      referenced functions are copied regardless.

  Returns:
    A tuple of (GraphDef, dict<string, string>). The dict contains a mapping
    from nested tensor names (in FunctionDef) to flattened names (in GraphDef).

  Raises:
    ValueError: If the length of input_shapes does not match the number of
      input_args or if the FunctionDef is invalid.
  """
  graph_def = graph_pb2.GraphDef()
  graph_def.versions.CopyFrom(
      versions_pb2.VersionDef(
          producer=versions.GRAPH_DEF_VERSION,
          min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER))

  default_graph = ops.get_default_graph()

  copied_functions = set()

  # Copy *all* functions from outer graph to `graph_def` so that both direct
  # and indirect references are safely handled.
  if copy_functions:
    # pylint: disable=protected-access
    default_graph._copy_functions_to_graph_def(graph_def, 0)
    for function_name in default_graph._functions.keys():
      copied_functions.add(function_name)
    # pylint: enable=protected-access

  if input_shapes and len(input_shapes) != len(fdef.signature.input_arg):
    raise ValueError("Length of input_shapes must match the number of " +
                     "input_args. len(input_shapes): {} len(input_arg): {}".
                     format(len(input_shapes), len(fdef.signature.input_arg)))

  # 1. Create placeholders for input nodes.
  for i, arg_def in enumerate(fdef.signature.input_arg):
    node_def = graph_def.node.add()
    node_def.name = arg_def.name
    node_def.op = "Placeholder"
    node_def.attr["dtype"].type = arg_def.type
    if input_shapes and input_shapes[i] is not None:
      input_shape = input_shapes[i]
      if not isinstance(input_shape, tensor_shape_pb2.TensorShapeProto):
        input_shape = input_shape.as_proto()
      node_def.attr["shape"].shape.CopyFrom(input_shape)
    arg_attrs = fdef.arg_attr[i].attr
    for k in arg_attrs:
      # Only copy internal attributes. Normal attributes for nodes cannot be
      # applied to these Placeholder nodes.
      if k.startswith("_"):
        node_def.attr[k].CopyFrom(arg_attrs[k])

  # 2. Copy all body NodeDefs to the GraphDef.
  graph_def.node.extend(fdef.node_def)

  # 3. Perform the renaming.

  # Build the tensor name mapping then flatten the tensor names.
  # See comment on `FunctionDef.node_def` on how the tensor naming in
  # FunctionDefs is different from GraphDefs.
  nested_to_flat_tensor_name = {}

  for arg_def in fdef.signature.input_arg:
    nested_to_flat_tensor_name[arg_def.name] = "{}:0".format(arg_def.name)
    control_name = "^" + arg_def.name
    nested_to_flat_tensor_name[control_name] = control_name

  for node_def in fdef.node_def:
    f = default_graph._functions.get(node_def.op, None)  # pylint: disable=protected-access
    if f is not None and hasattr(f, "signature"):
      op_def = f.signature
      if node_def.op not in copied_functions:
        # Since this function is referenced as an op type, we have no choice but
        # to copy it into the GraphDef if we want downstream tools to process
        # it.
        graph_def.library.function.add().CopyFrom(f.definition)
        copied_functions.add(node_def.op)
    else:
      op_def = ops.get_default_graph()._get_op_def(node_def.op)  # pylint: disable=protected-access

    for attr in op_def.attr:
      if attr.type == "func":
        fname = node_def.attr[attr.name].func.name
        if not is_function(fname):
          raise ValueError("%s function not found." % fname)
      elif attr.type == "list(func)":
        for fn in node_def.attr[attr.name].list.func:
          fname = fn.name
          if not is_function(fname):
            raise ValueError("%s function not found." % fname)

    # Iterate over output_args in op_def to build the map.
    # Index of the output tensor in the flattened list of *all* output
    # tensors of the op.
    flattened_index = 0
    for arg_def in op_def.output_arg:
      num_args = _get_num_args(arg_def, node_def)
      for i in range(num_args):
        # Map tensor names from "node_name:output_arg_name:index" to
        # "node_name:flattened_index".
        nested_name = "{}:{}:{}".format(node_def.name, arg_def.name, i)
        flat_name = "{}:{}".format(node_def.name, flattened_index)
        nested_to_flat_tensor_name[nested_name] = flat_name
        flattened_index += 1
    control_name = "^" + node_def.name
    nested_to_flat_tensor_name[control_name] = control_name

  # Update inputs of all nodes in graph.
  for node_def in graph_def.node:
    for i in range(len(node_def.input)):
      node_def.input[i] = nested_to_flat_tensor_name[node_def.input[i]]

  return graph_def, nested_to_flat_tensor_name


# Based on implementation in core/framework/node_def_util.cc::ComputeArgRange.
def _get_num_args(arg_def, node_def):
  if arg_def.number_attr:
    return node_def.attr[arg_def.number_attr].i
  elif arg_def.type_list_attr:
    return len(node_def.attr[arg_def.type_list_attr].list.type)
  elif arg_def.type_attr or arg_def.type != types_pb2.DT_INVALID:
    return 1
  else:
    raise ValueError("Invalid arg_def:\n\n{}".format(str(arg_def)))
