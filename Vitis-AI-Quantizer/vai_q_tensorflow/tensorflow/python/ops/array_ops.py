# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# Tests for this file live in python/kernel_tests/array_ops_test.py
"""Support for manipulating tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import six

from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
# 'Constant' gets imported in the module 'array_ops'.
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
# pylint: enable=wildcard-import

# Used for slicing to specify a new 1 size dimension
newaxis = None
tf_export("newaxis").export_constant(__name__, "newaxis")

# We override the 'slice' for the "slice" op, so we keep python's
# existing 'slice' for later use in this module.
_BaseSlice = slice


@tf_export("reshape", v1=["reshape", "manip.reshape"])
def reshape(tensor, shape, name=None):  # pylint: disable=redefined-outer-name
  r"""Reshapes a tensor.

  Given `tensor`, this operation returns a tensor that has the same values
  as `tensor` with shape `shape`.

  If one component of `shape` is the special value -1, the size of that
  dimension is computed so that the total size remains constant.  In particular,
  a `shape` of `[-1]` flattens into 1-D.  At most one component of `shape` can
  be -1.

  If `shape` is 1-D or higher, then the operation returns a tensor with shape
  `shape` filled with the values of `tensor`. In this case, the number of
  elements implied by `shape` must be the same as the number of elements in
  `tensor`.

  For example:

  ```
  # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
  # tensor 't' has shape [9]
  reshape(t, [3, 3]) ==> [[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]]

  # tensor 't' is [[[1, 1], [2, 2]],
  #                [[3, 3], [4, 4]]]
  # tensor 't' has shape [2, 2, 2]
  reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                          [3, 3, 4, 4]]

  # tensor 't' is [[[1, 1, 1],
  #                 [2, 2, 2]],
  #                [[3, 3, 3],
  #                 [4, 4, 4]],
  #                [[5, 5, 5],
  #                 [6, 6, 6]]]
  # tensor 't' has shape [3, 2, 3]
  # pass '[-1]' to flatten 't'
  reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

  # -1 can also be used to infer the shape

  # -1 is inferred to be 9:
  reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]
  # -1 is inferred to be 2:
  reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]
  # -1 is inferred to be 3:
  reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                                [2, 2, 2],
                                [3, 3, 3]],
                               [[4, 4, 4],
                                [5, 5, 5],
                                [6, 6, 6]]]

  # tensor 't' is [7]
  # shape `[]` reshapes to a scalar
  reshape(t, []) ==> 7
  ```

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Defines the shape of the output tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  result = gen_array_ops.reshape(tensor, shape, name)
  tensor_util.maybe_set_static_shape(result, shape)
  return result


@tf_export("fill")
def fill(dims, value, name=None):
  r"""Creates a tensor filled with a scalar value.

  This operation creates a tensor of shape `dims` and fills it with `value`.

  For example:

  ```
  # Output tensor has shape [2, 3].
  fill([2, 3], 9) ==> [[9, 9, 9]
                       [9, 9, 9]]
  ```

  `tf.fill` differs from `tf.constant` in a few ways:

  *   `tf.fill` only supports scalar contents, whereas `tf.constant` supports
      Tensor values.
  *   `tf.fill` creates an Op in the computation graph that constructs the
  actual
      Tensor value at runtime. This is in contrast to `tf.constant` which embeds
      the entire Tensor into the graph with a `Const` node.
  *   Because `tf.fill` evaluates at graph runtime, it supports dynamic shapes
      based on other runtime Tensors, unlike `tf.constant`.

  Args:
    dims: A `Tensor`. Must be one of the following types: `int32`, `int64`. 1-D.
      Represents the shape of the output tensor.
    value: A `Tensor`. 0-D (scalar). Value to fill the returned tensor.
      @compatibility(numpy) Equivalent to np.full @end_compatibility
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  """
  result = gen_array_ops.fill(dims, value, name=name)
  tensor_util.maybe_set_static_shape(result, dims)
  return result


@tf_export("identity")
@dispatch.add_dispatch_support
def identity(input, name=None):  # pylint: disable=redefined-builtin
  r"""Return a tensor with the same shape and contents as input.

  For example:

  ```python
  import tensorflow as tf
  val0 = tf.ones((1,), dtype=tf.float32)
  a = tf.atan2(val0, val0)
  a_identity = tf.identity(a)
  print(a.numpy())          #[0.7853982]
  print(a_identity.numpy()) #[0.7853982]
  ```

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  if context.executing_eagerly() and not hasattr(input, "graph"):
    # Make sure we get an input with handle data attached from resource
    # variables. Variables have correct handle data when graph building.
    input = ops.convert_to_tensor(input)
  ret = gen_array_ops.identity(input, name=name)
  # Propagate handle data for happier shape inference for resource variables.
  if hasattr(input, "_handle_data"):
    ret._handle_data = input._handle_data  # pylint: disable=protected-access
  return ret


# pylint: disable=redefined-builtin,protected-access
@tf_export(v1=["expand_dims"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None, "Use the `axis` argument instead", "dim")
def expand_dims(input, axis=None, name=None, dim=None):
  """Inserts a dimension of 1 into a tensor's shape.

  Given a tensor `input`, this operation inserts a dimension of 1 at the
  dimension index `axis` of `input`'s shape. The dimension index `axis` starts
  at zero; if you specify a negative number for `axis` it is counted backward
  from the end.

  This operation is useful if you want to add a batch dimension to a single
  element. For example, if you have a single image of shape `[height, width,
  channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
  which will make the shape `[1, height, width, channels]`.

  Other examples:

  ```python
  # 't' is a tensor of shape [2]
  tf.shape(tf.expand_dims(t, 0))  # [1, 2]
  tf.shape(tf.expand_dims(t, 1))  # [2, 1]
  tf.shape(tf.expand_dims(t, -1))  # [2, 1]

  # 't2' is a tensor of shape [2, 3, 5]
  tf.shape(tf.expand_dims(t2, 0))  # [1, 2, 3, 5]
  tf.shape(tf.expand_dims(t2, 2))  # [2, 3, 1, 5]
  tf.shape(tf.expand_dims(t2, 3))  # [2, 3, 5, 1]
  ```

  This operation requires that:

  `-1-input.dims() <= dim <= input.dims()`

  This operation is related to `squeeze()`, which removes dimensions of
  size 1.

  Args:
    input: A `Tensor`.
    axis: 0-D (scalar). Specifies the dimension index at which to expand the
      shape of `input`. Must be in the range `[-rank(input) - 1, rank(input)]`.
    name: The name of the output `Tensor` (optional).
    dim: 0-D (scalar). Equivalent to `axis`, to be deprecated.

  Returns:
    A `Tensor` with the same data as `input`, but its shape has an additional
    dimension of size 1 added.

  Raises:
    ValueError: if either both or neither of `dim` and `axis` are specified.
  """
  axis = deprecation.deprecated_argument_lookup("axis", axis, "dim", dim)
  if axis is None:
    raise ValueError("Must specify an axis argument to tf.expand_dims()")
  return expand_dims_v2(input, axis, name)


@tf_export("expand_dims", v1=[])
@dispatch.add_dispatch_support
def expand_dims_v2(input, axis, name=None):
  """Inserts a dimension of 1 into a tensor's shape.

  Given a tensor `input`, this operation inserts a dimension of 1 at the
  dimension index `axis` of `input`'s shape. The dimension index `axis` starts
  at zero; if you specify a negative number for `axis` it is counted backward
  from the end.

  This operation is useful if you want to add a batch dimension to a single
  element. For example, if you have a single image of shape `[height, width,
  channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
  which will make the shape `[1, height, width, channels]`.

  Other examples:

  ```python
  # 't' is a tensor of shape [2]
  tf.shape(tf.expand_dims(t, 0))  # [1, 2]
  tf.shape(tf.expand_dims(t, 1))  # [2, 1]
  tf.shape(tf.expand_dims(t, -1))  # [2, 1]

  # 't2' is a tensor of shape [2, 3, 5]
  tf.shape(tf.expand_dims(t2, 0))  # [1, 2, 3, 5]
  tf.shape(tf.expand_dims(t2, 2))  # [2, 3, 1, 5]
  tf.shape(tf.expand_dims(t2, 3))  # [2, 3, 5, 1]
  ```

  This operation requires that:

  `-1-input.dims() <= dim <= input.dims()`

  This operation is related to `squeeze()`, which removes dimensions of
  size 1.

  Args:
    input: A `Tensor`.
    axis: 0-D (scalar). Specifies the dimension index at which to expand the
      shape of `input`. Must be in the range `[-rank(input) - 1, rank(input)]`.
    name: The name of the output `Tensor` (optional).

  Returns:
    A `Tensor` with the same data as `input`, but its shape has an additional
    dimension of size 1 added.
  """
  return gen_array_ops.expand_dims(input, axis, name)


# pylint: enable=redefined-builtin,protected-access


# Aliases for some automatically-generated names.
# pylint: disable=protected-access
@deprecation.deprecated("2016-11-30",
                        "This op will be removed after the deprecation date. "
                        "Please switch to tf.setdiff1d().")
def listdiff(x, y, out_idx=None, name=None):
  return gen_array_ops.list_diff(x, y, out_idx, name)


listdiff.__doc__ = gen_array_ops.list_diff.__doc__ + "\n" + listdiff.__doc__

# pylint: enable=protected-access


# pylint: disable=undefined-variable
@deprecation.deprecated("2018-11-30",
                        "This op will be removed after the deprecation date. "
                        "Please switch to tf.sets.difference().")
@tf_export(v1=["setdiff1d"])
def setdiff1d(x, y, index_dtype=dtypes.int32, name=None):
  return gen_array_ops.list_diff(x, y, index_dtype, name)


setdiff1d.__doc__ = gen_array_ops.list_diff.__doc__


@tf_export("broadcast_dynamic_shape")
def broadcast_dynamic_shape(shape_x, shape_y):
  """Computes the shape of a broadcast given symbolic shapes.

  When shape_x and shape_y are Tensors representing shapes (i.e. the result of
  calling tf.shape on another Tensor) this computes a Tensor which is the shape
  of the result of a broadcasting op applied in tensors of shapes shape_x and
  shape_y.

  For example, if shape_x is [1, 2, 3] and shape_y is [5, 1, 3], the result is a
  Tensor whose value is [5, 2, 3].

  This is useful when validating the result of a broadcasting operation when the
  tensors do not have statically known shapes.

  Args:
    shape_x: A rank 1 integer `Tensor`, representing the shape of x.
    shape_y: A rank 1 integer `Tensor`, representing the shape of y.

  Returns:
    A rank 1 integer `Tensor` representing the broadcasted shape.
  """
  return gen_array_ops.broadcast_args(shape_x, shape_y)


@tf_export("broadcast_static_shape")
def broadcast_static_shape(shape_x, shape_y):
  """Computes the shape of a broadcast given known shapes.

  When shape_x and shape_y are fully known TensorShapes this computes a
  TensorShape which is the shape of the result of a broadcasting op applied in
  tensors of shapes shape_x and shape_y.

  For example, if shape_x is [1, 2, 3] and shape_y is [5, 1, 3], the result is a
  TensorShape whose value is [5, 2, 3].

  This is useful when validating the result of a broadcasting operation when the
  tensors have statically known shapes.

  Args:
    shape_x: A `TensorShape`
    shape_y: A `TensorShape`

  Returns:
    A `TensorShape` representing the broadcasted shape.

  Raises:
    ValueError: If the two shapes can not be broadcasted.
  """
  return common_shapes.broadcast_shape(shape_x, shape_y)


@tf_export("shape", v1=[])
def shape_v2(input, out_type=dtypes.int32, name=None):
  # pylint: disable=redefined-builtin
  """Returns the shape of a tensor.

  This operation returns a 1-D integer tensor representing the shape of `input`.

  For example:

  ```python
  t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
  tf.shape(t)  # [2, 2, 3]
  ```

  Args:
    input: A `Tensor` or `SparseTensor`.
    out_type: (Optional) The specified output type of the operation (`int32` or
      `int64`). Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
  return shape(input, name, out_type)


@tf_export(v1=["shape"])
def shape(input, name=None, out_type=dtypes.int32):
  # pylint: disable=redefined-builtin
  """Returns the shape of a tensor.

  This operation returns a 1-D integer tensor representing the shape of `input`.

  For example:

  ```python
  t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
  tf.shape(t)  # [2, 2, 3]
  ```

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    out_type: (Optional) The specified output type of the operation (`int32` or
      `int64`). Defaults to `tf.int32`.

  Returns:
    A `Tensor` of type `out_type`.
  """
  return shape_internal(input, name, optimize=True, out_type=out_type)


def shape_internal(input, name=None, optimize=True, out_type=dtypes.int32):
  # pylint: disable=redefined-builtin
  """Returns the shape of a tensor.

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    optimize: if true, encode the shape as a constant when possible.
    out_type: (Optional) The specified output type of the operation (`int32` or
      `int64`). Defaults to tf.int32.

  Returns:
    A `Tensor` of type `out_type`.

  """
  with ops.name_scope(name, "Shape", [input]) as name:
    if isinstance(
        input, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
      return gen_math_ops.cast(input.dense_shape, out_type)
    else:
      if not context.executing_eagerly():
        input_tensor = ops.convert_to_tensor(input)
        input_shape = input_tensor.get_shape()
        if optimize and input_shape.is_fully_defined():
          return constant(input_shape.as_list(), out_type, name=name)
      return gen_array_ops.shape(input, name=name, out_type=out_type)


@tf_export("shape_n")
def shape_n(input, out_type=dtypes.int32, name=None):
  # pylint: disable=redefined-builtin
  """Returns shape of tensors.

  Args:
    input: A list of at least 1 `Tensor` object with the same type.
    out_type: The specified output type of the operation (`int32` or `int64`).
      Defaults to `tf.int32`(optional).
    name: A name for the operation (optional).

  Returns:
    A list with the same length as `input` of `Tensor` objects with
      type `out_type`.
  """

  return gen_array_ops.shape_n(input, out_type=out_type, name=name)


@tf_export("size", v1=[])
@dispatch.add_dispatch_support
def size_v2(input, out_type=dtypes.int32, name=None):
  # pylint: disable=redefined-builtin
  return size(input, name, out_type)


@tf_export(v1=["size"])
@dispatch.add_dispatch_support
def size(input, name=None, out_type=dtypes.int32):
  # pylint: disable=redefined-builtin
  """Returns the size of a tensor.

  Returns a 0-D `Tensor` representing the number of elements in `input`
  of type `out_type`. Defaults to tf.int32.

  For example:

  ```python
  t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
  tf.size(t)  # 12
  ```

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    out_type: (Optional) The specified non-quantized numeric output type of the
      operation. Defaults to `tf.int32`.

  Returns:
    A `Tensor` of type `out_type`. Defaults to `tf.int32`.

  @compatibility(numpy)
  Equivalent to np.size()
  @end_compatibility
  """
  return size_internal(input, name, optimize=True, out_type=out_type)


def size_internal(input, name=None, optimize=True, out_type=dtypes.int32):
  # pylint: disable=redefined-builtin,protected-access
  """Returns the size of a tensor.

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    optimize: if true, encode the size as a constant when possible.
    out_type: (Optional) The specified non-quantized numeric output type of the
      operation. Defaults to `tf.int32`.

  Returns:
    A `Tensor` of type `out_type`. Defaults to `tf.int32`.
  """
  if (context.executing_eagerly()
      and not hasattr(input, "graph")
      and not isinstance(
          input, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue))
     ):
    input = ops.convert_to_tensor(input)
    np_out_type = out_type.as_numpy_dtype
    num_elements = np.prod(input._shape_tuple(), dtype=np_out_type)  # pylint: disable=protected-access
    return ops.convert_to_tensor(num_elements, dtype=out_type)
  with ops.name_scope(name, "Size", [input]) as name:
    if isinstance(
        input, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
      return gen_math_ops.prod(
          gen_math_ops.cast(input.dense_shape, out_type), 0, name=name)
    else:
      input_tensor = ops.convert_to_tensor(input)
      input_shape = input_tensor.get_shape()
      if optimize:
        if input_shape.is_fully_defined():
          return constant(input_shape.num_elements(), out_type, name=name)
        if input_shape.dims and any(dim == 0 for dim in input_shape.dims):
          return constant(0, out_type, name=name)
      return gen_array_ops.size(input, name=name, out_type=out_type)


@tf_export("rank")
@dispatch.add_dispatch_support
def rank(input, name=None):
  # pylint: disable=redefined-builtin
  """Returns the rank of a tensor.

  Returns a 0-D `int32` `Tensor` representing the rank of `input`.

  For example:

  ```python
  # shape of tensor 't' is [2, 2, 3]
  t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
  tf.rank(t)  # 3
  ```

  **Note**: The rank of a tensor is not the same as the rank of a matrix. The
  rank of a tensor is the number of indices required to uniquely select each
  element of the tensor. Rank is also known as "order", "degree", or "ndims."

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.

  @compatibility(numpy)
  Equivalent to np.ndim
  @end_compatibility
  """
  return rank_internal(input, name, optimize=True)


def rank_internal(input, name=None, optimize=True):
  # pylint: disable=redefined-builtin
  """Returns the rank of a tensor.

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    optimize: if true, encode the rank as a constant when possible.

  Returns:
    A `Tensor` of type `int32`.
  """
  with ops.name_scope(name, "Rank", [input]) as name:
    if isinstance(
        input, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
      return gen_array_ops.size(input.dense_shape, name=name)
    else:
      input_tensor = ops.convert_to_tensor(input)
      input_shape = input_tensor.get_shape()
      if optimize and input_shape.ndims is not None:
        return constant(input_shape.ndims, dtypes.int32, name=name)
      return gen_array_ops.rank(input, name=name)


_SLICE_TYPE_ERROR = (
    "Only integers, slices (`:`), ellipsis (`...`), "
    "tf.newaxis (`None`) and scalar tf.int32/tf.int64 tensors are valid "
    "indices")

_SUPPORTED_SLICE_DTYPES = (dtypes.int32, dtypes.int32_ref, dtypes.int64,
                           dtypes.int64_ref)


def _check_index(idx):
  """Check if a given value is a valid index into a tensor."""
  if isinstance(idx, (six.integer_types, tensor_shape.Dimension)):
    return

  # Optimistic check. Assumptions:
  # * any object with a dtype is supported
  # * any object with a dtype has a sizeable shape attribute.
  dtype = getattr(idx, "dtype", None)
  if (dtype is None or dtypes.as_dtype(dtype) not in _SUPPORTED_SLICE_DTYPES or
      idx.shape and len(idx.shape) == 1):
    # TODO(slebedev): IndexError seems more appropriate here, but it
    # will break `_slice_helper` contract.
    raise TypeError(_SLICE_TYPE_ERROR + ", got {!r}".format(idx))


def _slice_helper(tensor, slice_spec, var=None):
  """Overload for Tensor.__getitem__.

  This operation extracts the specified region from the tensor.
  The notation is similar to NumPy with the restriction that
  currently only support basic indexing. That means that
  using a non-scalar tensor as input is not currently allowed.

  Some useful examples:

  ```python
  # Strip leading and trailing 2 elements
  foo = tf.constant([1,2,3,4,5,6])
  print(foo[2:-2].eval())  # => [3,4]

  # Skip every other row and reverse the order of the columns
  foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
  print(foo[::2,::-1].eval())  # => [[3,2,1], [9,8,7]]

  # Use scalar tensors as indices on both dimensions
  print(foo[tf.constant(0), tf.constant(2)].eval())  # => 3

  # Insert another dimension
  foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
  print(foo[tf.newaxis, :, :].eval()) # => [[[1,2,3], [4,5,6], [7,8,9]]]
  print(foo[:, tf.newaxis, :].eval()) # => [[[1,2,3]], [[4,5,6]], [[7,8,9]]]
  print(foo[:, :, tf.newaxis].eval()) # => [[[1],[2],[3]], [[4],[5],[6]],
  [[7],[8],[9]]]

  # Ellipses (3 equivalent operations)
  foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
  print(foo[tf.newaxis, :, :].eval())  # => [[[1,2,3], [4,5,6], [7,8,9]]]
  print(foo[tf.newaxis, ...].eval())  # => [[[1,2,3], [4,5,6], [7,8,9]]]
  print(foo[tf.newaxis].eval())  # => [[[1,2,3], [4,5,6], [7,8,9]]]

  # Masks
  foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
  print(foo[foo > 2].eval())  # => [3, 4, 5, 6, 7, 8, 9]
  ```

  Notes:
    - `tf.newaxis` is `None` as in NumPy.
    - An implicit ellipsis is placed at the end of the `slice_spec`
    - NumPy advanced indexing is currently not supported.

  Args:
    tensor: An ops.Tensor object.
    slice_spec: The arguments to Tensor.__getitem__.
    var: In the case of variable slice assignment, the Variable object to slice
      (i.e. tensor is the read-only view of this variable).

  Returns:
    The appropriate slice of "tensor", based on "slice_spec".

  Raises:
    ValueError: If a slice range is negative size.
    TypeError: If the slice indices aren't int, slice, ellipsis,
      tf.newaxis or scalar int32/int64 tensors.
  """
  if isinstance(slice_spec, bool) or \
  (isinstance(slice_spec, ops.Tensor) and slice_spec.dtype == dtypes.bool) or \
  (isinstance(slice_spec, np.ndarray) and slice_spec.dtype == bool):
    return boolean_mask(tensor=tensor, mask=slice_spec)

  if not isinstance(slice_spec, (list, tuple)):
    slice_spec = [slice_spec]

  begin, end, strides = [], [], []
  index = 0

  new_axis_mask, shrink_axis_mask = 0, 0
  begin_mask, end_mask = 0, 0
  ellipsis_mask = 0
  for s in slice_spec:
    if isinstance(s, _BaseSlice):
      # python doesn't always use None when constructing ranges
      # for example a[:] gives slice(None,sys.maxsize,None)
      # whereas a[::1] gives slice(None,None,None)
      if s.start is not None and (isinstance(s.start, ops.Tensor) or
                                  s.start != sys.maxsize):
        _check_index(s.start)
        begin.append(s.start)
      else:
        begin.append(0)
        begin_mask |= (1 << index)
      if s.stop is not None and (isinstance(s.stop, ops.Tensor) or
                                 s.stop != sys.maxsize):
        _check_index(s.stop)
        end.append(s.stop)
      else:
        end.append(0)
        end_mask |= (1 << index)
      if s.step is not None:
        _check_index(s.step)
        strides.append(s.step)
      else:
        strides.append(1)
    elif s is Ellipsis:
      begin.append(0)
      end.append(0)
      strides.append(1)
      ellipsis_mask |= (1 << index)
    elif s is newaxis:
      begin.append(0)
      end.append(0)
      strides.append(1)
      new_axis_mask |= (1 << index)
    else:
      _check_index(s)
      begin.append(s)
      end.append(s + 1)
      strides.append(1)
      shrink_axis_mask |= (1 << index)
    index += 1

  # stack possibly involves no tensors, so we must use op_scope correct graph.
  with ops.name_scope(None, "strided_slice",
                      [tensor] + begin + end + strides) as name:
    if begin:
      packed_begin, packed_end, packed_strides = (stack(begin), stack(end),
                                                  stack(strides))
      if (packed_begin.dtype == dtypes.int64 or
          packed_end.dtype == dtypes.int64 or
          packed_strides.dtype == dtypes.int64):
        if packed_begin.dtype != dtypes.int64:
          packed_begin = gen_math_ops.cast(packed_begin, dtypes.int64)
        if packed_end.dtype != dtypes.int64:
          packed_end = gen_math_ops.cast(packed_end, dtypes.int64)
        if packed_strides.dtype != dtypes.int64:
          packed_strides = gen_math_ops.cast(packed_strides, dtypes.int64)
    else:
      var_empty = constant([], dtype=dtypes.int32)
      packed_begin = packed_end = packed_strides = var_empty
    return strided_slice(
        tensor,
        packed_begin,
        packed_end,
        packed_strides,
        begin_mask=begin_mask,
        end_mask=end_mask,
        shrink_axis_mask=shrink_axis_mask,
        new_axis_mask=new_axis_mask,
        ellipsis_mask=ellipsis_mask,
        var=var,
        name=name)


# pylint: disable=undefined-variable,protected-access,redefined-outer-name
@tf_export("slice")
def slice(input_, begin, size, name=None):
  # pylint: disable=redefined-builtin
  """Extracts a slice from a tensor.

  This operation extracts a slice of size `size` from a tensor `input_` starting
  at the location specified by `begin`. The slice `size` is represented as a
  tensor shape, where `size[i]` is the number of elements of the 'i'th dimension
  of `input_` that you want to slice. The starting location (`begin`) for the
  slice is represented as an offset in each dimension of `input_`. In other
  words, `begin[i]` is the offset into the i'th dimension of `input_` that you
  want to slice from.

  Note that `tf.Tensor.__getitem__` is typically a more pythonic way to
  perform slices, as it allows you to write `foo[3:7, :-2]` instead of
  `tf.slice(foo, [3, 0], [4, foo.get_shape()[1]-2])`.

  `begin` is zero-based; `size` is one-based. If `size[i]` is -1,
  all remaining elements in dimension i are included in the
  slice. In other words, this is equivalent to setting:

  `size[i] = input_.dim_size(i) - begin[i]`

  This operation requires that:

  `0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n]`

  For example:

  ```python
  t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                   [[3, 3, 3], [4, 4, 4]],
                   [[5, 5, 5], [6, 6, 6]]])
  tf.slice(t, [1, 0, 0], [1, 1, 3])  # [[[3, 3, 3]]]
  tf.slice(t, [1, 0, 0], [1, 2, 3])  # [[[3, 3, 3],
                                     #   [4, 4, 4]]]
  tf.slice(t, [1, 0, 0], [2, 1, 3])  # [[[3, 3, 3]],
                                     #  [[5, 5, 5]]]
  ```

  Args:
    input_: A `Tensor`.
    begin: An `int32` or `int64` `Tensor`.
    size: An `int32` or `int64` `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` the same type as `input_`.
  """
  return gen_array_ops._slice(input_, begin, size, name=name)


# pylint: disable=invalid-name
@tf_export("strided_slice")
def strided_slice(input_,
                  begin,
                  end,
                  strides=None,
                  begin_mask=0,
                  end_mask=0,
                  ellipsis_mask=0,
                  new_axis_mask=0,
                  shrink_axis_mask=0,
                  var=None,
                  name=None):
  """Extracts a strided slice of a tensor (generalized python array indexing).

  **Instead of calling this op directly most users will want to use the
  NumPy-style slicing syntax (e.g. `tensor[..., 3:4:-1, tf.newaxis, 3]`), which
  is supported via `tf.Tensor.__getitem__` and `tf.Variable.__getitem__`.**
  The interface of this op is a low-level encoding of the slicing syntax.

  Roughly speaking, this op extracts a slice of size `(end-begin)/stride`
  from the given `input_` tensor. Starting at the location specified by `begin`
  the slice continues by adding `stride` to the index until all dimensions are
  not less than `end`.
  Note that a stride can be negative, which causes a reverse slice.

  Given a Python slice `input[spec0, spec1, ..., specn]`,
  this function will be called as follows.

  `begin`, `end`, and `strides` will be vectors of length n.
  n in general is not equal to the rank of the `input_` tensor.

  In each mask field (`begin_mask`, `end_mask`, `ellipsis_mask`,
  `new_axis_mask`, `shrink_axis_mask`) the ith bit will correspond to
  the ith spec.

  If the ith bit of `begin_mask` is set, `begin[i]` is ignored and
  the fullest possible range in that dimension is used instead.
  `end_mask` works analogously, except with the end range.

  `foo[5:,:,:3]` on a 7x8x9 tensor is equivalent to `foo[5:7,0:8,0:3]`.
  `foo[::-1]` reverses a tensor with shape 8.

  If the ith bit of `ellipsis_mask` is set, as many unspecified dimensions
  as needed will be inserted between other dimensions. Only one
  non-zero bit is allowed in `ellipsis_mask`.

  For example `foo[3:5,...,4:5]` on a shape 10x3x3x10 tensor is
  equivalent to `foo[3:5,:,:,4:5]` and
  `foo[3:5,...]` is equivalent to `foo[3:5,:,:,:]`.

  If the ith bit of `new_axis_mask` is set, then `begin`,
  `end`, and `stride` are ignored and a new length 1 dimension is
  added at this point in the output tensor.

  For example,
  `foo[:4, tf.newaxis, :2]` would produce a shape `(4, 1, 2)` tensor.

  If the ith bit of `shrink_axis_mask` is set, it implies that the ith
  specification shrinks the dimensionality by 1, taking on the value at index
  `begin[i]`. `end[i]` and `strides[i]` are ignored in this case. For example in
  Python one might do `foo[:, 3, :]` which would result in `shrink_axis_mask`
  equal to 2.


  NOTE: `begin` and `end` are zero-indexed.
  `strides` entries must be non-zero.


  ```python
  t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                   [[3, 3, 3], [4, 4, 4]],
                   [[5, 5, 5], [6, 6, 6]]])
  tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1])  # [[[3, 3, 3]]]
  tf.strided_slice(t, [1, 0, 0], [2, 2, 3], [1, 1, 1])  # [[[3, 3, 3],
                                                        #   [4, 4, 4]]]
  tf.strided_slice(t, [1, -1, 0], [2, -3, 3], [1, -1, 1])  # [[[4, 4, 4],
                                                           #   [3, 3, 3]]]
  ```

  Args:
    input_: A `Tensor`.
    begin: An `int32` or `int64` `Tensor`.
    end: An `int32` or `int64` `Tensor`.
    strides: An `int32` or `int64` `Tensor`.
    begin_mask: An `int32` mask.
    end_mask: An `int32` mask.
    ellipsis_mask: An `int32` mask.
    new_axis_mask: An `int32` mask.
    shrink_axis_mask: An `int32` mask.
    var: The variable corresponding to `input_` or None
    name: A name for the operation (optional).

  Returns:
    A `Tensor` the same type as `input`.
  """

  if strides is None:
    strides = ones_like(begin)

  op = gen_array_ops.strided_slice(
      input=input_,
      begin=begin,
      end=end,
      strides=strides,
      name=name,
      begin_mask=begin_mask,
      end_mask=end_mask,
      ellipsis_mask=ellipsis_mask,
      new_axis_mask=new_axis_mask,
      shrink_axis_mask=shrink_axis_mask)

  parent_name = name

  if not (var is None and isinstance(op, ops.EagerTensor)):

    def assign(val, name=None):
      """Closure that holds all the arguments to create an assignment."""

      if var is None:
        raise ValueError("Sliced assignment is only supported for variables")
      else:
        if name is None:
          name = parent_name + "_assign"

        return var._strided_slice_assign(
            begin=begin,
            end=end,
            strides=strides,
            value=val,
            name=name,
            begin_mask=begin_mask,
            end_mask=end_mask,
            ellipsis_mask=ellipsis_mask,
            new_axis_mask=new_axis_mask,
            shrink_axis_mask=shrink_axis_mask)

    op.assign = assign
  return op


def _SliceHelperVar(var, slice_spec):
  """Creates a slice helper object given a variable.

  This allows creating a sub-tensor from part of the current contents
  of a variable. See `tf.Tensor.__getitem__` for detailed examples
  of slicing.

  This function in addition also allows assignment to a sliced range.
  This is similar to `__setitem__` functionality in Python. However,
  the syntax is different so that the user can capture the assignment
  operation for grouping or passing to `sess.run()`.
  For example,

  ```python
  import tensorflow as tf
  A = tf.Variable([[1,2,3], [4,5,6], [7,8,9]], dtype=tf.float32)
  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run(A[:2, :2]))  # => [[1,2], [4,5]]

    op = A[:2,:2].assign(22. * tf.ones((2, 2)))
    print(sess.run(op))  # => [[22, 22, 3], [22, 22, 6], [7,8,9]]
  ```

  Note that assignments currently do not support NumPy broadcasting
  semantics.

  Args:
    var: An `ops.Variable` object.
    slice_spec: The arguments to `Tensor.__getitem__`.

  Returns:
    The appropriate slice of "tensor", based on "slice_spec".
    As an operator. The operator also has a `assign()` method
    that can be used to generate an assignment operator.

  Raises:
    ValueError: If a slice range is negative size.
    TypeError: TypeError: If the slice indices aren't int, slice,
      ellipsis, tf.newaxis or int32/int64 tensors.

  """

  return _slice_helper(var.value(), slice_spec, var)


ops.Tensor._override_operator("__getitem__", _slice_helper)


@tf_export("parallel_stack")
def parallel_stack(values, name="parallel_stack"):
  """Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor in parallel.

  Requires that the shape of inputs be known at graph construction time.

  Packs the list of tensors in `values` into a tensor with rank one higher than
  each tensor in `values`, by packing them along the first dimension.
  Given a list of length `N` of tensors of shape `(A, B, C)`; the `output`
  tensor will have the shape `(N, A, B, C)`.

  For example:

  ```python
  x = tf.constant([1, 4])
  y = tf.constant([2, 5])
  z = tf.constant([3, 6])
  tf.parallel_stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]]
  ```

  The difference between `stack` and `parallel_stack` is that `stack` requires
  all the inputs be computed before the operation will begin but doesn't require
  that the input shapes be known during graph construction.

  `parallel_stack` will copy pieces of the input into the output as they become
  available, in some situations this can provide a performance benefit.

  Unlike `stack`, `parallel_stack` does NOT support backpropagation.

  This is the opposite of unstack.  The numpy equivalent is

      tf.parallel_stack([x, y, z]) = np.asarray([x, y, z])

  Args:
    values: A list of `Tensor` objects with the same shape and type.
    name: A name for this operation (optional).

  Returns:
    output: A stacked `Tensor` with the same type as `values`.
  """
  with ops.name_scope(name):
    value_t = ops.convert_to_tensor(values[0])
    value_shape = ops.convert_to_tensor(value_t).get_shape()

    output_shape = tensor_shape.TensorShape([len(values)])
    output_shape = output_shape.concatenate(value_shape)
    # expand_dims converts concat to stack.
    return gen_array_ops.parallel_concat(
        [expand_dims(value, 0) for value in values], shape=output_shape)


@tf_export("stack")
@dispatch.add_dispatch_support
def stack(values, axis=0, name="stack"):
  """Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.

  Packs the list of tensors in `values` into a tensor with rank one higher than
  each tensor in `values`, by packing them along the `axis` dimension.
  Given a list of length `N` of tensors of shape `(A, B, C)`;

  if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
  if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
  Etc.

  For example:

  ```python
  x = tf.constant([1, 4])
  y = tf.constant([2, 5])
  z = tf.constant([3, 6])
  tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
  tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
  ```

  This is the opposite of unstack.  The numpy equivalent is

  ```python
  tf.stack([x, y, z]) = np.stack([x, y, z])
  ```

  Args:
    values: A list of `Tensor` objects with the same shape and type.
    axis: An `int`. The axis to stack along. Defaults to the first dimension.
      Negative values wrap around, so the valid range is `[-(R+1), R+1)`.
    name: A name for this operation (optional).

  Returns:
    output: A stacked `Tensor` with the same type as `values`.

  Raises:
    ValueError: If `axis` is out of the range [-(R+1), R+1).
  """
  if axis == 0:
    try:
      # If the input is a constant list, it can be converted to a constant op
      return ops.convert_to_tensor(values, name=name)
    except (TypeError, ValueError):
      pass  # Input list contains non-constant tensors

  value_shape = ops.convert_to_tensor(values[0], name=name)._shape_tuple()  # pylint: disable=protected-access
  if value_shape is not None:
    expanded_num_dims = len(value_shape) + 1
    if axis < -expanded_num_dims or axis >= expanded_num_dims:
      raise ValueError("axis = %d not in [%d, %d)" %
                       (axis, -expanded_num_dims, expanded_num_dims))

  return gen_array_ops.pack(values, axis=axis, name=name)


# pylint: disable=invalid-name
def _autopacking_helper(list_or_tuple, dtype, name):
  """Converts the given list or tuple to a tensor by packing.

  Args:
    list_or_tuple: A (possibly nested) list or tuple containing a tensor.
    dtype: The element type of the returned tensor.
    name: A name for the returned tensor.

  Returns:
    A `tf.Tensor` with value equivalent to `list_or_tuple`.
  """
  if context.executing_eagerly():
    # NOTE: Fast path when all the items are tensors, this doesn't do any type
    # checking.
    if all(ops.is_dense_tensor_like(elem) for elem in list_or_tuple):
      return gen_array_ops.pack(list_or_tuple, name=name)
  must_pack = False
  converted_elems = []
  with ops.name_scope(name) as scope:
    for i, elem in enumerate(list_or_tuple):
      if ops.is_dense_tensor_like(elem):
        if dtype is not None and elem.dtype.base_dtype != dtype:
          raise TypeError("Cannot convert a list containing a tensor of dtype "
                          "%s to %s (Tensor is: %r)" %
                          (elem.dtype, dtype, elem))
        converted_elems.append(elem)
        must_pack = True
      elif isinstance(elem, (list, tuple)):
        converted_elem = _autopacking_helper(elem, dtype, str(i))
        if ops.is_dense_tensor_like(converted_elem):
          must_pack = True
        converted_elems.append(converted_elem)
      else:
        converted_elems.append(elem)
    if must_pack:
      elems_as_tensors = []
      for i, elem in enumerate(converted_elems):
        if ops.is_dense_tensor_like(elem):
          elems_as_tensors.append(elem)
        else:
          # NOTE(mrry): This is inefficient, but it enables us to
          # handle the case where the list arguments are other
          # convertible-to-tensor types, such as numpy arrays.
          elems_as_tensors.append(
              constant_op.constant(elem, dtype=dtype, name=str(i)))
      return gen_array_ops.pack(elems_as_tensors, name=scope)
    else:
      return converted_elems


def _get_dtype_from_nested_lists(list_or_tuple):
  """Returns the dtype of any tensor-like object in `list_or_tuple`, if found.

  Args:
    list_or_tuple: A list or tuple representing an object that can be converted
      to a `tf.Tensor`.

  Returns:
    The dtype of any tensor-like object in `list_or_tuple`, or `None` if no
    such object exists.
  """
  for elem in list_or_tuple:
    if ops.is_dense_tensor_like(elem):
      return elem.dtype.base_dtype
    elif isinstance(elem, (list, tuple)):
      maybe_dtype = _get_dtype_from_nested_lists(elem)
      if maybe_dtype is not None:
        return maybe_dtype
  return None


def _cast_nested_seqs_to_dtype(dtype):

  def _maybe_cast(elem):
    if ops.is_dense_tensor_like(elem):
      if dtype != elem.dtype.base_dtype:
        elem = gen_math_ops.cast(elem, dtype)
    return elem

  return _maybe_cast


_NON_AUTOPACKABLE_TYPES = set(np.core.numerictypes.ScalarType)
_NON_AUTOPACKABLE_TYPES.add(np.ndarray)


def _should_not_autopack(v):
  # The condition we really want is
  #    ops.is_dense_tensor_like(...)
  # but it is >5x slower due to abc.ABCMeta.__instancecheck__.
  # pylint: disable=unidiomatic-typecheck
  # TODO(slebedev): add nest.all?
  return all(type(elem) in _NON_AUTOPACKABLE_TYPES for elem in nest.flatten(v))
  # pylint: enable=unidiomatic-typecheck


def _autopacking_conversion_function(v, dtype=None, name=None, as_ref=False):
  """Tensor conversion function that automatically packs arguments."""
  if as_ref or _should_not_autopack(v):
    return NotImplemented
  inferred_dtype = _get_dtype_from_nested_lists(v)
  if inferred_dtype is None:
    # We did not find any tensor-like objects in the nested lists, so defer to
    # other conversion functions.
    return NotImplemented
  if dtype is None:
    dtype = inferred_dtype
  elif dtype != inferred_dtype:
    v = nest.map_structure(_cast_nested_seqs_to_dtype(dtype), v)
  return _autopacking_helper(v, dtype, name or "packed")


# pylint: enable=invalid-name

# NOTE: Register this conversion function to run *before* one that
# assumes every element is a value.
ops.register_tensor_conversion_function((list, tuple),
                                        _autopacking_conversion_function, 99)


@tf_export("unstack")
def unstack(value, num=None, axis=0, name="unstack"):
  """Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.

  Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
  If `num` is not specified (the default), it is inferred from `value`'s shape.
  If `value.shape[axis]` is not known, `ValueError` is raised.

  For example, given a tensor of shape `(A, B, C, D)`;

  If `axis == 0` then the i'th tensor in `output` is the slice
    `value[i, :, :, :]` and each tensor in `output` will have shape `(B, C, D)`.
    (Note that the dimension unpacked along is gone, unlike `split`).

  If `axis == 1` then the i'th tensor in `output` is the slice
    `value[:, i, :, :]` and each tensor in `output` will have shape `(A, C, D)`.
  Etc.

  This is the opposite of stack.

  Args:
    value: A rank `R > 0` `Tensor` to be unstacked.
    num: An `int`. The length of the dimension `axis`. Automatically inferred if
      `None` (the default).
    axis: An `int`. The axis to unstack along. Defaults to the first dimension.
      Negative values wrap around, so the valid range is `[-R, R)`.
    name: A name for the operation (optional).

  Returns:
    The list of `Tensor` objects unstacked from `value`.

  Raises:
    ValueError: If `num` is unspecified and cannot be inferred.
    ValueError: If `axis` is out of the range [-R, R).
  """
  if num is None:
    value = ops.convert_to_tensor(value)
    value_shape = value.get_shape()
    if value_shape.ndims is not None:
      if axis < -value_shape.ndims or axis >= value_shape.ndims:
        raise ValueError("axis = %d not in [%d, %d)" %
                         (axis, -value_shape.ndims, value_shape.ndims))
      num = value_shape.dims[axis].value
  if num is None:
    raise ValueError("Cannot infer num from shape %s" % value_shape)
  return gen_array_ops.unpack(value, num=num, axis=axis, name=name)


@tf_export("concat")
@dispatch.add_dispatch_support
def concat(values, axis, name="concat"):
  """Concatenates tensors along one dimension.

  Concatenates the list of tensors `values` along dimension `axis`.  If
  `values[i].shape = [D0, D1, ... Daxis(i), ...Dn]`, the concatenated
  result has shape

      [D0, D1, ... Raxis, ...Dn]

  where

      Raxis = sum(Daxis(i))

  That is, the data from the input tensors is joined along the `axis`
  dimension.

  The number of dimensions of the input tensors must match, and all dimensions
  except `axis` must be equal.

  For example:

  ```python
  t1 = [[1, 2, 3], [4, 5, 6]]
  t2 = [[7, 8, 9], [10, 11, 12]]
  tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
  tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

  # tensor t3 with shape [2, 3]
  # tensor t4 with shape [2, 3]
  tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
  tf.shape(tf.concat([t3, t4], 1))  # [2, 6]
  ```
  As in Python, the `axis` could also be negative numbers. Negative `axis`
  are interpreted as counting from the end of the rank, i.e.,
   `axis + rank(values)`-th dimension.

  For example:

  ```python
  t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
  t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
  tf.concat([t1, t2], -1)
  ```

  would produce:

  ```python
  [[[ 1,  2,  7,  4],
    [ 2,  3,  8,  4]],

   [[ 4,  4,  2, 10],
    [ 5,  3, 15, 11]]]
  ```

  Note: If you are concatenating along a new axis consider using stack.
  E.g.

  ```python
  tf.concat([tf.expand_dims(t, axis) for t in tensors], axis)
  ```

  can be rewritten as

  ```python
  tf.stack(tensors, axis=axis)
  ```

  Args:
    values: A list of `Tensor` objects or a single `Tensor`.
    axis: 0-D `int32` `Tensor`.  Dimension along which to concatenate. Must be
      in the range `[-rank(values), rank(values))`. As in Python, indexing for
      axis is 0-based. Positive axis in the rage of `[0, rank(values))` refers
      to `axis`-th dimension. And negative axis refers to `axis +
      rank(values)`-th dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` resulting from concatenation of the input tensors.
  """
  if not isinstance(values, (list, tuple)):
    values = [values]
  # TODO(mrry): Change to return values?
  if len(values) == 1:  # Degenerate case of one tensor.
    # Make a throwaway call to convert_to_tensor to make sure
    # that axis is of the correct type, and make sure that
    # the returned tensor is a scalar.
    # TODO(keveman): Implement a standalone type and shape checker.
    with ops.name_scope(name) as scope:
      ops.convert_to_tensor(
          axis, name="concat_dim",
          dtype=dtypes.int32).get_shape().assert_has_rank(0)
      return identity(values[0], name=name)
  return gen_array_ops.concat_v2(values=values, axis=axis, name=name)


@tf_export(v1=["boolean_mask"])
def boolean_mask(tensor, mask, name="boolean_mask", axis=None):
  """Apply boolean mask to tensor.

  Numpy equivalent is `tensor[mask]`.

  ```python
  # 1-D example
  tensor = [0, 1, 2, 3]
  mask = np.array([True, False, True, False])
  boolean_mask(tensor, mask)  # [0, 2]
  ```

  In general, `0 < dim(mask) = K <= dim(tensor)`, and `mask`'s shape must match
  the first K dimensions of `tensor`'s shape.  We then have:
    `boolean_mask(tensor, mask)[i, j1,...,jd] = tensor[i1,...,iK,j1,...,jd]`
  where `(i1,...,iK)` is the ith `True` entry of `mask` (row-major order).
  The `axis` could be used with `mask` to indicate the axis to mask from.
  In that case, `axis + dim(mask) <= dim(tensor)` and `mask`'s shape must match
  the first `axis + dim(mask)` dimensions of `tensor`'s shape.

  See also: `tf.ragged.boolean_mask`, which can be applied to both dense and
  ragged tensors, and can be used if you need to preserve the masked dimensions
  of `tensor` (rather than flattening them, as `tf.boolean_mask` does).

  Args:
    tensor:  N-D tensor.
    mask:  K-D boolean tensor, K <= N and K must be known statically.
    name:  A name for this operation (optional).
    axis:  A 0-D int Tensor representing the axis in `tensor` to mask from. By
      default, axis is 0 which will mask from the first dimension. Otherwise K +
      axis <= N.

  Returns:
    (N-K+1)-dimensional tensor populated by entries in `tensor` corresponding
    to `True` values in `mask`.

  Raises:
    ValueError:  If shapes do not conform.

  Examples:

  ```python
  # 2-D example
  tensor = [[1, 2], [3, 4], [5, 6]]
  mask = np.array([True, False, True])
  boolean_mask(tensor, mask)  # [[1, 2], [5, 6]]
  ```
  """

  def _apply_mask_1d(reshaped_tensor, mask, axis=None):
    """Mask tensor along dimension 0 with a 1-D mask."""
    indices = squeeze(where(mask), axis=[1])
    return gather(reshaped_tensor, indices, axis=axis)

  with ops.name_scope(name, values=[tensor, mask]):
    tensor = ops.convert_to_tensor(tensor, name="tensor")
    mask = ops.convert_to_tensor(mask, name="mask")

    shape_mask = mask.get_shape()
    ndims_mask = shape_mask.ndims
    shape_tensor = tensor.get_shape()
    if ndims_mask == 0:
      raise ValueError("mask cannot be scalar.")
    if ndims_mask is None:
      raise ValueError(
          "Number of mask dimensions must be specified, even if some dimensions"
          " are None.  E.g. shape=[None] is ok, but shape=None is not.")
    axis = 0 if axis is None else axis
    shape_tensor[axis:axis + ndims_mask].assert_is_compatible_with(shape_mask)

    leading_size = gen_math_ops.prod(shape(tensor)[axis:axis + ndims_mask], [0])
    tensor = reshape(
        tensor,
        concat([
            shape(tensor)[:axis], [leading_size],
            shape(tensor)[axis + ndims_mask:]
        ], 0))
    first_dim = shape_tensor[axis:axis + ndims_mask].num_elements()
    tensor.set_shape(
        tensor_shape.as_shape(shape_tensor[:axis]).concatenate(
            [first_dim]).concatenate(shape_tensor[axis + ndims_mask:]))

    mask = reshape(mask, [-1])
    return _apply_mask_1d(tensor, mask, axis)


@tf_export("boolean_mask", v1=[])
@dispatch.add_dispatch_support
def boolean_mask_v2(tensor, mask, axis=None, name="boolean_mask"):
  """Apply boolean mask to tensor.

  Numpy equivalent is `tensor[mask]`.

  ```python
  # 1-D example
  tensor = [0, 1, 2, 3]
  mask = np.array([True, False, True, False])
  boolean_mask(tensor, mask)  # [0, 2]
  ```

  In general, `0 < dim(mask) = K <= dim(tensor)`, and `mask`'s shape must match
  the first K dimensions of `tensor`'s shape.  We then have:
    `boolean_mask(tensor, mask)[i, j1,...,jd] = tensor[i1,...,iK,j1,...,jd]`
  where `(i1,...,iK)` is the ith `True` entry of `mask` (row-major order).
  The `axis` could be used with `mask` to indicate the axis to mask from.
  In that case, `axis + dim(mask) <= dim(tensor)` and `mask`'s shape must match
  the first `axis + dim(mask)` dimensions of `tensor`'s shape.

  See also: `tf.ragged.boolean_mask`, which can be applied to both dense and
  ragged tensors, and can be used if you need to preserve the masked dimensions
  of `tensor` (rather than flattening them, as `tf.boolean_mask` does).

  Args:
    tensor:  N-D tensor.
    mask:  K-D boolean tensor, K <= N and K must be known statically.
    axis:  A 0-D int Tensor representing the axis in `tensor` to mask from. By
      default, axis is 0 which will mask from the first dimension. Otherwise K +
      axis <= N.
    name:  A name for this operation (optional).

  Returns:
    (N-K+1)-dimensional tensor populated by entries in `tensor` corresponding
    to `True` values in `mask`.

  Raises:
    ValueError:  If shapes do not conform.

  Examples:

  ```python
  # 2-D example
  tensor = [[1, 2], [3, 4], [5, 6]]
  mask = np.array([True, False, True])
  boolean_mask(tensor, mask)  # [[1, 2], [5, 6]]
  ```
  """
  return boolean_mask(tensor, mask, name, axis)


@tf_export("sparse.mask", v1=["sparse.mask", "sparse_mask"])
@deprecation.deprecated_endpoints("sparse_mask")
def sparse_mask(a, mask_indices, name=None):
  """Masks elements of `IndexedSlices`.

  Given an `IndexedSlices` instance `a`, returns another `IndexedSlices` that
  contains a subset of the slices of `a`. Only the slices at indices not
  specified in `mask_indices` are returned.

  This is useful when you need to extract a subset of slices in an
  `IndexedSlices` object.

  For example:

  ```python
  # `a` contains slices at indices [12, 26, 37, 45] from a large tensor
  # with shape [1000, 10]
  a.indices  # [12, 26, 37, 45]
  tf.shape(a.values)  # [4, 10]

  # `b` will be the subset of `a` slices at its second and third indices, so
  # we want to mask its first and last indices (which are at absolute
  # indices 12, 45)
  b = tf.sparse.mask(a, [12, 45])

  b.indices  # [26, 37]
  tf.shape(b.values)  # [2, 10]
  ```

  Args:
    a: An `IndexedSlices` instance.
    mask_indices: Indices of elements to mask.
    name: A name for the operation (optional).

  Returns:
    The masked `IndexedSlices` instance.
  """
  with ops.name_scope(name, "sparse_mask", [a, mask_indices]) as name:
    indices = a.indices
    out_indices, to_gather = gen_array_ops.list_diff(indices, mask_indices)
    out_values = gather(a.values, to_gather, name=name)
    return ops.IndexedSlices(out_values, out_indices, a.dense_shape)


@tf_export("unique")
def unique(x, out_idx=dtypes.int32, name=None):
  # TODO(yongtang): switch to v2 once API deprecation
  # period (3 weeks) pass.
  # TODO(yongtang): The documentation should also
  # be updated when switch  to v2.
  return gen_array_ops.unique(x, out_idx, name)


unique.__doc__ = gen_array_ops.unique.__doc__


@tf_export("unique_with_counts")
def unique_with_counts(x, out_idx=dtypes.int32, name=None):
  # TODO(yongtang): switch to v2 once API deprecation
  # period (3 weeks) pass.
  # TODO(yongtang): The documentation should also
  # be updated when switch  to v2.
  return gen_array_ops.unique_with_counts(x, out_idx, name)


unique_with_counts.__doc__ = gen_array_ops.unique_with_counts.__doc__


@tf_export("split")
def split(value, num_or_size_splits, axis=0, num=None, name="split"):
  """Splits a tensor into sub tensors.

  If `num_or_size_splits` is an integer, then `value` is split along dimension
  `axis` into `num_split` smaller tensors. This requires that `num_split` evenly
  divides `value.shape[axis]`.

  If `num_or_size_splits` is a 1-D Tensor (or list), we call it `size_splits`
  and `value` is split into `len(size_splits)` elements. The shape of the `i`-th
  element has the same size as the `value` except along dimension `axis` where
  the size is `size_splits[i]`.

  For example:

  ```python
  # 'value' is a tensor with shape [5, 30]
  # Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
  split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
  tf.shape(split0)  # [5, 4]
  tf.shape(split1)  # [5, 15]
  tf.shape(split2)  # [5, 11]
  # Split 'value' into 3 tensors along dimension 1
  split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
  tf.shape(split0)  # [5, 10]
  ```

  Args:
    value: The `Tensor` to split.
    num_or_size_splits: Either an integer indicating the number of splits along
      split_dim or a 1-D integer `Tensor` or Python list containing the sizes of
      each output tensor along split_dim. If a scalar then it must evenly divide
      `value.shape[axis]`; otherwise the sum of sizes along the split dimension
      must match that of the `value`.
    axis: An integer or scalar `int32` `Tensor`. The dimension along which to
      split. Must be in the range `[-rank(value), rank(value))`. Defaults to 0.
    num: Optional, used to specify the number of outputs when it cannot be
      inferred from the shape of `size_splits`.
    name: A name for the operation (optional).

  Returns:
    if `num_or_size_splits` is a scalar returns `num_or_size_splits` `Tensor`
    objects; if `num_or_size_splits` is a 1-D Tensor returns
    `num_or_size_splits.get_shape[0]` `Tensor` objects resulting from splitting
    `value`.

  Raises:
    ValueError: If `num` is unspecified and cannot be inferred.
  """
  size_splits = ops.convert_to_tensor(num_or_size_splits)
  if isinstance(num_or_size_splits,
                six.integer_types + (tensor_shape.Dimension,)):
    return gen_array_ops.split(
        axis=axis, num_split=num_or_size_splits, value=value, name=name)

  if size_splits._rank() == 0:
    raise ValueError(
        "Rank-0 tensors are not supported as the num_or_size_splits argument "
        "to split. Argument provided: %s" % (num_or_size_splits,))

  if num is None:
    size_splits_shape = size_splits._shape_tuple()
    if size_splits_shape:
      num = size_splits_shape[0]
    if num is None:
      raise ValueError("Cannot infer num from shape %s" % num_or_size_splits)

  return gen_array_ops.split_v(
      value=value, size_splits=size_splits, axis=axis, num_split=num, name=name)


@tf_export("transpose", v1=[])
def transpose_v2(a, perm=None, conjugate=False, name="transpose"):
  """Transposes `a`.

  Permutes the dimensions according to `perm`.

  The returned tensor's dimension i will correspond to the input dimension
  `perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is
  the rank of the input tensor. Hence by default, this operation performs a
  regular matrix transpose on 2-D input Tensors. If conjugate is True and
  `a.dtype` is either `complex64` or `complex128` then the values of `a`
  are conjugated and transposed.

  @compatibility(numpy)
  In `numpy` transposes are memory-efficient constant time operations as they
  simply return a new view of the same data with adjusted `strides`.

  TensorFlow does not support strides, so `transpose` returns a new tensor with
  the items permuted.
  @end_compatibility

  For example:

  ```python
  x = tf.constant([[1, 2, 3], [4, 5, 6]])
  tf.transpose(x)  # [[1, 4]
                   #  [2, 5]
                   #  [3, 6]]

  # Equivalently
  tf.transpose(x, perm=[1, 0])  # [[1, 4]
                                #  [2, 5]
                                #  [3, 6]]

  # If x is complex, setting conjugate=True gives the conjugate transpose
  x = tf.constant([[1 + 1j, 2 + 2j, 3 + 3j],
                   [4 + 4j, 5 + 5j, 6 + 6j]])
  tf.transpose(x, conjugate=True)  # [[1 - 1j, 4 - 4j],
                                   #  [2 - 2j, 5 - 5j],
                                   #  [3 - 3j, 6 - 6j]]

  # 'perm' is more useful for n-dimensional tensors, for n > 2
  x = tf.constant([[[ 1,  2,  3],
                    [ 4,  5,  6]],
                   [[ 7,  8,  9],
                    [10, 11, 12]]])

  # Take the transpose of the matrices in dimension-0
  # (this common operation has a shorthand `linalg.matrix_transpose`)
  tf.transpose(x, perm=[0, 2, 1])  # [[[1,  4],
                                   #   [2,  5],
                                   #   [3,  6]],
                                   #  [[7, 10],
                                   #   [8, 11],
                                   #   [9, 12]]]
  ```

  Args:
    a: A `Tensor`.
    perm: A permutation of the dimensions of `a`.
    conjugate: Optional bool. Setting it to `True` is mathematically equivalent
      to tf.math.conj(tf.transpose(input)).
    name: A name for the operation (optional).

  Returns:
    A transposed `Tensor`.
  """
  return transpose(a=a, perm=perm, name=name, conjugate=conjugate)


@tf_export(v1=["transpose"])
def transpose(a, perm=None, name="transpose", conjugate=False):
  """Transposes `a`.

  Permutes the dimensions according to `perm`.

  The returned tensor's dimension i will correspond to the input dimension
  `perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is
  the rank of the input tensor. Hence by default, this operation performs a
  regular matrix transpose on 2-D input Tensors. If conjugate is True and
  `a.dtype` is either `complex64` or `complex128` then the values of `a`
  are conjugated and transposed.

  @compatibility(numpy)
  In `numpy` transposes are memory-efficient constant time operations as they
  simply return a new view of the same data with adjusted `strides`.

  TensorFlow does not support strides, so `transpose` returns a new tensor with
  the items permuted.
  @end_compatibility

  For example:

  ```python
  x = tf.constant([[1, 2, 3], [4, 5, 6]])
  tf.transpose(x)  # [[1, 4]
                   #  [2, 5]
                   #  [3, 6]]

  # Equivalently
  tf.transpose(x, perm=[1, 0])  # [[1, 4]
                                #  [2, 5]
                                #  [3, 6]]

  # If x is complex, setting conjugate=True gives the conjugate transpose
  x = tf.constant([[1 + 1j, 2 + 2j, 3 + 3j],
                   [4 + 4j, 5 + 5j, 6 + 6j]])
  tf.transpose(x, conjugate=True)  # [[1 - 1j, 4 - 4j],
                                   #  [2 - 2j, 5 - 5j],
                                   #  [3 - 3j, 6 - 6j]]

  # 'perm' is more useful for n-dimensional tensors, for n > 2
  x = tf.constant([[[ 1,  2,  3],
                    [ 4,  5,  6]],
                   [[ 7,  8,  9],
                    [10, 11, 12]]])

  # Take the transpose of the matrices in dimension-0
  # (this common operation has a shorthand `linalg.matrix_transpose`)
  tf.transpose(x, perm=[0, 2, 1])  # [[[1,  4],
                                   #   [2,  5],
                                   #   [3,  6]],
                                   #  [[7, 10],
                                   #   [8, 11],
                                   #   [9, 12]]]
  ```

  Args:
    a: A `Tensor`.
    perm: A permutation of the dimensions of `a`.
    name: A name for the operation (optional).
    conjugate: Optional bool. Setting it to `True` is mathematically equivalent
      to tf.math.conj(tf.transpose(input)).

  Returns:
    A transposed `Tensor`.
  """
  with ops.name_scope(name, "transpose", [a]) as name:
    transpose_fn = (
        gen_array_ops.conjugate_transpose if
        (conjugate and a.dtype.is_complex) else gen_array_ops.transpose)
    if perm is None:
      a = ops.convert_to_tensor(a, name="a")
      if not a.get_shape().ndims:
        rank = gen_array_ops.rank(a)
        perm = (rank - 1) - gen_math_ops._range(0, rank, 1)
      else:
        rank = a.get_shape().ndims
        perm = (rank - 1) - np.arange(rank)
      ret = transpose_fn(a, perm, name=name)
      # NOTE(mrry): Setting the shape explicitly because
      #   reverse is not handled by the shape function.
      if not context.executing_eagerly():
        input_shape = ret.op.inputs[0].get_shape().dims
        if input_shape is not None:
          ret.set_shape(input_shape[::-1])
    else:
      ret = transpose_fn(a, perm, name=name)
    return ret


# pylint: disable=invalid-name
@tf_export(
    "linalg.matrix_transpose",
    v1=["linalg.transpose", "linalg.matrix_transpose", "matrix_transpose"])
@deprecation.deprecated_endpoints("matrix_transpose", "linalg.transpose")
def matrix_transpose(a, name="matrix_transpose", conjugate=False):
  """Transposes last two dimensions of tensor `a`.

  For example:

  ```python
  x = tf.constant([[1, 2, 3], [4, 5, 6]])
  tf.linalg.matrix_transpose(x)  # [[1, 4],
                                 #  [2, 5],
                                 #  [3, 6]]

  x = tf.constant([[1 + 1j, 2 + 2j, 3 + 3j],
                   [4 + 4j, 5 + 5j, 6 + 6j]])
  tf.linalg.matrix_transpose(x, conjugate=True)  # [[1 - 1j, 4 - 4j],
                                                 #  [2 - 2j, 5 - 5j],
                                                 #  [3 - 3j, 6 - 6j]]

  # Matrix with two batch dimensions.
  # x.shape is [1, 2, 3, 4]
  # tf.linalg.matrix_transpose(x) is shape [1, 2, 4, 3]
  ```

  Note that `tf.matmul` provides kwargs allowing for transpose of arguments.
  This is done with minimal cost, and is preferable to using this function. E.g.

  ```python
  # Good!  Transpose is taken at minimal additional cost.
  tf.matmul(matrix, b, transpose_b=True)

  # Inefficient!
  tf.matmul(matrix, tf.linalg.matrix_transpose(b))
  ```

  @compatibility(numpy)
  In `numpy` transposes are memory-efficient constant time operations as they
  simply return a new view of the same data with adjusted `strides`.

  TensorFlow does not support strides, `linalg.matrix_transpose` returns a new
  tensor with the items permuted.
  @end_compatibility

  Args:
    a: A `Tensor` with `rank >= 2`.
    name: A name for the operation (optional).
    conjugate: Optional bool. Setting it to `True` is mathematically equivalent
      to tf.math.conj(tf.linalg.matrix_transpose(input)).

  Returns:
    A transposed batch matrix `Tensor`.

  Raises:
    ValueError:  If `a` is determined statically to have `rank < 2`.
  """
  with ops.name_scope(name, values=[a]):
    a = ops.convert_to_tensor(a, name="a")

    # If we know the number of dimensions (statically), we can do two things:
    # 1. Check that `a` is a (batch) matrix.
    # 2. Use a python list for perm.  This preserves static shape information
    #    and avoids extra computations.
    a_shape = a.get_shape()
    ndims = a_shape.ndims
    if ndims is not None:
      if ndims < 2:
        raise ValueError(
            "Argument 'a' should be a (batch) matrix, with rank >= 2.  Found: "
            "%s" % a_shape)
      perm = list(range(ndims - 2)) + [ndims - 1] + [ndims - 2]
    else:
      a_rank = rank(a)
      perm = concat(
          (gen_math_ops._range(0, a_rank - 2, 1), [a_rank - 1, a_rank - 2]), 0)

    return transpose(a, perm=perm, conjugate=conjugate)


@tf_export("linalg.diag", v1=["linalg.diag", "matrix_diag"])
@deprecation.deprecated_endpoints("matrix_diag")
def matrix_diag(diagonal,
                name="diag",
                k=0,
                num_rows=-1,
                num_cols=-1,
                padding_value=0):
  """Returns a batched diagonal tensor with given batched diagonal values.

  Returns a tensor with the contents in `diagonal` as `k[0]`-th to `k[1]`-th
  diagonals of a matrix, with everything else padded with `padding`. `num_rows`
  and `num_cols` specify the dimension of the innermost matrix of the output. If
  both are not specified, the op assumes the innermost matrix is square and
  infers its size from `k` and the innermost dimension of `diagonal`. If only
  one of them is specified, the op assumes the unspecified value is the smallest
  possible based on other criteria.

  Let `diagonal` have `r` dimensions `[I, J, ..., L, M, N]`. The output tensor
  has rank `r+1` with shape `[I, J, ..., L, M, num_rows, num_cols]` when only
  one diagonal is given (`k` is an integer or `k[0] == k[1]`). Otherwise, it has
  rank `r` with shape `[I, J, ..., L, num_rows, num_cols]`.

  The second innermost dimension of `diagonal` has double meaning. When `k` is
  scalar or `k[0] == k[1]`, `M` is part of the batch size [I, J, ..., M], and
  the output tensor is:

  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, n-max(d_upper, 0)] ; if n - m == d_upper
      output[i, j, ..., l, m, n]                ; otherwise
  ```

  Otherwise, `M` is treated as the number of diagonals for the matrix in the
  same batch (`M = k[1]-k[0]+1`), and the output tensor is:

  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, k[1]-d, n-max(d, 0)] ; if d_lower <= d <= d_upper
      input[i, j, ..., l, m, n]                   ; otherwise
  ```
  where `d = n - m`

  For example:

  ```
  # The main diagonal.
  diagonal = np.array([[1, 2, 3, 4],            # Input shape: (2, 4)
                       [5, 6, 7, 8]])
  tf.matrix_diag(diagonal) ==> [[[1, 0, 0, 0],  # Output shape: (2, 4, 4)
                                 [0, 2, 0, 0],
                                 [0, 0, 3, 0],
                                 [0, 0, 0, 4]],
                                [[5, 0, 0, 0],
                                 [0, 6, 0, 0],
                                 [0, 0, 7, 0],
                                 [0, 0, 0, 8]]]

  # A superdiagonal (per batch).
  diagonal = np.array([[1, 2, 3],  # Input shape: (2, 3)
                       [4, 5, 6]])
  tf.matrix_diag(diagonal, k = 1)
    ==> [[[0, 1, 0, 0],  # Output shape: (2, 4, 4)
          [0, 0, 2, 0],
          [0, 0, 0, 3],
          [0, 0, 0, 0]],
         [[0, 4, 0, 0],
          [0, 0, 5, 0],
          [0, 0, 0, 6],
          [0, 0, 0, 0]]]

  # A band of diagonals.
  diagonals = np.array([[[1, 2, 3],  # Input shape: (2, 2, 3)
                         [4, 5, 0]],
                        [[6, 7, 9],
                         [9, 1, 0]]])
  tf.matrix_diag(diagonals, k = (-1, 0))
    ==> [[[1, 0, 0],  # Output shape: (2, 3, 3)
          [4, 2, 0],
          [0, 5, 3]],
         [[6, 0, 0],
          [9, 7, 0],
          [0, 1, 9]]]

  # Rectangular matrix.
  diagonal = np.array([1, 2])  # Input shape: (2)
  tf.matrix_diag(diagonal, k = -1, num_rows = 3, num_cols = 4)
    ==> [[0, 0, 0, 0],  # Output shape: (3, 4)
         [1, 0, 0, 0],
         [0, 2, 0, 0]]

  # Rectangular matrix with inferred num_cols and padding = 9.
  tf.matrix_diag(diagonal, k = -1, num_rows = 3, padding = 9)
    ==> [[9, 9],  # Output shape: (3, 2)
         [1, 9],
         [9, 2]]
  ```

  Args:
    diagonal: A `Tensor` with `rank k >= 1`.
    name: A name for the operation (optional).
    k: Diagonal offset(s). Positive value means superdiagonal, 0 refers to the
      main diagonal, and negative value means subdiagonals. `k` can be a single
      integer (for a single diagonal) or a pair of integers specifying the low
      and high ends of a matrix band. `k[0]` must not be larger than `k[1]`.
    num_rows: The number of rows of the output matrix. If it is not provided,
      the op assumes the output matrix is a square matrix and infers the matrix
      size from `d_lower`, `d_upper`, and the innermost dimension of `diagonal`.
    num_cols: The number of columns of the output matrix. If it is not provided,
      the op assumes the output matrix is a square matrix and infers the matrix
      size from `d_lower`, `d_upper`, and the innermost dimension of `diagonal`.
    padding_value: The value to fill the area outside the specified diagonal
      band with. Default is 0.

  Returns:
    A Tensor. Has the same type as `diagonal`.
  """
  # LINT.IfChange
  if compat.forward_compatible(2019, 8, 31):
    # LINT.ThenChange(//tensorflow/python/kernel_tests/diag_op_test.py)

    # Special case to sidestep the tf.constant conversion error:
    # TypeError: Expected bool, got 0 of type 'int' instead.
    if hasattr(diagonal, "dtype") and diagonal.dtype == "bool":
      padding_value = bool(padding_value)
    return gen_array_ops.matrix_diag_v2(
        diagonal=diagonal,
        k=k,
        num_rows=num_rows,
        num_cols=num_cols,
        padding_value=padding_value,
        name=name)

  # Call v1 to maintain forward compatibility.
  return gen_array_ops.matrix_diag(diagonal=diagonal, name=name)


@tf_export("linalg.diag_part", v1=["linalg.diag_part", "matrix_diag_part"])
@deprecation.deprecated_endpoints("matrix_diag_part")
@dispatch.add_dispatch_support
def matrix_diag_part(
    input,  # pylint:disable=redefined-builtin
    name="diag_part",
    k=0,
    padding_value=0):
  """Returns the batched diagonal part of a batched tensor.

  Returns a tensor with the `k[0]`-th to `k[1]`-th diagonals of the batched
  `input`.

  Assume `input` has `r` dimensions `[I, J, ..., L, M, N]`.
  Let `max_diag_len` be the maximum length among all diagonals to be extracted,
  `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
  Let `num_diags` be the number of diagonals to extract,
  `num_diags = k[1] - k[0] + 1`.

  If `num_diags == 1`, the output tensor is of rank `r - 1` with shape
  `[I, J, ..., L, max_diag_len]` and values:

  ```
  diagonal[i, j, ..., l, n]
    = input[i, j, ..., l, n+y, n+x] ; when 0 <= n-y < M and 0 <= n-x < N,
      0                             ; otherwise.
  ```
  where `y = max(-k[1], 0)`, `x = max(k[1], 0)`.

  Otherwise, the output tensor has rank `r` with dimensions
  `[I, J, ..., L, num_diags, max_diag_len]` with values:

  ```
  diagonal[i, j, ..., l, m, n]
    = input[i, j, ..., l, n+y, n+x] ; when 0 <= n-y < M and 0 <= n-x < N,
      0                             ; otherwise.
  ```
  where `d = k[1] - m`, `y = max(-d, 0)`, and `x = max(d, 0)`.

  The input must be at least a matrix.

  For example:

  ```
  input = np.array([[[1, 2, 3, 4],  # Input shape: (2, 3, 4)
                     [5, 6, 7, 8],
                     [9, 8, 7, 6]],
                    [[5, 4, 3, 2],
                     [1, 2, 3, 4],
                     [5, 6, 7, 8]]])

  # A main diagonal from each batch.
  tf.matrix_diag_part(input) ==> [[1, 6, 7],  # Output shape: (2, 3)
                                  [5, 2, 7]]

  # A superdiagonal from each batch.
  tf.matrix_diag_part(input, k = 1)
    ==> [[2, 7, 6],  # Output shape: (2, 3)
         [4, 3, 8]]

  # A tridiagonal band from each batch.
  tf.matrix_diag_part(input, k = (-1, 1))
    ==> [[[2, 7, 6],  # Output shape: (2, 3, 3)
          [1, 6, 7],
          [5, 8, 0]],
         [[4, 3, 8],
          [5, 2, 7],
          [1, 6, 0]]]

  # Padding = 9
  tf.matrix_diag_part(input, k = (1, 3), padding = 9)
    ==> [[[4, 9, 9],  # Output shape: (2, 3, 3)
          [3, 8, 9],
          [2, 7, 6]],
         [[2, 9, 9],
          [3, 4, 9],
          [4, 3, 8]]]
  ```

  Args:
    input: A `Tensor` with `rank k >= 2`.
    name: A name for the operation (optional).
    k: Diagonal offset(s). Positive value means superdiagonal, 0 refers to the
      main diagonal, and negative value means subdiagonals. `k` can be a single
      integer (for a single diagonal) or a pair of integers specifying the low
      and high ends of a matrix band. `k[0]` must not be larger than `k[1]`.
    padding_value: The value to fill the area outside the specified diagonal
      band with. Default is 0.

  Returns:
    A Tensor containing diagonals of `input`. Has the same type as `input`.
  """
  # LINT.IfChange
  if compat.forward_compatible(2019, 8, 31):
    # LINT.ThenChange(//tensorflow/python/kernel_tests/diag_op_test.py)

    # Special case to sidestep the tf.constant conversion error:
    # TypeError: Expected bool, got 0 of type 'int' instead.
    if hasattr(input, "dtype") and input.dtype == "bool":
      padding_value = bool(padding_value)

    return gen_array_ops.matrix_diag_part_v2(
        input=input, k=k, padding_value=padding_value, name=name)

  # Call v1 to maintain forward compatibility.
  return gen_array_ops.matrix_diag_part(input=input, name=name)


@tf_export("linalg.set_diag", v1=["linalg.set_diag", "matrix_set_diag"])
@deprecation.deprecated_endpoints("matrix_set_diag")
def matrix_set_diag(
    input,  # pylint:disable=redefined-builtin
    diagonal,
    name="set_diag",
    k=0):
  """Returns a batched matrix tensor with new batched diagonal values.

  Given `input` and `diagonal`, this operation returns a tensor with the
  same shape and values as `input`, except for the specified diagonals of the
  innermost matrices. These will be overwritten by the values in `diagonal`.

  `input` has `r+1` dimensions `[I, J, ..., L, M, N]`. When `k` is scalar or
  `k[0] == k[1]`, `diagonal` has `r` dimensions `[I, J, ..., L, max_diag_len]`.
  Otherwise, it has `r+1` dimensions `[I, J, ..., L, num_diags, max_diag_len]`.
  `num_diags` is the number of diagonals, `num_diags = k[1] - k[0] + 1`.
  `max_diag_len` is the longest diagonal in the range `[k[0], k[1]]`,
  `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`

  The output is a tensor of rank `k+1` with dimensions `[I, J, ..., L, M, N]`.
  If `k` is scalar or `k[0] == k[1]`:

  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, n-max(k[1], 0)] ; if n - m == k[1]
      output[i, j, ..., l, m, n]             ; otherwise
  ```

  Otherwise,

  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, k[1]-d, n-max(d, 0)] ; if d_lower <= d <= d_upper
      input[i, j, ..., l, m, n]                   ; otherwise
  ```
  where `d = n - m`

  For example:

  ```
  # The main diagonal.
  input = np.array([[[7, 7, 7, 7],              # Input shape: (2, 3, 4)
                     [7, 7, 7, 7],
                     [7, 7, 7, 7]],
                    [[7, 7, 7, 7],
                     [7, 7, 7, 7],
                     [7, 7, 7, 7]]])
  diagonal = np.array([[1, 2, 3],               # Diagonal shape: (2, 3)
                       [4, 5, 6]])
  tf.matrix_diag(diagonal) ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
                                 [7, 2, 7, 7],
                                 [7, 7, 3, 7]],
                                [[4, 7, 7, 7],
                                 [7, 5, 7, 7],
                                 [7, 7, 6, 7]]]

  # A superdiagonal (per batch).
  tf.matrix_diag(diagonal, k = 1)
    ==> [[[7, 1, 7, 7],  # Output shape: (2, 3, 4)
          [7, 7, 2, 7],
          [7, 7, 7, 3]],
         [[7, 4, 7, 7],
          [7, 7, 5, 7],
          [7, 7, 7, 6]]]

  # A band of diagonals.
  diagonals = np.array([[[1, 2, 3],  # Diagonal shape: (2, 2, 3)
                         [4, 5, 0]],
                        [[6, 1, 2],
                         [3, 4, 0]]])
  tf.matrix_diag(diagonals, k = (-1, 0))
    ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
          [4, 2, 7, 7],
          [0, 5, 3, 7]],
         [[6, 7, 7, 7],
          [3, 1, 7, 7],
          [7, 4, 2, 7]]]

  ```

  Args:
    input: A `Tensor` with rank `k + 1`, where `k >= 1`.
    diagonal:  A `Tensor` with rank `k`, when `d_lower == d_upper`, or `k + 1`,
      otherwise. `k >= 1`.
    name: A name for the operation (optional).
    k: Diagonal offset(s). Positive value means superdiagonal, 0 refers to the
      main diagonal, and negative value means subdiagonals. `k` can be a single
      integer (for a single diagonal) or a pair of integers specifying the low
      and high ends of a matrix band. `k[0]` must not be larger than `k[1]`.
  """
  # LINT.IfChange
  if compat.forward_compatible(2019, 8, 31):
    # LINT.ThenChange(//tensorflow/python/kernel_tests/diag_op_test.py)
    return gen_array_ops.matrix_set_diag_v2(
        input=input, diagonal=diagonal, k=k, name=name)

  # Call v1 to maintain forward compatibility.
  return gen_array_ops.matrix_set_diag(
      input=input, diagonal=diagonal, name=name)

# pylint: enable=invalid-name


def _constant_if_small(value, shape, dtype, name):
  try:
    if np.prod(shape) < 1000:
      return constant(value, shape=shape, dtype=dtype, name=name)
  except TypeError:
    # Happens when shape is a Tensor, list with Tensor elements, etc.
    pass
  return None


@tf_export("zeros")
def zeros(shape, dtype=dtypes.float32, name=None):
  """Creates a tensor with all elements set to zero.

  This operation returns a tensor of type `dtype` with shape `shape` and
  all elements set to zero.

  For example:

  ```python
  tf.zeros([3, 4], tf.int32)  # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
  ```

  Args:
    shape: A list of integers, a tuple of integers, or a 1-D `Tensor` of type
      `int32`.
    dtype: The type of an element in the resulting `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to zero.
  """
  dtype = dtypes.as_dtype(dtype).base_dtype
  with ops.name_scope(name, "zeros", [shape]) as name:
    if dtype == dtypes.bool:
      zero = False
    elif dtype == dtypes.string:
      zero = ""
    else:
      zero = 0

    if not isinstance(shape, ops.Tensor):
      try:
        # Create a constant if it won't be very big. Otherwise create a fill op
        # to prevent serialized GraphDefs from becoming too large.
        output = _constant_if_small(zero, shape, dtype, name)
        if output is not None:
          return output

        # Go through tensor shapes to get int64-if-needed semantics
        shape = constant_op._tensor_shape_tensor_conversion_function(
            tensor_shape.TensorShape(shape))
      except (TypeError, ValueError):
        # Happens when shape is a list with tensor elements
        shape = ops.convert_to_tensor(shape, dtype=dtypes.int32)
    if not shape._shape_tuple():
      shape = reshape(shape, [-1])  # Ensure it's a vector
    output = fill(shape, constant(zero, dtype=dtype), name=name)
  assert output.dtype.base_dtype == dtype
  return output


@tf_export(v1=["zeros_like"])
@dispatch.add_dispatch_support
def zeros_like(tensor, dtype=None, name=None, optimize=True):
  """Creates a tensor with all elements set to zero.

  Given a single tensor (`tensor`), this operation returns a tensor of the
  same type and shape as `tensor` with all elements set to zero. Optionally,
  you can use `dtype` to specify a new type for the returned tensor.

  For example:

  ```python
  tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
  tf.zeros_like(tensor)  # [[0, 0, 0], [0, 0, 0]]
  ```

  Args:
    tensor: A `Tensor`.
    dtype: A type for the returned `Tensor`. Must be `float16`, `float32`,
      `float64`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`,
      `complex64`, `complex128`, `bool` or `string`.
    name: A name for the operation (optional).
    optimize: if true, attempt to statically determine the shape of 'tensor' and
      encode it as a constant.

  Returns:
    A `Tensor` with all elements set to zero.
  """
  return zeros_like_impl(tensor, dtype, name, optimize)


@tf_export("zeros_like", v1=[])
@dispatch.add_dispatch_support
def zeros_like_v2(
    input,  # pylint: disable=redefined-builtin
    dtype=None,
    name=None):
  """Creates a tensor with all elements set to zero.

  Given a single tensor (`tensor`), this operation returns a tensor of the
  same type and shape as `tensor` with all elements set to zero. Optionally,
  you can use `dtype` to specify a new type for the returned tensor.

  For example:

  ```python
  tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
  tf.zeros_like(tensor)  # [[0, 0, 0], [0, 0, 0]] with dtype=int32

  If dtype of input `tensor` is `float32`, then the output is also of `float32`
  tensor = tf.constant([[1.0, 2.0, 3.0], [4, 5, 6]])
  tf.zeros_like(tensor)  # [[0., 0., 0.], [0., 0., 0.]] with dtype=floa32

  If you want to specify desired dtype of output `tensor`, then specify it in
  the op tensor = tf.constant([[1.0, 2.0, 3.0], [4, 5, 6]])
  tf.zeros_like(tensor,dtype=tf.int32)  # [[0, 0, 0], [0, 0, 0]] with
  dtype=int32
  ```

  Args:
    input: A `Tensor`.
    dtype: A type for the returned `Tensor`. Must be `float16`, `float32`,
      `float64`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`,
      `complex64`, `complex128`, `bool` or `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to zero.
  """
  return zeros_like_impl(input, dtype, name, optimize=True)


def zeros_like_impl(tensor, dtype, name, optimize=True):
  """Internal implementation for the v1/v2 zeros_like API calls."""
  with ops.name_scope(name, "zeros_like", [tensor]) as name:
    tensor = ops.convert_to_tensor(tensor, name="tensor")

    if context.executing_eagerly():
      if dtype is not None and dtype != tensor.dtype:
        return zeros(
            shape_internal(tensor, optimize=optimize), dtype=dtype, name=name)
      with ops.device(tensor.device):
        return gen_array_ops.zeros_like(tensor, name=name)

    # For now, variant types must be created via zeros_like; as we need to
    # pass the input variant object to the proper zeros callback.

    if (optimize and tensor.shape.is_fully_defined() and
        tensor.dtype != dtypes.variant):
      # We can produce a zeros tensor independent of the value of 'tensor',
      # since the shape is known statically.
      return zeros(tensor.shape, dtype=dtype or tensor.dtype, name=name)

    if dtype is not None and dtype != tensor.dtype and dtype != dtypes.variant:
      return zeros(
          shape_internal(tensor, optimize=optimize), dtype=dtype, name=name)
    else:
      return gen_array_ops.zeros_like(tensor, name=name)


@tf_export(v1=["ones_like"])
@dispatch.add_dispatch_support
def ones_like(tensor, dtype=None, name=None, optimize=True):
  """Creates a tensor with all elements set to 1.

  Given a single tensor (`tensor`), this operation returns a tensor of the same
  type and shape as `tensor` with all elements set to 1. Optionally, you can
  specify a new type (`dtype`) for the returned tensor.

  For example:

  ```python
  tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
  tf.ones_like(tensor)  # [[1, 1, 1], [1, 1, 1]]
  ```

  Args:
    tensor: A `Tensor`.
    dtype: A type for the returned `Tensor`. Must be `float32`, `float64`,
      `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`, `complex64`,
      `complex128` or `bool`.
    name: A name for the operation (optional).
    optimize: if true, attempt to statically determine the shape of 'tensor' and
      encode it as a constant.

  Returns:
    A `Tensor` with all elements set to 1.
  """
  return ones_like_impl(tensor, dtype, name, optimize)


@tf_export("ones_like", v1=[])
@dispatch.add_dispatch_support
def ones_like_v2(
    input,  # pylint: disable=redefined-builtin
    dtype=None,
    name=None):
  """Creates a tensor with all elements set to one.

  Given a single tensor (`tensor`), this operation returns a tensor of the
  same type and shape as `tensor` with all elements set to 1. Optionally,
  you can use `dtype` to specify a new type for the returned tensor.

  For example:

  ```python
  tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
  tf.ones_like(tensor)  # [[1, 1, 1], [1, 1, 1]]
  ```

  Args:
    input: A `Tensor`.
    dtype: A type for the returned `Tensor`. Must be `float16`, `float32`,
      `float64`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`,
      `complex64`, `complex128`, `bool` or `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to one.
  """
  return ones_like_impl(input, dtype, name, optimize=True)


def ones_like_impl(tensor, dtype, name, optimize=True):
  """Internal implementation for the v1/v2 ones_like API calls."""
  with ops.name_scope(name, "ones_like", [tensor]) as name:
    tensor = ops.convert_to_tensor(tensor, name="tensor")
    ones_shape = shape_internal(tensor, optimize=optimize)
    if dtype is None:
      dtype = tensor.dtype
    ret = ones(ones_shape, dtype=dtype, name=name)
    if not context.executing_eagerly():
      ret.set_shape(tensor.get_shape())
    return ret


@tf_export("ones")
def ones(shape, dtype=dtypes.float32, name=None):
  """Creates a tensor with all elements set to 1.

  This operation returns a tensor of type `dtype` with shape `shape` and all
  elements set to 1.

  For example:

  ```python
  tf.ones([2, 3], tf.int32)  # [[1, 1, 1], [1, 1, 1]]
  ```

  Args:
    shape: A list of integers, a tuple of integers, or a 1-D `Tensor` of type
      `int32`.
    dtype: The type of an element in the resulting `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to 1.
  """
  dtype = dtypes.as_dtype(dtype).base_dtype
  with ops.name_scope(name, "ones", [shape]) as name:
    one = True if dtype == dtypes.bool else 1
    if not isinstance(shape, ops.Tensor):
      try:
        # Create a constant if it won't be very big. Otherwise create a fill op
        # to prevent serialized GraphDefs from becoming too large.
        output = _constant_if_small(one, shape, dtype, name)
        if output is not None:
          return output

        # Go through tensor shapes to get int64-if-needed semantics
        shape = constant_op._tensor_shape_tensor_conversion_function(
            tensor_shape.TensorShape(shape))
      except (TypeError, ValueError):
        # Happens when shape is a list with tensor elements
        shape = ops.convert_to_tensor(shape, dtype=dtypes.int32)
    if not shape._shape_tuple():
      shape = reshape(shape, [-1])  # Ensure it's a vector
    output = fill(shape, constant(one, dtype=dtype), name=name)
  assert output.dtype.base_dtype == dtype
  return output


@tf_export(v1=["placeholder"])
def placeholder(dtype, shape=None, name=None):
  """Inserts a placeholder for a tensor that will be always fed.

  **Important**: This tensor will produce an error if evaluated. Its value must
  be fed using the `feed_dict` optional argument to `Session.run()`,
  `Tensor.eval()`, or `Operation.run()`.

  For example:

  ```python
  x = tf.compat.v1.placeholder(tf.float32, shape=(1024, 1024))
  y = tf.matmul(x, x)

  with tf.compat.v1.Session() as sess:
    print(sess.run(y))  # ERROR: will fail because x was not fed.

    rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
  ```

  @compatibility(eager)
  Placeholders are not compatible with eager execution.
  @end_compatibility

  Args:
    dtype: The type of elements in the tensor to be fed.
    shape: The shape of the tensor to be fed (optional). If the shape is not
      specified, you can feed a tensor of any shape.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` that may be used as a handle for feeding a value, but not
    evaluated directly.

  Raises:
    RuntimeError: if eager execution is enabled
  """
  if context.executing_eagerly():
    raise RuntimeError("tf.placeholder() is not compatible with "
                       "eager execution.")

  return gen_array_ops.placeholder(dtype=dtype, shape=shape, name=name)


@tf_export(v1=["placeholder_with_default"])
def placeholder_with_default(input, shape, name=None):  # pylint: disable=redefined-builtin
  """A placeholder op that passes through `input` when its output is not fed.

  Args:
    input: A `Tensor`. The default value to produce when output is not fed.
    shape: A `tf.TensorShape` or list of `int`s. The (possibly partial) shape of
      the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  return gen_array_ops.placeholder_with_default(input, shape, name)


# pylint: disable=redefined-outer-name
def _normalize_sparse_shape(shape, name):
  """Returns a tuple of (Tensor or None, rank or None)."""
  if shape is None:
    return (None, None)
  rank = shape.get_shape()[0] if isinstance(shape, ops.Tensor) else len(shape)
  if not isinstance(shape, ops.Tensor) and None in shape:
    return (None, rank)
  return (ops.convert_to_tensor(shape, dtype=dtypes.int64, name=name), rank)


@tf_export(v1=["sparse.placeholder", "sparse_placeholder"])
@deprecation.deprecated_endpoints("sparse_placeholder")
def sparse_placeholder(dtype, shape=None, name=None):
  """Inserts a placeholder for a sparse tensor that will be always fed.

  **Important**: This sparse tensor will produce an error if evaluated.
  Its value must be fed using the `feed_dict` optional argument to
  `Session.run()`, `Tensor.eval()`, or `Operation.run()`.

  For example:

  ```python
  x = tf.compat.v1.sparse.placeholder(tf.float32)
  y = tf.sparse.reduce_sum(x)

  with tf.compat.v1.Session() as sess:
    print(sess.run(y))  # ERROR: will fail because x was not fed.

    indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float32)
    shape = np.array([7, 9, 2], dtype=np.int64)
    print(sess.run(y, feed_dict={
      x: tf.compat.v1.SparseTensorValue(indices, values, shape)}))  # Will
      succeed.
    print(sess.run(y, feed_dict={
      x: (indices, values, shape)}))  # Will succeed.

    sp = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    sp_value = sp.eval(session=sess)
    print(sess.run(y, feed_dict={x: sp_value}))  # Will succeed.
  ```

  @compatibility{eager} Placeholders are not compatible with eager execution.

  Args:
    dtype: The type of `values` elements in the tensor to be fed.
    shape: The shape of the tensor to be fed (optional). If the shape is not
      specified, you can feed a sparse tensor of any shape.
    name: A name for prefixing the operations (optional).

  Returns:
    A `SparseTensor` that may be used as a handle for feeding a value, but not
    evaluated directly.

  Raises:
    RuntimeError: if eager execution is enabled
  """
  if context.executing_eagerly():
    raise RuntimeError("tf.placeholder() is not compatible with "
                       "eager execution.")

  shape_name = (name + "/shape") if name is not None else None
  shape, rank = _normalize_sparse_shape(shape, shape_name)
  if shape is None:
    shape = placeholder(dtypes.int64, shape=[rank], name=shape_name)
  return sparse_tensor.SparseTensor(
      values=placeholder(
          dtype,
          shape=[None],
          name=(name + "/values") if name is not None else None),
      indices=placeholder(
          dtypes.int64,
          shape=[None, rank],
          name=(name + "/indices") if name is not None else None),
      dense_shape=shape)


# pylint: enable=redefined-outer-name


@tf_export("pad", v1=[])
def pad_v2(tensor, paddings, mode="CONSTANT", constant_values=0, name=None):
  """Pads a tensor.

  This operation pads a `tensor` according to the `paddings` you specify.
  `paddings` is an integer tensor with shape `[n, 2]`, where n is the rank of
  `tensor`. For each dimension D of `input`, `paddings[D, 0]` indicates how
  many values to add before the contents of `tensor` in that dimension, and
  `paddings[D, 1]` indicates how many values to add after the contents of
  `tensor` in that dimension. If `mode` is "REFLECT" then both `paddings[D, 0]`
  and `paddings[D, 1]` must be no greater than `tensor.dim_size(D) - 1`. If
  `mode` is "SYMMETRIC" then both `paddings[D, 0]` and `paddings[D, 1]` must be
  no greater than `tensor.dim_size(D)`.

  The padded size of each dimension D of the output is:

  `paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1]`

  For example:

  ```python
  t = tf.constant([[1, 2, 3], [4, 5, 6]])
  paddings = tf.constant([[1, 1,], [2, 2]])
  # 'constant_values' is 0.
  # rank of 't' is 2.
  tf.pad(t, paddings, "CONSTANT")  # [[0, 0, 0, 0, 0, 0, 0],
                                   #  [0, 0, 1, 2, 3, 0, 0],
                                   #  [0, 0, 4, 5, 6, 0, 0],
                                   #  [0, 0, 0, 0, 0, 0, 0]]

  tf.pad(t, paddings, "REFLECT")  # [[6, 5, 4, 5, 6, 5, 4],
                                  #  [3, 2, 1, 2, 3, 2, 1],
                                  #  [6, 5, 4, 5, 6, 5, 4],
                                  #  [3, 2, 1, 2, 3, 2, 1]]

  tf.pad(t, paddings, "SYMMETRIC")  # [[2, 1, 1, 2, 3, 3, 2],
                                    #  [2, 1, 1, 2, 3, 3, 2],
                                    #  [5, 4, 4, 5, 6, 6, 5],
                                    #  [5, 4, 4, 5, 6, 6, 5]]
  ```

  Args:
    tensor: A `Tensor`.
    paddings: A `Tensor` of type `int32`.
    mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
    constant_values: In "CONSTANT" mode, the scalar pad value to use. Must be
      same type as `tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.

  Raises:
    ValueError: When mode is not one of "CONSTANT", "REFLECT", or "SYMMETRIC".
  """
  return pad(tensor, paddings, mode, name, constant_values)


@tf_export(v1=["pad"])
def pad(tensor, paddings, mode="CONSTANT", name=None, constant_values=0):  # pylint: disable=invalid-name
  """Pads a tensor.

  This operation pads a `tensor` according to the `paddings` you specify.
  `paddings` is an integer tensor with shape `[n, 2]`, where n is the rank of
  `tensor`. For each dimension D of `input`, `paddings[D, 0]` indicates how
  many values to add before the contents of `tensor` in that dimension, and
  `paddings[D, 1]` indicates how many values to add after the contents of
  `tensor` in that dimension. If `mode` is "REFLECT" then both `paddings[D, 0]`
  and `paddings[D, 1]` must be no greater than `tensor.dim_size(D) - 1`. If
  `mode` is "SYMMETRIC" then both `paddings[D, 0]` and `paddings[D, 1]` must be
  no greater than `tensor.dim_size(D)`.

  The padded size of each dimension D of the output is:

  `paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1]`

  For example:

  ```python
  t = tf.constant([[1, 2, 3], [4, 5, 6]])
  paddings = tf.constant([[1, 1,], [2, 2]])
  # 'constant_values' is 0.
  # rank of 't' is 2.
  tf.pad(t, paddings, "CONSTANT")  # [[0, 0, 0, 0, 0, 0, 0],
                                   #  [0, 0, 1, 2, 3, 0, 0],
                                   #  [0, 0, 4, 5, 6, 0, 0],
                                   #  [0, 0, 0, 0, 0, 0, 0]]

  tf.pad(t, paddings, "REFLECT")  # [[6, 5, 4, 5, 6, 5, 4],
                                  #  [3, 2, 1, 2, 3, 2, 1],
                                  #  [6, 5, 4, 5, 6, 5, 4],
                                  #  [3, 2, 1, 2, 3, 2, 1]]

  tf.pad(t, paddings, "SYMMETRIC")  # [[2, 1, 1, 2, 3, 3, 2],
                                    #  [2, 1, 1, 2, 3, 3, 2],
                                    #  [5, 4, 4, 5, 6, 6, 5],
                                    #  [5, 4, 4, 5, 6, 6, 5]]
  ```

  Args:
    tensor: A `Tensor`.
    paddings: A `Tensor` of type `int32`.
    mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
    name: A name for the operation (optional).
    constant_values: In "CONSTANT" mode, the scalar pad value to use. Must be
      same type as `tensor`.

  Returns:
    A `Tensor`. Has the same type as `tensor`.

  Raises:
    ValueError: When mode is not one of "CONSTANT", "REFLECT", or "SYMMETRIC".
  """

  # Convert lower/mixed case to upper for NumPy compatibility
  # NumPy uses all lower-case modes.
  mode = mode.upper()
  if mode == "CONSTANT":
    # TODO(rjryan): Once the forward compatibility period (3 weeks) have passed
    # remove the "Pad" fallback here.
    if not tensor_util.is_tensor(constant_values) and constant_values == 0:
      result = gen_array_ops.pad(tensor, paddings, name=name)
    else:
      result = gen_array_ops.pad_v2(
          tensor, paddings, constant_values, name=name)
  elif mode == "REFLECT":
    result = gen_array_ops.mirror_pad(
        tensor, paddings, mode="REFLECT", name=name)
  elif mode == "SYMMETRIC":
    result = gen_array_ops.mirror_pad(
        tensor, paddings, mode="SYMMETRIC", name=name)
  else:
    raise ValueError("Unknown padding mode: %s" % mode)

  # Restore shape information where possible.
  if not context.executing_eagerly():
    paddings_constant = tensor_util.constant_value(
        result.op.inputs[1], partial=True)
    input_shape = result.op.inputs[0].shape
    if (input_shape.ndims is not None and
        not result.shape.is_fully_defined() and paddings_constant is not None):
      new_shape = []
      for padding, dim in zip(paddings_constant, input_shape.as_list()):
        if padding is None or dim is None or any((x is None for x in padding)):
          new_shape.append(None)
        else:
          new_shape.append(sum(padding) + dim)
      result.set_shape(new_shape)

  return result


@tf_export("meshgrid")
def meshgrid(*args, **kwargs):
  """Broadcasts parameters for evaluation on an N-D grid.

  Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
  of N-D coordinate arrays for evaluating expressions on an N-D grid.

  Notes:

  `meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
  When the `indexing` argument is set to 'xy' (the default), the broadcasting
  instructions for the first two dimensions are swapped.

  Examples:

  Calling `X, Y = meshgrid(x, y)` with the tensors

  ```python
  x = [1, 2, 3]
  y = [4, 5, 6]
  X, Y = tf.meshgrid(x, y)
  # X = [[1, 2, 3],
  #      [1, 2, 3],
  #      [1, 2, 3]]
  # Y = [[4, 4, 4],
  #      [5, 5, 5],
  #      [6, 6, 6]]
  ```

  Args:
    *args: `Tensor`s with rank 1.
    **kwargs:
      - indexing: Either 'xy' or 'ij' (optional, default: 'xy').
      - name: A name for the operation (optional).

  Returns:
    outputs: A list of N `Tensor`s with rank N.

  Raises:
    TypeError: When no keyword arguments (kwargs) are passed.
    ValueError: When indexing keyword argument is not one of `xy` or `ij`.
  """

  indexing = kwargs.pop("indexing", "xy")
  name = kwargs.pop("name", "meshgrid")
  if kwargs:
    key = list(kwargs.keys())[0]
    raise TypeError("'{}' is an invalid keyword argument "
                    "for this function".format(key))

  if indexing not in ("xy", "ij"):
    raise ValueError("indexing parameter must be either 'xy' or 'ij'")

  with ops.name_scope(name, "meshgrid", args) as name:
    ndim = len(args)
    s0 = (1,) * ndim

    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
      output.append(reshape(stack(x), (s0[:i] + (-1,) + s0[i + 1::])))
    # Create parameters for broadcasting each tensor to the full size
    shapes = [size(x) for x in args]

    output_dtype = ops.convert_to_tensor(args[0]).dtype.base_dtype

    if indexing == "xy" and ndim > 1:
      output[0] = reshape(output[0], (1, -1) + (1,) * (ndim - 2))
      output[1] = reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
      shapes[0], shapes[1] = shapes[1], shapes[0]

    # TODO(nolivia): improve performance with a broadcast
    mult_fact = ones(shapes, output_dtype)
    return [x * mult_fact for x in output]


NEW_AXIS = -1
SHRINK_AXIS = -2


# PEP-8 naming
# pylint: disable=invalid-name,redefined-outer-name
def _compute_size_of_strided_dim(shrink, spec, size):
  """Computes the size of a single strided slice dimension."""

  unknown = None  # Document what None means here.
  use_full_range = None  # Document other use of None.
  # if this is a shrink axis (i.e. a non-range index)
  # it either will produce an error or return 1
  if shrink:
    return 1
  if size is unknown or size.value is unknown:
    return unknown
  size = size.value
  stride = spec.step
  if stride is not unknown:
    if stride == 0:
      return unknown
    stride = spec.step
    valid_range = [0, size] if stride > 0 else [-1, size - 1]

    # PEP-8 naming
    # pylint: disable=invalid-name
    def canonical(x, c):
      if x is use_full_range:
        return valid_range[c] if stride > 0 else valid_range[(c + 1) & 1]
      else:
        x_fwd = size + x if x < 0 else x  # make negative indices positive
        return max(valid_range[0], min(valid_range[1], x_fwd))

    begin = canonical(spec.start, 0)
    end = canonical(spec.stop, 1)
    interval_length = end - begin
    if interval_length == 0 or ((interval_length < 0) != (stride < 0)):
      return 0
    else:
      remainder = 1 if interval_length % stride != 0 else 0
      return interval_length // stride + remainder
  else:
    return unknown  # unknown because stride is unknown


def _TileGradShape(op):
  """Shape function for the TileGrad op."""
  multiples_shape = op.inputs[1].get_shape().with_rank(1)
  input_shape = op.inputs[0].get_shape().with_rank(multiples_shape[0])
  # NOTE(mrry): Represent `multiples` as a `TensorShape` because (i)
  # it is a vector of non-negative integers, and (ii) doing so allows
  # us to handle partially-known multiples.
  multiples = tensor_util.constant_value_as_shape(op.inputs[1]).with_rank(
      input_shape.ndims)
  if multiples.ndims is None:
    return [tensor_shape.unknown_shape()]
  else:
    output_dims = []
    for dim, multiple in zip(input_shape.dims, multiples.dims):
      output_dims.append(dim // multiple)
    return [tensor_shape.TensorShape(output_dims)]


@tf_export("edit_distance")
def edit_distance(hypothesis, truth, normalize=True, name="edit_distance"):
  """Computes the Levenshtein distance between sequences.

  This operation takes variable-length sequences (`hypothesis` and `truth`),
  each provided as a `SparseTensor`, and computes the Levenshtein distance.
  You can normalize the edit distance by length of `truth` by setting
  `normalize` to true.

  For example, given the following input:

  ```python
  # 'hypothesis' is a tensor of shape `[2, 1]` with variable-length values:
  #   (0,0) = ["a"]
  #   (1,0) = ["b"]
  hypothesis = tf.SparseTensor(
      [[0, 0, 0],
       [1, 0, 0]],
      ["a", "b"],
      (2, 1, 1))

  # 'truth' is a tensor of shape `[2, 2]` with variable-length values:
  #   (0,0) = []
  #   (0,1) = ["a"]
  #   (1,0) = ["b", "c"]
  #   (1,1) = ["a"]
  truth = tf.SparseTensor(
      [[0, 1, 0],
       [1, 0, 0],
       [1, 0, 1],
       [1, 1, 0]],
      ["a", "b", "c", "a"],
      (2, 2, 2))

  normalize = True
  ```

  This operation would return the following:

  ```python
  # 'output' is a tensor of shape `[2, 2]` with edit distances normalized
  # by 'truth' lengths.
  output ==> [[inf, 1.0],  # (0,0): no truth, (0,1): no hypothesis
             [0.5, 1.0]]  # (1,0): addition, (1,1): no hypothesis
  ```

  Args:
    hypothesis: A `SparseTensor` containing hypothesis sequences.
    truth: A `SparseTensor` containing truth sequences.
    normalize: A `bool`. If `True`, normalizes the Levenshtein distance by
      length of `truth.`
    name: A name for the operation (optional).

  Returns:
    A dense `Tensor` with rank `R - 1`, where R is the rank of the
    `SparseTensor` inputs `hypothesis` and `truth`.

  Raises:
    TypeError: If either `hypothesis` or `truth` are not a `SparseTensor`.
  """
  if not isinstance(
      hypothesis,
      (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
    raise TypeError("Hypothesis must be a SparseTensor.")
  if not isinstance(
      truth, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
    raise TypeError("Truth must be a SparseTensor.")

  return gen_array_ops.edit_distance(
      hypothesis.indices,
      hypothesis.values,
      hypothesis.dense_shape,
      truth.indices,
      truth.values,
      truth.dense_shape,
      normalize=normalize,
      name=name)


@ops.RegisterGradient("FakeQuantWithMinMaxArgs")
def _FakeQuantWithMinMaxArgsGradient(op, grad):
  """Gradient for FakeQuantWithMinMaxArgs op."""
  return fake_quant_with_min_max_args_gradient(
      grad,
      op.inputs[0],
      min=op.get_attr("min"),
      max=op.get_attr("max"),
      num_bits=op.get_attr("num_bits"),
      narrow_range=op.get_attr("narrow_range"))


@ops.RegisterGradient("FakeQuantWithMinMaxVars")
def _FakeQuantWithMinMaxVarsGradient(op, grad):
  """Gradient for FakeQuantWithMinMaxVars op."""
  return fake_quant_with_min_max_vars_gradient(
      grad,
      op.inputs[0],
      op.inputs[1],
      op.inputs[2],
      num_bits=op.get_attr("num_bits"),
      narrow_range=op.get_attr("narrow_range"))


@ops.RegisterGradient("FakeQuantWithMinMaxVarsPerChannel")
def _FakeQuantWithMinMaxVarsPerChannelGradient(op, grad):
  """Gradient for FakeQuantWithMinMaxVarsPerChannel op."""
  return fake_quant_with_min_max_vars_per_channel_gradient(
      grad,
      op.inputs[0],
      op.inputs[1],
      op.inputs[2],
      num_bits=op.get_attr("num_bits"),
      narrow_range=op.get_attr("narrow_range"))


@tf_export("required_space_to_batch_paddings")
def required_space_to_batch_paddings(input_shape,
                                     block_shape,
                                     base_paddings=None,
                                     name=None):
  """Calculate padding required to make block_shape divide input_shape.

  This function can be used to calculate a suitable paddings argument for use
  with space_to_batch_nd and batch_to_space_nd.

  Args:
    input_shape: int32 Tensor of shape [N].
    block_shape: int32 Tensor of shape [N].
    base_paddings: Optional int32 Tensor of shape [N, 2].  Specifies the minimum
      amount of padding to use.  All elements must be >= 0.  If not specified,
      defaults to 0.
    name: string.  Optional name prefix.

  Returns:
    (paddings, crops), where:

    `paddings` and `crops` are int32 Tensors of rank 2 and shape [N, 2]
    satisfying:

        paddings[i, 0] = base_paddings[i, 0].
        0 <= paddings[i, 1] - base_paddings[i, 1] < block_shape[i]
        (input_shape[i] + paddings[i, 0] + paddings[i, 1]) % block_shape[i] == 0

        crops[i, 0] = 0
        crops[i, 1] = paddings[i, 1] - base_paddings[i, 1]

  Raises: ValueError if called with incompatible shapes.
  """
  with ops.name_scope(name, "required_space_to_batch_paddings",
                      [input_shape, block_shape]):
    input_shape = ops.convert_to_tensor(
        input_shape, dtype=dtypes.int32, name="input_shape")
    block_shape = ops.convert_to_tensor(
        block_shape, dtype=dtypes.int32, name="block_shape")

    block_shape.get_shape().assert_is_fully_defined()
    block_shape.get_shape().assert_has_rank(1)
    num_block_dims = block_shape.get_shape().dims[0].value
    if num_block_dims == 0:
      return zeros([0, 2], dtypes.int32), zeros([0, 2], dtypes.int32)

    input_shape.get_shape().assert_is_compatible_with([num_block_dims])

    if base_paddings is not None:
      base_paddings = ops.convert_to_tensor(
          base_paddings, dtype=dtypes.int32, name="base_paddings")
      base_paddings.get_shape().assert_is_compatible_with([num_block_dims, 2])
    else:
      base_paddings = zeros([num_block_dims, 2], dtypes.int32)

    const_block_shape = tensor_util.constant_value(block_shape)
    const_input_shape = tensor_util.constant_value(input_shape)
    const_base_paddings = tensor_util.constant_value(base_paddings)
    if (const_block_shape is not None and const_input_shape is not None and
        const_base_paddings is not None):
      block_shape = const_block_shape
      input_shape = const_input_shape
      base_paddings = const_base_paddings

    # Use same expression for both constant and non-constant case.
    pad_start = base_paddings[:, 0]
    orig_pad_end = base_paddings[:, 1]
    full_input_shape = input_shape + pad_start + orig_pad_end
    pad_end_extra = (block_shape - full_input_shape % block_shape) % block_shape
    pad_end = orig_pad_end + pad_end_extra

    result_paddings = stack(
        [[pad_start[i], pad_end[i]] for i in range(num_block_dims)],
        name="paddings")
    result_crops = stack([[0, pad_end_extra[i]] for i in range(num_block_dims)],
                         name="crops")
    return result_paddings, result_crops


@tf_export(v1=["nn.space_to_batch", "space_to_batch"])
@deprecation.deprecated_endpoints("space_to_batch")
def space_to_batch(  # pylint: disable=missing-docstring
    input, paddings, block_size=None, name=None, block_shape=None):  # pylint: disable=redefined-builtin
  block_size = deprecation.deprecated_argument_lookup("block_shape",
                                                      block_shape, "block_size",
                                                      block_size)
  result = space_to_batch_nd(
      input,
      paddings=paddings,
      block_shape=np.array([block_size, block_size], dtype=np.int64),
      name=name)
  result.set_shape(result.get_shape().with_rank(4))
  return result


space_to_batch.__doc__ = gen_array_ops.space_to_batch.__doc__


@tf_export("space_to_batch", "nn.space_to_batch", v1=[])
def space_to_batch_v2(input, block_shape, paddings, name=None):  # pylint: disable=redefined-builtin
  return space_to_batch_nd(input, block_shape, paddings, name)


space_to_batch_v2.__doc__ = gen_array_ops.space_to_batch_nd.__doc__


@tf_export(v1=["nn.space_to_depth", "space_to_depth"])
@deprecation.deprecated_endpoints("space_to_depth")
def space_to_depth(input, block_size, name=None, data_format="NHWC"):  # pylint: disable=redefined-builtin
  return gen_array_ops.space_to_depth(input, block_size, data_format, name=name)


space_to_depth.__doc__ = gen_array_ops.space_to_depth.__doc__


@tf_export("nn.space_to_depth", v1=[])
def space_to_depth_v2(input, block_size, data_format="NHWC", name=None):  # pylint: disable=redefined-builtin
  return gen_array_ops.space_to_depth(input, block_size, data_format, name=name)


space_to_depth_v2.__doc__ = gen_array_ops.space_to_depth.__doc__


@tf_export(v1=["nn.depth_to_space", "depth_to_space"])
@deprecation.deprecated_endpoints("depth_to_space")
def depth_to_space(input, block_size, name=None, data_format="NHWC"):  # pylint: disable=redefined-builtin
  return gen_array_ops.depth_to_space(input, block_size, data_format, name=name)


depth_to_space.__doc__ = gen_array_ops.depth_to_space.__doc__


@tf_export("nn.depth_to_space", v1=[])
def depth_to_space_v2(input, block_size, data_format="NHWC", name=None):  # pylint: disable=redefined-builtin
  return gen_array_ops.depth_to_space(input, block_size, data_format, name=name)


depth_to_space_v2.__doc__ = gen_array_ops.depth_to_space.__doc__


@tf_export(v1=["batch_to_space"])
def batch_to_space(input, crops, block_size, name=None, block_shape=None):  # pylint: disable=redefined-builtin,missing-docstring
  block_size = deprecation.deprecated_argument_lookup("block_shape",
                                                      block_shape, "block_size",
                                                      block_size)
  result = batch_to_space_nd(
      input,
      crops=crops,
      block_shape=np.array([block_size, block_size], dtype=np.int64),
      name=name)
  result.set_shape(result.get_shape().with_rank(4))
  return result


batch_to_space.__doc__ = gen_array_ops.batch_to_space.__doc__


@tf_export("batch_to_space", v1=[])
def batch_to_space_v2(input, block_shape, crops, name=None):  # pylint: disable=redefined-builtin
  """BatchToSpace for N-D tensors of type T.

  This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of
  shape `block_shape + [batch]`, interleaves these blocks back into the grid
  defined by the spatial dimensions `[1, ..., M]`, to obtain a result with the
  same rank as the input.  The spatial dimensions of this intermediate result
  are then optionally cropped according to `crops` to produce the output.  This
  is the reverse of SpaceToBatch.  See below for a precise description.

  Args:
    input: A `Tensor`. N-D with shape `input_shape = [batch] + spatial_shape +
      remaining_shape`, where spatial_shape has M dimensions.
    block_shape: A `Tensor`. Must be one of the following types: `int32`,
      `int64`. 1-D with shape `[M]`, all values must be >= 1. For backwards
      compatibility with TF 1.0, this parameter may be an int, in which case it
      is converted to `numpy.array([block_shape, block_shape],
      dtype=numpy.int64)`.
    crops: A `Tensor`. Must be one of the following types: `int32`, `int64`. 2-D
      with shape `[M, 2]`, all values must be >= 0. `crops[i] = [crop_start,
      crop_end]` specifies the amount to crop from input dimension `i + 1`,
      which corresponds to spatial dimension `i`.  It is required that
      `crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i + 1]`.
      This operation is equivalent to the following steps:
      1. Reshape `input` to `reshaped` of shape: [block_shape[0], ...,
        block_shape[M-1], batch / prod(block_shape), input_shape[1], ...,
        input_shape[N-1]]  2. Permute dimensions of `reshaped` to produce
        `permuted` of shape [batch / prod(block_shape),  input_shape[1],
        block_shape[0], ..., input_shape[M], block_shape[M-1],
        input_shape[M+1], ..., input_shape[N-1]]  3. Reshape `permuted` to
        produce `reshaped_permuted` of shape [batch / prod(block_shape),
        input_shape[1] * block_shape[0], ..., input_shape[M] * block_shape[M-1],
        input_shape[M+1], ..., input_shape[N-1]]  4. Crop the start and end of
        dimensions `[1, ..., M]` of `reshaped_permuted` according to `crops` to
        produce the
         output of shape: [batch / prod(block_shape),  input_shape[1] *
           block_shape[0] - crops[0,0] - crops[0,1], ..., input_shape[M] *
           block_shape[M-1] - crops[M-1,0] - crops[M-1,1],  input_shape[M+1],
           ..., input_shape[N-1]]
      Some examples:  (1) For the following input of shape `[4, 1, 1, 1]`,
          `block_shape = [2, 2]`, and `crops = [[0, 0], [0, 0]]`:  ``` [[[[1]]],
            [[[2]]], [[[3]]], [[[4]]]] ```
      The output tensor has shape `[1, 2, 2, 1]` and value:  ``` x = [[[[1],
        [2]], [[3], [4]]]] ```  (2) For the following input of shape `[4, 1, 1,
        3]`,
          `block_shape = [2, 2]`, and `crops = [[0, 0], [0, 0]]`:  ``` [[[1, 2,
            3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]] ```
      The output tensor has shape `[1, 2, 2, 3]` and value:  ``` x = [[[[1, 2,
        3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]] ```  (3) For the following
        input of shape `[4, 2, 2, 1]`,
          `block_shape = [2, 2]`, and `crops = [[0, 0], [0, 0]]`:  ``` x =
            [[[[1], [3]], [[9], [11]]], [[[2], [4]], [[10], [12]]], [[[5], [7]],
            [[13], [15]]], [[[6], [8]], [[14], [16]]]] ```
      The output tensor has shape `[1, 4, 4, 1]` and value:  ``` x = [[[1],
        [2],  [3],  [4]], [[5],   [6],  [7],  [8]], [[9],  [10], [11],  [12]],
        [[13], [14], [15],  [16]]] ```  (4) For the following input of shape
        `[8, 1, 3, 1]`,
          `block_shape = [2, 2]`, and `crops = [[0, 0], [2, 0]]`:  ``` x =
            [[[[0], [1], [3]]], [[[0], [9], [11]]], [[[0], [2], [4]]], [[[0],
            [10], [12]]], [[[0], [5], [7]]], [[[0], [13], [15]]], [[[0], [6],
            [8]]], [[[0], [14], [16]]]] ```
      The output tensor has shape `[2, 2, 4, 1]` and value:  ``` x = [[[[1],
        [2],  [3],  [4]], [[5],   [6],  [7],  [8]]], [[[9],  [10], [11],  [12]],
        [[13], [14], [15],  [16]]]] ```
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  if isinstance(block_shape, int):
    block_shape = np.array([block_shape, block_shape], dtype=np.int64)

  return batch_to_space_nd(
      input=input, block_shape=block_shape, crops=crops, name=name)


@tf_export("one_hot")
@dispatch.add_dispatch_support
def one_hot(indices,
            depth,
            on_value=None,
            off_value=None,
            axis=None,
            dtype=None,
            name=None):
  """Returns a one-hot tensor.

  The locations represented by indices in `indices` take value `on_value`,
  while all other locations take value `off_value`.

  `on_value` and `off_value` must have matching data types. If `dtype` is also
  provided, they must be the same data type as specified by `dtype`.

  If `on_value` is not provided, it will default to the value `1` with type
  `dtype`

  If `off_value` is not provided, it will default to the value `0` with type
  `dtype`

  If the input `indices` is rank `N`, the output will have rank `N+1`. The
  new axis is created at dimension `axis` (default: the new axis is appended
  at the end).

  If `indices` is a scalar the output shape will be a vector of length `depth`

  If `indices` is a vector of length `features`, the output shape will be:

  ```
    features x depth if axis == -1
    depth x features if axis == 0
  ```

  If `indices` is a matrix (batch) with shape `[batch, features]`, the output
  shape will be:

  ```
    batch x features x depth if axis == -1
    batch x depth x features if axis == 1
    depth x batch x features if axis == 0
  ```

  If `indices` is a RaggedTensor, the 'axis' argument must be positive and refer
  to a non-ragged axis. The output will be equivalent to applying 'one_hot' on
  the values of the RaggedTensor, and creating a new RaggedTensor from the
  result.

  If `dtype` is not provided, it will attempt to assume the data type of
  `on_value` or `off_value`, if one or both are passed in. If none of
  `on_value`, `off_value`, or `dtype` are provided, `dtype` will default to the
  value `tf.float32`.

  Note: If a non-numeric data type output is desired (`tf.string`, `tf.bool`,
  etc.), both `on_value` and `off_value` _must_ be provided to `one_hot`.

  For example:

  ```python
  indices = [0, 1, 2]
  depth = 3
  tf.one_hot(indices, depth)  # output: [3 x 3]
  # [[1., 0., 0.],
  #  [0., 1., 0.],
  #  [0., 0., 1.]]

  indices = [0, 2, -1, 1]
  depth = 3
  tf.one_hot(indices, depth,
             on_value=5.0, off_value=0.0,
             axis=-1)  # output: [4 x 3]
  # [[5.0, 0.0, 0.0],  # one_hot(0)
  #  [0.0, 0.0, 5.0],  # one_hot(2)
  #  [0.0, 0.0, 0.0],  # one_hot(-1)
  #  [0.0, 5.0, 0.0]]  # one_hot(1)

  indices = [[0, 2], [1, -1]]
  depth = 3
  tf.one_hot(indices, depth,
             on_value=1.0, off_value=0.0,
             axis=-1)  # output: [2 x 2 x 3]
  # [[[1.0, 0.0, 0.0],   # one_hot(0)
  #   [0.0, 0.0, 1.0]],  # one_hot(2)
  #  [[0.0, 1.0, 0.0],   # one_hot(1)
  #   [0.0, 0.0, 0.0]]]  # one_hot(-1)

  indices = tf.ragged.constant([[0, 1], [2]])
  depth = 3
  tf.one_hot(indices, depth)  # output: [2 x None x 3]
  # [[[1., 0., 0.],
  #   [0., 1., 0.]],
  #  [[0., 0., 1.]]]
  ```

  Args:
    indices: A `Tensor` of indices.
    depth: A scalar defining the depth of the one hot dimension.
    on_value: A scalar defining the value to fill in output when `indices[j]
      = i`. (default: 1)
    off_value: A scalar defining the value to fill in output when `indices[j]
      != i`. (default: 0)
    axis: The axis to fill (default: -1, a new inner-most axis).
    dtype: The data type of the output tensor.
    name: A name for the operation (optional).

  Returns:
    output: The one-hot tensor.

  Raises:
    TypeError: If dtype of either `on_value` or `off_value` don't match `dtype`
    TypeError: If dtype of `on_value` and `off_value` don't match one another
  """
  with ops.name_scope(
      name, "one_hot",
      [indices, depth, on_value, off_value, axis, dtype]) as name:
    on_exists = on_value is not None
    off_exists = off_value is not None

    on_dtype = (
        ops.convert_to_tensor(on_value).dtype.base_dtype if on_exists else None)
    off_dtype = (
        ops.convert_to_tensor(off_value).dtype.base_dtype
        if off_exists else None)

    if on_exists or off_exists:
      if dtype is not None:
        # Ensure provided on_value and/or off_value match dtype
        if on_exists and on_dtype != dtype:
          raise TypeError("dtype {0} of on_value does not match "
                          "dtype parameter {1}".format(on_dtype, dtype))
        if off_exists and off_dtype != dtype:
          raise TypeError("dtype {0} of off_value does not match "
                          "dtype parameter {1}".format(off_dtype, dtype))
      else:
        # dtype not provided: automatically assign it
        dtype = on_dtype if on_exists else off_dtype
    elif dtype is None:
      # None of on_value, off_value, or dtype provided. Default dtype to float32
      dtype = dtypes.float32

    if not on_exists:
      # on_value not provided: assign to value 1 of type dtype
      on_value = ops.convert_to_tensor(1, dtype, name="on_value")
      on_dtype = dtype
    if not off_exists:
      # off_value not provided: assign to value 0 of type dtype
      off_value = ops.convert_to_tensor(0, dtype, name="off_value")
      off_dtype = dtype

    if on_dtype != off_dtype:
      raise TypeError("dtype {0} of on_value does not match "
                      "dtype {1} of off_value".format(on_dtype, off_dtype))

    return gen_array_ops.one_hot(indices, depth, on_value, off_value, axis,
                                 name)


def _all_dimensions(x):
  """Returns a 1D-tensor listing all dimensions in x."""
  # Fast path: avoid creating Rank and Range ops if ndims is known.
  if isinstance(x, ops.Tensor) and x.get_shape().ndims is not None:
    return constant_op.constant(
        np.arange(x.get_shape().ndims), dtype=dtypes.int32)
  if (isinstance(x, sparse_tensor.SparseTensor) and
      x.dense_shape.get_shape().is_fully_defined()):
    r = x.dense_shape.get_shape().dims[0].value  # sparse.dense_shape is 1-D.
    return constant_op.constant(np.arange(r), dtype=dtypes.int32)

  # Otherwise, we rely on `range` and `rank` to do the right thing at runtime.
  return gen_math_ops._range(0, rank(x), 1)


@tf_export("sequence_mask")
def sequence_mask(lengths, maxlen=None, dtype=dtypes.bool, name=None):
  """Returns a mask tensor representing the first N positions of each cell.

  If `lengths` has shape `[d_1, d_2, ..., d_n]` the resulting tensor `mask` has
  dtype `dtype` and shape `[d_1, d_2, ..., d_n, maxlen]`, with

  ```
  mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
  ```

  Examples:

  ```python
  tf.sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                  #  [True, True, True, False, False],
                                  #  [True, True, False, False, False]]

  tf.sequence_mask([[1, 3],[2,0]])  # [[[True, False, False],
                                    #   [True, True, True]],
                                    #  [[True, True, False],
                                    #   [False, False, False]]]
  ```

  Args:
    lengths: integer tensor, all its values <= maxlen.
    maxlen: scalar integer tensor, size of last dimension of returned tensor.
      Default is the maximum value in `lengths`.
    dtype: output type of the resulting tensor.
    name: name of the op.

  Returns:
    A mask tensor of shape `lengths.shape + (maxlen,)`, cast to specified dtype.
  Raises:
    ValueError: if `maxlen` is not a scalar.
  """
  with ops.name_scope(name, "SequenceMask", [lengths, maxlen]):
    lengths = ops.convert_to_tensor(lengths)

    if maxlen is None:
      maxlen = gen_math_ops._max(lengths, _all_dimensions(lengths))
      maxlen = gen_math_ops.maximum(constant(0, maxlen.dtype), maxlen)
    else:
      maxlen = ops.convert_to_tensor(maxlen)
    if maxlen.get_shape().ndims is not None and maxlen.get_shape().ndims != 0:
      raise ValueError("maxlen must be scalar for sequence_mask")

    # The basic idea is to compare a range row vector of size maxlen:
    # [0, 1, 2, 3, 4]
    # to length as a matrix with 1 column: [[1], [3], [2]].
    # Because of broadcasting on both arguments this comparison results
    # in a matrix of size (len(lengths), maxlen)
    row_vector = gen_math_ops._range(
        constant(0, maxlen.dtype), maxlen, constant(1, maxlen.dtype))
    # Since maxlen >= max(lengths), it is safe to use maxlen as a cast
    # authoritative type. Whenever maxlen fits into tf.int32, so do the lengths.
    matrix = gen_math_ops.cast(expand_dims(lengths, -1), maxlen.dtype)
    result = row_vector < matrix

    if dtype is None or result.dtype.base_dtype == dtype.base_dtype:
      return result
    else:
      return gen_math_ops.cast(result, dtype)


@tf_export(v1=["squeeze"])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None, "Use the `axis` argument instead",
                             "squeeze_dims")
def squeeze(input, axis=None, name=None, squeeze_dims=None):
  # pylint: disable=redefined-builtin
  """Removes dimensions of size 1 from the shape of a tensor.

  Given a tensor `input`, this operation returns a tensor of the same type with
  all dimensions of size 1 removed. If you don't want to remove all size 1
  dimensions, you can remove specific size 1 dimensions by specifying
  `axis`.

  For example:

  ```python
  # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  tf.shape(tf.squeeze(t))  # [2, 3]
  ```

  Or, to remove specific size 1 dimensions:

  ```python
  # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]
  ```

  Note: if `input` is a `tf.RaggedTensor`, then this operation takes `O(N)`
  time, where `N` is the number of elements in the squeezed dimensions.

  Args:
    input: A `Tensor`. The `input` to squeeze.
    axis: An optional list of `ints`. Defaults to `[]`. If specified, only
      squeezes the dimensions listed. The dimension index starts at 0. It is an
      error to squeeze a dimension that is not 1. Must be in the range
      `[-rank(input), rank(input))`.
      Must be specified if `input` is a `RaggedTensor`.
    name: A name for the operation (optional).
    squeeze_dims: Deprecated keyword argument that is now axis.

  Returns:
    A `Tensor`. Has the same type as `input`.
    Contains the same data as `input`, but has one or more dimensions of
    size 1 removed.

  Raises:
    ValueError: When both `squeeze_dims` and `axis` are specified.
  """
  axis = deprecation.deprecated_argument_lookup("axis", axis, "squeeze_dims",
                                                squeeze_dims)
  if np.isscalar(axis):
    axis = [axis]
  return gen_array_ops.squeeze(input, axis, name)


@tf_export("squeeze", v1=[])
@dispatch.add_dispatch_support
def squeeze_v2(input, axis=None, name=None):
  """Removes dimensions of size 1 from the shape of a tensor.

  Given a tensor `input`, this operation returns a tensor of the same type with
  all dimensions of size 1 removed. If you don't want to remove all size 1
  dimensions, you can remove specific size 1 dimensions by specifying
  `axis`.

  For example:

  ```python
  # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  tf.shape(tf.squeeze(t))  # [2, 3]
  ```

  Or, to remove specific size 1 dimensions:

  ```python
  # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]
  ```

  Unlike the older op `tf.compat.v1.squeeze`, this op does not accept a
  deprecated `squeeze_dims` argument.

  Note: if `input` is a `tf.RaggedTensor`, then this operation takes `O(N)`
  time, where `N` is the number of elements in the squeezed dimensions.

  Args:
    input: A `Tensor`. The `input` to squeeze.
    axis: An optional list of `ints`. Defaults to `[]`. If specified, only
      squeezes the dimensions listed. The dimension index starts at 0. It is an
      error to squeeze a dimension that is not 1. Must be in the range
      `[-rank(input), rank(input))`. Must be specified if `input` is a
      `RaggedTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Contains the same data as `input`, but has one or more dimensions of
    size 1 removed.

  Raises:
    ValueError: The input cannot be converted to a tensor, or the specified
      axis cannot be squeezed.
  """
  # pylint: disable=redefined-builtin
  return squeeze(input, axis, name)


@tf_export(v1=["where"])
@deprecation.deprecated(
    date=None,
    instructions="Use tf.where in 2.0, "
    "which has the same broadcast rule as np.where")
@dispatch.add_dispatch_support
def where(condition, x=None, y=None, name=None):
  """Return the elements, either from `x` or `y`, depending on the `condition`.

  If both `x` and `y` are None, then this operation returns the coordinates of
  true elements of `condition`.  The coordinates are returned in a 2-D tensor
  where the first dimension (rows) represents the number of true elements, and
  the second dimension (columns) represents the coordinates of the true
  elements. Keep in mind, the shape of the output tensor can vary depending on
  how many true values there are in input. Indices are output in row-major
  order.

  If both non-None, `x` and `y` must have the same shape.
  The `condition` tensor must be a scalar if `x` and `y` are scalar.
  If `x` and `y` are tensors of higher rank, then `condition` must be either a
  vector with size matching the first dimension of `x`, or must have the same
  shape as `x`.

  The `condition` tensor acts as a mask that chooses, based on the value at each
  element, whether the corresponding element / row in the output should be taken
  from `x` (if true) or `y` (if false).

  If `condition` is a vector and `x` and `y` are higher rank matrices, then it
  chooses which row (outer dimension) to copy from `x` and `y`. If `condition`
  has the same shape as `x` and `y`, then it chooses which element to copy from
  `x` and `y`.

  Args:
    condition: A `Tensor` of type `bool`
    x: A Tensor which may have the same shape as `condition`. If `condition` is
      rank 1, `x` may have higher rank, but its first dimension must match the
      size of `condition`.
    y: A `tensor` with the same shape and type as `x`.
    name: A name of the operation (optional)

  Returns:
    A `Tensor` with the same type and shape as `x`, `y` if they are non-None.
    Otherwise, a `Tensor` with shape `(num_true, rank(condition))`.

  Raises:
    ValueError: When exactly one of `x` or `y` is non-None.
  """
  if x is None and y is None:
    with ops.name_scope(name, "Where", [condition]) as name:
      condition = ops.convert_to_tensor(
          condition, preferred_dtype=dtypes.bool, name="condition")
      return gen_array_ops.where(condition=condition, name=name)
  elif x is not None and y is not None:
    return gen_math_ops.select(condition=condition, x=x, y=y, name=name)
  else:
    raise ValueError("x and y must both be non-None or both be None.")


@tf_export("where", v1=["where_v2"])
def where_v2(condition, x=None, y=None, name=None):
  """Return the elements, either from `x` or `y`, depending on the `condition`.

  If both `x` and `y` are None, then this operation returns the coordinates of
  true elements of `condition`.  The coordinates are returned in a 2-D tensor
  where the first dimension (rows) represents the number of true elements, and
  the second dimension (columns) represents the coordinates of the true
  elements. Keep in mind, the shape of the output tensor can vary depending on
  how many true values there are in input. Indices are output in row-major
  order.

  If both non-None, `condition`, `x` and `y` must be broadcastable to the same
  shape.

  The `condition` tensor acts as a mask that chooses, based on the value at each
  element, whether the corresponding element / row in the output should be taken
  from `x` (if true) or `y` (if false).

  Args:
    condition: A `Tensor` of type `bool`
    x: A Tensor which is of the same type as `y`, and may be broadcastable with
      `condition` and `y`.
    y: A Tensor which is of the same type as `x`, and may be broadcastable with
      `condition` and `x`.
    name: A name of the operation (optional).

  Returns:
    A `Tensor` with the same type as `x` and `y`, and shape that
      is broadcast from `condition`, `x`, and `y`, if `x`, `y` are non-None.
    A `Tensor` with shape `(num_true, dim_size(condition))`.

  Raises:
    ValueError: When exactly one of `x` or `y` is non-None.
  """
  if x is None and y is None:
    with ops.name_scope(name, "Where", [condition]) as name:
      condition = ops.convert_to_tensor(
          condition, preferred_dtype=dtypes.bool, name="condition")
      return gen_array_ops.where(condition=condition, name=name)
  elif x is not None and y is not None:
    return gen_math_ops.select_v2(condition=condition, t=x, e=y, name=name)
  else:
    raise ValueError("x and y must both be non-None or both be None.")


# pylint: disable=redefined-builtin
@tf_export(v1=["reverse_sequence"])
@deprecation.deprecated_args(None,
                             "seq_dim is deprecated, use seq_axis instead",
                             "seq_dim")
@deprecation.deprecated_args(None,
                             "batch_dim is deprecated, use batch_axis instead",
                             "batch_dim")
def reverse_sequence(input,
                     seq_lengths,
                     seq_axis=None,
                     batch_axis=None,
                     name=None,
                     seq_dim=None,
                     batch_dim=None):
  seq_axis = deprecation.deprecated_argument_lookup("seq_axis", seq_axis,
                                                    "seq_dim", seq_dim)
  batch_axis = deprecation.deprecated_argument_lookup("batch_axis", batch_axis,
                                                      "batch_dim", batch_dim)
  return gen_array_ops.reverse_sequence(
      input=input,
      seq_lengths=seq_lengths,
      seq_dim=seq_axis,
      batch_dim=batch_axis,
      name=name)


reverse_sequence.__doc__ = deprecation.rewrite_argument_docstring(
    deprecation.rewrite_argument_docstring(
        gen_array_ops.reverse_sequence.__doc__, "batch_dim", "batch_axis"),
    "seq_dim", "seq_axis")


@tf_export("reverse_sequence", v1=[])
def reverse_sequence_v2(input,
                        seq_lengths,
                        seq_axis=None,
                        batch_axis=None,
                        name=None):
  return gen_array_ops.reverse_sequence(
      input=input,
      seq_lengths=seq_lengths,
      seq_dim=seq_axis,
      batch_dim=batch_axis,
      name=name)


reverse_sequence_v2.__doc__ = deprecation.rewrite_argument_docstring(
    deprecation.rewrite_argument_docstring(
        gen_array_ops.reverse_sequence.__doc__, "batch_dim", "batch_axis"),
    "seq_dim", "seq_axis")

# pylint: enable=redefined-builtin


@tf_export(v1=["gather"])
@dispatch.add_dispatch_support
def gather(params,
           indices,
           validate_indices=None,
           name=None,
           axis=None,
           batch_dims=0):  # pylint: disable=g-doc-args
  r"""Gather slices from params axis axis according to indices.

  Gather slices from params axis `axis` according to `indices`.  `indices` must
  be an integer tensor of any dimension (usually 0-D or 1-D).

  For 0-D (scalar) `indices`:

  > `output`$$[p_0,          ..., p_{axis-1},        \hspace{5.1em}
  >            p_{axis + 1}, ..., p_{N-1}]$$ =\
  > `params`$$[p_0,          ..., p_{axis-1},        \hspace{1em}
  >            indices,                              \hspace{1em}
  >            p_{axis + 1}, ..., p_{N-1}]$$.

  For 1-D (vector) `indices` with `batch_dims=0`:

  > `output`$$[p_0,          ..., p_{axis-1},        \hspace{2.6em}
  >            i,                                    \hspace{2.6em}
  >            p_{axis + 1}, ..., p_{N-1}]$$ =\
  > `params`$$[p_0,          ..., p_{axis-1},        \hspace{1em}
  >            indices[i],                           \hspace{1em}
  >            p_{axis + 1}, ..., p_{N-1}]$$.

  In the general case, produces an output tensor where:

  $$\begin{align*}
  output[p_0,             &..., p_{axis-1},                       &
       &i_{B},           ..., i_{M-1},                          &
       p_{axis + 1},    &..., p_{N-1}]                          = \\
  params[p_0,             &..., p_{axis-1},                       &
       indices[p_0, ..., p_{B-1}, &i_{B}, ..., i_{M-1}],        &
       p_{axis + 1},    &..., p_{N-1}]
  \end{align*}$$

  Where $$N$$=`ndims(params)`, $$M$$=`ndims(indices)`, and $$B$$=`batch_dims`.
  Note that params.shape[:batch_dims] must be identical to
  indices.shape[:batch_dims].

  The shape of the output tensor is:

  > `output.shape = params.shape[:axis] + indices.shape[batch_dims:] +
  > params.shape[axis + 1:]`.

  Note that on CPU, if an out of bound index is found, an error is returned.
  On GPU, if an out of bound index is found, a 0 is stored in the corresponding
  output value.

  See also `tf.gather_nd`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/Gather.png"
  alt>
  </div>

  Args:
    params: The `Tensor` from which to gather values. Must be at least rank
      `axis + 1`.
    indices: The index `Tensor`.  Must be one of the following types: `int32`,
      `int64`. Must be in range `[0, params.shape[axis])`.
    validate_indices: Deprecated, does nothing.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`. The
      `axis` in `params` to gather `indices` from. Must be greater than or equal
      to `batch_dims`.  Defaults to the first non-batch dimension. Supports
      negative indexes.
    batch_dims: An `integer`.  The number of batch dimensions.  Must be less
      than `rank(indices)`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `params`.
  """
  del validate_indices

  if axis is None:
    axis = batch_dims
  if tensor_util.constant_value(axis) != 0:
    return gen_array_ops.gather_v2(
        params, indices, axis, batch_dims=batch_dims, name=name)
  try:
    # TODO(apassos) find a less bad way of detecting resource variables
    # without introducing a circular dependency.
    return params.sparse_read(indices, name=name)
  except AttributeError:
    return gen_array_ops.gather_v2(
        params, indices, axis, name=name)


@tf_export("gather", v1=[])
@dispatch.add_dispatch_support
def gather_v2(params,
              indices,
              validate_indices=None,
              axis=None,
              batch_dims=0,
              name=None):
  return gather(
      params,
      indices,
      validate_indices=validate_indices,
      name=name,
      axis=axis,
      batch_dims=batch_dims)


gather_v2.__doc__ = gather.__doc__


@tf_export(v1=["batch_gather"])
@dispatch.add_dispatch_support
@deprecation.deprecated(
    "2017-10-25", "`tf.batch_gather` is deprecated, please use `tf.gather` "
    "with `batch_dims=-1` instead.")  # pylint: disable=missing-docstring
def batch_gather(params, indices, name=None):
  """Gather slices from params according to indices with leading batch dims."""
  with ops.name_scope(name, "BatchGather", [params, indices]):
    indices = ops.convert_to_tensor(indices, name="indices")
    params = ops.convert_to_tensor(params, name="params")
    if indices.shape.ndims is None:
      raise ValueError(
          "batch_gather does not allow indices with unknown shape.")
    return _batch_gather(params, indices, batch_dims=indices.shape.ndims - 1)


def _batch_gather(params, indices, batch_dims, axis=None):
  r"""Gather slices from params according to indices with leading batch dims.

  This operation assumes that the leading `batch_dims` dimensions of `indices`
  and `params` are batch dimensions; and performs a `tf.gather` operation within
  each batch. (If `batch_dims` is not specified, then it defaults to
  `rank(indices)-1`.)  In the case in which `batch_dims==0`, this operation
  is equivalent to `tf.gather`.

  Args:
    params: A Tensor. The tensor from which to gather values.
    indices: A Tensor. Must be one of the following types: int32, int64. Index
      tensor. Must be in range `[0, params.shape[batch_dims]]`.
    batch_dims: An integer or none.  The number of batch dimensions.  Must be
      less than `rank(indices)`.  Defaults to `rank(indices) - 1` if None.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`. The
      `axis` in `params` to gather `indices` from. Must be greater than or equal
      to `batch_dims`.  Defaults to the first non-batch dimension. Supports
      negative indexes.

  Returns:
    A Tensor. Has the same type as `params`.

  Raises:
    ValueError: if `indices` has an unknown shape.
  """
  if batch_dims is not None and not isinstance(batch_dims, int):
    raise TypeError("batch_dims must be an int; got %r" % (batch_dims,))
  indices = ops.convert_to_tensor(indices, name="indices")
  params = ops.convert_to_tensor(params, name="params")

  indices_ndims = indices.shape.ndims
  if indices_ndims is None:
    raise ValueError("tf.gather does not allow indices with unknown "
                     "rank when batch_dims is specified.")
  if batch_dims is None:
    batch_dims = indices_ndims - 1
  if batch_dims < 0:
    batch_dims += indices_ndims
  if batch_dims < 0 or batch_dims >= indices_ndims:
    raise ValueError("batch_dims = %d must be less than rank(indices) = %d" %
                     (batch_dims, indices_ndims))
  if params.shape.ndims is not None and batch_dims >= params.shape.ndims:
    raise ValueError("batch_dims = %d must be less than rank(params) = %d" %
                     (batch_dims, params.shape.ndims))

  # Handle axis by transposing the axis dimension to be the first non-batch
  # dimension, recursively calling batch_gather with axis=0, and then
  # transposing the result to put the pre-axis dimensions before the indices
  # dimensions.
  if axis is not None and axis != batch_dims:
    # Adjust axis to be positive.
    if not isinstance(axis, int):
      axis = tf.where(axis < 0, axis + array_ops.rank(params), axis)
    elif axis < 0 and params.shape.ndims is None:
      axis = axis + array_ops.rank(params)
    else:
      if (axis < -params.shape.ndims) or (axis >= params.shape.ndims):
        raise ValueError("axis (%d) out of range [%d, %d)" %
                         (axis, -params.shape.ndims, params.shape.ndims))
      if axis < 0:
        axis += params.shape.ndims
      if axis < batch_dims:
        raise ValueError("batch_dims = %d must be less than or equal to "
                         "axis = %d" % (batch_dims, axis))

    # Move params[axis] up to params[batch_dims].
    perm = [
        list(range(batch_dims)), [axis],
        gen_math_ops._range(batch_dims, axis, 1),
        gen_math_ops._range(axis + 1, rank(params), 1)
    ]
    params = transpose(params, concat(perm, axis=0))

    result = _batch_gather(params, indices, batch_dims=batch_dims)

    # Move the result dimensions corresponding to params[batch_dims:axis]
    # to just before the dimensions corresponding to indices[batch_dims:].
    params_start = indices_ndims + axis - batch_dims
    perm = [
        list(range(batch_dims)),
        gen_math_ops._range(indices_ndims, params_start, 1),
        list(range(batch_dims, indices_ndims)),
        gen_math_ops._range(params_start, rank(result), 1)
    ]
    return transpose(result, perm=concat(perm, axis=0))

  indices_shape = shape(indices)
  params_shape = shape(params)
  batch_indices = indices
  indices_dtype = indices.dtype.base_dtype
  accum_dim_value = ones((), dtype=indices_dtype)
  # Use correct type for offset index computation
  casted_params_shape = gen_math_ops.cast(params_shape, indices_dtype)
  for dim in range(batch_dims, 0, -1):
    dim_value = casted_params_shape[dim - 1]
    accum_dim_value *= casted_params_shape[dim]
    start = zeros((), dtype=indices_dtype)
    step = ones((), dtype=indices_dtype)
    dim_indices = gen_math_ops._range(start, dim_value, step)
    dim_indices *= accum_dim_value
    dim_shape = stack(
        [1] * (dim - 1) + [dim_value] + [1] * (indices_ndims - dim), axis=0)
    batch_indices += reshape(dim_indices, dim_shape)

  flat_indices = reshape(batch_indices, [-1])
  outer_shape = params_shape[batch_dims + 1:]
  flat_inner_shape = gen_math_ops.prod(params_shape[:batch_dims + 1], [0],
                                       False)

  flat_params = reshape(params, concat([[flat_inner_shape], outer_shape],
                                       axis=0))
  flat_result = gather(flat_params, flat_indices)
  result = reshape(flat_result, concat([indices_shape, outer_shape], axis=0))
  final_shape = indices.get_shape()[:batch_dims].merge_with(
      params.get_shape()[:batch_dims])
  final_shape = final_shape.concatenate(indices.get_shape().dims[batch_dims:])
  final_shape = final_shape.concatenate(params.get_shape()[batch_dims + 1:])
  result.set_shape(final_shape)
  return result


@tf_export(v1=["gather_nd", "manip.gather_nd"])
@dispatch.add_dispatch_support
@deprecated_endpoints("manip.gather_nd")
def gather_nd(params, indices, name=None, batch_dims=0):
  r"""Gather slices from `params` into a Tensor with shape specified by `indices`.

  `indices` is an K-dimensional integer tensor, best thought of as a
  (K-1)-dimensional tensor of indices into `params`, where each element defines
  a slice of `params`:

      output[\\(i_0, ..., i_{K-2}\\)] = params[indices[\\(i_0, ..., i_{K-2}\\)]]

  Whereas in `tf.gather` `indices` defines slices into the first
  dimension of `params`, in `tf.gather_nd`, `indices` defines slices into the
  first `N` dimensions of `params`, where `N = indices.shape[-1]`.

  The last dimension of `indices` can be at most the rank of
  `params`:

      indices.shape[-1] <= params.rank

  The last dimension of `indices` corresponds to elements
  (if `indices.shape[-1] == params.rank`) or slices
  (if `indices.shape[-1] < params.rank`) along dimension `indices.shape[-1]`
  of `params`.  The output tensor has shape

      indices.shape[:-1] + params.shape[indices.shape[-1]:]

  Additionally both 'params' and 'indices' can have M leading batch
  dimensions that exactly match. In this case 'batch_dims' must be M.

  Note that on CPU, if an out of bound index is found, an error is returned.
  On GPU, if an out of bound index is found, a 0 is stored in the
  corresponding output value.

  Some examples below.

  Simple indexing into a matrix:

  ```python
      indices = [[0, 0], [1, 1]]
      params = [['a', 'b'], ['c', 'd']]
      output = ['a', 'd']
  ```

  Slice indexing into a matrix:

  ```python
      indices = [[1], [0]]
      params = [['a', 'b'], ['c', 'd']]
      output = [['c', 'd'], ['a', 'b']]
  ```

  Indexing into a 3-tensor:

  ```python
      indices = [[1]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [[['a1', 'b1'], ['c1', 'd1']]]


      indices = [[0, 1], [1, 0]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [['c0', 'd0'], ['a1', 'b1']]


      indices = [[0, 0, 1], [1, 0, 1]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = ['b0', 'b1']
  ```

  The examples below are for the case when only indices have leading extra
  dimensions. If both 'params' and 'indices' have leading batch dimensions, use
  the 'batch_dims' parameter to run gather_nd in batch mode.

  Batched indexing into a matrix:

  ```python
      indices = [[[0, 0]], [[0, 1]]]
      params = [['a', 'b'], ['c', 'd']]
      output = [['a'], ['b']]
  ```

  Batched slice indexing into a matrix:

  ```python
      indices = [[[1]], [[0]]]
      params = [['a', 'b'], ['c', 'd']]
      output = [[['c', 'd']], [['a', 'b']]]
  ```

  Batched indexing into a 3-tensor:

  ```python
      indices = [[[1]], [[0]]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [[[['a1', 'b1'], ['c1', 'd1']]],
                [[['a0', 'b0'], ['c0', 'd0']]]]

      indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [[['c0', 'd0'], ['a1', 'b1']],
                [['a0', 'b0'], ['c1', 'd1']]]


      indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [['b0', 'b1'], ['d0', 'c1']]
  ```

  Examples with batched 'params' and 'indices':

  ```python
      batch_dims = 1
      indices = [[1], [0]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [['c0', 'd0'], ['a1', 'b1']]

      batch_dims = 1
      indices = [[[1]], [[0]]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [[['c0', 'd0']], [['a1', 'b1']]]

      batch_dims = 1
      indices = [[[1, 0]], [[0, 1]]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [['c0'], ['b1']]
  ```

  See also `tf.gather`.

  Args:
    params: A `Tensor`. The tensor from which to gather values.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Index tensor.
    name: A name for the operation (optional).
    batch_dims: An integer or a scalar 'Tensor'. The number of batch dimensions.

  Returns:
    A `Tensor`. Has the same type as `params`.
  """
  batch_dims_ = tensor_util.constant_value(batch_dims)
  if batch_dims_ is not None:
    batch_dims = int(batch_dims_)
  if batch_dims == 0:
    if compat.forward_compatible(2019, 4, 29):
      try:
        # TODO(apassos) find a less bad way of detecting resource variables
        # without introducing a circular dependency.
        return params.gather_nd(indices, name=name)
      except AttributeError:
        return gen_array_ops.gather_nd(params, indices, name=name)
    else:
      return gen_array_ops.gather_nd(params, indices, name=name)
  else:
    return batch_gather_nd(params, indices, batch_dims=batch_dims, name=name)


@tf_export("gather_nd", v1=[])
@dispatch.add_dispatch_support
def gather_nd_v2(params, indices, batch_dims=0, name=None):
  return gather_nd(params, indices, name=name, batch_dims=batch_dims)


gather_nd_v2.__doc__ = gather_nd.__doc__


def batch_gather_nd(params, indices, batch_dims, name=None):
  """gather_nd implementation with batch support."""
  with ops.name_scope(name, "BatchGatherND", [params, indices]):
    indices = ops.convert_to_tensor(indices, name="indices")
    params = ops.convert_to_tensor(params, name="params")

    if not isinstance(batch_dims, int):
      raise TypeError("batch_dims must be an int; got %r" % (batch_dims,))
    if batch_dims < 0:
      raise ValueError("tf.gather_nd does not allow negative batch_dims.")
    params_ndims = params.shape.ndims
    indices_ndims = indices.shape.ndims
    if indices_ndims is not None and batch_dims >= indices_ndims:
      raise ValueError("batch_dims = %d must be less than rank(indices) = %d" %
                       (batch_dims, indices_ndims))
    if params_ndims is not None and batch_dims >= params_ndims:
      raise ValueError("batch_dims = %d must be less than rank(params) = %d" %
                       (batch_dims, params_ndims))

    expand = batch_dims == 0
    if expand:
      # Normally gather_nd will be called when batch_dims == 0.
      # But if this function is called with batch_dims = 0, e.g. for testing
      # purposes, this adds a dummy batch dimension to make batch_dims = 1.
      params = expand_dims(params, axis=0)
      indices = expand_dims(indices, axis=0)
      batch_dims = 1

    params_shape = shape(params)
    indices_shape = shape(indices)
    batch_shape = params_shape[:batch_dims]
    batch_size = gen_math_ops.prod(batch_shape, [0])
    index_internal_ndims = rank(indices) - batch_dims - 1
    indices_internal_shape = indices_shape[batch_dims:-1]

    # Assuming a 'params' with shape [b1, ..., bM, g1, ..., gN] and an 'indices'
    # with shape [b1, ..., bM, i1, ..., iK, C], where C <= N, we need to modify
    # 'indices' s.t. it has shape [i1, ..., iK, D], where D <= M + N and slices
    # to the entire 'params' tensor.
    # Assuming we have a batch of shape [B1, B2], we use meshgrid to create a
    # grid of size B1 x B2.
    batch_dim_list = unstack(batch_shape, axis=0)
    dim_ranges = [
        gen_math_ops.cast(gen_math_ops._range(0, x, 1), indices.dtype)
        for x in batch_dim_list
    ]
    mesh_list = meshgrid(*dim_ranges, indexing="ij") if dim_ranges else []
    # Then we flatten and stack the tensors to form a (B1.B2) by 2 matrix.
    flat_list = [reshape(x, shape=(-1,)) for x in mesh_list]
    index_grid = transpose(stack(flat_list, axis=0))
    # We need to concatenate these batch coordinates with the internal indices.
    # concat -> index_grid [B1.B2, 2] with indices [i1, ..., iK, C]
    # So we reshape them both to [(B1.B2), i1, ..., iK, *]
    index_grid_shape = shape(index_grid)
    index_grid = reshape(
        index_grid,
        concat([
            index_grid_shape[:1],
            ones(index_internal_ndims, dtype=dtypes.int32), index_grid_shape[1:]
        ],
               axis=0))
    tile_shape = concat(((1,), indices_internal_shape, (1,)), axis=0)
    index_grid = tile(index_grid, multiples=tile_shape)
    # index_grid now has shape [(B1.B2), i1, ..., iK, 2]
    flat_shape = concat(([batch_size], indices_shape[batch_dims:]), axis=0)
    flat_indices = reshape(indices, shape=flat_shape)
    # flat_indices now has shape [(B1.B2), i1, ..., iK, C]
    indices = concat((index_grid, flat_indices), axis=-1)
    # indices has shape [(B1.B2), i1, ..., iK, 2+C]
    out = gen_array_ops.gather_nd(params, indices)
    # out has shape [(B1.B2), i1, ..., iK, N-C]. Now we reshape batch to
    # its original form.
    out_shape = shape(out)
    out = reshape(out, shape=concat((batch_shape, out_shape[1:]), axis=0))
    if expand:
      out = squeeze(out, axis=0)
  return out


# Define quantize_v2 here in order to make name the second-to-last attribute,
# because round_mode was added later.
@tf_export(v1=["quantize_v2"])
@deprecation.deprecated(
    "2017-10-25",
    "`tf.quantize_v2` is deprecated, please use `tf.quantization.quantize` "
    "instead.")  # pylint: disable=missing-docstring
def quantize_v2(
    input,  # pylint: disable=redefined-builtin
    min_range,
    max_range,
    T,
    mode="MIN_COMBINED",
    name=None,
    round_mode="HALF_AWAY_FROM_ZERO"):
  return gen_array_ops.quantize_v2(
      input,
      min_range,
      max_range,
      T=T,
      mode=mode,
      name=name,
      round_mode=round_mode)


quantize_v2.__doc__ = """Please use `tf.quantization.quantize` instead."""


# We want to expose tf.quantization.quantize instead of
# tf.quantization.quantize; we can deprecate tf.quantization.quantize in next
# version of TensorFlow.
@tf_export("quantization.quantize", v1=["quantization.quantize", "quantize"])
@deprecation.deprecated_endpoints("quantize")
def quantize(input,  # pylint: disable=redefined-builtin
             min_range,
             max_range,
             T,
             mode="MIN_COMBINED",
             round_mode="HALF_AWAY_FROM_ZERO",
             name=None):
  return gen_array_ops.quantize_v2(
      input,
      min_range,
      max_range,
      T,
      mode=mode,
      round_mode=round_mode,
      name=name)


@tf_export("quantization.quantize_and_dequantize")
def quantize_and_dequantize(input,  # pylint: disable=redefined-builtin
                            input_min,
                            input_max,
                            signed_input=True,
                            num_bits=8,
                            range_given=False,
                            round_mode="HALF_TO_EVEN",
                            name=None,
                            narrow_range=False):
  """Quantizes then dequantizes a tensor.

  Args:
    input: A `Tensor` to quantize and dequantize.
    input_min: If range_given=True, the minimum input value that needs to be
      represented in the quantized representation.
    input_max: If range_given=True, the maximum input value that needs to be
      represented in the quantized representation.
    signed_input: True if the quantization is signed or unsigned.
    num_bits: The bitwidth of the quantization.
    range_given: If true use `input_min` and `input_max` for the range of the
      input, otherwise determine min and max from the input `Tensor`.
    round_mode: Rounding mode when rounding from float values to quantized ones.
    name: Optional name for the operation.
    narrow_range: If true, then the absolute value of the quantized minimum
      value is the same as the quantized maximum value, instead of 1 greater.
      i.e. for 8 bit quantization, the minimum value is -127 instead of -128.

  Returns:
    A `Tensor`. Each element is the result of quantizing and dequantizing the
    corresponding element of `input`.
  """
  return gen_array_ops.quantize_and_dequantize_v2(
      input,
      input_min=input_min,
      input_max=input_max,
      signed_input=signed_input,
      num_bits=num_bits,
      range_given=range_given,
      round_mode=round_mode,
      narrow_range=narrow_range,
      name=name)


@tf_export("searchsorted")
def searchsorted(sorted_sequence,
                 values,
                 side="left",
                 out_type=dtypes.int32,
                 name=None):
  """Searches input tensor for values on the innermost dimension.

  A 2-D example:

  ```
    sorted_sequence = [[0, 3, 9, 9, 10],
                       [1, 2, 3, 4, 5]]
    values = [[2, 4, 9],
              [0, 2, 6]]

    result = searchsorted(sorted_sequence, values, side="left")

    result == [[1, 2, 2],
               [0, 1, 5]]

    result = searchsorted(sorted_sequence, values, side="right")

    result == [[1, 2, 4],
               [0, 2, 5]]
  ```

  Args:
    sorted_sequence: N-D `Tensor` containing a sorted sequence.
    values: N-D `Tensor` containing the search values.
    side: 'left' or 'right'; 'left' corresponds to lower_bound and 'right' to
      upper_bound.
    out_type: The output type (`int32` or `int64`).  Default is `tf.int32`.
    name: Optional name for the operation.

  Returns:
    An N-D `Tensor` the size of values containing the result of applying either
    lower_bound or upper_bound (depending on side) to each value.  The result
    is not a global index to the entire `Tensor`, but the index in the last
    dimension.

  Raises:
    ValueError: If the last dimension of `sorted_sequence >= 2^31-1` elements.
                If the total size of values exceeds `2^31 - 1` elements.
                If the first `N-1` dimensions of the two tensors don't match.
  """
  sequence_size = shape_internal(sorted_sequence)[-1]
  values_size = shape_internal(values)[-1]
  sorted_sequence_2d = reshape(sorted_sequence, [-1, sequence_size])
  values_2d = reshape(values, [-1, values_size])
  if side == "right":
    output = gen_array_ops.upper_bound(sorted_sequence_2d, values_2d, out_type,
                                       name)
  elif side == "left":
    output = gen_array_ops.lower_bound(sorted_sequence_2d, values_2d, out_type,
                                       name)
  else:
    raise ValueError("side must be either 'right' or 'left'.  Saw: %s." % side)
  return reshape(output, shape_internal(values))


quantize.__doc__ = gen_array_ops.quantize_v2.__doc__


@tf_export("image.extract_patches")
def extract_image_patches_v2(images, sizes, strides, rates, padding, name=None):
  r"""Extract `patches` from `images`.

  This op collects patches from the input image, as if applying a
  convolution. All extracted patches are stacked in the depth (last) dimension
  of the output.

  Specifically, the op extracts patches of shape `sizes` which are `strides`
  apart in the input image. The output is subsampled using the `rates` argument,
  in the same manner as "atrous" or "dilated" convolutions.

  The result is a 4D tensor which is indexed by batch, row, and column.
  `output[i, x, y]` contains a flattened patch of size `sizes[1], sizes[2]`
  which is taken from the input starting at
  `images[i, x*strides[1], y*strides[2]]`.

  Each output patch can be reshaped to `sizes[1], sizes[2], depth`, where
  `depth` is `images.shape[3]`.

  The output elements are taken from the input at intervals given by the `rate`
  argument, as in dilated convolutions.

  The `padding` argument has no effect on the size of each patch, it determines
  how many patches are extracted. If `VALID`, only patches which are fully
  contained in the input image are included. If `SAME`, all patches whose
  starting point is inside the input are included, and areas outside the input
  default to zero.

  Example:

  ```
    n = 10
    # images is a 1 x 10 x 10 x 1 array that contains the numbers 1 through 100
    images = [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]

    # We generate two outputs as follows:
    # 1. 3x3 patches with stride length 5
    # 2. Same as above, but the rate is increased to 2
    tf.extract_image_patches(images=images,
                             ksizes=[1, 3, 3, 1],
                             strides=[1, 5, 5, 1],
                             rates=[1, 1, 1, 1],
                             padding='VALID')

    # Yields:
    [[[[ 1  2  3 11 12 13 21 22 23]
       [ 6  7  8 16 17 18 26 27 28]]
      [[51 52 53 61 62 63 71 72 73]
       [56 57 58 66 67 68 76 77 78]]]]
  ```

  If we mark the pixels in the input image which are taken for the output with
  `*`, we see the pattern:

  ```
     *  *  *  4  5  *  *  *  9 10
     *  *  * 14 15  *  *  * 19 20
     *  *  * 24 25  *  *  * 29 30
    31 32 33 34 35 36 37 38 39 40
    41 42 43 44 45 46 47 48 49 50
     *  *  * 54 55  *  *  * 59 60
     *  *  * 64 65  *  *  * 69 70
     *  *  * 74 75  *  *  * 79 80
    81 82 83 84 85 86 87 88 89 90
    91 92 93 94 95 96 97 98 99 100
  ```

  ```
    tf.extract_image_patches(images=images,
                             sizes=[1, 3, 3, 1],
                             strides=[1, 5, 5, 1],
                             rates=[1, 2, 2, 1],
                             padding='VALID')

    # Yields:
    [[[[  1   3   5  21  23  25  41  43  45]
       [  6   8  10  26  28  30  46  48  50]]

      [[ 51  53  55  71  73  75  91  93  95]
       [ 56  58  60  76  78  80  96  98 100]]]]
  ```

  We can again draw the effect, this time using the symbols `*`, `x`, `+` and
  `o` to distinguish the patches:

  ```
     *  2  *  4  *  x  7  x  9  x
    11 12 13 14 15 16 17 18 19 20
     * 22  * 24  *  x 27  x 29  x
    31 32 33 34 35 36 37 38 39 40
     * 42  * 44  *  x 47  x 49  x
     + 52  + 54  +  o 57  o 59  o
    61 62 63 64 65 66 67 68 69 70
     + 72  + 74  +  o 77  o 79  o
    81 82 83 84 85 86 87 88 89 90
     + 92  + 94  +  o 97  o 99  o
  ```

  Args:
    images: A 4-D Tensor with shape `[batch, in_rows, in_cols, depth]
    sizes: The size of the extracted patches. Must
      be [1, size_rows, size_cols, 1].
    strides: A 1-D Tensor of length 4. How far the centers of two consecutive
      patches are in the images. Must be: `[1, stride_rows, stride_cols, 1]`.
    rates: A 1-D Tensor of length 4. Must be: `[1, rate_rows, rate_cols, 1]`.
      This is the input stride, specifying how far two consecutive patch samples
      are in the input. Equivalent to extracting patches with `patch_sizes_eff =
      patch_sizes + (patch_sizes - 1) * (rates - 1)`, followed by subsampling
      them spatially by a factor of `rates`. This is equivalent to `rate` in
      dilated (a.k.a. Atrous) convolutions.
    padding: The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A 4-D Tensor of the same type as the input.
  """
  return gen_array_ops.extract_image_patches(images, sizes, strides, rates,
                                             padding, name)


@tf_export(v1=["image.extract_image_patches", "extract_image_patches"])
@deprecation.deprecated_args(None, "ksizes is deprecated, use sizes instead",
                             "ksizes")
def extract_image_patches(  # pylint: disable=missing-docstring
    images,
    ksizes=None,
    strides=None,
    rates=None,
    padding=None,
    name=None,
    sizes=None):
  ksizes = deprecation.deprecated_argument_lookup("sizes", sizes, "ksizes",
                                                  ksizes)
  return gen_array_ops.extract_image_patches(images, ksizes, strides, rates,
                                             padding, name)


extract_image_patches.__doc__ = gen_array_ops.extract_image_patches.__doc__


@tf_export("fingerprint")
def fingerprint(data, method="farmhash64", name=None):
  r"""Generates fingerprint values.

  Generates fingerprint values of `data`.

  Fingerprint op considers the first dimension of `data` as the batch dimension,
  and `output[i]` contains the fingerprint value generated from contents in
  `data[i, ...]` for all `i`.

  Fingerprint op writes fingerprint values as byte arrays. For example, the
  default method `farmhash64` generates a 64-bit fingerprint value at a time.
  This 8-byte value is written out as an `tf.uint8` array of size 8, in
  little-endian order.

  For example, suppose that `data` has data type `tf.int32` and shape (2, 3, 4),
  and that the fingerprint method is `farmhash64`. In this case, the output
  shape is (2, 8), where 2 is the batch dimension size of `data`, and 8 is the
  size of each fingerprint value in bytes. `output[0, :]` is generated from
  12 integers in `data[0, :, :]` and similarly `output[1, :]` is generated from
  other 12 integers in `data[1, :, :]`.

  Note that this op fingerprints the raw underlying buffer, and it does not
  fingerprint Tensor's metadata such as data type and/or shape. For example, the
  fingerprint values are invariant under reshapes and bitcasts as long as the
  batch dimension remain the same:

  ```python
  tf.fingerprint(data) == tf.fingerprint(tf.reshape(data, ...))
  tf.fingerprint(data) == tf.fingerprint(tf.bitcast(data, ...))
  ```

  For string data, one should expect `tf.fingerprint(data) !=
  tf.fingerprint(tf.string.reduce_join(data))` in general.

  Args:
    data: A `Tensor`. Must have rank 1 or higher.
    method: A `Tensor` of type `tf.string`. Fingerprint method used by this op.
      Currently available method is `farmhash64`.
    name: A name for the operation (optional).

  Returns:
    A two-dimensional `Tensor` of type `tf.uint8`. The first dimension equals to
    `data`'s first dimension, and the second dimension size depends on the
    fingerprint algorithm.
  """
  return gen_array_ops.fingerprint(data, method, name)


def convert_to_int_tensor(tensor, name, dtype=dtypes.int32):
  """Converts the given value to an integer Tensor."""
  tensor = ops.convert_to_tensor(tensor, name=name, preferred_dtype=dtype)
  if tensor.dtype.is_integer:
    tensor = gen_math_ops.cast(tensor, dtype)
  else:
    raise TypeError("%s must be an integer tensor; dtype=%s" %
                    (name, tensor.dtype))
  return tensor


def get_positive_axis(axis, ndims):
  """Validate an `axis` parameter, and normalize it to be positive.

  If `ndims` is known (i.e., not `None`), then check that `axis` is in the
  range `-ndims <= axis < ndims`, and return `axis` (if `axis >= 0`) or
  `axis + ndims` (otherwise).
  If `ndims` is not known, and `axis` is positive, then return it as-is.
  If `ndims` is not known, and `axis` is negative, then report an error.

  Args:
    axis: An integer constant
    ndims: An integer constant, or `None`

  Returns:
    The normalized `axis` value.

  Raises:
    ValueError: If `axis` is out-of-bounds, or if `axis` is negative and
      `ndims is None`.
  """
  if not isinstance(axis, int):
    raise TypeError("axis must be an int; got %s" % type(axis).__name__)
  if ndims is not None:
    if 0 <= axis < ndims:
      return axis
    elif -ndims <= axis < 0:
      return axis + ndims
    else:
      raise ValueError("axis=%s out of bounds: expected %s<=axis<%s" %
                       (axis, -ndims, ndims))
  elif axis < 0:
    raise ValueError("axis may only be negative if ndims is statically known.")
  return axis


# This op is intended to exactly match the semantics of numpy.repeat, with
# one exception: numpy.repeat has special (and somewhat non-intuitive) behavior
# when axis is not specified.  Rather than implement that special behavior, we
# simply make `axis` be a required argument.
#
# External (OSS) `tf.repeat` feature request:
# https://github.com/tensorflow/tensorflow/issues/8246
def repeat_with_axis(data, repeats, axis, name=None):
  """Repeats elements of `data`.

  Args:
    data: An `N`-dimensional tensor.
    repeats: A 1-D integer tensor specifying how many times each element in
      `axis` should be repeated.  `len(repeats)` must equal `data.shape[axis]`.
      Supports broadcasting from a scalar value.
    axis: `int`.  The axis along which to repeat values.  Must be less than
      `max(N, 1)`.
    name: A name for the operation.

  Returns:
    A tensor with `max(N, 1)` dimensions.  Has the same shape as `data`,
    except that dimension `axis` has size `sum(repeats)`.
  #### Examples:
    ```python
    >>> repeat(['a', 'b', 'c'], repeats=[3, 0, 2], axis=0)
    ['a', 'a', 'a', 'c', 'c']
    >>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=0)
    [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]
    >>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=1)
    [[1, 1, 2, 2, 2], [3, 3, 4, 4, 4]]
    ```
  """
  if not isinstance(axis, int):
    raise TypeError("axis must be an int; got %s" % type(axis).__name__)

  with ops.name_scope(name, "Repeat", [data, repeats]):
    data = ops.convert_to_tensor(data, name="data")
    repeats = convert_to_int_tensor(repeats, name="repeats")
    repeats.shape.with_rank_at_most(1)

    # If `data` is a scalar, then upgrade it to a vector.
    data = _with_nonzero_rank(data)
    data_shape = shape(data)

    # If `axis` is negative, then convert it to a positive value.
    axis = get_positive_axis(axis, data.shape.ndims)

    # Check data Tensor shapes.
    if repeats.shape.ndims == 1:
      data.shape.dims[axis].assert_is_compatible_with(repeats.shape[0])

    # If we know that `repeats` is a scalar, then we can just tile & reshape.
    if repeats.shape.ndims == 0:
      expanded = expand_dims(data, axis + 1)
      tiled = tile_one_dimension(expanded, axis + 1, repeats)
      result_shape = concat([data_shape[:axis], [-1], data_shape[axis + 1:]],
                            axis=0)
      return reshape(tiled, result_shape)

    # Broadcast the `repeats` tensor so rank(repeats) == axis + 1.
    if repeats.shape.ndims != axis + 1:
      repeats_shape = shape(repeats)
      repeats_ndims = rank(repeats)
      broadcast_shape = concat(
          [data_shape[:axis + 1 - repeats_ndims], repeats_shape], axis=0)
      repeats = broadcast_to(repeats, broadcast_shape)
      repeats.set_shape([None] * (axis + 1))

    # Create a "sequence mask" based on `repeats`, where slices across `axis`
    # contain one `True` value for each repetition.  E.g., if
    # `repeats = [3, 1, 2]`, then `mask = [[1, 1, 1], [1, 0, 0], [1, 1, 0]]`.
    max_repeat = gen_math_ops.maximum(
        0, gen_math_ops._max(repeats, _all_dimensions(repeats)))
    mask = sequence_mask(repeats, max_repeat)

    # Add a new dimension around each value that needs to be repeated, and
    # then tile that new dimension to match the maximum number of repetitions.
    expanded = expand_dims(data, axis + 1)
    tiled = tile_one_dimension(expanded, axis + 1, max_repeat)

    # Use `boolean_mask` to discard the extra repeated values.  This also
    # flattens all dimensions up through `axis`.
    masked = boolean_mask(tiled, mask)

    # Reshape the output tensor to add the outer dimensions back.
    if axis == 0:
      result = masked
    else:
      result_shape = concat([data_shape[:axis], [-1], data_shape[axis + 1:]],
                            axis=0)
      result = reshape(masked, result_shape)

    # Preserve shape information.
    if data.shape.ndims is not None:
      new_axis_size = 0 if repeats.shape[0] == 0 else None
      result.set_shape(data.shape[:axis].concatenate(
          [new_axis_size]).concatenate(data.shape[axis + 1:]))

    return result


def tile_one_dimension(data, axis, multiple):
  """Tiles a single dimension of a tensor."""
  # Assumes axis is a nonnegative int.
  if data.shape.ndims is not None:
    multiples = [1] * data.shape.ndims
    multiples[axis] = multiple
  else:
    ones_value = ones(rank(data), dtypes.int32)
    multiples = concat([ones_value[:axis], [multiple], ones_value[axis + 1:]],
                       axis=0)
  return tile(data, multiples)


def _with_nonzero_rank(data):
  """If `data` is scalar, then add a dimension; otherwise return as-is."""
  if data.shape.ndims is not None:
    if data.shape.ndims == 0:
      return stack([data])
    else:
      return data
  else:
    data_shape = shape(data)
    data_ndims = rank(data)
    return reshape(data, concat([[1], data_shape], axis=0)[-data_ndims:])


@tf_export("repeat")
def repeat(input, repeats, axis=None, name=None):  # pylint: disable=redefined-builtin
  """Repeat elements of `input`

  Args:
    input: An `N`-dimensional Tensor.
    repeats: An 1-D `int` Tensor. The number of repetitions for each element.
      repeats is broadcasted to fit the shape of the given axis. `len(repeats)`
      must equal `input.shape[axis]` if axis is not None.
    axis: An int. The axis along which to repeat values. By default (axis=None),
      use the flattened input array, and return a flat output array.
    name: A name for the operation.

  Returns:
    A Tensor which has the same shape as `input`, except along the given axis.
      If axis is None then the output array is flattened to match the flattened
      input array.
  #### Examples:
    ```python
    >>> repeat(['a', 'b', 'c'], repeats=[3, 0, 2], axis=0)
    ['a', 'a', 'a', 'c', 'c']
    >>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=0)
    [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]
    >>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=1)
    [[1, 1, 2, 2, 2], [3, 3, 4, 4, 4]]
    >>> repeat(3, repeats=4)
    [3, 3, 3, 3]
    >>> repeat([[1,2], [3,4]], repeats=2)
    [1, 1, 2, 2, 3, 3, 4, 4]
    ```
  """
  if axis is None:
    input = reshape(input, [-1])
    axis = 0
  return repeat_with_axis(input, repeats, axis, name)
