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
"""Utility functions for adding pruning related ops to the graph.

"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import g3
import numpy as np
import tensorflow as tf


def kronecker_product(mat1, mat2):
  """Computes the Kronecker product of two matrices mat1 and mat2.

  Args:
    mat1: A matrix of size m x n
    mat2: A matrix of size p x q

  Returns:
    Kronecker product of matrices mat1 and mat2 of size mp x nq
  """

  m1, n1 = mat1.get_shape().as_list()
  mat1_rsh = tf.reshape(mat1, [m1, 1, n1, 1])
  m2, n2 = mat2.get_shape().as_list()
  mat2_rsh = tf.reshape(mat2, [1, m2, 1, n2])
  return tf.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2])


def expand_tensor(tensor, block_size):
  """Expands a 2D tensor by replicating the tensor values.

  This is equivalent to the kronecker product of the tensor and a matrix of
  ones of size block_size.

  Example:

  tensor = [[1,2]
            [3,4]]
  block_size = [2,2]

  result = [[1 1 2 2]
            [1 1 2 2]
            [3 3 4 4]
            [3 3 4 4]]

  Args:
    tensor: A 2D tensor that needs to be expanded.
    block_size: List of integers specifying the expansion factor.

  Returns:
    The expanded tensor

  Raises:
    ValueError: if tensor is not rank-2 or block_size is does not have 2
    elements.
  """
  if tensor.get_shape().ndims != 2:
    raise ValueError('Input tensor must be rank 2')

  if len(block_size) != 2:
    raise ValueError('block_size must have 2 elements')

  block_height, block_width = block_size

  def _tile_rows(tensor, multiple):
    """Create a new tensor by tiling the tensor along rows."""
    return tf.tile(tensor, [multiple, 1])

  def _generate_indices(num_rows, block_dim):
    indices = np.zeros(shape=[num_rows * block_dim, 1], dtype=np.int32)
    for k in range(block_dim):
      for r in range(num_rows):
        indices[k * num_rows + r] = r * block_dim + k
    return indices

  def _replicate_rows(tensor, multiple):
    tensor_shape = tensor.shape.as_list()
    expanded_shape = [tensor_shape[0] * multiple, tensor_shape[1]]
    indices = tf.constant(_generate_indices(tensor_shape[0], multiple))
    return tf.scatter_nd(indices, _tile_rows(tensor, multiple), expanded_shape)

  expanded_tensor = tensor

  # Expand rows by factor block_height.
  if block_height > 1:
    expanded_tensor = _replicate_rows(tensor, block_height)

  # Transpose and expand by factor block_width. Transpose the result.
  if block_width > 1:
    expanded_tensor = tf.transpose(
        _replicate_rows(tf.transpose(expanded_tensor), block_width))

  return expanded_tensor


def factorized_pool(input_tensor,
                    window_shape,
                    pooling_type,
                    strides,
                    padding,
                    name=None):
  """Performs m x n pooling through a combination of 1xm and 1xn pooling.

  Args:
    input_tensor: Input tensor. Must be rank 2
    window_shape: Pooling window shape
    pooling_type: Either 'MAX' or 'AVG'
    strides: The stride of the pooling window
    padding: 'SAME' or 'VALID'.
    name: Name of the op

  Returns:
    A rank 2 tensor containing the pooled output

  Raises:
    ValueError: if the input tensor is not rank 2
  """
  if input_tensor.get_shape().ndims != 2:
    raise ValueError('factorized_pool() accepts tensors of rank 2 only')

  [height, width] = input_tensor.get_shape()
  if name is None:
    name = 'factorized_pool'
  with tf.name_scope(name):
    input_tensor_aligned = tf.reshape(input_tensor, [1, 1, height, width])

    height_pooling = tf.nn.pool(
        input_tensor_aligned,
        window_shape=[1, window_shape[0]],
        pooling_type=pooling_type,
        strides=[1, strides[0]],
        padding=padding)
    swap_height_width = tf.transpose(height_pooling, perm=[0, 1, 3, 2])

    width_pooling = tf.nn.pool(
        swap_height_width,
        window_shape=[1, window_shape[1]],
        pooling_type=pooling_type,
        strides=[1, strides[1]],
        padding=padding)

  return tf.squeeze(tf.transpose(width_pooling, perm=[0, 1, 3, 2]))
