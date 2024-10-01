# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf

from tf_nndct.pruning import compat as tf_compat

class Pruning(object):
  """Implementation of magnitude-based weight pruning."""

  def __init__(self, pruning_vars):
    """The logic for magnitude-based pruning weight tensors.

    Args:
      pruning_vars: A list of (weight, mask) tuples
    """
    self._pruning_vars = pruning_vars

  def _weight_assign_objs(self):
    """Gather the assign objs for assigning weights<=weights*mask.

    The objs are ops for graph execution and tensors for eager
    execution.

    Returns:
      group of objs for weight assignment.
    """

    def update_fn(distribution, values_and_vars):
      # TODO(yunluli): Need this ReduceOp because the weight is created by the
      # layer wrapped, so we don't have control of its aggregation policy. May
      # be able to optimize this when distribution strategy supports easier
      # update to mirrored variables in replica context.
      reduced_values = distribution.extended.batch_reduce_to(
          tf.distribute.ReduceOp.MEAN, values_and_vars)
      var_list = [v for _, v in values_and_vars]
      values_and_vars = zip(reduced_values, var_list)

      def update_var(variable, reduced_value):
        return tf_compat.assign(variable, reduced_value)

      update_objs = []
      for value, var in values_and_vars:
        update_objs.append(
            distribution.extended.update(var, update_var, args=(value,)))

      return tf.group(update_objs)

    assign_objs = []

    if tf.distribute.get_replica_context():
      values_and_vars = []
      for weight, mask in self._pruning_vars:
        masked_weight = tf.math.multiply(weight, mask)
        values_and_vars.append((masked_weight, weight))
      if values_and_vars:
        assign_objs.append(tf.distribute.get_replica_context().merge_call(
            update_fn, args=(values_and_vars,)))
    else:
      for weight, mask in self._pruning_vars:
        masked_weight = tf.math.multiply(weight, mask)
        assign_objs.append(tf_compat.assign(weight, masked_weight))

    return assign_objs

  def weight_mask_op(self):
    return tf.group(self._weight_assign_objs())

  def add_pruning_summaries(self):
    """Adds summaries of weight sparsities and thresholds."""
    # b/(139939526): update to use public API.
    summary = tf.summary
    if not tf.executing_eagerly():
      summary = tf.compat.v1.summary
    for _, mask in self._pruning_vars:
      summary.scalar(mask.name + '/sparsity', 1.0 - tf.math.reduce_mean(mask))
