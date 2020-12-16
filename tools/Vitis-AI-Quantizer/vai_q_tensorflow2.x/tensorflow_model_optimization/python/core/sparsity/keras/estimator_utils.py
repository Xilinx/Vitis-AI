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
"""Utility functions for making pruning wrapper work with estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import g3

from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import monitored_session
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_wrapper import PruneLowMagnitude


class PruningEstimatorSpec(EstimatorSpec):
  """Returns an EstimatorSpec modified to prune the model while training."""

  def __new__(cls, model, step=None, train_op=None, **kwargs):
    if "mode" not in kwargs:
      raise ValueError("Must provide a mode (TRAIN/EVAL/PREDICT) when "
                       "creating an EstimatorSpec")

    if train_op is None:
      raise ValueError(
          "Must provide train_op for creating a PruningEstimatorSpec")

    def _get_step_increment_ops(model, step=None):
      """Returns ops to increment the pruning_step in the prunable layers."""
      increment_ops = []

      for layer in model.layers:
        if isinstance(layer, PruneLowMagnitude):
          if step is None:
            # Add ops to increment the pruning_step by 1
            increment_ops.append(state_ops.assign_add(layer.pruning_step, 1))
          else:
            increment_ops.append(
                state_ops.assign(layer.pruning_step,
                                 math_ops.cast(step, dtypes.int32)))

      return control_flow_ops.group(increment_ops)

    pruning_ops = []
    # Grab the ops to update pruning step in every prunable layer
    step_increment_ops = _get_step_increment_ops(model, step)
    pruning_ops.append(step_increment_ops)
    # Grab the model updates.
    pruning_ops.append(model.updates)

    kwargs["train_op"] = control_flow_ops.group(pruning_ops, train_op)

    def init_fn(scaffold, session):  # pylint: disable=unused-argument
      return session.run(step_increment_ops)

    def get_new_scaffold(old_scaffold):
      if old_scaffold.init_fn is None:
        return monitored_session.Scaffold(
            init_fn=init_fn, copy_from_scaffold=old_scaffold)
      # TODO(suyoggupta): Figure out a way to merge the init_fn of the
      # original scaffold with the one defined above.
      raise ValueError("Scaffold provided to PruningEstimatorSpec must not "
                       "set an init_fn.")

    scaffold = monitored_session.Scaffold(init_fn=init_fn)
    if "scaffold" in kwargs:
      scaffold = get_new_scaffold(kwargs["scaffold"])

    kwargs["scaffold"] = scaffold

    return super(PruningEstimatorSpec, cls).__new__(cls, **kwargs)


def add_pruning_summaries(model):
  """Add pruning summaries to the graph for the given model."""

  with ops.name_scope("pruning_summaries"):
    for layer in model.layers:
      if isinstance(layer, PruneLowMagnitude):
        # Add the summary under the underlying layer's name_scope.
        # TODO(suyoggupta): Look for a less ugly way of doing this.
        with ops.name_scope(layer.layer.name):
          layer.pruning_obj.add_pruning_summaries()
