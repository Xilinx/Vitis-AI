# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Pruning Schedule classes to control pruning rate during training."""

import abc
import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class PruningSchedule(object):
  """Specifies when to prune layer and the sparsity(%) at each training step.

  PruningSchedule controls pruning during training by notifying at each step
  whether the layer's weights should be pruned or not, and the sparsity(%) at
  which they should be pruned.

  It can be invoked as a `callable` by providing the training `step` Tensor. It
  returns a tuple of bool and float tensors.

  ```python
    should_prune, sparsity = pruning_schedule(step)
  ```

  You can inherit this class to write your own custom pruning schedule.
  """

  @staticmethod
  def _should_prune_in_step(step, begin_step, end_step, frequency):
    """Checks if pruning should be applied in the current training step.

    Pruning should only occur within the [`begin_step`, `end_step`] range every
    `frequency` number of steps.

    Args:
      step: Current training step.
      begin_step: Step at which to begin pruning.
      end_step: Step at which to end pruning.
      frequency: Only apply pruning every `frequency` steps.

    Returns:
      True/False, if pruning should be applied in current step.
    """
    is_in_pruning_range = tf.math.logical_and(
        tf.math.greater_equal(step, begin_step),
        # If end_pruning_step is negative, keep pruning forever!
        tf.math.logical_or(
            tf.math.less_equal(step, end_step), tf.math.less(end_step, 0)))

    is_pruning_turn = tf.math.equal(
        tf.math.floormod(tf.math.subtract(step, begin_step), frequency), 0)

    return tf.math.logical_and(is_in_pruning_range, is_pruning_turn)

  @staticmethod
  def _validate_step(begin_step, end_step, frequency, allow_negative_1):
    """Checks whether the parameters for pruning schedule are valid.

    Args:
      begin_step: Step at which to begin pruning.
      end_step: Step at which to end pruning. Special value of `-1` implies
        pruning can continue forever.
      frequency: Only apply pruning every `frequency` steps.
      allow_negative_1: Whether end_step is allowed to be `-1` or not.

    Returns:
      None
    """

    if begin_step < 0:
      raise ValueError('begin_step should be >= 0')

    # In cases like PolynomialDecay, continuing to prune forever does not make
    # sense. The function needs an end_step to decay the sparsity.
    if not allow_negative_1 and end_step == -1:
      raise ValueError('end_step cannot be -1.')

    if end_step != -1:
      if end_step < 0:
        raise ValueError('end_step can be -1 or >= 0')
      if end_step < begin_step:
        raise ValueError('begin_step should be <= end_step if end_step != -1')

    if frequency <= 0:
      raise ValueError('frequency should be > 0')

  @staticmethod
  def _validate_sparsity(sparsity, variable_name):
    if not 0.0 <= sparsity < 1.0:
      raise ValueError('{} must be in range [0,1)'.format(variable_name))

  @abc.abstractmethod
  def __call__(self, step):
    """Returns the sparsity(%) to be applied.

    If the returned sparsity(%) is 0, pruning is ignored for the step.

    Args:
      step: Current step in graph execution.

    Returns:
      Sparsity (%) that should be applied to the weights for the step.
    """
    raise NotImplementedError(
        'PruningSchedule implementation must override __call__')

  @abc.abstractmethod
  def get_config(self):
    raise NotImplementedError(
        'PruningSchedule implementation override get_config')

  @classmethod
  def from_config(cls, config):
    """Instantiates a `PruningSchedule` from its config.

    Args:
        config: Output of `get_config()`.

    Returns:
        A `PruningSchedule` instance.
    """
    return cls(**config)


class ConstantSparsity(PruningSchedule):
  """Pruning schedule with constant sparsity(%) throughout training."""

  def __init__(self,
               target_sparsity,
               begin_step,
               end_step=-1,
               frequency=100):
    """Initializes a Pruning schedule with constant sparsity.

    Sparsity is applied in the interval [`begin_step`, `end_step`] every
    `frequency` steps. At each applicable step, the sparsity(%) is constant.

    Args:
      target_sparsity: A scalar float representing the target sparsity value.
      begin_step: Step at which to begin pruning.
      end_step: Step at which to end pruning. `-1` by default. `-1` implies
        continuing to prune till the end of training.
      frequency: Only apply pruning every `frequency` steps.
    """

    self.target_sparsity = target_sparsity
    self.begin_step = begin_step
    self.end_step = end_step
    self.frequency = frequency

    self._validate_step(self.begin_step, self.end_step, self.frequency, True)
    self._validate_sparsity(target_sparsity, 'target_sparsity')

  def __call__(self, step):
    return (self._should_prune_in_step(step, self.begin_step, self.end_step,
                                       self.frequency),
            tf.constant(self.target_sparsity, dtype=tf.float32))

  def get_config(self):
    return {
        'class_name': self.__class__.__name__,
        'config': {
            'target_sparsity': self.target_sparsity,
            'begin_step': self.begin_step,
            'end_step': self.end_step,
            'frequency': self.frequency
        }
    }


class PolynomialDecay(PruningSchedule):
  """Pruning Schedule with a PolynomialDecay function."""

  def __init__(self,
               initial_sparsity,
               final_sparsity,
               begin_step,
               end_step,
               power=3,
               frequency=100):
    """Initializes a Pruning schedule with a PolynomialDecay function.

    Pruning rate grows rapidly in the beginning from initial_sparsity, but then
    plateaus slowly to the target sparsity. The function applied is

    current_sparsity = final_sparsity + (initial_sparsity - final_sparsity)
          * (1 - (step - begin_step)/(end_step - begin_step)) ^ exponent

    which is a polynomial decay function. See
    [paper](https://arxiv.org/abs/1710.01878).

    Args:
      initial_sparsity: Sparsity (%) at which pruning begins.
      final_sparsity: Sparsity (%) at which pruning ends.
      begin_step: Step at which to begin pruning.
      end_step: Step at which to end pruning.
      power: Exponent to be used in the sparsity function.
      frequency: Only apply pruning every `frequency` steps.
    """

    self.initial_sparsity = initial_sparsity
    self.final_sparsity = final_sparsity
    self.power = power

    self.begin_step = begin_step
    self.end_step = end_step
    self.frequency = frequency

    self._validate_step(self.begin_step, self.end_step, self.frequency, False)
    self._validate_sparsity(initial_sparsity, 'initial_sparsity')
    self._validate_sparsity(final_sparsity, 'final_sparsity')

  def __call__(self, step):
    # TODO(tf-mot): consider switch to divide for 1.XX also.
    if hasattr(tf, 'div'):
      divide = tf.div
    else:
      divide = tf.math.divide

    # TODO(pulkitb): Replace function with tf.polynomial_decay
    with tf.name_scope('polynomial_decay_pruning_schedule'):
      p = tf.math.minimum(
          1.0,
          tf.math.maximum(
              0.0,
              divide(
                  tf.dtypes.cast(step - self.begin_step, tf.float32),
                  self.end_step - self.begin_step)))
      sparsity = tf.math.add(
          tf.math.multiply(self.initial_sparsity - self.final_sparsity,
                           tf.math.pow(1 - p, self.power)),
          self.final_sparsity,
          name='sparsity')

    return (self._should_prune_in_step(step, self.begin_step, self.end_step,
                                       self.frequency),
            sparsity)

  def get_config(self):
    return {
        'class_name': self.__class__.__name__,
        'config': {
            'initial_sparsity': self.initial_sparsity,
            'final_sparsity': self.final_sparsity,
            'power': self.power,
            'begin_step': self.begin_step,
            'end_step': self.end_step,
            'frequency': self.frequency
        }
    }
