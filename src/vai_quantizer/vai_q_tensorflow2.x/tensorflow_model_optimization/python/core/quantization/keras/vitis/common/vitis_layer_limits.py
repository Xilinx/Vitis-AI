# Copyright 2019 Xilinx Inc.
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
"""Vitis layer limit class."""

import abc
import six
import enum

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

register_keras_serializable = tf.keras.utils.register_keras_serializable
logger = common_utils.VAILogger


class LimitType(enum.Enum):
  """Enum class for limit types."""
  INT_RANGE = 1
  INT_RANGE_PAIR = 2
  INT_CHOICE = 3
  INT_CHOICE_PAIR = 4
  STR_CHOICE = 5


class Limit(object):
  """Class to handle different kinds of limits."""

  def __init__(self,
               bound_str=None,
               bound_str_sep='-',
               limit_type=LimitType.INT_RANGE):
    if not isinstance(limit_type, LimitType):
      limit_type = LimitType(limit_type)
    self._limit_type = limit_type

    if self._limit_type == LimitType.INT_RANGE:
      self._parse_int_range(bound_str, bound_str_sep)
    elif self._limit_type == LimitType.INT_RANGE_PAIR:
      self._parse_int_range_pair(bound_str, bound_str_sep)
    elif self._limit_type == LimitType.INT_CHOICE:
      raise NotImplementedError()
    elif self._limit_type == LimitType.INT_CHOICE_PAIR:
      self._parse_int_choice_pair(bound_str, bound_str_sep)
    elif self._limit_type == LimitType.STR_CHOICE:
      raise NotImplementedError()
    else:
      logger.error('Unsupported limit type {}.'.format(self._limit_type))

  def _parse_int_range(self, bound_str, bound_str_sep):
    """Parse bound string to int range. 

    E.g. '1-16' to [1, 16].
    """
    if not bound_str:
      logger.error('Invalid bound_str {}'.format(bound_str))
    tmp = [int(i) for i in bound_str.split(bound_str_sep)]
    assert len(tmp) == 2
    self._lower_bound, self._upper_bound = tmp

  def _parse_int_range_pair(self, bound_str, bound_str_sep):
    """Parse bound string to int range pair. Pair should be seperated by ";".

    E.g. '1-16;2-8' to [1, 16], [2, 8].
    """
    if not bound_str:
      logger.error('Invalid bound_str {}'.format(bound_str))
    bound_str_1, bound_str_2 = bound_str.split(';')
    tmp_1 = [int(i) for i in bound_str_1.split(bound_str_sep)]
    tmp_2 = [int(i) for i in bound_str_2.split(bound_str_sep)]
    self._lower_bound = [tmp_1[0], tmp_2[0]]
    self._upper_bound = [tmp_1[1], tmp_2[1]]

  def _parse_int_choice_pair(self, bound_str, bound_str_sep):
    """Parse bound string to int choices pair. Pair should be seperated by ";".

    E.g. '2,3,5,7,8;2,3,5,7,8' to [2, 3, 5, 7, 8], [2, 3, 5, 7, 8].
    """
    if not bound_str:
      logger.error('Invalid bound_str {}'.format(bound_str))
    bound_str_1, bound_str_2 = bound_str.split(';')
    tmp_1 = [int(i) for i in bound_str_1.split(',')]
    tmp_2 = [int(i) for i in bound_str_2.split(',')]
    self._choices = [tmp_1, tmp_2]

  def in_limit(self, inp):
    if self._limit_type == LimitType.INT_RANGE:
      return self._in_int_range(inp)
    elif self._limit_type == LimitType.INT_RANGE_PAIR:
      return self._in_int_range_pair(inp)
    elif self._limit_type == LimitType.INT_CHOICE:
      return self._in_int_choice(inp)
    elif self._limit_type == LimitType.INT_CHOICE_PAIR:
      return self._in_int_choice_pair(inp)
    elif self._limit_type == LimitType.STR_CHOICE:
      return self._in_str_choice(inp)
    else:
      logger.error('Unsupported limit type {}.'.format(self._limit_type))

  def _in_int_range(self, inp):
    inp = int(inp)
    is_in_limit = inp >= self._lower_bound and inp <= self._upper_bound
    msg = None
    if not is_in_limit:
      msg = '{} exceed limit {}'.format(inp, self)
    return is_in_limit, msg

  def _in_int_range_pair(self, inp):
    if not isinstance(inp, tuple):
      logger.error('Expected inp to be tuple, but found {}.'.format(type(inp)))
    inp1, inp2 = inp

    is_in_limit1 = inp1 >= self._lower_bound[0] and inp1 <= self._upper_bound[0]
    is_in_limit2 = inp2 >= self._lower_bound[1] and inp2 <= self._upper_bound[1]
    is_in_limit = is_in_limit1 and is_in_limit2

    msg = None
    if not is_in_limit:
      msg = '{} exceed limit {}'.format(inp, self)
    return is_in_limit, msg

  def _in_int_choice(self, inp):
    inp = int(inp)
    return inp in self._choices

  def _in_int_choice_pair(self, inp):
    if not isinstance(inp, tuple):
      logger.error('Expected inp to be tuple, but found {}.'.format(type(inp)))
    inp1, inp2 = inp

    is_in_limit1 = inp1 in self._choices[0]
    is_in_limit2 = inp2 in self._choices[1]
    is_in_limit = is_in_limit1 and is_in_limit2

    msg = None
    if not is_in_limit:
      msg = '{} exceed limit {}'.format(inp, self)
    return is_in_limit, msg

  def _in_str_choice(self, inp):
    return inp in self._choices

  def __repr__(self):
    if self._limit_type == LimitType.INT_RANGE:
      return '{}-{}'.format(self._lower_bound, self._upper_bound)
    elif self._limit_type == LimitType.INT_RANGE_PAIR:
      return '{}-{};{}-{}'.format(self._lower_bound[0], self._upper_bound[0],
                                  self._lower_bound[1], self._upper_bound[1])
    elif self._limit_type == LimitType.INT_CHOICE_PAIR:
      return '{};{}'.format(self._choices[0], self._choices[1])
    else:
      raise NotImplementedError()


@six.add_metaclass(abc.ABCMeta)
class BaseLayerLimits(object):
  """Base class to handle different kinds of limits of a layer."""

  def __init__(self, limits={}):
    self._limits = limits

  def add_attr_limit(self, attr, limit):
    if not isinstance(limit, Limit):
      logger.error('limit should be Limit instance, but got type {}.'.format(
          type(limit)))
    self._limits[attr] = limit

  def merge(self, other):
    for attr, limit in other._limits.items():
      self.add_limit(attr, limit)

  def in_attr_limits(self, layer):
    """Check if all attributes of layer is in limit."""
    is_in_limit = True
    msgs = []
    for attr, limit in self._limits.items():
      _is_in_limit, _msg = self._in_attr_limit(layer, attr)
      is_in_limit &= _is_in_limit
      if not _is_in_limit:
        msgs.append(_msg)
    return is_in_limit, msgs

  def _in_attr_limit(self, layer, attr):
    """Check if layer.attr is in limit."""
    if not hasattr(layer, attr):
      logger.error('Fail to get attr {} from layer {}({}).'.format(
          attr, layer, layer.__class__.__name__))
    value = getattr(layer, attr)

    is_in_limit = None
    msg = None

    limit = self._limits.get(attr)
    if not limit:
      is_in_limit = True
      msg = 'No limit for {}'.format(attr)
      logger.info('No limit for {}'.format(attr))

    is_in_limit, _ = limit.in_limit(value)
    if not is_in_limit:
      msg = '{}({}) exceed limit {}'.format(attr, value, limit)
    return is_in_limit, msg

  @abc.abstractmethod
  def get_layer_type(self):
    """Get current layer type."""
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def is_supported(self):
    """Check if current layer type is supported by target."""
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def build_attr_limits(self, layer):
    """Build attr limits."""
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def in_other_limits(self, layer):
    """Check limits not representable in attr limits."""
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def in_limits(self, layer):
    """Main entrance to check attr limits and other limits."""
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def in_act_limits(self, layer, act_layer):
    """Check layer + act_layer limits."""
    raise NotImplementedError('Must be implemented in subclasses.')

  def __repr__(self):
    msgs = []
    for attr, limit in self._limits.items():
      msgs.append("'{}': '{}'".format(attr, limit))
    msgs = ', '.join(msgs)
    return "{" + msgs + "}"

  @abc.abstractmethod
  def get_config(self):
    """Get config for serialization."""
    raise NotImplementedError('Must be implemented in subclasses.')
