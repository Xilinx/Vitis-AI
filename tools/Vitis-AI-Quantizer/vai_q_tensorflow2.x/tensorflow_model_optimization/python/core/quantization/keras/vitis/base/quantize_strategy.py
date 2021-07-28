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
"""Interface of Quantization Strategy."""

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class QuantizeStrategy(object):
  """ABC interface for Quantize Strategy.

  QuantizeStrategy encapsulates all the configurations for quantization.

  It mainly contains of below parts:
  1) quantize_registry: A QuantizeRegistry object contains the input quantizer
  configurations and detailed configurations for different keras layer types.
  2) optimize_pipeline: A TransformsPipeline object to do the pre-quantization optimization transformations.
  3) quantize_pipeline: A TransformsPipeline object to do the main quantize transformations.

  Users can create new custom quantize strategies to override the default ones, for a full example, see [GUIDE]
  """

  @abc.abstractmethod
  def update(self, qs_configs):
    """Update the current configurations by overriding.

    Args:
      new_config: String, file name of the new quantize strategy configurations.

    Returns:
      None
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def get_configs(self):
    """Update the current configurations by overriding.

    Args:
      None

    Returns:
      None
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def get_quantize_registry(self):
    """Return the quantize registry including the input quantize configurations
    and the detailed configurations for keras and layers.

    Args:
      None

    Returns:
      A QuantizeRegistry object.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def get_optimize_pipeline(self):
    """Return the TransformsPipeline of the pre-quantization optimization processes.

    Args:
      None

    Returns:
      A TransformsPipeline object.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def get_quantize_pipeline(self):
    """Return the TransformsPipeline of the main quantization processes.

    Args:
      None

    Returns:
      A TransformsPipeline object.
    """
    raise NotImplementedError('Must be implemented in subclasses.')
