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
"""Abstract Base Class for model transformations pipeline to a keras model.

Keras models need certain transformations to exactly match the behavior of the 
backend they will be implemented on. This is important for improving model performance.

This interface abstracts that behavior. Different backends can implement their
own version.
"""
import abc
import six

from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

logger = common_utils.VAILogger


@six.add_metaclass(abc.ABCMeta)
class TransformsPipeline(object):
  """Wrapper of transforms to the model, apply in sequence.

  Transforms the original model to perform better during quantization.
  """

  def __init__(self, configs):
    """Init.

    Args:
      configs: Dict objects containing the detailed configurations.
    """
    self._configs = configs

  def update(self, configs):
    """Update configurations.

    Args:
      configs: Dict objects containing the detailed configurations.

    Returns:
      None
    """
    if not isinstance(configs, dict):
      if isinstance(configs, tuple) and len(configs) == 2:
        configs = {configs[0]: configs[1]}
      else:
        logger.error('Invalid format of configs: {}'.format(configs))

    for k, v in configs.items():
      if not self.is_valid_config((k, v)):
        logger.error('Invalid config for {}: ({}: {}).'.format(
            self.__class__.__name__, k, v))
      else:
        logger.debug('Update config for {}: {}: {}.'.format(
            self.__class__.__name__, k, v))
        self._configs.update({k: v})

  def get_configs(self):
    """Get the configurations.

    Args:
      None

    Returns:
      Dict of configurations
    """
    return self._configs

  def is_valid_config(self, config):
    """Check if the config is valid."""
    return config[0] in self.get_configs()

  def print_configs(self):
    """Print the configurations."""
    for k, v in self._configs.items():
      logger.debug('- {}: {}'.format(k, v))

  @abc.abstractmethod
  def apply(self, model, candidate_layers, layer_metadata):
    """Apply list of transforms to keras model.

    Args:
      model: Keras model to be quantized.

    Returns:
      New keras model based on `model` which has been transformed.
    """
    raise NotImplementedError('Must be implemented in subclasses.')
