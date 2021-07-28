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
"""Vitis Quantization Strategy."""

from tensorflow_model_optimization.python.core.quantization.keras.vitis.base import quantize_strategy
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

QuantizeStrategy = quantize_strategy.QuantizeStrategy
logger = common_utils.VAILogger


class VitisQuantizeStrategy(QuantizeStrategy):
  """Vitis Quantize Strategy."""

  def update(self, qs_configs):
    """Update the current configurations by overriding.

    Args:
      new_config: String, file name of the new quantize strategy configurations.

    Returns:
      None
    """

    if 'quantize_registry_config' in qs_configs:
      self._quantize_registry.update(qs_configs.pop('quantize_registry_config'))
    if 'optimize_pipeline_config' in qs_configs:
      self._optimize_pipeline.update(qs_configs.pop('optimize_pipeline_config'))
    if 'quantize_pipeline_config' in qs_configs:
      self._quantize_pipeline.update(qs_configs.pop('quantize_pipeline_config'))

    invalid_configs = []
    while qs_configs:
      config = qs_configs.popitem()
      if self._quantize_registry.is_valid_config(config):
        self._quantize_registry.update(config)
      elif self._optimize_pipeline.is_valid_config(config):
        self._optimize_pipeline.update(config)
      elif self._quantize_pipeline.is_valid_config(config):
        self._quantize_pipeline.update(config)
      else:
        invalid_configs.append(config)

    # Check for invalid configurations
    if invalid_configs:
      logger.error('Invalid configs: {}'.format(invalid_configs))

    self._qs_configs.update({
        'quantize_registry_config': self._quantize_registry.get_configs(),
        'optimize_pipeline_config': self._optimize_pipeline.get_configs(),
        'quantize_pipeline_config': self._quantize_pipeline.get_configs()
    })

  # Interface functions
  def get_configs(self):
    return self._qs_configs

  def get_quantize_registry(self):
    """Return the quantize registry including the input quantize configurations
    and the detailed configurations for keras and vitis layers.

    Args:
      None

    Returns:
      A VitisQuantizeRegistry object.
    """
    return self._quantize_registry

  def get_optimize_pipeline(self):
    """Return the TransformsPipeline of the pre-quantization optimization processes.

    Args:
      None

    Returns:
      A TransformsPipeline object.
    """
    return self._optimize_pipeline

  def get_quantize_pipeline(self):
    """Return the TransformsPipeline of the main quantization processes.

    Args:
      None

    Returns:
      A TransformsPipeline object.
    """
    return self._quantize_pipeline
