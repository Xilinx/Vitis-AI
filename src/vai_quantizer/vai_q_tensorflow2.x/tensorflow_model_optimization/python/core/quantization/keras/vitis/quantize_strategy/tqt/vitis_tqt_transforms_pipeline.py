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
"""Vitis trained quantization threshold transformations pipeline for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_refine_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_transforms_pipeline
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.tqt import vitis_tqt_refine_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_fast_finetune
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_bias_correction
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import model_utils

VitisPof2SOptimizeTransformsPipeline = vitis_pof2s_transforms_pipeline.VitisPof2SOptimizeTransformsPipeline
VitisPof2SQuantizeTransformsPipeline = vitis_pof2s_transforms_pipeline.VitisPof2SQuantizeTransformsPipeline
VitisPof2SRefineTransformsPipeline = vitis_pof2s_transforms_pipeline.VitisPof2SRefineTransformsPipeline
VitisPof2SFinalizeTransformsPipeline = vitis_pof2s_transforms_pipeline.VitisPof2SFinalizeTransformsPipeline
logger = common_utils.VAILogger


class VitisTQTOptimizeTransformsPipeline(VitisPof2SOptimizeTransformsPipeline):
  """Vitis trained quantization threshold optimize model transformmations."""
  pass


class VitisTQTQuantizeTransformsPipeline(VitisPof2SQuantizeTransformsPipeline):
  """Vitis trained quantization threshold quantize model transformmations."""
  pass


class VitisTQTRefineTransformsPipeline(VitisPof2SRefineTransformsPipeline):
  """Vitis trained quantization threshold refine model transformmations."""

  def apply(self, quantized_model, candidate_layers, layer_metadata,
            optimized_model, dataset, batch_size, steps, add_shape_info,
            input_shape):
    """Implement vitis 8-bit float scale refine transforms.

    Args:
      qunantized_model: the quantized model to be refined.
      optimized_model: the optimized float model used in fast finetune to generate fake label.
      dataset: the dataset used in fast finetune.
      batch_size: the batch size of dataset used in fast finetune.
      steps: the steps of dataste used in fast finetune.
      add_shape_info: bool, whether to add shape information to the refined model. Must be set True
        for models with custom layers.
      input_shape: the shape of the model inputs, if not set, the default shape in the model inputs
        will be used.

    Returns:
      (Refined quantized model.)
    """
    refined_model = quantized_model
    configs = self.get_configs()

    # Convert TQTQuantizer to Pof2SQuantizer
    if configs['convert_to_pof2s_quantize_strategy']:
      logger.info("Start Converting To Pof2S Quantize Strategy...")
      transforms = [
          vitis_tqt_refine_transforms.ConvertTQTToPof2SQuantizeStrategy()
      ]
      refined_model, _ = model_transformer.ModelTransformer(
          refined_model, transforms, None, None).recursive_transform()
      logger.info("Converting To Pof2S Quantize Strategy Done.")

      # Adjust quantize positions only when convert_to_pof2s_quantize_strategy is True
      logger.info("Start Quantize Position Ajustment...")
      adjust_vs = configs['adjust_dpu_sigmoid']
      adjust_sc = configs['adjust_dpu_shift_cut']
      adjust_sb = configs['adjust_dpu_shift_bias']
      adjust_sr = configs['adjust_dpu_shift_read']
      adjust_sw = configs['adjust_dpu_shift_write']
      adjust_sh = configs['adjust_dpu_shift_swish']
      align_concat = configs['align_concat']
      align_pool = configs['align_pool']
      quantize_info = model_utils.get_quantize_info(refined_model)
      adjusted_quantize_info = vitis_pof2s_refine_transforms.adjust_quantize_info(
          refined_model, quantize_info, adjust_vs, adjust_sc, adjust_sb,
          adjust_sr, adjust_sw, adjust_sh, align_concat, align_pool)
      model_utils.set_quantize_info(refined_model, adjusted_quantize_info)
      logger.info("Quantize Position Ajustment Done.")

    # Fast finetune
    include_fast_ft = configs['include_fast_ft']
    fast_ft_epochs = configs['fast_ft_epochs']
    if include_fast_ft:
      logger.info("Start Fast Finetuning...")
      vitis_fast_finetune.fast_finetune(refined_model, optimized_model, dataset,
                                        batch_size, steps, fast_ft_epochs)
      logger.info("Fast Finetuning Done.")

    #  # Bias correction
    #  include_bias_corr = configs['quantize_pipeline_config'][
    #      'include_bias_corr']
    #  if include_bias_corr:
    #    logger.info("Start Bias Correction...")
    #    vitis_bias_correction.bias_correction(self._qcbev_model,
    #                                          self._optimized_model,
    #                                          calib_dataset, calib_batch_size,
    #                                          calib_steps)
    #    logger.info("Bias Correction Done.")

    if add_shape_info:
      logger.info("Start Getting Shape Information...")
      shape_info = model_utils.get_shape(
          model=refined_model, calib_dataset=dataset, input_shape=input_shape)
      if logger.debug_enabled():
        model_utils.save_shape_info(shape_info, './debug/')
        model_utils.save_model(refined_model, 'calibrated_model_add_shape.h5',
                               './debug/')
      logger.info("Getting Shape Information Done.")

    return refined_model, layer_metadata


class VitisTQTFinalizeTransformsPipeline(VitisPof2SFinalizeTransformsPipeline):
  """Vitis trained quantization threshold finalize model transformmations."""
  pass
