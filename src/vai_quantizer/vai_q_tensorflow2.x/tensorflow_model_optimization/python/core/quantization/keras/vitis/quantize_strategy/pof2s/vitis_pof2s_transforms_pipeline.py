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
"""Vitis pof2s transformations pipeline for quantization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import model_transformer
from tensorflow_model_optimization.python.core.quantization.keras.vitis.graph_transformations import transforms_pipeline
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_optimize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_equalization_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_fast_finetune
from tensorflow_model_optimization.python.core.quantization.keras.vitis.optimizations import vitis_bias_correction
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_quantize_transforms_with_xcompiler
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_quantize_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.quantize_strategy.pof2s import vitis_pof2s_refine_transforms
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import model_utils

TransformsPipeline = transforms_pipeline.TransformsPipeline
keras = tf.keras
logger = common_utils.VAILogger


def _apply_availables(model, configs, available_transforms, candidate_layers,
                      layer_metadata):
  transforms = []
  for key in available_transforms:
    if configs.get(key):
      new_trans = available_transforms.get(key)
      if isinstance(new_trans, list):
        transforms.extend(new_trans)
      else:
        transforms.append(new_trans)

  transformed_model, layer_metadata = model_transformer.ModelTransformer(
      model, transforms, candidate_layers,
      layer_metadata).recursive_transform()
  return transformed_model, layer_metadata


class VitisPof2SOptimizeTransformsPipeline(TransformsPipeline):
  """Vitis pof2s pre-quantization optimize model transformations."""

  def apply(self, model, candidate_layers, layer_metadata):
    """Implement vitis 8-bit optimize transforms to make it more quantize-friendly.

    All the transforms should not break the float model structure, and
    the output of the transformed model should be consistant with the float
    model.
    """
    configs = self.get_configs()

    available_transforms = collections.OrderedDict({
        'remove_dropout':
            vitis_optimize_transforms.RemoveDropout(),
        'separate_conv_act':
            vitis_optimize_transforms.SeparateConvAct(),
        'convert_relu6_to_relu':
            vitis_optimize_transforms.ConvertReLU6ToReLU(),
        'convert_tf_op_to_keras':
            vitis_optimize_transforms.ConvertTFOpToKeras(),
        'separable_conv':
            vitis_optimize_transforms.SeparableConv(),
    })

    transformed_model, layer_metadata = _apply_availables(
        model, configs, available_transforms, candidate_layers, layer_metadata)

    # Train with bn is conflict with fold bn params
    model = transformed_model
    if configs['train_with_bn']:
      transforms = [
          vitis_optimize_transforms.FakeConvBNFold(),
      ]
      transformed_model, layer_metadata = model_transformer.ModelTransformer(
          model, transforms, candidate_layers,
          layer_metadata).recursive_transform()
    else:
      available_transforms = {
          'fold_conv_bn': vitis_optimize_transforms.ConvBNFold(),
          'convert_bn_to_dwconv': vitis_optimize_transforms.ConvertBNToDWConv(),
      }
      transformed_model, layer_metadata = _apply_availables(
          model, configs, available_transforms, candidate_layers,
          layer_metadata)

    # Cross Layer Equalization
    if configs['include_cle']:
      cle_to_relu6 = configs['cle_to_relu6']
      cle_balance_method = configs['cle_balance_method']
      cle_weight_threshold = configs['cle_weight_threshold']
      cle_transforms = [
          vitis_equalization_transforms.ConvConvCLE(cle_to_relu6,
                                                    cle_balance_method,
                                                    cle_weight_threshold),
          vitis_equalization_transforms.ConvActConvCLE(cle_to_relu6,
                                                       cle_balance_method,
                                                       cle_weight_threshold),
          vitis_equalization_transforms.ConvReLUConvCLE(cle_to_relu6,
                                                        cle_balance_method,
                                                        cle_weight_threshold),
          vitis_equalization_transforms.ConvReLUPadConvCLE(
              cle_to_relu6, cle_balance_method, cle_weight_threshold)
      ]

      cle_steps = configs['cle_steps']
      progbar = keras.utils.Progbar(cle_steps)
      logger.info('Start CrossLayerEqualization...')
      for i in range(cle_steps):
        progbar.update(i + 1)
        tmp_model = transformed_model
        transformed_model, layer_metadata = model_transformer.ModelTransformer(
            tmp_model, cle_transforms, candidate_layers,
            layer_metadata).recursive_transform()
      logger.info('CrossLayerEqualization Done.')

    if logger.debug_enabled():
      model_utils.save_model(transformed_model, 'optimized_model.h5',
                             './debug/')
    return transformed_model, layer_metadata


class VitisPof2SQuantizeTransformsPipeline(TransformsPipeline):
  """Vitis pof2s model quantize transformations pipeline."""

  def apply(self, model, candidate_layers, layer_metadata, quantize_registry,
            mode, target):
    """Implement vitis 8-bit quantize transforms.

    Args:
      model: Keras model to be quantized.
      quantize_registry: QuantizeRegistry object containing the quantize configs
        for each layer.
      mode: String object indicating the mode of the quantized model.

    Returns:
      (Quantized Keras model.)
    """
    configs = self.get_configs()

    available_pre_annotate_transforms = collections.OrderedDict({
        'convert_sigmoid_to_hard_sigmoid': [
            vitis_pof2s_quantize_transforms.ConvertConvSigmoid(),
            vitis_pof2s_quantize_transforms.ConvertActivationSigmoid(),
            vitis_pof2s_quantize_transforms.ConvertConvSwish(),
            vitis_pof2s_quantize_transforms.ConvertActivationSwish(),
        ],
        'convert_hard_sigmoid_to_dpu_version': [
            vitis_pof2s_quantize_transforms.ConvertHardSigmoidToDpuVersion(),
        ],
        'convert_average_pooling2d_to_dpu_version': [
            vitis_pof2s_quantize_transforms.ConvertAveragePooling2DToDpuVersion(
            ),
            vitis_pof2s_quantize_transforms
            .ConvertGlobalAveragePooling2DToDpuVersion(),
        ],
        'convert_leaky_relu_to_dpu_version': [
            vitis_pof2s_quantize_transforms.ConvertLeakyReLUToDpuVersion(),
        ],
    })

    pre_annotated_model, layer_metadata = _apply_availables(
        model, configs, available_pre_annotate_transforms, candidate_layers,
        layer_metadata)

    if logger.debug_enabled():
      model_utils.save_model(pre_annotated_model, 'pre_annotated_model.h5',
                             './debug/')

    annotate_transforms = [
        vitis_pof2s_quantize_transforms.AnnotateConvBNAct(),
        vitis_pof2s_quantize_transforms.AnnotateConvAct(pre_annotated_model,
                                                        quantize_registry),
        vitis_pof2s_quantize_transforms.AnnotateAddAct(pre_annotated_model,
                                                       quantize_registry),
        vitis_pof2s_quantize_transforms.AnnotatePoolAct(pre_annotated_model,
                                                        quantize_registry),
    ]
    annotated_model, layer_metadata = model_transformer.ModelTransformer(
        pre_annotated_model, annotate_transforms, candidate_layers,
        layer_metadata).recursive_transform()

    if logger.debug_enabled():
      model_utils.save_model(annotated_model, 'annotated_model.h5', './debug/')

    quantize_transforms = [
        vitis_pof2s_quantize_transforms.InputLayerQuantize(
            quantize_registry, mode),
        vitis_pof2s_quantize_transforms.ConvBNQuantize(
            quantize_registry, mode, configs['freeze_bn_delay']),
    ]
    if "quantize_with_xcompiler" in configs and configs["quantize_with_xcompiler"]:
      for _, LayerQuantize in vitis_pof2s_quantize_transforms_with_xcompiler.quantize_pattern_dict.items(
      ):
        quantize_transforms.append(LayerQuantize(annotated_model, mode, target))
    else:
      quantize_transforms.append(
          vitis_pof2s_quantize_transforms.LayersQuantize(
              annotated_model, quantize_registry, mode))
    quantize_transforms.extend([
        vitis_pof2s_quantize_transforms.LayersInputQuantize(
            quantize_registry, mode),
        vitis_pof2s_quantize_transforms.CustomLayerWrapper(quantize_registry)
    ])

    quantized_model, layer_metadata = model_transformer.ModelTransformer(
        annotated_model, quantize_transforms, candidate_layers,
        layer_metadata).recursive_transform()
    return quantized_model, layer_metadata


class VitisPof2SRefineTransformsPipeline(TransformsPipeline):
  """Vitis pof2s model refine transformations pipeline."""

  def apply(self, quantized_model, candidate_layers, layer_metadata,
            optimized_model, dataset, batch_size, steps, add_shape_info,
            input_shape):
    """Implement vitis model refine transforms.

    Args:
      qunantized_model: the quantized model to be refined.
      optimized_model: the optimized float model used in fast finetune to generate fake label.
      dataset: the dataset used in fast finetune.
      batch_size: the batch size of dataset used in fast finetune.
      steps: the steps of dataste used in fast finetune.
      add_shape_info: bool, whether to add shape information to the refined model. Must be set True
        for models with custom layers.
      input_shape: the shape of the model inputs, if not set, the pof2s shape in the model inputs
        will be used.

    Returns:
      (Refined quantized model.)
    """
    refined_model = quantized_model
    configs = self.get_configs()

    # Adjust quantize positions
    logger.info("Start Quantize Position Ajustment...")
    adjust_vs = configs['adjust_dpu_sigmoid']
    adjust_sc = configs['adjust_dpu_shift_cut']
    adjust_sb = configs['adjust_dpu_shift_bias']
    adjust_sr = configs['adjust_dpu_shift_read']
    adjust_sw = configs['adjust_dpu_shift_write']
    adjust_sh = configs['adjust_dpu_shift_swish']
    align_concat = configs['align_concat']
    align_pool = configs['align_pool']
    adjust_sb_leakyrelu = configs['adjust_dpu_shift_bias_leakyrelu']
    quantize_info = model_utils.get_quantize_info(refined_model)
    adjusted_quantize_info = vitis_pof2s_refine_transforms.adjust_quantize_info(
        refined_model, quantize_info, adjust_vs, adjust_sc, adjust_sb,
        adjust_sr, adjust_sw, adjust_sh, align_concat, align_pool,
        adjust_sb_leakyrelu)
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

    # Bias correction
    #include_bias_corr = configs['include_bias_corr']
    #if include_bias_corr:
    #logger.info("Start Bias Correction...")
    #vitis_bias_correction.bias_correction(refined_model,
    #                                      optimized_model,
    #                                      dataset, batch_size,
    #                                      steps)
    #logger.info("Bias Correction Done.")

    # Convert Pof2SQuantizer to FSQuantizer
    if configs['convert_to_fs_quantize_strategy']:
      logger.info("Start Converting To FS Quantize Strategy...")
      transforms = [
          vitis_pof2s_refine_transforms.ConvertPof2SToFSQuantizeStrategy()
      ]
      refined_model, _ = model_transformer.ModelTransformer(
          refined_model, transforms, None, None).recursive_transform()
      logger.info("Converting To FS Quantize Strategy Done.")
    # Make sure model is built

    if input_shape:
      refined_model, input_tensor_list = model_utils.modify_input_shape(
          refined_model, input_shape, calib_dataset=dataset)
      input_tensor_list = dataset
      refined_model.predict(input_tensor_list, batch_size=1, steps=1);

    if add_shape_info:
      logger.info("Start Getting Shape Information...")
      shape_info = model_utils.get_shape(
          model=refined_model, calib_dataset=dataset, input_shape=input_shape)
      if logger.debug_enabled():
        model_utils.save_shape_info(shape_info, './debug/')
        model_utils.save_model(refined_model, 'calibrated_model_add_shape.h5',
                               './debug/')
      logger.info("Getting Shape Information Done.")
    if input_shape:
      if isinstance(input_shape, list) or isinstance(input_shape,tuple):
        input_shape_slice = input_shape[1:]
      elif isinstance(input_shape, dict):
        for k,v in input_shape.items():
          input_shape_slice = v[1:]
          input_shape = v

      if input_shape[0] not in (None, 1):
        logger.warning(
            'the model is quantized, but compiler does not support batch_num not in [None, 1]. the input_shape is: ({})'
            .format(input_shape))

      if input_shape_slice.count(None) > 0 or input_shape_slice.count(-1) > 0:
        logger.warning(
            'the model is quantized, but compiler does not support input_shape ({}) with None.'
            .format(input_shape))

    return refined_model, layer_metadata


class VitisPof2SFinalizeTransformsPipeline(TransformsPipeline):
  """Vitis pof2s model finalize transformations pipeline."""

  def apply(self, refined_model, candidate_layers, layer_metadata):
    """Implement vitis model refine transforms.

    Args:
      refined_model: the refined model to be finalized.

    Returns:
      (Finalized quantized model.)
    """
    finalized_model = refined_model
    configs = self.get_configs()

    # Convert model format
    if configs['output_format'] == '':
      logger.debug(
          "No `output_format` found, skip model format conversion and output.")
      return finalized_model, layer_metadata

    formats = {'h5': '.h5', 'tf': '', 'onnx': '.onnx'}
    if configs['output_format'] not in formats:
      logger.error(
          "Invalid output_format: {}, supported output_format are: {}".format(
              configs['output_format'], list(formats.keys())))

    finalized_model_name = 'quantized_model'
    if configs['output_format'] == 'onnx':
      onnx_opset_version = configs['onnx_opset_version']
      model_utils.convert_to_onnx(finalized_model, configs['output_dir'],
                                  finalized_model_name, onnx_opset_version)
    else:
      filepath = os.path.join(
          configs['output_dir'],
          'quantized_model' + formats[configs['output_format']])
      finalized_model.save(filepath, save_format=configs['output_format'])

    return finalized_model, layer_metadata
