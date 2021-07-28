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
"""Correct bias for quantized tf.keras models."""

import tensorflow as tf
import numpy as np

from tensorflow.python.keras.engine import data_adapter
from tensorflow_model_optimization.python.core.quantization.keras.vitis.common import vitis_quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.vitis.utils import common_utils

keras = tf.keras
logger = common_utils.VAILogger


def _get_layer_output(model, layer_name, input_data, batch_size, steps):
  """Get the predict result of layer's ouput."""
  target_layer = model.get_layer(layer_name)
  layer_model = tf.keras.Model(inputs=model.input, outputs=target_layer.output)
  return layer_model.predict(input_data, batch_size, steps)


def _has_bias(layer):
  conv_like_layers = (keras.layers.Conv2D, keras.layers.DepthwiseConv2D,
                      keras.layers.Conv2DTranspose, keras.layers.Dense)
  if isinstance(layer, conv_like_layers):
    return layer.use_bias
  else:
    return False


def bias_correction(quant_model, float_model, calib_dataset, calib_batch_size,
                    calib_steps):
  """Bias correction."""

  target_layers = []
  for layer in quant_model.layers:
    if isinstance(
        layer,
        vitis_quantize_wrapper.QuantizeWrapper) and layer.trainable_weights:
      if _has_bias(layer.layer):
        target_layers.append(layer)

  # Main loop
  for i, layer in enumerate(target_layers):
    logger.info("Bias Correction ({}/{}): {}".format(i, len(target_layers),
                                                     layer.layer.name))

    f_outputs = _get_layer_output(float_model, layer.layer.name, calib_dataset,
                                  calib_batch_size, calib_steps)
    q_outputs = _get_layer_output(quant_model, layer.name, calib_dataset,
                                  calib_batch_size, calib_steps)

    n_ch = f_outputs.shape[-1]
    f_means = f_outputs.reshape(-1, n_ch).mean(0)
    q_means = q_outputs.reshape(-1, n_ch).mean(0)
    diff_means = q_means - f_means
    if diff_means.mean() > 10:
      import pdb; pdb.set_trace()

    q_weights = layer.get_weights()
    q_weights[1] += diff_means

    layer.set_weights(q_weights)

    q_outputs_after = _get_layer_output(quant_model, layer.name, calib_dataset,
                                        calib_batch_size, calib_steps)
    q_means_after = q_outputs_after.reshape(-1, n_ch).mean(0)
    diff_means_after = q_means - f_means
    print('diff means {} -> {}'.format(diff_means.mean(), diff_means_after.mean()))
  return
