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
# pylint: disable=protected-access
"""Quantization specific utilities for generating, saving, testing, and evaluating models."""

import tempfile

import tensorflow as tf

from tensorflow_model_optimization.python.core.keras import compat


def convert_keras_to_tflite(model,
                            output_path,
                            custom_objects=None,
                            is_quantized=True,
                            inference_type=None,
                            inference_input_type=None,
                            input_quant_params=(-128., 255.),
                            experimental_new_converter=True):
  """Convert Keras model to TFLite."""
  if custom_objects is None:
    custom_objects = {}

  if not compat.is_v1_apis():
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
  else:
    _, keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model, keras_file)
    converter = tf.lite.TFLiteConverter.from_keras_model_file(
        keras_file, custom_objects=custom_objects)

  converter.experimental_new_converter = experimental_new_converter

  if is_quantized:
    if not compat.is_v1_apis():
      converter.optimizations = [tf.lite.Optimize.DEFAULT]
    else:
      converter.inference_type = tf.lite.constants.INT8
      converter.inference_input_type = tf.lite.constants.FLOAT
      # TODO(tfmot): should be able to make everything use the
      # same inference_type in TF 1.X tests.
      if inference_type:
        converter.inference_type = inference_type
      if inference_input_type:
        converter.inference_input_type = inference_input_type

      input_arrays = converter.get_input_arrays()
      converter.quantized_input_stats = {
          input_arrays[0]: input_quant_params
      }  # mean, std_dev values for float to quantized int8 values.

  tflite_model = converter.convert()

  if output_path is not None:
    with open(output_path, 'wb') as f:
      f.write(tflite_model)

  return tflite_model
