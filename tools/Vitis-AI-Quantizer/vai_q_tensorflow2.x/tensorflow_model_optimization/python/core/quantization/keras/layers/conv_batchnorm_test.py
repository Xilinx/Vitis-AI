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
"""ConvBatchNorm layer tests.

See FoldedBatchNormTest for shared test cases between different
classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import numpy as np
from six.moves import range
import tensorflow as tf

from tensorflow_model_optimization.python.core.keras import compat
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras import utils
from tensorflow_model_optimization.python.core.quantization.keras.layers import conv_batchnorm
from tensorflow_model_optimization.python.core.quantization.keras.layers import conv_batchnorm_test_utils

activations = tf.keras.activations
keras = tf.keras

_ConvBatchNorm2D = conv_batchnorm._ConvBatchNorm2D
_DepthwiseConvBatchNorm2D = conv_batchnorm._DepthwiseConvBatchNorm2D
Conv2DModel = conv_batchnorm_test_utils.Conv2DModel
DepthwiseConv2DModel = conv_batchnorm_test_utils.DepthwiseConv2DModel


class FoldedBatchNormTestBase(tf.test.TestCase):

  @staticmethod
  def _get_asymmetric_quant_params(real_min, real_max, quant_min, quant_max):
    # TODO(alanchiao): remove this once the converter for training-time
    # quantization supports producing a TFLite model with a float input/output.

    # Code clones quantization logic from TFLite.
    # third_party/tensorflow/lite/tools/optimize/quantization_utils.cc

    real_min = min(real_min, 0.0)
    real_max = max(real_max, 0.0)

    scale = (real_max - real_min) / (quant_max - quant_min)

    zero_point_from_min = quant_min
    if scale != 0:
      zero_point_from_min = quant_min - real_min / scale

    if zero_point_from_min < quant_min:
      zero_point = quant_min
    elif zero_point_from_min > quant_max:
      zero_point = quant_max
    else:
      zero_point = round(zero_point_from_min)

    return scale, zero_point

  # This does a basic serialize/deserialize test since deserialization
  # occurs during conversion to TFLite.
  def _test_equal_tf_and_tflite_outputs(self,
                                        tf_model,
                                        is_tflite_quantized=False):
    _, tflite_file = tempfile.mkstemp('.tflite')

    batched_input_shape = self._get_batched_input_shape()
    output_shape = self._get_output_shape()

    tf_model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    tf_model.fit(
        np.random.uniform(0, 1, size=batched_input_shape),
        np.random.uniform(0, 10, size=output_shape),
        epochs=1,
        callbacks=[])
    # Prepare for inference.
    inp = np.random.uniform(0, 1, size=batched_input_shape)
    inp = inp.astype(np.float32)

    if is_tflite_quantized:
      real_min = keras.backend.eval(tf_model.layers[-1]._activation_min_var)
      real_max = keras.backend.eval(tf_model.layers[-1]._activation_max_var)
      scale, zero_point = self._get_asymmetric_quant_params(
          real_min, real_max, -128.0, 127.0)

      # TFLite input needs to be quantized.
      real_input_min = 0.0
      real_input_max = 1.0
      inp_scale, inp_zp = self._get_asymmetric_quant_params(
          real_input_min, real_input_max, -128.0, 127.0)

      inp8 = np.round(inp / inp_scale + inp_zp)
      inp8 = inp8.astype(np.int8)

      # Dequant
      inp = (inp8.astype(np.float32) - inp_zp) * inp_scale

    # TensorFlow inference.
    tf_out = tf_model.predict(inp)

    # TensorFlow Lite inference.
    with quantize.quantize_scope():
      utils.convert_keras_to_tflite(
          tf_model,
          tflite_file,
          custom_objects={
              '_ConvBatchNorm2D': _ConvBatchNorm2D,
              '_DepthwiseConvBatchNorm2D': _DepthwiseConvBatchNorm2D,
          },
          is_quantized=is_tflite_quantized,
          inference_input_type=tf.lite.constants.INT8)

    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    if is_tflite_quantized:
      interpreter.set_tensor(input_index, inp8)
    else:
      interpreter.set_tensor(input_index, inp)

    interpreter.invoke()
    tflite_out = interpreter.get_tensor(output_index)

    if is_tflite_quantized:
      # dequantize outputs
      tflite_out = [scale * (x - zero_point) for x in tflite_out]

      # TODO(pulkitb): DConv quantized test somehow has a single value (0.065%)
      # of total values, which falls off by 1 scale. Investigate further and
      # introduce stricter testing by removing atol=scale.
      self.assertAllClose(tf_out, tflite_out, atol=scale)
    else:
      # Taken from testFoldFusedBatchNorms from
      # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference_test.py#L230
      self.assertAllClose(tf_out, tflite_out, rtol=1e-04, atol=1e-06)

  def _test_equal_outputs(self, model, model2):
    for _ in range(2):
      inp = np.random.uniform(0, 10, size=self._get_batched_input_shape())
      model_out = model.predict(inp)
      model2_out = model2.predict(inp)

      # Taken from testFoldFusedBatchNorms from
      # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference_test.py#L230
      self.assertAllClose(model_out, model2_out, rtol=1e-04, atol=1e-06)


class ConvBatchNorm2DTest(FoldedBatchNormTestBase):

  def _get_folded_batchnorm_model(self,
                                  is_quantized=False,
                                  post_bn_activation=None):
    return Conv2DModel.get_folded_batchnorm_model(
        is_quantized=is_quantized, post_bn_activation=post_bn_activation)

  def _get_nonfolded_batchnorm_model(self):
    return Conv2DModel.get_nonfolded_batchnorm_model()

  def _get_batched_input_shape(self):
    return Conv2DModel.get_batched_input_shape()

  def _get_output_shape(self):
    return Conv2DModel.get_output_shape()

  def testEquivalentToNonFoldedBatchNorm(self):
    self._test_equal_outputs(
        self._get_folded_batchnorm_model(is_quantized=False),
        self._get_nonfolded_batchnorm_model())

  def testEquivalentToFloatTFLite(self):
    if not compat.is_v1_apis():
      return

    tf_model = self._get_folded_batchnorm_model(is_quantized=False)
    self._test_equal_tf_and_tflite_outputs(tf_model)

  def testQuantizedEquivalentToQuantizedTFLite(self):
    if not compat.is_v1_apis():
      return

    tf_model = self._get_folded_batchnorm_model(is_quantized=True)
    self._test_equal_tf_and_tflite_outputs(tf_model, is_tflite_quantized=True)

  # TODO(pulkitb): Implement FakeQuant addition for keras Input layers.
  # That will remove the need to do Int8 tests for TFLite, and push input
  # quantization into the kernels, and remove the need for quantized_input_stats

  # TODO(pulkitb): Enable tests once TFLite converter supports new spec.
  # TFLite Converter does not support quantizing/de-quantizing based on
  # per-channel FakeQuants.
  #
  # def testQuantizedEquivalentToFloatTFLite(self):
  #   tf_model = self._get_folded_batchnorm_model(is_quantized=True)
  #   self._test_equal_tf_and_tflite_outputs(tf_model)
  #
  # def testQuantizedWithReLUEquivalentToFloatTFLite(self):
  #   tf_model = self._get_folded_batchnorm_model(
  #       is_quantized=True, post_bn_activation=activations.get('relu'))
  #   self._test_equal_tf_and_tflite_outputs(tf_model)
  #
  # def testQuantizedWithAdvancedReLUEquivalentToFloatTFLite(self):
  #   tf_model = self._get_folded_batchnorm_model(
  #       is_quantized=True,
  #       post_bn_activation=keras.layers.ReLU(max_value=6.0))
  #   self._test_equal_tf_and_tflite_outputs(tf_model)
  #
  # def testQuantizedWithSoftmaxEquivalentToFloatTfLite(self):
  #   tf_model = self._get_folded_batchnorm_model(
  #       is_quantized=True, post_bn_activation=activations.get('softmax'))
  #   self._test_equal_tf_and_tflite_outputs(tf_model)


class DepthwiseConvBatchNorm2DTest(FoldedBatchNormTestBase):

  def _get_folded_batchnorm_model(self,
                                  is_quantized=False,
                                  post_bn_activation=None):
    return DepthwiseConv2DModel.get_folded_batchnorm_model(
        is_quantized=is_quantized, post_bn_activation=post_bn_activation)

  def _get_nonfolded_batchnorm_model(self):
    return DepthwiseConv2DModel.get_nonfolded_batchnorm_model()

  def _get_batched_input_shape(self):
    return DepthwiseConv2DModel.get_batched_input_shape()

  def _get_output_shape(self):
    return DepthwiseConv2DModel.get_output_shape()

  def testEquivalentToNonFoldedBatchNorm(self):
    self._test_equal_outputs(
        self._get_folded_batchnorm_model(is_quantized=False),
        self._get_nonfolded_batchnorm_model())

  def testEquivalentToFloatTFLite(self):
    if not compat.is_v1_apis():
      return

    tf_model = self._get_folded_batchnorm_model(is_quantized=False)
    self._test_equal_tf_and_tflite_outputs(tf_model)

  def testQuantizedEquivalentToQuantizedTFLite(self):
    if not compat.is_v1_apis():
      return

    tf_model = self._get_folded_batchnorm_model(is_quantized=True)
    self._test_equal_tf_and_tflite_outputs(tf_model, is_tflite_quantized=True)

  # TODO(pulkitb): Enable tests once TFLite converter supports new spec.
  # TFLite Converter does not support quantizing/de-quantizing based on
  # per-channel FakeQuants.

  # def testQuantizedEquivalentToFloatTFLite(self):
  #   tf_model = self._get_folded_batchnorm_model(is_quantized=True)
  #   self._test_equal_tf_and_tflite_outputs(tf_model)
  #
  # def testQuantizedWithSoftmaxEquivalentToFloatTfLite(self):
  #   tf_model = self._get_folded_batchnorm_model(
  #       is_quantized=True, post_bn_activation=activations.get('softmax'))
  #   self._test_equal_tf_and_tflite_outputs(tf_model)
  #
  # def testQuantizedWithReLUEquivalentToFloatTFLite(self):
  #   tf_model = self._get_folded_batchnorm_model(
  #       is_quantized=True, post_bn_activation=activations.get('relu'))
  #   self._test_equal_tf_and_tflite_outputs(tf_model)
  #
  # def testQuantizedWithAdvancedReLUEquivalentToFloatTFLite(self):
  #   tf_model = self._get_folded_batchnorm_model(
  #       is_quantized=True,
  #       post_bn_activation=keras.layers.ReLU(max_value=6.0))
  #   self._test_equal_tf_and_tflite_outputs(tf_model)


if __name__ == '__main__':
  tf.test.main()
