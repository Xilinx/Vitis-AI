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
"""Functional test which fully trains quantized models and verifies accuracy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

# TODO(b/139939526): move to public API.
from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.keras import compat
from tensorflow_model_optimization.python.core.keras.testing import test_utils_mnist
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow_model_optimization.python.core.quantization.keras import utils as test_utils

layers = tf.keras.layers


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class QuantizeFunctionalTest(tf.test.TestCase, parameterized.TestCase):

  # TODO(pulkitb): Parameterize test and include functional mnist, and
  # other RNN models.
  def testQuantizesMnist(self):
    if not compat.is_v1_apis():
      return

    model = test_utils_mnist.sequential_model()
    x_train, y_train, x_test, y_test = test_utils_mnist.preprocessed_data()

    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=500)
    _, model_accuracy = model.evaluate(x_test, y_test, verbose=0)

    quantized_model = quantize.quantize_model(model)
    quantized_model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    quantized_model.fit(x_train, y_train, batch_size=500)
    _, quantized_model_accuracy = quantized_model.evaluate(
        x_test, y_test, verbose=0)

    self.assertGreater(quantized_model_accuracy, 0.6)

    _, quantized_tflite_file = tempfile.mkstemp('.tflite')

    with quantize.quantize_scope():
      test_utils.convert_keras_to_tflite(
          model=quantized_model,
          output_path=quantized_tflite_file,
          is_quantized=True)
    quantized_model_tflite_accuracy = test_utils_mnist.eval_tflite(
        quantized_tflite_file)

    # Ensure accuracy for quantized TF and TFLite models are similar to original
    # model. There is no clear way to measure quantization, but for MNIST
    # results which differ a lot likely suggest an error in quantization.
    self.assertAllClose(
        model_accuracy, quantized_model_accuracy,
        rtol=0.2, atol=0.2)
    self.assertAllClose(
        quantized_model_accuracy, quantized_model_tflite_accuracy,
        rtol=0.2, atol=0.2)


# Set of tests to determine what we can include in the allowlisted layers
# for the default API.
#
# TFLite in TF 2.X currently does not support creation of full-integer models.
# However, having every layer pass these tests ensures that the resulting
# quantization-aware trained model will have a path to deployment once
# TFLite adds support.
#
# Note these tests are not perfect yet.
# 1. Some Keras layers use different
# TensorFlow ops depending on the initialization parameters. This
# tests the most noticable ones, but unlikely all.
#
# TODO(tfmot): merge with test class above when run_all_keras_modes works
# with V1.
class QuantizeFullIntegerModelTest(tf.test.TestCase, parameterized.TestCase):

  _LAYER_PARAMS = [
      (layers.ReLU, {}),
      (layers.Softmax, {}),
      (layers.Conv1D, {
          'input_shape': (3, 6),
          'filters': 4,
          'kernel_size': 2,
      }),
      (layers.Conv2D, {
          'input_shape': (4, 6, 1),
          'filters': 4,
          'kernel_size': (2, 2)
      }),
      (layers.Conv3D, {
          'input_shape': (3, 4, 6, 1),
          'filters': 4,
          'kernel_size': (2, 2, 2)
      }),
      (layers.Conv2DTranspose, {
          'input_shape': (4, 6, 1),
          'filters': 4,
          'kernel_size': (2, 2)
      }),
      (layers.Conv3DTranspose, {
          'input_shape': (3, 4, 6, 1),
          'filters': 4,
          'kernel_size': (2, 2, 2)
      }),
      (layers.Cropping1D, {
          'input_shape': (3, 6),
      }),
      (layers.Cropping2D, {
          'input_shape': (4, 6, 1),
      }),
      (layers.Cropping3D, {
          'input_shape': (3, 4, 6, 1),
      }),
      (layers.UpSampling1D, {
          'input_shape': (3, 6)
      }),
      (layers.UpSampling2D, {
          'input_shape': (4, 6, 1),
      }),
      (layers.UpSampling3D, {
          'input_shape': (4, 6, 1),
      }),
      (layers.ZeroPadding1D, {
          'input_shape': (3, 6),
      }),
      (layers.ZeroPadding2D, {
          'input_shape': (4, 6, 1),
      }),
      (layers.ZeroPadding3D, {
          'input_shape': (3, 4, 6, 1),
      }),
      (layers.ActivityRegularization, {}),
      (layers.Dense, {
          'units': 2
      }),
      (layers.Dropout, {
          'rate': 0.2
      }),
      (layers.Flatten, {}),
      (layers.Masking, {}),
      (layers.Permute, {
          'input_shape': (10, 64),
          'dims': (2, 1)
      }),
      (layers.RepeatVector, {
          'n': 3
      }),
      (layers.Reshape, {
          'target_shape': [5, 1, 1]
      }),
      (layers.SpatialDropout1D, {
          'input_shape': (3, 6),
          'rate': 0.2,
      }),
      (layers.SpatialDropout2D, {
          'input_shape': (4, 6, 1),
          'rate': 0.2,
      }),
      (layers.SpatialDropout3D, {
          'input_shape': (3, 4, 6, 1),
          'rate': 0.2,
      }),
      (layers.AveragePooling1D, {
          'input_shape': (3, 6),
      }),
      (layers.AveragePooling2D, {
          'input_shape': (4, 6, 1),
      }),
      (layers.AveragePooling3D, {
          'input_shape': (3, 4, 6, 1),
      }),
      (layers.GlobalAveragePooling1D, {
          'input_shape': (3, 6),
      }),
      (layers.GlobalAveragePooling2D, {
          'input_shape': (4, 6, 1),
      }),
      (layers.GlobalAveragePooling3D, {
          'input_shape': (3, 4, 6, 1),
      }),
      (layers.GlobalMaxPooling1D, {
          'input_shape': (3, 6),
      }),
      (layers.GlobalMaxPooling2D, {
          'input_shape': (4, 6, 1),
      }),
      (layers.GlobalMaxPooling3D, {
          'input_shape': (3, 4, 6, 1),
      }),
      (layers.MaxPooling1D, {
          'input_shape': (3, 6),
      }),
      (layers.MaxPooling2D, {
          'input_shape': (4, 6, 1),
      }),
      (layers.MaxPooling3D, {
          'input_shape': (3, 4, 6, 1),
      }),
      # LocallyConnected1D implementations use significantly different TF
      # operations underneath, so they should be all tested.
      (layers.LocallyConnected1D, {
          'input_shape': (3, 6),
          'implementation': 1,
          'filters': 4,
          'kernel_size': 2
      }),
      (layers.LocallyConnected1D, {
          'input_shape': (3, 6),
          'implementation': 2,
          'filters': 4,
          'kernel_size': 2
      }),
      (layers.LocallyConnected1D, {
          'input_shape': (3, 6),
          'implementation': 3,
          'filters': 4,
          'kernel_size': 2
      }),
      (layers.LocallyConnected2D, {
          'input_shape': (4, 6, 1),
          'implementation': 1,
          'filters': 4,
          'kernel_size': (2, 2)
      }),
      (layers.LocallyConnected2D, {
          'input_shape': (4, 6, 1),
          'implementation': 2,
          'filters': 4,
          'kernel_size': (2, 2)
      }),
      (layers.LocallyConnected2D, {
          'input_shape': (4, 6, 1),
          'implementation': 3,
          'filters': 4,
          'kernel_size': (2, 2)
      }),
  ]

  # pylint: disable=g-complex-comprehension,undefined-variable

  @parameterized.parameters([
      l for l in _LAYER_PARAMS if l[0] not in [
          # Not done since TFLite converter doesn't support in TF2 yet.
          layers.UpSampling2D,
          layers.Conv3D,
          layers.Conv3DTranspose,
          layers.AveragePooling3D,
          layers.MaxPooling3D,
          layers.LocallyConnected1D,
          layers.LocallyConnected2D,
          # Not done since TFLite inference doesn't support yet.
          layers.ZeroPadding3D,  # Does not support 5D inputs yet.
          # Not done because converter transforms graph until there are
          # zero ops, and then an error is thrown because it cannot handle
          # zero op graphs.
          layers.ActivityRegularization,
          layers.Dropout,
          layers.Flatten,
          layers.SpatialDropout1D,
          layers.SpatialDropout2D,
          layers.SpatialDropout3D,
          # Not done since there are float tensors besides
          # the inputs and outputs (e.g. FakeQuant not placed in
          # all areas or converter support not there).
          layers.Masking,
          layers.RepeatVector,
          layers.MaxPooling1D,
          layers.UpSampling1D,
          layers.UpSampling3D,
          # Not done since not registered since not per-axis yet.
          layers.Conv1D,
          layers.Conv2DTranspose,
      ]
  ])
  def testQuantizeSingleLayer_ProducesFullIntegerModel_TF2(
      self, layer_type, kwargs):
    # "FullInteger" in the sense that ignores inputs and outputs.
    if compat.is_v1_apis():
      return

    if 'input_shape' not in kwargs:
      kwargs['input_shape'] = (5,)

    layer = layer_type(**kwargs)
    model = tf.keras.Sequential([layer])
    quantized_model = quantize.quantize_model(model)

    _, quantized_tflite_file = tempfile.mkstemp('.tflite')

    with quantize.quantize_scope():
      test_utils.convert_keras_to_tflite(
          model=quantized_model,
          output_path=quantized_tflite_file,
          is_quantized=True,
          input_quant_params=(0., 1.),
          experimental_new_converter=True)

    interpreter = tf.lite.Interpreter(model_path=quantized_tflite_file)
    interpreter.allocate_tensors()

    input_tensor_details = interpreter.get_input_details()
    self.assertEqual(input_tensor_details[0]['dtype'], np.float32)

    output_tensor_details = interpreter.get_output_details()
    self.assertEqual(output_tensor_details[0]['dtype'], np.float32)

    tensor_details = interpreter.get_tensor_details()
    float_tensor_details = [
        t for t in tensor_details if t['dtype'] == np.float32
    ]
    # Only the input and outputs are float. The rest are integer.
    #
    # TODO(tfmot): update this test to use the full-integer path when available,
    # so that float_tensor_details should be length 0.
    self.assertLen(float_tensor_details, 2)

  # This unit test runs in TF1. While we don't publicly support this path in
  # the Keras tooling, this is useful for two reasons:
  # 1. TOCO has better debugging functionality than MLIR, for incrementally
  # adding new layers.
  # 2. It's useful to track supported layers in TF1 converter in case we
  # want to eventually support V1 conversion.
  # 3. This also tracks more layers where FakeQuant placement is incorrect,
  # given that the TF2 converter doesn't support all layers that TF1 did.
  @parameterized.parameters([
      l for l in _LAYER_PARAMS if l[0] not in [
          # Not done since per-channel not supported in TF1 without MLIR.
          # By temporarily switching layers to be per-tensor instead of
          # per-channel, some minimum testing can be done.
          #
          # TODO(tfmot): add Conv1D/Conv3D/Conv with Transpose after they
          # are made per-channel by quantization scheme.
          layers.Conv2D,
          # Not done since FakeQuants are not placed in right areas or
          # converter doesn't handle it properly yet.
          layers.Conv3D,
          layers.Conv3DTranspose,
          layers.Masking,
          layers.LocallyConnected1D,
          # TODO(tfmot): find reason.
          layers.LocallyConnected2D,
          # Not done because TF1 converter doesn't support quantized op.
          layers.AveragePooling3D,
          layers.MaxPooling3D,
          # Not done because TF1 converter transforms graph until there are
          # zero ops, and then an error is thrown because it cannot handle
          # zero op graphs.
          layers.ActivityRegularization,
          layers.Dropout,
          layers.Flatten,
          layers.SpatialDropout1D,
          layers.SpatialDropout2D,
          layers.SpatialDropout3D,
          # Note done because not support in TF2, so we had disabled it.
          # Works fine in TF1 TOCO.
          layers.MaxPooling1D,
          layers.UpSampling3D,
          layers.RepeatVector,
          layers.ZeroPadding3D,
          layers.Conv1D,
          layers.Conv2DTranspose,
          layers.UpSampling1D,
          layers.UpSampling2D,
      ]
  ])
  def testQuantizeSingleLayer_ProducesFullIntegerModel_TF1(
      self, layer_type, kwargs):
    if not compat.is_v1_apis():
      return

    if 'input_shape' not in kwargs:
      kwargs['input_shape'] = (5,)

    layer = layer_type(**kwargs)
    model = tf.keras.Sequential([layer])
    quantized_model = quantize.quantize_model(model)

    with quantize.quantize_scope():
      test_utils.convert_keras_to_tflite(
          model=quantized_model,
          output_path=None,
          is_quantized=True,
          inference_type=tf.uint8,
          inference_input_type=tf.uint8,
          input_quant_params=(0., 1.),
          # Set to False to throw errors when FakeQuants are
          # not placed everywhere to create full-integer model. Errors
          # are not thrown when set to True.
          experimental_new_converter=False)

  # pylint: enable=g-complex-comprehension,undefined-variable


if __name__ == '__main__':
  tf.test.main()
