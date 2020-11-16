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
"""Test utilities for generating, saving, and evaluating models."""
# TODO(tf-mot): dedup and migrate to testing/ directory.

import numpy as np
import tensorflow as tf

l = tf.keras.layers


class ModelCompare(object):
  """Mixin with helper functions for model comparison.

  Needs to be used with tf.test.TestCase.

  Note the following test only trainable_weights.
  """

  def _assert_weights_same_objects(self, model1, model2):
    self.assertEqual(
        len(model1.trainable_weights), len(model2.trainable_weights))
    for w1, w2 in zip(model1.trainable_weights, model2.trainable_weights):
      self.assertEqual(id(w1), id(w2))

  def _assert_weights_different_objects(self, model1, model2):
    self.assertEqual(
        len(model1.trainable_weights), len(model2.trainable_weights))
    for w1, w2 in zip(model1.trainable_weights, model2.trainable_weights):
      self.assertNotEqual(id(w1), id(w2))

  def _assert_weights_same_values(self, model1, model2):
    self.assertEqual(
        len(model1.trainable_weights), len(model2.trainable_weights))

    model1_weights = tf.keras.backend.batch_get_value(model1.trainable_weights)
    model2_weights = tf.keras.backend.batch_get_value(model2.trainable_weights)
    for w1, w2 in zip(model1_weights, model2_weights):
      self.assertAllClose(w1, w2)

  def _assert_weights_different_values(self, model1, model2):
    self.assertEqual(
        len(model1.trainable_weights), len(model2.trainable_weights))

    model1_weights = tf.keras.backend.batch_get_value(model1.trainable_weights)
    model2_weights = tf.keras.backend.batch_get_value(model2.trainable_weights)
    for w1, w2 in zip(model1_weights, model2_weights):
      self.assertNotAllClose(w1, w2)


def build_simple_dense_model():
  return tf.keras.Sequential([
      l.Dense(8, activation='relu', input_shape=(10,)),
      l.Dense(5, activation='softmax')
  ])


def get_preprocessed_mnist_data(img_rows=28,
                                img_cols=28,
                                num_classes=10,
                                is_quantized_model=False):
  """Get data for mnist training and evaluation."""
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
  else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

  if not is_quantized_model:
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

  # convert class vectors to binary class matrices
  y_train = tf.keras.utils.to_categorical(y_train, num_classes)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes)

  return (x_train, y_train), (x_test, y_test), input_shape


def eval_mnist_tflite(model_path=None, model_content=None, is_quantized=False):
  """Evaluate mnist in TFLite for accuracy."""
  interpreter = tf.lite.Interpreter(
      model_path=model_path, model_content=model_content)
  interpreter.allocate_tensors()
  input_index = interpreter.get_input_details()[0]['index']
  output_index = interpreter.get_output_details()[0]['index']

  _, test_data, _ = get_preprocessed_mnist_data(is_quantized_model=is_quantized)
  x_test, y_test = test_data

  total_seen = 0
  num_correct = 0

  for img, label in zip(x_test, y_test):
    inp = img.reshape((1, 28, 28, 1))
    total_seen += 1
    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    if np.argmax(predictions) == np.argmax(label):
      num_correct += 1

    if total_seen % 1000 == 0:
      print('Accuracy after %i images: %f' %
            (total_seen, float(num_correct) / float(total_seen)))

  return float(num_correct) / float(total_seen)
