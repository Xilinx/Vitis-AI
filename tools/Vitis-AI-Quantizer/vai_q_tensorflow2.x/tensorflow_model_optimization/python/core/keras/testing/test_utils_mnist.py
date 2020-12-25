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
"""Utils for testing MOT code with MNIST model/dataset."""

import numpy as np
import tensorflow as tf

from tensorflow.python import keras
l = keras.layers


def layers_list():
  return [
      l.Conv2D(32, 5, padding='same', activation='relu',
               input_shape=input_shape()),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      # TODO(pulkitb): Add BatchNorm when transformations are ready.
      # l.BatchNormalization(),
      l.Conv2D(64, 5, padding='same', activation='relu'),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.Flatten(),
      l.Dense(1024, activation='relu'),
      l.Dropout(0.4),
      l.Dense(10, activation='softmax')
  ]


def sequential_model():
  return keras.Sequential(layers_list())


def functional_model():
  """Builds an MNIST functional model."""
  inp = keras.Input(input_shape())
  x = l.Conv2D(32, 5, padding='same', activation='relu')(inp)
  x = l.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
  # TODO(pulkitb): Add BatchNorm when transformations are ready.
  # x = l.BatchNormalization()(x)
  x = l.Conv2D(64, 5, padding='same', activation='relu')(x)
  x = l.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
  x = l.Flatten()(x)
  x = l.Dense(1024, activation='relu')(x)
  x = l.Dropout(0.4)(x)
  out = l.Dense(10, activation='softmax')(x)

  return keras.models.Model([inp], [out])


def input_shape(img_rows=28, img_cols=28):
  if tf.keras.backend.image_data_format() == 'channels_first':
    return 1, img_rows, img_cols
  else:
    return img_rows, img_cols, 1


def preprocessed_data(img_rows=28,
                      img_cols=28,
                      num_classes=10):
  """Get data for mnist training and evaluation."""
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255

  # convert class vectors to binary class matrices
  y_train = tf.keras.utils.to_categorical(y_train, num_classes)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes)

  return x_train, y_train, x_test, y_test


def eval_tflite(model_path):
  """Evaluate mnist in TFLite for accuracy."""
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
  input_index = interpreter.get_input_details()[0]['index']
  output_index = interpreter.get_output_details()[0]['index']

  _, _, x_test, y_test = preprocessed_data()
  # Testing the entire dataset is too slow. Verifying only 300 of 10k samples.
  x_test = x_test[0:300, :]
  y_test = y_test[0:300, :]

  total_seen = 0
  num_correct = 0

  for img, label in zip(x_test, y_test):
    batch_input_shape = (1,) + input_shape()
    inp = img.reshape(batch_input_shape)
    total_seen += 1
    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    if np.argmax(predictions) == np.argmax(label):
      num_correct += 1

  return float(num_correct) / float(total_seen)
