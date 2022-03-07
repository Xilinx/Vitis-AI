# Copyright 2021 Xilinx Inc.
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

from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow as tf

num_classes = 10
K.set_image_data_format('channels_first')
# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

def build_model():
  if K.image_data_format() == 'channels_first':
    inputs = tf.keras.Input(shape=(1, img_rows, img_cols))  # Returns a placeholder tensor
  else:
    inputs = tf.keras.Input(shape=(img_rows, img_cols, 1))  # Returns a placeholder tensor

  # A layer instance is callable on a tensor, and returns a tensor.
  conv1 = layers.Conv2D(filters=32,
                        kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape)(inputs)
  conv2 = layers.Conv2D(64, (3, 3), activation='relu')(conv1)
  conv3 = layers.Conv2D(64, (3, 3), strides=(2,2), activation='relu')(conv2)
  dropout1 = layers.Dropout(0.25)(conv3)
  flat = layers.Flatten()(dropout1)
  dense = layers.Dense(128, activation='relu')(flat)
  dropout2 = layers.Dropout(0.5)(dense)
  predictions = layers.Dense(num_classes, activation='softmax')(dropout2)
  return tf.keras.Model(inputs=inputs, outputs=predictions)

def evaluate(ckpt_path):
  model = build_model()
  model.load_weights(ckpt_path)
  model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  res = model.evaluate(x_test, y_test, verbose=0)
  return {'acc5': res[1]}
