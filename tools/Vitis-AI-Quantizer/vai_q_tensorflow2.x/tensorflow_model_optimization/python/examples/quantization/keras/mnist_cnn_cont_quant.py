# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Train a simple convnet on the MNISt dataset.

Only the first layer has quantization annotation and quantized trained. A
representative dataset is set to invoke the post-training quantization as well.
The model should be fully quantized at the end.
"""
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

from tensorflow_model_optimization.python.core.quantization.keras import quantize

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

if tf.keras.backend.image_data_format() == 'channels_first':
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

batch_input_shape = (1,) + input_shape

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

l = tf.keras.layers

keras_file = '/tmp/quantized_mnist.h5'
if not os.path.exists(keras_file):
  model = tf.keras.Sequential([
      # Only the fisrt layer is quantized trained.
      # The rest of the layers are not quantization-aware.
      quantize.quantize_annotate_layer(
          l.Conv2D(
              32, 5, padding='same', activation='relu',
              input_shape=input_shape)),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.Conv2D(64, 5, padding='same', activation='relu'),
      l.BatchNormalization(),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.Flatten(),
      l.Dense(1024, activation='relu'),
      l.Dropout(0.4),
      l.Dense(num_classes),
      l.Softmax(),
  ])
  model = quantize.quantize_apply(model)
  model.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer=tf.keras.optimizers.Adadelta(),
      metrics=['accuracy'])

  model.fit(
      x_train,
      y_train,
      batch_size=batch_size,
      epochs=epochs,
      verbose=1,
      validation_data=(x_test, y_test))

  # Export to Keras.
  tf.keras.models.save_model(model, keras_file)

with quantize.quantize_scope():
  model = tf.keras.models.load_model(keras_file)

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Use the first 300 images in the post-training quantization.
def calibration_gen():
  for i in range(300):
    image = x_train[i].reshape(batch_input_shape)
    yield [image]

# Convert to TFLite model.
with quantize.quantize_scope():
  # It is complex to set the flags with converter v1:
  #
  #  converter = tf.lite.TFLiteConverter.from_keras_model_file(
  #      keras_file, input_shapes={'quant_conv2d_input': batch_input_shape})
  #
  # Must set the inference_input_type to float, so we can still use the floating
  # point training data. Set the inference_type to int8, to partially quantize
  # the model.
  # converter.inference_type = tf.lite.constants.INT8
  # converter.inference_input_type = tf.lite.constants.FLOAT
  # input_arrays = converter.get_input_arrays()
  # print(input_arrays)
  # converter.quantized_input_stats = {
  #     input_arrays[0]: (-128., 255.)
  # }  # mean, std_dev values for float [0, 1] quantized to [-128, 127]
  # Set the representative dataset for post-training quantization.

  model = tf.keras.models.load_model(keras_file)
  converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.representative_dataset = calibration_gen
converter._experimental_new_quantizer = True  # pylint: disable=protected-access
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]  # to enable post-training quantization with the representative dataset

print('Convert TFLite model.')
tflite_model = converter.convert()
print('Write TFLite model.')
tflite_file = '/tmp/quantized_mnist.tflite'
open(tflite_file, 'wb').write(tflite_model)

# Evaluate the fully quantized model.
interpreter = tf.lite.Interpreter(model_path=tflite_file)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

total_seen = 0
num_correct = 0

# Testing the entire dataset is too slow. Verifying only 300 of 10k samples.
print('Evaluate TFLite model.')
x_test = x_test[0:300, :]
y_test = y_test[0:300, :]
for img, label in zip(x_test, y_test):
  inp = img.reshape(batch_input_shape)
  total_seen += 1
  interpreter.set_tensor(input_index, inp)
  interpreter.invoke()
  predictions = interpreter.get_tensor(output_index)
  if np.argmax(predictions) == np.argmax(label):
    num_correct += 1

quantized_score = float(num_correct) / float(total_seen)
print('Quantized accuracy:', quantized_score)

# Ensure accuracy for quantized TF and TFLite models are similar to original
# model. There is no clear way to measure quantization, but for MNIST
# results which differ a lot likely suggest an error in quantization.
np.testing.assert_allclose(score[1], quantized_score, rtol=0.2, atol=0.2)

