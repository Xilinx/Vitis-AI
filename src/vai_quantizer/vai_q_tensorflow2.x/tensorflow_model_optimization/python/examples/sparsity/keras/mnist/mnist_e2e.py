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
# pylint: disable=missing-docstring,protected-access
"""Train a simple convnet on the MNIST dataset."""
from __future__ import print_function

from absl import app as absl_app
from absl import flags

import tensorflow as tf

from tensorflow_model_optimization.python.core.keras import test_utils as keras_test_utils
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

ConstantSparsity = pruning_schedule.ConstantSparsity
keras = tf.keras
l = keras.layers

FLAGS = flags.FLAGS

batch_size = 128
num_classes = 10
epochs = 1

flags.DEFINE_float('sparsity', '0.0', 'Target sparsity level.')


def build_layerwise_model(input_shape, **pruning_params):
  return tf.keras.Sequential([
      l.Conv2D(
          32, 5, padding='same', activation='relu', input_shape=input_shape),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.Conv2D(64, 5, padding='same'),
      l.BatchNormalization(),
      l.ReLU(),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.Flatten(),
      prune.prune_low_magnitude(
          l.Dense(1024, activation='relu'), **pruning_params),
      l.Dropout(0.4),
      prune.prune_low_magnitude(
          l.Dense(num_classes, activation='softmax'), **pruning_params)
  ])


def train(model, x_train, y_train, x_test, y_test):
  model.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer='adam',
      metrics=['accuracy'])

  # Print the model summary.
  model.summary()

  # Add a pruning step callback to peg the pruning step to the optimizer's
  # step. Also add a callback to add pruning summaries to tensorboard
  callbacks = [
      pruning_callbacks.UpdatePruningStep(),
      pruning_callbacks.PruningSummaries(log_dir='/tmp/logs')
  ]

  model.fit(
      x_train,
      y_train,
      batch_size=batch_size,
      epochs=epochs,
      verbose=1,
      callbacks=callbacks,
      validation_data=(x_test, y_test))
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  model = prune.strip_pruning(model)
  return model


def main(unused_argv):
  ##############################################################################
  # Prepare training and testing data
  ##############################################################################
  (x_train, y_train), (
      x_test,
      y_test), input_shape = keras_test_utils.get_preprocessed_mnist_data()

  ##############################################################################
  # Train and convert a model with 2x2 block config. There's no kernel in tflite
  # supporting this block configuration, so the sparse tensor is densified and
  # the model falls back to dense execution.
  ##############################################################################
  pruning_params = {
      'pruning_schedule':
          ConstantSparsity(FLAGS.sparsity, begin_step=0, frequency=100),
      'block_size': (2, 2)
  }

  model = build_layerwise_model(input_shape, **pruning_params)
  model = train(model, x_train, y_train, x_test, y_test)

  converter = tf.lite.TFLiteConverter.from_keras_model(model)

  # Get a dense model as baseline
  tflite_model_dense = converter.convert()
  tflite_model_path = '/tmp/dense_mnist.tflite'
  with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model_dense)

  # The _experimental_sparsify_model is to enable sparse tensor encoding,
  # otherwise the model is converted as dense.
  converter._experimental_sparsify_model = True

  tflite_model = converter.convert()

  # Check the model is compressed
  print('Compression ratio: ', len(tflite_model) / len(tflite_model_dense))

  tflite_model_path = '/tmp/sparse_mnist_%s_2x2.tflite' % FLAGS.sparsity
  with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

  print('evaluate 2x2 model')
  print(keras_test_utils.eval_mnist_tflite(model_content=tflite_model))

  ##############################################################################
  # Train and convert a model with 1x4 block config. There's kernel in tflite
  # with this block configuration, so the model can take advantage of sparse
  # execution and see inference speed-up.
  ##############################################################################
  pruning_params = {
      'pruning_schedule':
          ConstantSparsity(FLAGS.sparsity, begin_step=0, frequency=100),
      # TFLite transposes the weight during conversion, so we need to specify
      # the block as (4, 1) in the training API.
      'block_size': (4, 1)
  }

  model = build_layerwise_model(input_shape, **pruning_params)
  model = train(model, x_train, y_train, x_test, y_test)

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter._experimental_sparsify_model = True

  tflite_model = converter.convert()
  # Check the model is compressed
  print('Compression ratio: ', len(tflite_model) / len(tflite_model_dense))

  tflite_model_path = '/tmp/sparse_mnist_%s_1x4.tflite' % FLAGS.sparsity
  with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

  print('evaluate 1x4 model')
  print(keras_test_utils.eval_mnist_tflite(model_content=tflite_model))


if __name__ == '__main__':
  absl_app.run(main)
