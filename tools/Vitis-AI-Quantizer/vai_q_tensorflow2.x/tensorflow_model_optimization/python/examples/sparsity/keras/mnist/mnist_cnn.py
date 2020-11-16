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
# pylint: disable=missing-docstring
"""Train a simple convnet on the MNIST dataset."""
from __future__ import print_function

from absl import app as absl_app
from absl import flags

import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

ConstantSparsity = pruning_schedule.ConstantSparsity
keras = tf.keras
l = keras.layers

FLAGS = flags.FLAGS

batch_size = 128
num_classes = 10
epochs = 12

flags.DEFINE_string('output_dir', '/tmp/mnist_train/',
                    'Output directory to hold tensorboard events')


def build_sequential_model(input_shape):
  return tf.keras.Sequential([
      l.Conv2D(
          32, 5, padding='same', activation='relu', input_shape=input_shape),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.BatchNormalization(),
      l.Conv2D(64, 5, padding='same', activation='relu'),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.Flatten(),
      l.Dense(1024, activation='relu'),
      l.Dropout(0.4),
      l.Dense(num_classes, activation='softmax')
  ])


def build_functional_model(input_shape):
  inp = tf.keras.Input(shape=input_shape)
  x = l.Conv2D(32, 5, padding='same', activation='relu')(inp)
  x = l.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
  x = l.BatchNormalization()(x)
  x = l.Conv2D(64, 5, padding='same', activation='relu')(x)
  x = l.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
  x = l.Flatten()(x)
  x = l.Dense(1024, activation='relu')(x)
  x = l.Dropout(0.4)(x)
  out = l.Dense(num_classes, activation='softmax')(x)

  return tf.keras.models.Model([inp], [out])


def build_layerwise_model(input_shape, **pruning_params):
  return tf.keras.Sequential([
      prune.prune_low_magnitude(
          l.Conv2D(32, 5, padding='same', activation='relu'),
          input_shape=input_shape,
          **pruning_params),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.BatchNormalization(),
      prune.prune_low_magnitude(
          l.Conv2D(64, 5, padding='same', activation='relu'), **pruning_params),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.Flatten(),
      prune.prune_low_magnitude(
          l.Dense(1024, activation='relu'), **pruning_params),
      l.Dropout(0.4),
      prune.prune_low_magnitude(
          l.Dense(num_classes, activation='softmax'), **pruning_params)
  ])


def train_and_save(models, x_train, y_train, x_test, y_test):
  for model in models:
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
        pruning_callbacks.PruningSummaries(log_dir=FLAGS.output_dir)
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

    # Export and import the model. Check that accuracy persists.
    saved_model_dir = '/tmp/saved_model'
    print('Saving model to: ', saved_model_dir)
    tf.keras.models.save_model(model, saved_model_dir, save_format='tf')
    print('Loading model from: ', saved_model_dir)
    loaded_model = tf.keras.models.load_model(saved_model_dir)

    score = loaded_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def main(unused_argv):
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

  pruning_params = {
      'pruning_schedule': ConstantSparsity(0.75, begin_step=2000, frequency=100)
  }

  layerwise_model = build_layerwise_model(input_shape, **pruning_params)
  sequential_model = build_sequential_model(input_shape)
  sequential_model = prune.prune_low_magnitude(
      sequential_model, **pruning_params)
  functional_model = build_functional_model(input_shape)
  functional_model = prune.prune_low_magnitude(
      functional_model, **pruning_params)

  models = [layerwise_model, sequential_model, functional_model]
  train_and_save(models, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
  absl_app.run(main)
