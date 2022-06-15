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

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tf_nndct.optimization import IterativePruningRunner

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def build_model(pretrained=None):
  # Implementation adapted from https://keras.io/examples/vision/mnist_convnet
  model = keras.Sequential([
      keras.Input(shape=input_shape),
      layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Flatten(), layers.Dropout(0.5),
      layers.Dense(num_classes, activation="softmax"),
  ])

  if pretrained:
    model.load_weights(pretrained)
  return model

def train(model, save_path, epochs=10):
  batch_size = 128
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  model.fit(
      x_train,
      y_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_split=0.1)

  model.evaluate(x_test, y_test, verbose=1)
  model.save_weights(save_path, save_format='tf')

def evaluate(model):
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  score = model.evaluate(x_test, y_test, verbose=0)
  print("Test accuracy:", score[1])
  return score[1]

def prune(model, ratio):
  input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
  runner = IterativePruningRunner(model, input_spec)
  runner.ana(evaluate)
  return runner.prune(ratio)

def transform(model):
  input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
  runner = IterativePruningRunner(model, input_spec)
  return runner.get_slim_model()

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-t",
      "--train",
      action="store_true",
      help="If true, train a baseline model."
  )
  parser.add_argument(
      "-p",
      "--prune",
      type=float,
      default=None,
      help="Pruning ratio",
  )
  parser.add_argument(
      "-tf",
      "--transform",
      action="store_true",
      help="Transforms a sparse model to a slim model."
  )
  parser.add_argument(
      "-sp",
      "--save_path",
      help="Path to save trained weights"
  )
  parser.add_argument(
      "-pr",
      "--pretrained",
      help="Pretrained weights loaded to model."
  )
  return parser.parse_args()

if __name__ == '__main__':

  args = parse_args()
  model = build_model(args.pretrained)

  if args.train:
    train(model, args.save_path)
  elif args.prune:
    pruned_model = prune(model, args.prune)
    train(pruned_model, args.save_path, epochs=5)
  elif args.transform:
    slim_model = transform(model)
    accuracy = evaluate(model)
    slim_accuracy = evaluate(slim_model)
    np.testing.assert_almost_equal(accuracy, slim_accuracy)
