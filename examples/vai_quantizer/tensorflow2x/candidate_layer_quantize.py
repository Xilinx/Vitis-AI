# Copyright 2019 Xilinx Inc.
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
import os, sys
import json
import numpy as np
import tensorflow as tf

from tensorflow import keras

print(tf.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow_model_optimization.quantization.keras import vitis_quantize

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0


def build_model():
  inputs = tf.keras.Input((28, 28, 1))
  x = tf.keras.layers.Conv2D(32, (7, 7))(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.ReLU(name="relu")(x)

  x = tf.keras.layers.Conv2D(32, (7, 7))(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(x)

  x_2 = tf.keras.layers.Conv2D(64, (3, 3))(x)
  x = tf.keras.layers.Conv2D(64, (3, 3))(x)
  x = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(x)

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(10, activation="sigmoid")(x)

  model = tf.keras.Model(inputs=inputs, outputs=x)
  return model


def main():
  #################################
  ##### build model
  #################################
  model = build_model()
  model.summary()
  #################################
  ##### compile train
  #################################
  model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"])
  model.fit(train_images, train_labels, epochs=2, shuffle=True)
  model.evaluate(test_images, test_labels)
  model.save("./float.h5")
  del model

  #################################
  ##### quantize model
  #################################
  loaded_model = tf.keras.models.load_model("./float.h5")
  loaded_model.summary()

  # quantize scope is determined by specify input_layers and output layers
  # ignore layers will not be quantized
  input_layers = ["relu"]
  output_layers = ["flatten"]
  ignore_layers = ["max_pooling2d"]

  quantizer = vitis_quantize.VitisQuantizer(loaded_model, 'pof2s')
  quant_model = quantizer.quantize_model(
      calib_dataset=test_images,
      input_layers=input_layers,
      output_layers=output_layers,
      ignore_layers=ignore_layers)
  quant_model.summary()
  quant_model.save('quantized.h5')

  with vitis_quantize.quantize_scope():
    quantized_model = tf.keras.models.load_model("quantized.h5")
    quantized_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    quantized_model.evaluate(test_images, test_labels)

  # Dump Quantized Model
  vitis_quantize.VitisQuantizer.dump_model(
      quant_model, test_images[0:1], "./dump_results", dump_float=True)


if __name__ == '__main__':
  main()
