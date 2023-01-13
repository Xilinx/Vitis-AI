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


def load_json(json_file):
  """Load json file."""
  with open(json_file, 'r') as f:
    try:
      data = json.loads(f.read())
    except Exception as e:
      raise (
          'Fail to load the json file `{}`, please check the format. \nError: {}'
          .format(json_file, e))
  return data


class PRelu(tf.keras.layers.Layer):
  """
  single input and single output custom op with weights
  """

  def __init__(self, name="param_relu", **kwargs):
    super().__init__(name=name, **kwargs)

  def build(self, input_shape):
    self.alpha = self.add_weight(
        shape=input_shape[1:],
        name='alpha',
        initializer="zeros",
        trainable=True,
    )

  def call(self, inputs, training=None, mask=None):
    pos = tf.nn.relu(inputs)
    neg = -self.alpha * tf.nn.relu(-inputs)
    return pos + neg


class TimeTwo(tf.keras.layers.Layer):
  """
  single input and single output custom op without weights
  """

  def __init__(self, name="time_two", **kwargs):
    super().__init__(name=name, **kwargs)

  def call(self, inputs, training=None, mask=None):
    return 2 * inputs


class MultiInput(tf.keras.layers.Layer):
  """
  multi input and single output custom op without weights
  wieh param in __init__
  """

  def __init__(self, name="multi_input", scale=2, **kwargs):
    super().__init__(name=name, **kwargs)
    self.scale = scale

  def call(self, inputs, training=None, mask=None):
    x_1 = inputs[0]
    x_2 = inputs[1] * self.scale
    return x_1 + x_2

  def get_config(self):
    return {"name": self.name, "scale": self.scale}


custom_objects = {"PRelu": PRelu, "TimeTwo": TimeTwo, "MultiInput": MultiInput}


def build_model():
  inputs = tf.keras.Input((28, 28, 1))
  x = tf.keras.layers.Conv2D(32, (7, 7))(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = PRelu(name="param_relu")(x)

  x = tf.keras.layers.Conv2D(32, (7, 7))(x)
  x = tf.keras.layers.BatchNormalization()(x)
  # x = PRelu(name="param_relu_1")(x)
  x = TimeTwo(name="time_two")(x)
  x = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(x)

  # x = TimeTwo(name="time_two_1")(x)
  x_2 = tf.keras.layers.Conv2D(64, (3, 3))(x)
  x = tf.keras.layers.Conv2D(64, (3, 3))(x)
  multi_input = MultiInput(scale=4)
  x = multi_input([x, x_2])
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

  # Use json format custom quantize strategy
  # if custom layer is defined in the same *.py file with quantize
  # tool then "layer_type": "__main__.PRelu", if not in the same
  # *.py file, then layer_type should be ${FILE_NAME}.PRelu
  custom_quantize_strategy = "./custom_quantize_strategy.json"
  my_quantize_strategy = load_json(custom_quantize_strategy)
  #################################
  ##### quantize model
  #################################
  loaded_model = tf.keras.models.load_model(
      "./float.h5", custom_objects=custom_objects)
  loaded_model.summary()

  # custom layer will be quantized according to custom_quantize_strategy
  # first, and then wrapped by custom layer wrapper
  quantizer = vitis_quantize.VitisQuantizer(
      loaded_model,
      'pof2s',
      custom_objects=custom_objects,
      custom_quantize_strategy=my_quantize_strategy)
  quant_model = quantizer.quantize_model(calib_dataset=test_images)
  quant_model.summary()
  quant_model.save('quantized.h5')

  with vitis_quantize.quantize_scope():
    quantized_model = tf.keras.models.load_model(
        "quantized.h5", custom_objects=custom_objects)
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
