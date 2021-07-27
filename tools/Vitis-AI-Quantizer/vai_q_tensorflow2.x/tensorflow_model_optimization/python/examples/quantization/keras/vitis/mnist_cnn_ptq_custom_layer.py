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
import tempfile
import os

import tensorflow as tf
import numpy as np
import datetime

from tensorflow import keras

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0


class MyLayer(keras.layers.Layer):

  def __init__(self, units=32, name=None, **kwargs):
    super(MyLayer, self).__init__(name=name, **kwargs)
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(
        shape=(input_shape[-1], self.units),
        initializer="random_normal",
        trainable=True,
        name='w')
    self.b = self.add_weight(
        shape=(self.units,), initializer="zeros", trainable=True, name='b')

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

  def get_config(self):
    base_config = super(MyLayer, self).get_config()
    config = {"units": self.units}
    return dict(list(base_config.items()) + list(config.items()))


# Define the model architecture.
inputs = keras.layers.Input(shape=(28, 28))
x = keras.layers.Reshape(target_shape=(28, 28, 1))(inputs)
x = keras.layers.Conv2D(
    filters=32, kernel_size=(3, 3), use_bias=True, activation='linear')(
        x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.DepthwiseConv2D(
    kernel_size=(3, 3), use_bias=True, activation='linear')(
        x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dropout(rate=0.1)(x)
x = MyLayer(10)(x)
predictions = x

model = keras.Model(inputs=inputs, outputs=predictions, name="mnist_model")

#  Train the float model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'])

model.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels))

model.save('float.h5')
del model

# Load Float Model
model = keras.models.load_model('float.h5', custom_objects={'MyLayer': MyLayer})

# Evaluate Float Model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'])
model.evaluate(test_images, test_labels, batch_size=500)

# Post-Training Quantization with custom quantize strategy
from tensorflow_model_optimization.quantization.keras import vitis_quantize

w_quantizer = {
    "quantizer_type": "LastValueQuantPosQuantizer",
    "quantizer_params": {
        "bit_width": 8,
        "method": 1,
        "round_mode": 0
    }
}
a_quantizer = {
    "quantizer_type": "LastValueQuantPosQuantizer",
    "quantizer_params": {
        "bit_width": 8,
        "method": 1,
        "round_mode": 1
    }
}
my_quantize_strategy = {
    "quantize_registry_config": {
        "layer_quantize_config": [{
            "layer_type": "__main__.MyLayer",
            "quantizable_weights": ["w", "b"],
            "weight_quantizers": [w_quantizer, w_quantizer],
            "quantizable_outputs": ["0"],
            "output_quantizers": [a_quantizer]
        }]
    }
}

quantizer = vitis_quantize.VitisQuantizer(
    model, custom_objects={'MyLayer': MyLayer})

#  # Use Dict format cust quantize strategy
#  quantizer = vitis_quantize.VitisQuantizer(
#      model,
#      custom_objects={'MyLayer': MyLayer},
#      custom_quantize_strategy=my_quantize_strategy)

#  # Use json format custom quantize strategy
#  quantizer = vitis_quantize.VitisQuantizer(
#      model,
#      custom_objects={'MyLayer': MyLayer},
#      custom_quantize_strategy='./my_quantize_strategy.json')

quantized_model = quantizer.quantize_model(calib_dataset=train_images[0:10])
quantized_model.save('quantized.h5')

# Load Quantized Model
quantized_model = keras.models.load_model('quantized.h5')

# Evaluate Quantized Model
quantized_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'])
quantized_model.evaluate(test_images, test_labels, batch_size=500)

# Dump Quantized Model
quantizer.dump_model(
    quantized_model, dataset=train_images[0:1], dump_float=True)
