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
import os
import onnx
import onnxruntime
import tensorflow as tf
import numpy as np
import tempfile

import datetime
from tensorflow import keras
from tensorflow.keras import mixed_precision

import tensorflow as tf
# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

from tensorflow_model_optimization.quantization.keras.vitis.layers import vitis_activation

from tensorflow.keras import layers, models

# Define the model architecture.
inputs = keras.layers.Input(shape=(28, 28), dtype=tf.float32)
x = keras.layers.Reshape(target_shape=(28, 28, 1))(inputs)
x = keras.layers.Conv2D(
    filters=32, kernel_size=(3, 3), use_bias=True, activation='linear')(
        x)
x = keras.layers.BatchNormalization(axis=[-1])(x)
y = x * 2
x = keras.layers.Concatenate()([x, y])
x = keras.layers.Activation('relu')(x)
x = keras.layers.DepthwiseConv2D(
    kernel_size=(3, 3), use_bias=True, activation='linear')(
        x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dropout(rate=0.1)(x)
x = keras.layers.Dense(10, dtype=tf.float32)(x)

predictions = x

model = keras.Model(inputs=inputs, outputs=predictions, name="mnist_model")

#  Train the float model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['sparse_categorical_accuracy'])

log_dir = "logs/float_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)
model.fit(
    train_images,
    train_labels,
    epochs=1,
    validation_data=(test_images, test_labels))
model.save('float.h5')

# Post-Training Quantize
from tensorflow_model_optimization.quantization.keras import vitis_quantize

quantizer = vitis_quantize.VitisQuantizer(model)

#concatenate layer config with bfloat16 datatype
quantized_model = quantizer.quantize_model(
    layer_config={"concatenate": "bfloat16"},
    calib_dataset=train_images[0:10],
    include_cle=False,
    cle_steps=10,
    convert_to_fs_quantize_strategy=True,
    output_format='onnx',
    onnx_opset_version=13,
    include_fast_ft=False)

onnx_model = onnx.load("quantize_results/quantized_model.onnx")
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession(
    "quantize_results/quantized_model.onnx")
num_correct_top1 = 0
total_num_test = len(test_images)
for i in range(total_num_test):
  test_data_x, test_data_y = test_images[i:i + 1], test_labels[i]
  ort_inputs = {
      ort_session.get_inputs()[0].name: test_data_x.astype(np.float32)
  }
  ort_outs = ort_session.run(None, ort_inputs)
  ort_outs_top1 = np.argmax(ort_outs[0])
  if test_data_y == ort_outs_top1:
    num_correct_top1 += 1
acc_top1 = round(num_correct_top1 / total_num_test, 4)
print("onnx val_sparse_categorical_accuracy: {} ".format(acc_top1))
