#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
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
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images / 255.0
test_images = test_images.reshape((-1, 28, 28, 1))

# Train the float model
retrain = 1
if retrain:
    mode = 'f'
    # Define the model architecture.
    inputs = keras.layers.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(filters=32,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            use_bias=True,
                            padding='same',
                            activation='linear')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(10)(x)
    x = keras.layers.Softmax()(x)
    predictions = x

    model = keras.Model(inputs=inputs, outputs=predictions, name="mnist_model")
    model.summary()

    #  Train the digit classification model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy'])

    log_dir = "logs/float_fit/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)
    model.fit(train_images,
              train_labels,
              epochs=1,
              validation_data=(test_images, test_labels))

    model.save('mnist.h5')
else:
    model = keras.models.load_model('mnist.h5')
    model.evaluate(test_images, test_labels)

# Convert to onnx
import tf2onnx

spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name='input'),)
output_path = "mnist.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model,
                                            input_signature=spec,
                                            opset=13,
                                            output_path=output_path)

# Quantize in onnxruntime
import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static, CalibrationDataReader, CalibrationMethod

input_model_path = './mnist.onnx'
qdq_model_path = './mnist_qdq.onnx'


class MnistDataReader(CalibrationDataReader):

    def __init__(self, model_path):
        self.enum_data = None

        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        (_, _, height, width) = session.get_inputs()[0].shape

        self.input_name = session.get_inputs()[0].name

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{
                self.input_name: [i]
            } for i in test_images.astype(np.float32)])
        return next(self.enum_data, None)


quantize_static(
    model_input=input_model_path,
    model_output=qdq_model_path,
    calibration_data_reader=MnistDataReader(input_model_path),
    quant_format=QuantFormat.QDQ,
    per_channel=False,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
    optimize_model=True,
    calibrate_method=CalibrationMethod.MinMax,
)
print('Quantize Finished.')
print('Quantized QDQ model saved in: {}'.format(qdq_model_path))

# Convert QDQ to QOperator
from vai_q_onnx.tools.convert_qdq_to_qop import convert_qdq_to_qop
import onnx

qop_model_path = './mnist_qop.onnx'
model = onnx.load_model(qdq_model_path)
qop_model = convert_qdq_to_qop(model)
onnx.save(qop_model, qop_model_path)
print('QDQ to QOP Convertsion Finished.')
print('Converted QOP model saved in: {}'.format(qop_model_path))
