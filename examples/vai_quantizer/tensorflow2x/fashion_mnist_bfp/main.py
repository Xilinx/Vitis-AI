#!/usr/bin/env python

import os
import numpy as np

from net import net_fn, x_train, y_train, x_test, y_test
from net import load_data
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.app.flags.DEFINE_integer(
    'batch_size', 120,
    'batch size.')
tf.compat.v1.app.flags.DEFINE_integer(
    'epochs', 5,
    'Epochs of float model training.')
tf.compat.v1.app.flags.DEFINE_integer(
    'qat_epochs', 2, 'Epochs of quantization-aware-training.')
tf.compat.v1.app.flags.DEFINE_integer(
    'bit_width', 13, 'Quantization bit-width.')
tf.compat.v1.app.flags.DEFINE_string(
    'data_format', 'bfp',
    'bfp/msfp/fp32/bf16')
tf.compat.v1.app.flags.DEFINE_string(
    'save_dir', './result',
    'Where to save the result.')
tf.compat.v1.app.flags.DEFINE_string(
    'origin_model_file', 'origin.h5',
    'Name of origin model.')
tf.compat.v1.app.flags.DEFINE_bool(
    'eager_mode', False, 'Whether to use eager mode.')
tf.compat.v1.app.flags.DEFINE_bool(
    'quantize', True, 'Whether to apply "quantize" to model.')
tf.compat.v1.app.flags.DEFINE_string(
    'quantize_strategy', 'bfp', 'Quantize strategy(bfp/pof2s/..., use bfp as default).')
tf.compat.v1.app.flags.DEFINE_bool(
    'use_qat', False, 'Whether to apply "quantization-aware-training" to model.')

FLAGS = tf.compat.v1.app.flags.FLAGS

print('TensroFlow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)

if FLAGS.eager_mode:
  tf.data.experimental.enable_debug_mode()
  tf.config.run_functions_eagerly(True)

def main(_):
  os.makedirs(FLAGS.save_dir, exist_ok=True)
  model = net_fn()
  model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  origin_model_path = os.path.join(FLAGS.save_dir, FLAGS.origin_model_file)
  if os.path.exists(origin_model_path):
    print("Loading trained weights:", origin_model_path)
    model.load_weights(origin_model_path)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('float model accuracy: ', score)

  else:
    model.fit(x_train, y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    model.save_weights(origin_model_path)
    model.load_weights(origin_model_path)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('float model accuracy: ', score)

  if FLAGS.quantize:
    quantizer = vitis_quantize.VitisQuantizer(model, quantize_strategy=FLAGS.quantize_strategy)
    quantized_model = quantizer.quantize_model(calib_dataset=None, calib_steps=0, calib_batch_size=0, input_bit=FLAGS.bit_width, weight_bit=FLAGS.bit_width, data_format=FLAGS.data_format)
    quantized_model.save(os.path.join(FLAGS.save_dir, 'bfp.h5'))
    quantized_model.compile(optimizer='adadelta',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'], run_eagerly=True)
    score = quantized_model.evaluate(x_test, y_test, verbose=0)
    print('bfp model accuracy: ', score)

  if FLAGS.use_qat:
    qat_model = quantizer.get_qat_model(input_bit=FLAGS.bit_width, weight_bit=FLAGS.bit_width, data_format=FLAGS.data_format)
    qat_model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    qat_model.fit(x_train, y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.qat_epochs,
              validation_data=(x_test, y_test))

    score = qat_model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    qat_model.save(os.path.join(FLAGS.save_dir, 'bfp_qat.h5'))

  
if __name__ == '__main__':
  tf.compat.v1.app.run()
