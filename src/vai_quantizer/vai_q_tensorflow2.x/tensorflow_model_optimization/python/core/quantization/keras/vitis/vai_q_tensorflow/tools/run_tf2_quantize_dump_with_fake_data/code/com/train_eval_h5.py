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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNet
from tensorflow.compat.v1 import flags
from tensorflow.keras.optimizers import RMSprop
from dataset import input_fn, NUM_IMAGES
from dataset import get_images_infor_from_file, ImagenetSequence

keras = tf.keras

flags.DEFINE_string(
    'model', './h5_model/xx.h5',
    'TensorFlow \'GraphDef\' file to load.')
flags.DEFINE_bool(
    'eval_tfrecords', True,
    'If True then use tf_records data .')
flags.DEFINE_string(
    'data_dir', '/scratch/workspace/dataset/imagenet/tf_records',
    'The directory where put the eval images')
flags.DEFINE_bool(
    'eval_images', False,
    'If True then use tf_records data .')
flags.DEFINE_string(
    'eval_image_path', '/scratch/workspace/dataset/val_datasets/imagenet/val_dataset',
    'The directory where put the eval images')
flags.DEFINE_string(
    'eval_image_list',  '/scratch/workspace/dataset/val_datasets/imagenet/val_list.txt', 'file has validation images list')
flags.DEFINE_string(
    'save_path', "train_dir",
    'The directory where save model')
flags.DEFINE_string(
    'filename', "trained_model_{epoch}.h5",
    'The name of sved model')
flags.DEFINE_integer(
    'label_offset', 1, 'label offset')
flags.DEFINE_string(
    'gpus', '1',
    'The gpus used for running evaluation.')
flags.DEFINE_bool(
    'eval_only', False,
    'If True then do not train model, only eval model.')
flags.DEFINE_bool(
    'save_whole_model', False,
    'as applications h5 file just include weights if true save whole model to h5 file.')
flags.DEFINE_bool(
    'use_synth_data', False,
    'If True then use synth data other than imagenet.')
flags.DEFINE_bool(
    'save_best_only', False,
    'If True then only save a model if `val_loss` has improved..')
flags.DEFINE_integer('train_step', None, 'Train step number')
flags.DEFINE_integer('batch_size', 32, 'Train batch size')
flags.DEFINE_integer('epochs', 10, 'Train epochs')
flags.DEFINE_integer('eval_batch_size', 1, 'Evaluate batch size')
flags.DEFINE_integer('save_every_epoch', 1, 'save every step number')
flags.DEFINE_integer('eval_every_epoch', 1, 'eval every step number')
flags.DEFINE_integer('steps_per_epoch', None, 'steps_per_epoch')
flags.DEFINE_integer('decay_steps', 10000, 'decay_steps')
flags.DEFINE_float('learning_rate', 1e-6, 'learning rate')
# Quantization Config
flags.DEFINE_bool('quantize', False, 'Whether to do quantization.')
flags.DEFINE_string('quantize_output_dir', './quantized/', 'Directory for quantize output results.')
flags.DEFINE_bool('quantize_eval', False, 'Whether to do quantize evaluation.')
flags.DEFINE_bool('quantize_train', False, 'Whether to do quantize training.')
flags.DEFINE_bool('dump', False, 'Whether to do dump.')
flags.DEFINE_string('dump_output_dir', './quantized/', 'Directory for dump output results.')

flags.DEFINE_integer('height', 224, 'height')
flags.DEFINE_integer('width', 224, 'width')

FLAGS = flags.FLAGS

TRAIN_NUM = NUM_IMAGES['train']
EVAL_NUM = NUM_IMAGES['validation']

def get_input_data(num_epochs=1):
  train_data = input_fn(
      is_training=True, data_dir=FLAGS.data_dir,
      batch_size=FLAGS.batch_size,
      num_epochs=num_epochs,
      num_gpus=1,
      dtype=tf.float32)

  eval_data = input_fn(
      is_training=False, data_dir=FLAGS.data_dir,
      batch_size=FLAGS.eval_batch_size,
      num_epochs=1,
      num_gpus=1,
      dtype=tf.float32)
  return train_data, eval_data



def main():
  ## run once to save h5 file (add model info)
  if FLAGS.save_whole_model:
    model = MobileNet(weights='imagenet')
    model.save(FLAGS.model)
    exit()

  if not FLAGS.eval_images:
    train_data, eval_data = get_input_data(FLAGS.epochs)

  strategy = tf.distribute.MirroredStrategy()
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

  with strategy.scope():
    if FLAGS.dump or FLAGS.quantize_eval or FLAGS.quantize_train:
        from tensorflow_model_optimization.quantization.keras import vitis_quantize
        with vitis_quantize.quantize_scope():
            model = keras.models.load_model(FLAGS.model)
    else:
        model = keras.models.load_model(FLAGS.model)

    if FLAGS.eval_images:
      img_paths, labels = get_images_infor_from_file(FLAGS.eval_image_path,
              FLAGS.eval_image_list, FLAGS.label_offset)
      calib_dataset = ImagenetSequence(img_paths[0:100], labels[0:100],
              FLAGS.eval_batch_size, output_height=FLAGS.height,
              output_width=FLAGS.width)
    else:
      calib_dataset = eval_data.take(10)


    if FLAGS.quantize:
        from tensorflow_model_optimization.quantization.keras import vitis_quantize
        if FLAGS.eval_only:
          # do quantization
          model = vitis_quantize.VitisQuantizer(model, '8bit').quantize_model(calib_dataset=calib_dataset,
              replace_relu6=False,
              include_cle=False,
              include_fast_ft=False,
              fast_ft_epochs=10)

          # save quantized model
          model.save(os.path.join(FLAGS.quantize_output_dir, 'quantized.h5'))
          print('Quantize finished, results in: {}'.format(FLAGS.quantize_output_dir))
          return
        else:
          model = vitis_quantize.VitisQuantizer(model).get_qat_model(remove_dropout=False, replace_relu6=False)

    if FLAGS.eval_images:
      dump_dataset = ImagenetSequence(img_paths[0:1], labels[0:1], FLAGS.eval_batch_size, output_height=FLAGS.height,
              output_width=FLAGS.width)
    else:
      dump_dataset = eval_data.take(1)

    if FLAGS.dump:
        # do quantize dump
        vitis_quantize.VitisQuantizer.dump_model(model, dump_dataset, FLAGS.dump_output_dir)

        print('Dump finished, results in: {}'.format(FLAGS.dump_output_dir))
        return

    initial_learning_rate = FLAGS.learning_rate
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, decay_steps=FLAGS.decay_steps, decay_rate=0.96,
                staircase=True

            )
    opt = RMSprop(learning_rate=lr_schedule)

    loss = keras.losses.SparseCategoricalCrossentropy()
    metric_top_5 = keras.metrics.SparseTopKCategoricalAccuracy()
    accuracy = keras.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer=opt, loss=loss,
            metrics=[accuracy, metric_top_5])

  if not FLAGS.eval_only:
    if not os.path.exists(FLAGS.save_path):
      os.makedirs(FLAGS.save_path)
    callbacks = [
      keras.callbacks.ModelCheckpoint(
          filepath=os.path.join(FLAGS.save_path,FLAGS.filename),
          save_best_only=True,
          monitor="sparse_categorical_accuracy",
          verbose=1,
      )]
    steps_per_epoch = FLAGS.steps_per_epoch if FLAGS.steps_per_epoch else np.ceil(TRAIN_NUM/FLAGS.batch_size)
    model.fit(train_data,
            epochs=FLAGS.epochs,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_freq=FLAGS.eval_every_epoch,
            validation_steps = EVAL_NUM/FLAGS.eval_batch_size,
            validation_data=eval_data)
  if not FLAGS.eval_images:
    print("evaluate model using tf_records data format")
    res = model.evaluate(eval_data, steps=EVAL_NUM/FLAGS.eval_batch_size)
    print(res)
  if FLAGS.eval_images and FLAGS.eval_only:
    img_paths, labels = get_images_infor_from_file(FLAGS.eval_image_path,
            FLAGS.eval_image_list, FLAGS.label_offset)
    imagenet_seq = ImagenetSequence(img_paths, labels, FLAGS.eval_batch_size)
    res = model.evaluate(imagenet_seq, steps=EVAL_NUM/FLAGS.eval_batch_size, verbose=1)
    print(res)


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpus
  main()
