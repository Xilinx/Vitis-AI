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

from net import build_model, x_train, y_train, x_test, y_test
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'ckpt_path', './model.ckpt',
    'Where to save the trained weights as checkpoint.')
tf.app.flags.DEFINE_string(
    'pretrained', '', 'Pretrained weights path to load.')
tf.app.flags.DEFINE_integer(
    'epochs', 15, 'Epochs used for training.')
tf.app.flags.DEFINE_boolean(
    'pruning', False,
    'If running with pruning masks.')

FLAGS = tf.app.flags.FLAGS

if FLAGS.pruning:
  tf.set_pruning_mode()

def main(_):
  batch_size = 120
  epochs = FLAGS.epochs

  model = build_model()
  model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  
  if FLAGS.pretrained:
    model.load_weights(FLAGS.pretrained)

  # Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)

  score = model.evaluate(x_test, y_test, verbose=1)
  model.save_weights(FLAGS.ckpt_path, save_format='tf')

if __name__ == '__main__':
  tf.app.run()
