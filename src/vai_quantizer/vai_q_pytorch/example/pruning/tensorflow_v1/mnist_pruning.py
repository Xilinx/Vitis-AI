# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from tf1_nndct.optimization.pruning import IterativePruningRunner
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


keras = tf.keras

def mnist_convnet():
  num_classes = 10
  input_shape = (28, 28, 1)

  model = keras.Sequential([
      layers.InputLayer(input_shape=input_shape),
      layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
      layers.BatchNormalization(),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Flatten(),
      layers.Dropout(0.5),
      layers.Dense(num_classes),
  ])
  return model, input_shape


def eval_fn(frozen_graph_def: tf.compat.v1.GraphDef) -> float:
  with tf.compat.v1.Session().as_default() as sess:
    tf.import_graph_def(frozen_graph_def, name="")
    inp = np.random.rand(1, 28, 28, 1)
    a = sess.run(sess.graph.get_tensor_by_name('dense/BiasAdd:0'), feed_dict={'input_1:0': inp})
    print(a)
    print("in eval_fn done")
    return 0.5


def prune():
  with tf.Session() as sess:
    model, input_shape = mnist_convnet()
    sess.run(tf.global_variables_initializer())
    input_specs={'input_1:0': tf.TensorSpec(shape=(1, 28, 28, 1), dtype=tf.dtypes.float32)}
    pruner = IterativePruningRunner("mnist", sess, input_specs, ["dense/BiasAdd"])
    pruner.ana(eval_fn, gpu_ids=['/GPU:0', '/GPU:1'])
    shape_tensors, masks = pruner.prune(sparsity=0.5)

    def loss_fn():
      images = np.ones((1, 28, 28, 1), dtype=np.float32)
      out = model(images, training=True)
      return tf.reduce_sum(out)

    opt = tf.compat.v1.train.GradientDescentOptimizer(3.0)
    sess.run(opt.minimize(loss_fn, var_list=tf.trainable_variables()))
    slim_graph_def = pruner.get_slim_graph_def(shape_tensors, masks)


if __name__ == "__main__":
  prune()

