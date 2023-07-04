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
import numpy as np
from nets.inception_v3 import inception_v3


def eval_fn(frozen_graph_def: tf.compat.v1.GraphDef) -> float:
  with tf.compat.v1.Session().as_default() as sess:
    return 0.5


def main():
  with tf.compat.v1.Session() as sess:
    opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    images = tf.convert_to_tensor(np.ones((1, 224, 224, 3), dtype=np.float32))
    net, _ = inception_v3(images, 1000)
    print(net)
    loss = tf.reduce_sum(net)
    sess.run(tf.global_variables_initializer())

    pruner = IterativePruningRunner("inception_v3", sess, {}, ["InceptionV3/Logits/SpatialSqueeze"])
    pruner.ana(eval_fn, gpu_ids=['/GPU:0', '/GPU:1'])
    shape_tensors, masks = pruner.prune(sparsity=0.5)

    variables = tf.trainable_variables()
    sess.run(opt.minimize(loss, var_list=variables))
    slim_graph_def = pruner.get_slim_graph_def(shape_tensors, masks)


if __name__ == "__main__":
  main()
