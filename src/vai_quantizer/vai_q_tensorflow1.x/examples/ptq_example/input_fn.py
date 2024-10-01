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

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class Calibrator(object):

    def __init__(self, input_height, input_width, channel_num,
                 calib_batch_size):
        self.input_height = input_height
        self.input_width = input_width
        self.channel_num = channel_num
        self.calib_batch_size = calib_batch_size

    def _calib_input(self, iter):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch_x, batch_y = mnist.train.next_batch(self.calib_batch_size)
            images = np.reshape(
                batch_x,
                (-1, self.input_height, self.input_width, self.channel_num))
            return {"input": images}


caliber = Calibrator(input_height=28,
                  input_width=28,
                  channel_num=1,
                  calib_batch_size=50)
calib_input = caliber._calib_input
