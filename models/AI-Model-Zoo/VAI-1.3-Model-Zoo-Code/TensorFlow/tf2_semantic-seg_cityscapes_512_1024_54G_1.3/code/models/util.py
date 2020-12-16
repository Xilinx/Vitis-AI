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

import random
import imageio
import collections
import tensorflow as tf
import numpy as np
import multiprocessing
import logging

def get_norm_by_name(name):
    if name == 'batch':
        return tf.keras.layers.BatchNormalization(axis=-1)
    else:
        raise Exception("unknown norm name %s" % name)


def get_logger(name='tf-semantic-segmentation'):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
