# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Functions for TF 1.X and 2.X compatibility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def assign(ref, value, name=None):
  if hasattr(tf, 'assign'):
    return tf.assign(ref, value, name=name)
  else:
    return ref.assign(value, name=name)


def initialize_variables(testcase):
  """Handle global variable initialization in TF 1.X.

  Arguments:
    testcase: instance of tf.test.TestCase
  """
  if hasattr(tf, 'global_variables_initializer') and not tf.executing_eagerly():
    testcase.evaluate(tf.global_variables_initializer())


def is_v1_apis():
  return hasattr(tf, 'assign')
