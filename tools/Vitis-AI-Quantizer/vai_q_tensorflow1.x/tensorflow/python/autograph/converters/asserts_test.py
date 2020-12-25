# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for asserts module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.autograph.converters import asserts
from tensorflow.python.autograph.converters import function_scopes
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class AssertsTest(converter_testing.TestCase):

  def test_basic(self):

    def test_fn(a):
      assert a, 'testmsg'
      return a

    with ops.Graph().as_default():
      with self.converted(test_fn, (function_scopes, asserts), {}) as result:
        op = result.test_fn(constant_op.constant(False))

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError, 'testmsg'):
        self.evaluate(op)


if __name__ == '__main__':
  test.main()
