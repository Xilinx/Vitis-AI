# python3
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
"""Tests for api module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test


class ApiTest(test.TestCase):

  def test_converted_call_kwonly_args(self):

    def test_fn(*, a):
      return a

    x = api.converted_call(test_fn, converter.ConversionOptions(recursive=True),
                           (), {'a': constant_op.constant(-1)})
    self.assertEqual(-1, self.evaluate(x))

  def test_super_with_no_arg(self):
    test_case_self = self

    class TestBase:

      def plus_three(self, x):
        return x + 3

    class TestSubclass(TestBase):

      def plus_three(self, x):
        test_case_self.fail('This should never be called.')

      def no_arg(self, x):
        return super().plus_three(x)

    tc = api.converted_call(TestSubclass,
                            converter.ConversionOptions(recursive=True), (), {})

    self.assertEqual(5, tc.no_arg(2))


if __name__ == '__main__':
  os.environ['AUTOGRAPH_STRICT_CONVERSION'] = '1'
  test.main()
