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
"""ScopedAnnotation allows the profiler to track python events.

Usage:
    with scoped_annotation.ScopedAnnotation('name'):
      ...
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.pywrap_tensorflow import PythonScopedAnnotation


class ScopedAnnotation(object):
  """Context manager that generates an annotation for the profiler."""

  def __init__(self, name, **kwargs):
    if PythonScopedAnnotation.IsEnabled():
      if kwargs:
        name += '#' + ','.join(
            [key + '=' + str(value) for key, value in kwargs.iteritems()]) + '#'
      self._scoped_annotation = PythonScopedAnnotation(name)
    else:
      self._scoped_annotation = None

  def __enter__(self):
    if self._scoped_annotation:
      self._scoped_annotation.Enter()

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self._scoped_annotation:
      self._scoped_annotation.Exit()
