# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Bring in all of the public TensorFlow interface into this module."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

import os as _os
import sys as _sys

from tensorflow.python.tools import module_util as _module_util

# pylint: disable=g-bad-import-order

# API IMPORTS PLACEHOLDER

# WRAPPER_PLACEHOLDER

# Hook external TensorFlow modules.
_current_module = _sys.modules[__name__]
try:
  from tensorflow_estimator.python.estimator.api._v1 import estimator
  _current_module.__path__ = (
      [_module_util.get_parent_dir(estimator)] + _current_module.__path__)
  setattr(_current_module, "estimator", estimator)
except ImportError:
  pass

try:
  from tensorflow.python.keras.api._v1 import keras
  _current_module.__path__ = (
      [_module_util.get_parent_dir(keras)] + _current_module.__path__)
  setattr(_current_module, "keras", keras)
except ImportError:
  pass


from tensorflow.python.platform import flags  # pylint: disable=g-import-not-at-top
_current_module.app.flags = flags  # pylint: disable=undefined-variable
setattr(_current_module, "flags", flags)
