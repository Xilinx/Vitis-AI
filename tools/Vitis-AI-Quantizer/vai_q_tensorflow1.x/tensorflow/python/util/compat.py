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
"""Functions for Python 2 vs. 3 compatibility.

## Conversion routines
In addition to the functions below, `as_str` converts an object to a `str`.


## Types
The compatibility module also provides the following types:

* `bytes_or_text_types`
* `complex_types`
* `integral_types`
* `real_types`
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers as _numbers

import numpy as _np
import six as _six

from tensorflow.python.util.tf_export import tf_export

try:
  # This import only works on python 3.3 and above.
  import collections.abc as collections_abc  # pylint: disable=unused-import
except ImportError:
  import collections as collections_abc  # pylint: disable=unused-import


def as_bytes(bytes_or_text, encoding='utf-8'):
  """Converts `bytearray`, `bytes`, or unicode python input types to `bytes`.

  Uses utf-8 encoding for text by default.

  Args:
    bytes_or_text: A `bytearray`, `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for encoding unicode.

  Returns:
    A `bytes` object.

  Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
  """
  if isinstance(bytes_or_text, bytearray):
    return bytes(bytes_or_text)
  elif isinstance(bytes_or_text, _six.text_type):
    return bytes_or_text.encode(encoding)
  elif isinstance(bytes_or_text, bytes):
    return bytes_or_text
  else:
    raise TypeError('Expected binary or unicode string, got %r' %
                    (bytes_or_text,))


def as_text(bytes_or_text, encoding='utf-8'):
  """Converts any string-like python input types to unicode.

  Returns the input as a unicode string. Uses utf-8 encoding for text
  by default.

  Args:
    bytes_or_text: A `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for decoding unicode.

  Returns:
    A `unicode` (Python 2) or `str` (Python 3) object.

  Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
  """
  if isinstance(bytes_or_text, _six.text_type):
    return bytes_or_text
  elif isinstance(bytes_or_text, bytes):
    return bytes_or_text.decode(encoding)
  else:
    raise TypeError('Expected binary or unicode string, got %r' % bytes_or_text)


# Convert an object to a `str` in both Python 2 and 3.
if _six.PY2:
  as_str = as_bytes
  tf_export('compat.as_bytes', 'compat.as_str')(as_bytes)
  tf_export('compat.as_text')(as_text)
else:
  as_str = as_text
  tf_export('compat.as_bytes')(as_bytes)
  tf_export('compat.as_text', 'compat.as_str')(as_text)


@tf_export('compat.as_str_any')
def as_str_any(value):
  """Converts input to `str` type.

     Uses `str(value)`, except for `bytes` typed inputs, which are converted
     using `as_str`.

  Args:
    value: A object that can be converted to `str`.

  Returns:
    A `str` object.
  """
  if isinstance(value, bytes):
    return as_str(value)
  else:
    return str(value)


@tf_export('compat.path_to_str')
def path_to_str(path):
  r"""Converts input which is a `PathLike` object to `str` type.

  Converts from any python constant representation of a `PathLike` object to
  a string. If the input is not a `PathLike` object, simply returns the input.

  Args:
    path: An object that can be converted to path representation.

  Returns:
    A `str` object.

  Usage:
    In case a simplified `str` version of the path is needed from an
    `os.PathLike` object

  Examples:
  ```python3
  >>> tf.compat.path_to_str('C:\XYZ\tensorflow\./.././tensorflow')
  'C:\XYZ\tensorflow\./.././tensorflow' # Windows OS
  >>> tf.compat.path_to_str(Path('C:\XYZ\tensorflow\./.././tensorflow'))
  'C:\XYZ\tensorflow\..\tensorflow' # Windows OS
  >>> tf.compat.path_to_str(Path('./corpus'))
  'corpus' # Linux OS
  >>> tf.compat.path_to_str('./.././Corpus')
  './.././Corpus' # Linux OS
  >>> tf.compat.path_to_str(Path('./.././Corpus'))
  '../Corpus' # Linux OS
  >>> tf.compat.path_to_str(Path('./..////../'))
  '../..' # Linux OS

  ```
  """
  if hasattr(path, '__fspath__'):
    path = as_str_any(path.__fspath__())
  return path


# Numpy 1.8 scalars don't inherit from numbers.Integral in Python 3, so we
# need to check them specifically.  The same goes from Real and Complex.
integral_types = (_numbers.Integral, _np.integer)
tf_export('compat.integral_types').export_constant(__name__, 'integral_types')
real_types = (_numbers.Real, _np.integer, _np.floating)
tf_export('compat.real_types').export_constant(__name__, 'real_types')
complex_types = (_numbers.Complex, _np.number)
tf_export('compat.complex_types').export_constant(__name__, 'complex_types')

# Either bytes or text.
bytes_or_text_types = (bytes, _six.text_type)
tf_export('compat.bytes_or_text_types').export_constant(__name__,
                                                        'bytes_or_text_types')
