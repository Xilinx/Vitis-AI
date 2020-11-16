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
"""Init module for TensorFlow Model Optimization Python API.

```
import tensorflow_model_optimization as tfmot
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# We need to put some imports inside a function call below, and the function
# call needs to come before the *actual* imports that populate the
# tensorflow_model_optimization namespace. Hence, we disable this lint check
# throughout the file.
#
# pylint: disable=g-import-not-at-top


# Ensure TensorFlow is importable and its version is sufficiently recent. This
# needs to happen before anything else, since the imports below will try to
# import tensorflow, too.
def _ensure_tf_install():  # pylint: disable=g-statement-before-imports
  """Attempt to import tensorflow, and ensure its version is sufficient.

  Raises:
    ImportError: if either tensorflow is not importable or its version is
    inadequate.
  """
  try:
    import tensorflow as tf
  except ImportError:
    # Print more informative error message, then reraise.
    print(
        '\n\nFailed to import TensorFlow. Please note that TensorFlow is not '
        'installed by default when you install TensorFlow Model Optimization. This '
        'is so that users can decide whether to install the GPU-enabled '
        'TensorFlow package. To use TensorFlow Model Optimization, please install '
        'the most recent version of TensorFlow, by following instructions at '
        'https://tensorflow.org/install.\n\n')
    raise

  import distutils.version

  #
  # Update this whenever we need to depend on a newer TensorFlow release.
  #
  required_tensorflow_version = '1.14.0'

  if (distutils.version.LooseVersion(tf.version.VERSION) <
      distutils.version.LooseVersion(required_tensorflow_version)):
    raise ImportError(
        'This version of TensorFlow Model Optimization requires TensorFlow '
        'version >= {required}; Detected an installation of version {present}. '
        'Please upgrade TensorFlow to proceed.'.format(
            required=required_tensorflow_version, present=tf.__version__))


_ensure_tf_install()


import inspect as _inspect
import os as _os
import sys as _sys


# To ensure users only access the expected public API, the API structure is
# created in the `api` directory. Import all api modules.
# pylint: disable=wildcard-import
from tensorflow_model_optimization.python.core.api import *
# pylint: enable=wildcard-import


# Use sparsity module to fetch the path for the `api` directory.
# This handles all techniques, not just sparsity.
_API_MODULE = sparsity  # pylint: disable=undefined-variable
# Returns $(install_dir)/tensorflow_model_optimization/api
_sparsity_api_dir = _os.path.dirname(
    _os.path.dirname(_inspect.getfile(_API_MODULE)))

# Add the `api` directory to `__path__` so that `from * import module` works.
_current_module = _sys.modules[__name__]
if not hasattr(_current_module, '__path__'):
  __path__ = [_sparsity_api_dir]
elif _os.path.dirname(_inspect.getfile(_API_MODULE)) not in __path__:
  __path__.append(_sparsity_api_dir)


# Delete python module so that users only access the code using the API path
# rather than using the code directory structure.
# This will disallow usage such as `tfmot.python.core.sparsity.keras`.
# pylint: disable=undefined-variable
try:
  del python
except NameError:
  pass
# pylint: enable=undefined-variable
