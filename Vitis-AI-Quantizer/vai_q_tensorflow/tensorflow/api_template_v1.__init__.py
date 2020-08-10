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

import distutils as _distutils
import inspect as _inspect
import os as _os
import site as _site
import sys as _sys

# pylint: disable=g-bad-import-order
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.platform import tf_logging as _logging

# API IMPORTS PLACEHOLDER

# WRAPPER_PLACEHOLDER

if "dev" in __version__:   # pylint: disable=undefined-variable
  _logging.warning("""

  TensorFlow's `tf-nightly` package will soon be updated to TensorFlow 2.0.

  Please upgrade your code to TensorFlow 2.0:
    * https://www.tensorflow.org/beta/guide/migration_guide

  Or install the latest stable TensorFlow 1.X release:
    * `pip install -U "tensorflow==1.*"`

  Otherwise your code may be broken by the change.

  """)

# Make sure directory containing top level submodules is in
# the __path__ so that "from tensorflow.foo import bar" works.
# We're using bitwise, but there's nothing special about that.
_API_MODULE = _sys.modules[__name__].bitwise  # pylint: disable=undefined-variable
_current_module = _sys.modules[__name__]
_tf_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
if not hasattr(_current_module, '__path__'):
  __path__ = [_tf_api_dir]
elif _tf_api_dir not in __path__:
  __path__.append(_tf_api_dir)

# Hook external TensorFlow modules.
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


from tensorflow.python.util.lazy_loader import LazyLoader  # pylint: disable=g-import-not-at-top
_CONTRIB_WARNING = """
The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
  * https://github.com/tensorflow/io (for I/O related ops)
If you depend on functionality not listed there, please file an issue.
"""
contrib = LazyLoader('contrib', globals(), 'tensorflow.contrib',
                     _CONTRIB_WARNING)
del LazyLoader
# The templated code that replaces the placeholder above sometimes
# sets the __all__ variable. If it does, we have to be sure to add
# "contrib".
if '__all__' in vars():
  vars()['__all__'].append('contrib')

from tensorflow.python.platform import flags  # pylint: disable=g-import-not-at-top
# The 'app' module will be imported as part of the placeholder section above.
_current_module.app.flags = flags  # pylint: disable=undefined-variable
setattr(_current_module, "flags", flags)

# Load all plugin libraries from site-packages/tensorflow-plugins if we are
# running under pip.
# TODO(gunan): Enable setting an environment variable to define arbitrary plugin
# directories.
# TODO(gunan): Find a better location for this code snippet.
from tensorflow.python.framework import load_library as _ll
from tensorflow.python.lib.io import file_io as _fi

# Get sitepackages directories for the python installation.
_site_packages_dirs = []
_site_packages_dirs += [_site.USER_SITE]
_site_packages_dirs += [_p for _p in _sys.path if 'site-packages' in _p]
if 'getsitepackages' in dir(_site):
  _site_packages_dirs += _site.getsitepackages()

if 'sysconfig' in dir(_distutils):
  _site_packages_dirs += [_distutils.sysconfig.get_python_lib()]

_site_packages_dirs = list(set(_site_packages_dirs))

# Find the location of this exact file.
_current_file_location = _inspect.getfile(_inspect.currentframe())

def _running_from_pip_package():
  return any(
      _current_file_location.startswith(dir_) for dir_ in _site_packages_dirs)

if _running_from_pip_package():
  for s in _site_packages_dirs:
    # TODO(gunan): Add sanity checks to loaded modules here.
    plugin_dir = _os.path.join(s, 'tensorflow-plugins')
    if _fi.file_exists(plugin_dir):
      _ll.load_library(plugin_dir)

# These symbols appear because we import the python package which
# in turn imports from tensorflow.core and tensorflow.python. They
# must come from this module. So python adds these symbols for the
# resolution to succeed.
# pylint: disable=undefined-variable
try:
  del python
except NameError:
  pass
try:
  del core
except NameError:
  pass
try:
  del compiler
except NameError:
  pass

_current_module.compat.v2.compat.v1 = _current_module.compat.v1
# pylint: enable=undefined-variable
