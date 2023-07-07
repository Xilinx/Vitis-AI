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
"""Install tensorflow_model_optimization."""
import datetime
import os
import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(
    os.path.dirname(__file__), 'tensorflow_model_optimization', 'python/core')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

# TODO(alanchiao): add explicit Tensorflow requirement once Tensorflow
# moves from a tf and tf-gpu packaging approach (where a user installs
# one of the two) to one where a user installs the tf package and then
# also installs the gpu package if they need gpu support. The latter allows
# us (and our dependents) to maintain a single package instead of two.
REQUIRED_PACKAGES = [
    'numpy~=1.14',
    'six~=1.10',
    'enum34~=1.1;python_version<"3.4"',
    'dm-tree~=0.1.1',
]

if '--release' in sys.argv:
  release = True
  sys.argv.remove('--release')
else:
  # Build a nightly package by default.
  release = False

if release:
  project_name = 'vai-q-tensorflow2'
else:
  # Nightly releases use date-based versioning of the form
  # '0.0.1.dev20180305'
  project_name = 'vai-q-tensorflow2-nightly'
  datestring = datetime.datetime.now().strftime('%Y%m%d')
  __version__ += datestring


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return False

setup(
    name=project_name,
    version=__version__,
    description='Xilinx Vitis AI Quantizer for Tensorflow 2.x. '
    'This is customized based on tensorflow-model-optimization('
    'https://github.com/tensorflow/model-optimization)'
    'A suite of tools that users, both novice and advanced'
    ' can use to optimize machine learning models for deployment'
    ' and execution.',
    author='Xiao Sheng',
    author_email='kylexiao@xilinx.com',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    package_data={'': ['*.so', '*.json']},
    exclude_package_data={'': ['BUILD', '*.h', '*.cc']},
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={
        'pip_pkg': InstallCommandBase,
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow model optimization machine learning',
)
