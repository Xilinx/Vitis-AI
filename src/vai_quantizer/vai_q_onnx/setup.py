#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Install vai_q_onnx."""
import datetime
import os
import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

version_number = "1.14.0"

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")) as req_file:
    requirements = req_file.read().splitlines()

if '--release' in sys.argv:
  release = True
  sys.argv.remove('--release')
else:
  # Build a nightly package by default.
  release = False

if release:
  project_name = 'vai-q-onnx'
else:
  # Nightly releases use date-based versioning of the form
  # '0.0.1.dev20180305'
  project_name = 'vai-q-onnx-nightly'
  datestring = datetime.datetime.now().strftime('%Y%m%d')
  version_number += datestring


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return False

setup(
    name=project_name,
    version=version_number,
    description='Xilinx Vitis AI Quantizer for ONNX. '
    'It is customized based on [Quantization Tool](https://github.com/microsoft/onnxruntime/tree/rel-1.14.0/onnxruntime/python/tools/quantization).',
    author='Xiao Sheng',
    author_email='kylexiao@xilinx.com',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=requirements,
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
    keywords='onnx model optimization machine learning',
)
