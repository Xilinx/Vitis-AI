# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from setuptools import setup
from setuptools import find_packages
from setuptools.command.build_ext import build_ext

import os
import shutil

try:
  import tensorflow as tf
except ImportError:
  raise RuntimeError(
      'Tensorflow must be installed to build the tensorflow wrapper.')

PROJECT_NAME = 'tf1_nndct'

def clean():
  try:
    for dir_name in ['build', 'dist', PROJECT_NAME + '.egg-info']:
      if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
  except:
    print('failed to do the cleaning, please clean up manully')

setup(
    name=PROJECT_NAME,
    description='Neural Network Deep Compression Toolkit(NNDCT) for TensorFlow.',
    author='Xilinx Inc.',
    version='1.0.0',
    url='https://github.com/Xilinx/Vitis-AI',
    packages=find_packages(),
    cmdclass={'build_ext': build_ext},
)

clean()
