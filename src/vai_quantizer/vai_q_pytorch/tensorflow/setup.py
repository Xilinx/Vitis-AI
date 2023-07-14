from setuptools import setup
from setuptools import find_packages
from setuptools.command.build_ext import build_ext

try:
  import tensorflow as tf
except ImportError:
  raise RuntimeError(
      'Tensorflow must be installed to build the tensorflow wrapper.')

setup(
    name='tf_nndct',
    description='Neural Network Deep Compression Toolkit(NNDCT) for TensorFlow.',
    author='Xilinx Inc.',
    version='3.5.0',
    url='https://github.com/Xilinx/Vitis-AI',
    packages=find_packages(),
    cmdclass={'build_ext': build_ext},
)
