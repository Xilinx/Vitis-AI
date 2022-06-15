from setuptools import setup
from setuptools import Extension
from setuptools import find_packages
from setuptools.command.build_ext import build_ext

import os
import platform
import shutil
import subprocess
import sys

try:
  import tensorflow as tf
except ImportError:
  raise RuntimeError(
      'Tensorflow must be installed to build the tensorflow wrapper.')

try:
  import numpy
  try:
    numpy_include = numpy.get_include()
  except AttributeError:
    numpy_include = numpy.get_numpy_include()
except ImportError:
  numpy_include = ''
  assert 'NUMPY_INCLUDE' in os.environ
numpy_include = os.getenv('NUMPY_INCLUDE', numpy_include)

if 'CUDA_HOME' in os.environ:
  cuda_dir = os.environ['CUDA_HOME']
  enable_gpu = True
elif os.path.exists('/usr/local/cuda'):
  cuda_dir = '/usr/local/cuda'
  enable_gpu = True
else:
  print(
      'CUDA_HOME not found in the environment so building '
      'without GPU support. To build with GPU support '
      'please define the CUDA_HOME environment variable. '
      'This should be a path which contains include/cuda.h',
      file=sys.stderr)
  enable_gpu = False
  sys.exit(1)

PROJECT_NAME = 'tf_nndct'
def clean():
  try:
    for dir_name in ['build', 'dist', PROJECT_NAME + '.egg-info']:
      if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
  except:
    print('failed to do the cleaning, please clean up manully')


libext = '.dylib' if platform.system() == 'Darwin' else '.so'
libnndct = 'libnndct' + libext
NNDCT_LIB_DIR = '../build'
if not os.path.exists(os.path.join(NNDCT_LIB_DIR, libnndct)):
  print('Could not find {} in {}'.format(libnndct, NNDCT_LIB_DIR),
      file=sys.stderr)
  sys.exit(1)

root_path = os.path.realpath(os.path.dirname(__file__))
include_dirs = [os.path.join(root_path, '../include/cuda'), tf.sysconfig.get_include()]
include_dirs += [os.path.join(root_path, '../include/cpu'), tf.sysconfig.get_include()]
tf_cflags = tf.sysconfig.get_compile_flags()
tf_lflags = tf.sysconfig.get_link_flags()
ver_list = tf.__version__.split('-')[0].split('.')
tf_version = int(ver_list[0]) * 1000 + int(ver_list[1]) * 10 + int(ver_list[2])

extra_compile_args = ['-std=c++14', '-fPIC']
# Currently tensorflow triggers return type errors.
extra_compile_args += ['-Wno-return-type', '-DTF_VERION={}'.format(tf_version)]
extra_compile_args += tf_cflags
extra_link_args = tf_lflags
extra_link_args.append('-Wl,-rpath='+os.path.realpath(NNDCT_LIB_DIR))

if enable_gpu:
  extra_compile_args += ['-DGOOGLE_CUDA']
  include_dirs += [os.path.join(cuda_dir, 'include')]
  include_dirs += [os.path.join(root_path, '../include')]

# Ensure that all expected files and directories exist.
for loc in include_dirs:
  if not os.path.exists(loc):
    print(
        ('Could not find file or directory {}.\n'
         'Check your environment variables and paths?').format(loc),
        file=sys.stderr)
    sys.exit(1)

#ext modules for pure c++
src_dir = PROJECT_NAME + '/kernels/'
sources = [src_dir + f for f in os.listdir(src_dir) if f.endswith('.cc')]
op_ext = Extension(
    PROJECT_NAME + '.nndct_kernels',
    sources=sources,
    language='c++',
    include_dirs=include_dirs,
    library_dirs=[NNDCT_LIB_DIR],
    #runtime_library_dirs=[os.path.realpath(NNDCT_LIB_DIR)],
    libraries=['nndct'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args)

setup(
    name=PROJECT_NAME,
    description='Neural Network Deep Compression Toolkit(NNDCT) for TensorFlow.',
    author='Xilinx Inc.',
    version='1.0.0',
    url='',
    packages=find_packages(),
    cmdclass={'build_ext': build_ext},
    ext_modules=[op_ext],
)

clean()
