import os
import shutil
import subprocess
import sys

import setuptools.command.develop
import setuptools.command.install
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from distutils import core
from distutils.core import Distribution
from distutils.errors import DistutilsArgError

INSTALL = False
DEVELOP = False
BDIST = False
EMIT_WARNING = False
CUDA_AVAILABLE = False
#MACROS = []
package_name = "pytorch_nndct"
version = open("version.txt", "r").read().strip()
build_update_message = """
  To install:
    $ python setup.py install
  To develop locally:
    $ python setup.py develop
  To make a wheel package:
    $ python setup.py sdist bdist_wheel -d $YOUR_TARGET

"""


class install(setuptools.command.install.install):
  def run(self):
    setuptools.command.install.install.run(self)


class develop(setuptools.command.develop.develop):
  def run(self):
    setuptools.command.develop.develop.run(self)


def check_env_args():
  global INSTALL, DEVELOP, BDIST, EMIT_WARNING, CUDA_AVAILABLE
  #if torch.cuda.is_available() and "CUDA_HOME" in os.environ:
  if "CUDA_HOME" in os.environ:
    CUDA_AVAILABLE = True
    #MACROS += [("WITH_CUDA", None)]
  else:
    CUDA_AVAILABLE = False
    print("CUDA is not available, or CUDA_HOME not found in the environment "
          "so building without GPU support.")
    '''
    print("CUDA_HOME not found in the environment so building "
          "without GPU support. To build with GPU support "
          "please define the CUDA_HOME environment variable. "
          "This should be a path which contains include/cuda.h")
    '''

  if "install" in sys.argv:
    INSTALL = True
  elif "develop" in sys.argv:
    DEVELOP = True
  elif "bdist_wheel" in sys.argv:
    BDIST = True
  else:
    args = []
    args.append(sys.argv[0])
    args += ["sdist", "bdist_wheel"]
    sys.argv = args
    BDIST = True
    EMIT_WARNING = True


def clean_install_info():
  try:
    for dir_name in [
        'build', 'dist', "pytorch_nndct" + '.egg-info', "nndct_shared",
        "pytorch_nndct/nn/kernel"
    ]:
      if dir_name == "nndct_shared":
        if INSTALL:
          if os.path.exists(dir_name):
            os.unlink(dir_name)
      else:
        if os.path.exists(dir_name):
          shutil.rmtree(dir_name)
  except Exception:
    print("failed to do the cleaning, please clean up manully")


def build_config_setup():
  global INSTALL, DEVELOP, BDIST, CUDA_AVAILABLE
  install_packages = ["nndct_shared"]
  for package in install_packages:
    if os.path.exists(package):
      try:
        os.unlink(package)
      except Exception:
        print("failed to do the cleaning, please clean up manully")
      else:
        os.symlink(f"../{package}", package)
    else:
      os.symlink(f"../{package}", package)

  if INSTALL:
    if not os.path.exists("pytorch_nndct/nn/kernel"):
      os.mkdir("pytorch_nndct/nn/kernel")
      with open("pytorch_nndct/nn/kernel/__init__.py", 'w') as f:
        cwd = os.path.dirname(os.path.realpath(__file__))
        nn_path = os.path.join(cwd, "pytorch_nndct/nn")
        f.write(f"NN_PATH='{nn_path}'")

  install_requires = []
  if not DEVELOP:
    install_requires += ["sklearn",
                         "scipy==1.3.1",
                         "numpy==1.17.2",
                         "tqdm",
                         "ninja"]
  extensions = []
  if not BDIST:
    cmdclass = {"install": install,
                "develop": develop
                }

  else:
    cmdclass = {"build_ext": BuildExtension}
    extra_compile_args = {'cxx': ['-std=c++14', '-fPIC']}
    cwd = os.path.dirname(os.path.realpath(__file__))
    cpu_src_path = os.path.join(cwd, "../csrc/cpu")

    source_files = []
    for name in os.listdir(cpu_src_path):
      if name.split(".")[-1] in ["cpp", "cc", "c"]:
        source_files.append(os.path.join(cpu_src_path, name))


    include_dir = [
        os.path.join(cwd, "../include/cpu"),
        os.path.join(cwd, "pytorch_nndct/nn/include")
    ]
    Extension = CppExtension

    if CUDA_AVAILABLE:
      extra_compile_args['nvcc'] = ['-O2','-arch=sm_35']
      cuda_src_path = os.path.join(cwd, "../csrc/cuda")
      for name in os.listdir(cuda_src_path):
        if name.split(".")[-1] in ["cu", "cpp", "cc", "c"]:
          source_files.append(os.path.join(cuda_src_path, name))

      cpp_src_path = os.path.join(cwd, "pytorch_nndct/nn/src/cuda")
      for name in os.listdir(cpp_src_path):
        if name.split(".")[-1] in ["cpp", "cc", "c"]:
          source_files.append(os.path.join(cpp_src_path, name))

      include_dir.append(os.path.join(cwd, "../include/cuda"))

      from torch.utils.cpp_extension import CUDAExtension
      Extension = CUDAExtension
    else:
      cpp_src_path = os.path.join(cwd, "pytorch_nndct/nn/src/cpu")
      for name in os.listdir(cpp_src_path):
        if name.split(".")[-1] in ["cpp", "cc", "c"]:
          source_files.append(os.path.join(cpp_src_path, name))

    kernel_ext = Extension(name='pytorch_nndct.nn._kernels',
                           language='c++',
                           sources=source_files,
                           include_dirs=include_dir,
                           extra_compile_args=extra_compile_args)
    extensions.append(kernel_ext)

    sources = []

  return extensions, cmdclass, install_requires


def get_version():
  global version
  sha = "Unknown"
  try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
  except Exception:
    pass
  if sha != "Unknown":
    version += "+" + sha[:7]
  return sha


if __name__ == '__main__':
  dist = Distribution()
  dist.script_name = sys.argv[0]
  dist.script_args = sys.argv[1:]
  try:
    is_valid_args = dist.parse_command_line()
  except DistutilsArgError as msg:
    raise SystemExit(f"{core.gen_usage(dist.script_name)}\nerror:{msg}")

  if not is_valid_args:
    sys.exit()

  check_env_args()
  extensions, cmdclass, install_requires = build_config_setup()
  cwd = os.path.dirname(os.path.abspath(__file__))
  sha = get_version()
  version_path = os.path.join(cwd, "pytorch_nndct", "version.py")
  with open(version_path, "w") as f:
    f.write(f"__version__ = '{version}'\n")
    f.write(f"git_version = '{sha}'\n")

  if BDIST:
    version += f"%torch{torch.__version__}"

  setup(
      name=package_name,
      version=version,
      description="A library Xilinx Tool nndct in pytorch",
      url="fill in later",
      author="NNDCT Team",
      author_email="wluo@xilinx.com, niuxj@xilinx.com",
      license="Xilinx",
      packages=find_packages(),
      ext_modules=extensions,
      cmdclass=cmdclass,
      install_requires=install_requires,
      python_requires='>=3.6.8'
  )

  if BDIST:
    print(f"Building wheel {package_name}--{version}")

  clean_install_info()
  if EMIT_WARNING:
    print(80 * "#")
    print(build_update_message)
