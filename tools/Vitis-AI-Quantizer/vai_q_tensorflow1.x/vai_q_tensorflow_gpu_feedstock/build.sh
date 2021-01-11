#!/bin/bash
set -x #echo on

echo "******************* BUILD ENV VARS!!!!!!!!!!!!!!!!!!!!!!!! *******************"
export CONDA_PREFIX=$PREFIX
export LIBDIR=$PREFIX/lib
export INCLUDEDIR=$PREFIX/include
export LD_LIBRARY_PATH="$LIBDIR:$LD_LIBRARY_PATH"

echo "SITE_PACKAGES: $SP_DIR"
echo "PREFIX: $PREFIX"
echo "BUILD_PREFIX: $BUILD_PREFIX"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "SRC_DIR: $SRC_DIR"
echo "PYTHON VERSION: $PY_VER"

# Python settings
export PYTHON_BIN_PATH=${PYTHON}
export PYTHON_LIB_PATH=${SP_DIR}
export USE_DEFAULT_PYTHON_LIB_PATH=1

# Configure
export TF_NEED_CUDA="1"
yes "" | ./configure

bazel build \
  --verbose_failures \
  --config=opt \
  --config=cuda \
  --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
  --copt="-march=x86-64" \
  --copt="-mno-sse3" \
  --copt="-mno-sse4.1" \
  --copt="-mno-sse4.2" \
  --copt="-mno-avx" \
  --copt="-mno-avx2" \
  --copt="-mno-avx512f" \
  --copt="-mno-fma" \
  //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package --gpu "$SRC_DIR/tensorflow_pkg"

pip install --no-deps $SRC_DIR/tensorflow_pkg/*.whl
