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

bash ./pip_pkg.sh ./pkgs/ --release
pip install --no-deps $SRC_DIR/pkgs/*.whl
