# Copyright 2019 Xilinx Inc.
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

pip install tensorflow==1.15.2

python -c "import tensorflow as tf; print(tf)"

bash build.sh --build_with_cpu --conda

pip install --no-deps $SRC_DIR/pkgs/*.whl
