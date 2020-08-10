#!/bin/bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
set -x

cd bazel_pip
virtualenv --system-site-packages --python=python .env
source .env/bin/activate
pip --version
pip install portpicker
pip install *.whl

# Use default configuration
yes "" | python configure.py

PIP_TEST_ROOT=pip_test_root
mkdir -p ${PIP_TEST_ROOT}
ln -s $(pwd)/tensorflow ${PIP_TEST_ROOT}/tensorflow
bazel test --define=no_tensorflow_py_deps=true \
      --test_lang_filters=py \
      --build_tests_only \
      -k \
      --test_tag_filters=-no_oss,-oss_serial,-no_pip,-nopip \
      --test_size_filters=small,medium \
      --test_timeout 300,450,1200,3600 \
      --test_output=errors \
      -- //${PIP_TEST_ROOT}/tensorflow/python/... \
      -//${PIP_TEST_ROOT}/tensorflow/python:virtual_gpu_test \
      -//${PIP_TEST_ROOT}/tensorflow/python:virtual_gpu_test_gpu \
      -//${PIP_TEST_ROOT}/tensorflow/python:collective_ops_gpu_test \
      -//${PIP_TEST_ROOT}/tensorflow/python:collective_ops_gpu_test_gpu

