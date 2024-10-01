"""
 Copyright 2019 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

#!/usr/bin/env bash

set -ex

GIT_VERSION=$(git rev-parse --short HEAD)
sed -i 's/__git_version__ = ".*"/__git_version__ = "'"$GIT_VERSION"'"/' xnnc/version.py

rm -rf dist/*
python setup.py sdist bdist_wheel
pip install --force-reinstall ./dist/*.whl

