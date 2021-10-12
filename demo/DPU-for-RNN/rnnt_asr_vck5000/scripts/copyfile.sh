#!/bin/sh
#
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
if [ ! -d "inference" ]; then
    git clone https://github.com/mlcommons/inference.git
    cd inference
    git reset --hard 819a825de093421eb1969a163e25f7abe72cd32e
    cd ..
fi

ROOT=./inference/speech_recognition/rnnt/pytorch
cp $ROOT/parts ./ -r
cp $ROOT/utils ./ -r
cp `ls $ROOT/*py | grep -v decoders.py` ./ -r
mv fbank.py ./parts
