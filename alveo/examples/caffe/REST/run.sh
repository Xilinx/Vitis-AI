#!/usr/bin/env bash
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

MODEL_WEIGHTS="/opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel"
MODEL_PROTOTXT="/opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt"
LABELS="/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt"
PORT="5000"

python ../run.py --prototxt ${MODEL_PROTOTXT} --caffemodel ${MODEL_WEIGHTS} --prepare

python app.py --caffemodel ${MODEL_WEIGHTS} --prototxt xfdnn_auto_cut_deploy.prototxt --synset_words ${LABELS} --port ${PORT}
