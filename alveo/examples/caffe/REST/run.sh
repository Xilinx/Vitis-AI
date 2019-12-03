#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash
#!/bin/bash

MODEL_WEIGHTS="/opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel"
MODEL_PROTOTXT="/opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt"
LABELS="/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt"
PORT="5000"

python ../run.py --prototxt ${MODEL_PROTOTXT} --caffemodel ${MODEL_WEIGHTS} --prepare

python app.py --caffemodel ${MODEL_WEIGHTS} --prototxt xfdnn_auto_cut_deploy.prototxt --synset_words ${LABELS} --port ${PORT} 


