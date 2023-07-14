#!/usr/bin/env bash

MODEL="resnet50"
BATCH_SIZE=128

DATA_DIR="/dataset/imagenet/pytorch"

python -u resnet.py --model ${MODEL} --batch_size ${BATCH_SIZE} --data_dir ${DATA_DIR}
