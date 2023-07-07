#!/usr/bin/env bash

MODEL="resnet50"
CONFIG_FILE="config.json"

python resnet.py --model ${MODEL} --config_file ${CONFIG_FILE}
