#!/bin/sh

export CF_NETWORK_PATH='cf_resnet50'
export TF_NETWORK_PATH='tf_resnet50'

cp ${TF_NETWORK_PATH}/input_fn.py ${TF_NETWORK_PATH}/input_fn.py.bak
cp tf_eval_script/input_fn.py ${TF_NETWORK_PATH}/

