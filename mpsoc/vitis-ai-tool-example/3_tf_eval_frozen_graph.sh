#!/bin/sh

EVAL_SCRIPT_PATH=tf_eval_script

python ${EVAL_SCRIPT_PATH}/resnet_eval.py \
       --input_frozen_graph ${TF_NETWORK_PATH}/float/frozen.pb \
       --input_node input \
       --output_node resnet_v1_50/predictions/Reshape_1 \
       --eval_batches 100 \
       --batch_size 10 \
       --eval_image_dir images/ \
       --eval_image_list images/list.txt \
