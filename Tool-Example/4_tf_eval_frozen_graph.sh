#!/bin/sh
#conda activate vitis-ai-tensorflow

EVAL_SCRIPT_PATH=example_file

python ${EVAL_SCRIPT_PATH}/resnet_eval.py \
       --input_frozen_graph ${TF_NETWORK_PATH}/float/resnet_v1_50_inference.pb \
       --input_node input \
       --output_node resnet_v1_50/predictions/Reshape_1 \
       --eval_batches 10 \
       --batch_size 10 \
       --eval_image_dir images/ \
       --eval_image_list images/list.txt \
