#!/bin/sh

EVAL_SCRIPT_PATH=example_file
EVAL_MODEL_PATH=vai_q_output

python ${EVAL_SCRIPT_PATH}/resnet_eval.py \
       --input_frozen_graph ${TF_NETWORK_PATH}/${EVAL_MODEL_PATH}/quantize_eval_model.pb  \
       --input_node input \
       --output_node resnet_v1_50/predictions/Reshape_1 \
       --eval_batches 100 \
       --batch_size 10 \
       --eval_image_dir images/ \
       --eval_image_list images/list.txt \

