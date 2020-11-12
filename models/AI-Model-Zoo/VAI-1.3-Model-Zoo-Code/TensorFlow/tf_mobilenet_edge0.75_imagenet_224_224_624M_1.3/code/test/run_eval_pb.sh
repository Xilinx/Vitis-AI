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



INPUT_PB=../../float/frozen_mobilenet_edgetpu_224_075.pb
INPUT_HEIGHT=224
INPUT_WIDTH=224
INPUT_NODE=input
OUTPUT_NODE=MobilenetEdgeTPU/Predictions/Reshape_1
LABEL_OFFSET=0
IMAGE_DIR=../../data/Imagenet/val_dataset
IMAGE_LIST=../../data/Imagenet/val.txt



python eval_tf_classification_models_alone.py \
    --input_graph $INPUT_PB \
    --eval_image_path $IMAGE_DIR \
    --eval_image_list $IMAGE_LIST \
    --input_node $INPUT_NODE \
    --output_node $OUTPUT_NODE \
    --input_height $INPUT_HEIGHT \
    --input_width $INPUT_WIDTH \
    --label_offset $LABEL_OFFSET \
    --gpus 0
