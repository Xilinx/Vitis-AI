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


# Note:
# First you should back up your exporter.py in YourPath/tensorflow/models/research/object_detection
# Then copy exporter_without_pre_post_process.py to YourPath/tensorflow/models/research/object_detection/exporter.py
# Finally run this script to export the inference model without the preprocess and postprocess

export CUDA_VISIBLE_DEVICES=0
MODEL_DIR=Your/model/directory
python tensorflow/models/research/object_detection/export_inference_graph.py \
    --input_type float_image_tensor \
    --input_shape -1,640,640,3 \
    --pipeline_config_path ${MODEL_DIR}/pipeline.config \
    --trained_checkpoint_prefix ${MODEL_DIR}/model.ckpt \
    --output_directory model_without_pre_post_process \
    --write_inference_graph True \
