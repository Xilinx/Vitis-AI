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


GPUS=0
WEIGHTS=float/float.pb
DATASET=data/coco2014_minival_8059
IMG_LIST=${DATASET}/minival2014_8059.txt
IMG_ROOT=${DATASET}/image
GT_JSON=${DATASET}/minival2014_8059.json
DT_JSON=data/ssd_prediction.json
TEST_LOG=data/log.txt
export PYTHONPATH=code/test/dataset_tools:${PYTHONPATH}
python code/test/ssd_detector.py \
    --model_type ssd_mobilenet_v2 \
    --input_graph ${WEIGHTS} \
    --eval_image_path ${IMG_ROOT} \
    --eval_image_list ${IMG_LIST} \
    --gt_json ${GT_JSON} \
    --det_json ${DT_JSON} \
    --gpus ${GPUS} | tee -a ${TEST_LOG}
echo "Test report is saved to ${TEST_LOG}"
