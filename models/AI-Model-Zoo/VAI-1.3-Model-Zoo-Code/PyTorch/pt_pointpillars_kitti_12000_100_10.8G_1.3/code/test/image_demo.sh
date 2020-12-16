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



# bash code/test/prepare_data.sh

echo "Conducting test..."

export CUDA_VISIBLE_DEVICES=0
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
export PYTHONPATH=${PWD}/code/test:${PYTHONPATH}
export CUDA_HOME=/usr/local/cuda

CONFIG=float/config.proto
WEIGHTS=float/float.pb
QUANT_DIR=quantized
DATA_ROOT=data/KITTI/KITTI_Raw_Data/2011_09_26/2011_09_26_drive_0095_sync
FRAME_IDX=0
CALIB_DIR=data/KITTI/KITTI_Raw_Data/2011_09_26
SCORE_THRESH=0.5
RESULT=data/preds

export W_QUANT=0
QUANT_MODE=float
# export W_QUANT=1
# QUANT_MODE=test
python code/test/image_demo.py -config ${CONFIG} -weights ${WEIGHTS} -data_root ${DATA_ROOT} -frame_idx ${FRAME_IDX} -result_dir ${RESULT} -quant_dir ${QUANT_DIR} -quant_mode ${QUANT_MODE} -score_thresh ${SCORE_THRESH} -calib_dir ${CALIB_DIR}
