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

echo  "********** Build classification c++ application ********"

export LD_LIBRARY_PATH=
make

echo -e " ****Setting up enviorment variables and run inference on Alveo FPGA CARD****\n\n"

VAI_ALVEO_ROOT=../../../
ROOT_PATH=$VAI_ALVEO_ROOT

export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$VAI_ALVEO_ROOT/vai/dpuv1/rt/xdnn_cpp/lib/:/wrk/acceleration/users/anup/anaconda2/envs/ml-suite-py3/lib:$VAI_ALVEO_ROOT/apps/yolo/nms/:$LD_LIBRARY_PATH

 
./classify.exe --xclbin /opt/xilinx/overlaybins/xdnnv3 --datadir $ROOT_PATH/examples/deployment_modes/data/googlenet_v1_data --netcfg $ROOT_PATH/examples/deployment_modes/data/googlenet_v1_56.json --quantizecfg $ROOT_PATH/examples/deployment_modes/data/googlenet_v1_8b.json --labels $ROOT_PATH/examples/deployment_modes/synset_words.txt --image $ROOT_PATH/examples/deployment_modes/dog.jpg --in_w 224 --in_h 224 --out_w 1 --out_h 1 --out_d 1024 --batch_sz 2


echo -e "cleaning up"

make clean
