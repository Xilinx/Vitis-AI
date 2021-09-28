#
# Copyright 2019-2020 Xilinx, Inc.
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
#
#
PY_SCRIPT=${XFLIB_DIR}/L1/tests/sw/python/rtm2d/operation.py
RTM_dataType      = float
RTM_width  = 128
RTM_depth  = 128
RTM_maxDim  = 1280
RTM_MaxB = 40
RTM_NXB = $(RTM_MaxB)
RTM_NZB = $(RTM_MaxB)
RTM_order  = 8
RTM_time = 10
RTM_shots = 1
RTM_verify=1
RTM_deviceId = 0

DATA_DIR= ./$(BUILD_DIR)/dataset_h${RTM_depth}_w${RTM_width}_t${RTM_time}/
HOST_ARGS = $(BINARY_CONTAINERS) ${DATA_DIR} $(RTM_depth) $(RTM_width) $(RTM_time) ${RTM_verify} ${RTM_deviceId}

data_gen:
	mkdir -p ${DATA_DIR} 
	python3 ${PY_SCRIPT} --func testBackward --path ${DATA_DIR} --depth ${RTM_depth} --width ${RTM_width} --time ${RTM_time} --nxb ${RTM_NXB} --nzb ${RTM_NZB} --order ${RTM_order} --verify ${RTM_verify} --shot ${RTM_shots} --type ${RTM_dataType}
