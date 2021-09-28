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

pyGenMat=${XFLIB_DIR}/L2/src/python/genGemv.py
p_n = 256
p_m = 256
BLAS_deviceId = 0
BLAS_dataType = double

dataDir = ./data/
HOST_ARGS += $(p_m) $(p_n) $(dataDir) ${BLAS_deviceId}

data_gen:
	mkdir -p ${dataDir} 
	python3 ${pyGenMat} --p_m $(p_m) --p_n $(p_n) --path ${dataDir} --datatype ${BLAS_dataType}
