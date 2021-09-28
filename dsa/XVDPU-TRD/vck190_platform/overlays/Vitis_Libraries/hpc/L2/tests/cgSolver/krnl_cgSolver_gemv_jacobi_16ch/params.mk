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

pyGenMat=${XFLIB_DIR}/L2/src/python/cgSolver/gemv/genMat.py
maxIter=100
vectorDim = 64
CG_deviceId = 0
CG_dataType = double
tol=1e-8

dataDir = ./$(BUILD_DIR)/data/
HOST_ARGS += $(maxIter) ${tol} $(vectorDim) $(dataDir) ${CG_deviceId}

data_gen:
	mkdir -p ${dataDir} 
	python3 ${pyGenMat} --dimension $(vectorDim) --path ${dataDir} --datatype ${CG_dataType} --maxIter ${maxIter} --preconditioner Jacobi --debug --verify
