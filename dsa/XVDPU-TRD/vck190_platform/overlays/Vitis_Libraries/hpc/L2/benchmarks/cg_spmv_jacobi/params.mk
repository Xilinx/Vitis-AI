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

pyGenSig=${XFLIB_DIR}/../sparse/L2/tests/fp64/spmv/python/gen_signature.py
pyGenVec=${XFLIB_DIR}/../sparse/L2/tests/fp64/spmv/python/gen_vectors.py
mtxList = ${XFLIB_DIR}/L2/benchmarks/cg_spmv_jacobi/test.txt

dataPath = ./$(BUILD_DIR)/data/
sigPath = ./$(BUILD_DIR)/sigs/

deviceID = 0
maxIter = 5000
tol=1e-12
mtxName = ted_B

HOST_ARGS += ${maxIter} ${tol} ${sigPath} ${dataPath} $(mtxName) ${deviceID}

data_gen:
	rm -rf ${sigPath}
	rm -rf ${dataPath}
	python ${pyGenSig}  --partition --mtx_list ${mtxList} --sig_path ${sigPath}
	python ${pyGenVec}  --gen_vec --pcg --mtx_list ${mtxList} --vec_path ${dataPath}
