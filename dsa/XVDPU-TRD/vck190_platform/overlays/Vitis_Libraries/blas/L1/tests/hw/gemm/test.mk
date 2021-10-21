#
# Copyright 2019 Xilinx, Inc.
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

BLAS_dataType=float
BLAS_parEntries=4
BLAS_m=16
BLAS_k=16
BLAS_n=16
BLAS_alpha=1
BLAS_beta=1
BLAS_matrixSizeA=${BLAS_m} * ${BLAS_k}
BLAS_matrixSizeB=${BLAS_k} * ${BLAS_n}
BLAS_matrixSizeC=${BLAS_m} * ${BLAS_n}
TEST_DIR=D${BLAS_dataType}_m${BLAS_m}_n${BLAS_n}_k${BLAS_k}_par${BLAS_parEntries}

PYTHON_SCRIPT=../../sw/python/data_gen.py
DESCRIPTION_SCRIPT=../../sw/python/description.py

params:
	@echo "#pragma once"					> params.hpp
	@echo "#define BLAS_dataType ${BLAS_dataType}"  	>> params.hpp
	@echo "#define BLAS_parEntries ${BLAS_parEntries}" 	>> params.hpp
	@echo "#define BLAS_matrixSizeA ${BLAS_matrixSizeA}"	>> params.hpp
	@echo "#define BLAS_matrixSizeB ${BLAS_matrixSizeB}"	>> params.hpp
	@echo "#define BLAS_matrixSizeC ${BLAS_matrixSizeC}"	>> params.hpp
	@echo "#define BLAS_m ${BLAS_m}"			>> params.hpp
	@echo "#define BLAS_k ${BLAS_k}"			>> params.hpp
	@echo "#define BLAS_n ${BLAS_n}"			>> params.hpp
	@echo "#define BLAS_alpha ${BLAS_alpha}"		>> params.hpp
	@echo "#define BLAS_beta ${BLAS_beta}"			>> params.hpp
	
hls: data params
	vitis_hls -f run_hls.tcl
	
data:
	python ${PYTHON_SCRIPT} --m ${BLAS_m} --n ${BLAS_n} --k ${BLAS_k} --alpha ${BLAS_alpha} --beta ${BLAS_beta} --func gemm --dirname ./data --dataType  ${BLAS_dataType}
	
description:
	python ${DESCRIPTION_SCRIPT} --testname ${TEST_DIR} --func gemm

generate: data params description
	mkdir -p ./${TEST_DIR}
	mv params.hpp ${TEST_DIR}/params.hpp
	mv ./data ${TEST_DIR}/data
	mv description.json ${TEST_DIR}/description.json
	mv ${TEST_DIR} ./tests/${TEST_DIR}

clean:
	@rm -rf gemm_test.prj/
	@rm -rf vitis_hls.log

cleanall: clean
	@rm -rf data/
	@rm -rf params.hpp