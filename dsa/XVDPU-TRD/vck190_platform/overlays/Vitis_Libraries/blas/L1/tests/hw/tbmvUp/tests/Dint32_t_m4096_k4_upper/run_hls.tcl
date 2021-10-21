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
#

source settings.tcl

set PROJ "tbmv_Dint32_t_m4096_k4_upper_test.prj"
set SOLN "sol"

if {![info exists CLKP]} {
  set CLKP 3.3333
}

open_project -reset $PROJ

add_files "${XF_PROJ_ROOT}/L1/tests/hw/tbmvUp/uut_top.cpp" -cflags "-I${XF_PROJ_ROOT}/L1/include/hw                -I${XF_PROJ_ROOT}/L1/include/hw/xf_blas                -I${XF_PROJ_ROOT}/L1/tests/ -I${XF_PROJ_ROOT}/L1/tests/hw -g -O0                -std=c++11 -DBLAS_L2=true -DBLAS_opName=tbmv -DBLAS_resDataType=uint32_t             -DBLAS_vectorSize=4096 -DBLAS_matrixSize=16777216             -DBLAS_memorySize=20480 -DBLAS_dataType=int32_t -DBLAS_logParEntries=-1             -DBLAS_parEntries=4"
add_files -tb "${XF_PROJ_ROOT}/L1/tests/sw/src/test.cpp" -cflags "-std=c++11 -I${XF_PROJ_ROOT}/L1/tests/hw/ -I${XF_PROJ_ROOT}/ -I${XF_PROJ_ROOT}/L1/tests/sw/include               -DBLAS_pageSizeBytes=4096               -DBLAS_memWidthBytes=64               -DBLAS_instrSizeBytes=8               -DBLAS_maxNumInstrs=16               -DBLAS_instrPageIdx=0                -DBLAS_paramPageIdx=1                -DBLAS_statsPageIdx=2  -DBLAS_L2=true -DBLAS_opName=tbmv -DBLAS_resDataType=uint32_t             -DBLAS_vectorSize=4096 -DBLAS_matrixSize=16777216             -DBLAS_memorySize=20480 -DBLAS_dataType=int32_t -DBLAS_logParEntries=-1             -DBLAS_parEntries=4"
set_top uut_top

open_solution -reset $SOLN



set_part $XPART
create_clock -period $CLKP

if {$CSIM == 1} {
  csim_design -argv "${XF_PROJ_ROOT}/L1/tests/hw/tbmvUp/tests/Dint32_t_m4096_k4_upper/TestBin.bin"
}

if {$CSYNTH == 1} {
  csynth_design
}

if {$COSIM == 1} {
  cosim_design -argv "${XF_PROJ_ROOT}/L1/tests/hw/tbmvUp/tests/Dint32_t_m4096_k4_upper/TestBin.bin"
}

if {$VIVADO_SYN == 1} {
  export_design -flow syn -rtl verilog
}

if {$VIVADO_IMPL == 1} {
  export_design -flow impl -rtl verilog
}

exit