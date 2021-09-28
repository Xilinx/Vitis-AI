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

set PROJ "gemm_test_Dfloat_m32_n16_k64_par8.prj"
set SOLN "sol"

if {![info exists CLKP]} {
  set CLKP 5
}

open_project -reset $PROJ

add_files "${XF_PROJ_ROOT}/L1/tests/hw/gemm/uut_top.cpp" -cflags "-I${XF_PROJ_ROOT}/L1/include/hw -I${XF_PROJ_ROOT}/L1/tests/hw/gemm -I${XF_PROJ_ROOT}/L1/include/hw/xf_blas/helpers/utils -I${XF_PROJ_ROOT}/L1/tests/hw/gemm/tests/Dfloat_m32_n16_k64_par8 -std=c++11 "
add_files -tb "${XF_PROJ_ROOT}/L1/tests/hw/gemm/test.cpp" -cflags "-std=c++11 -I${XF_PROJ_ROOT}/L1/include/hw -I${XF_PROJ_ROOT}/L1/tests/sw/include -I${XF_PROJ_ROOT}/L1/tests/hw/gemm -I${XF_PROJ_ROOT}/L1/include/hw/xf_blas/helpers/utils -I${XF_PROJ_ROOT}/L1/tests/hw/gemm/tests/Dfloat_m32_n16_k64_par8"
set_top uut_top

open_solution -reset $SOLN




set_part $XPART
create_clock -period $CLKP

if {$CSIM == 1} {
  csim_design -argv "${XF_PROJ_ROOT}/L1/tests/hw/gemm/tests/Dfloat_m32_n16_k64_par8/data/"
}

if {$CSYNTH == 1} {
  csynth_design
}

if {$COSIM == 1} {
  cosim_design -argv "${XF_PROJ_ROOT}/L1/tests/hw/gemm/tests/Dfloat_m32_n16_k64_par8/data/"
}

if {$VIVADO_SYN == 1} {
  export_design -flow syn -rtl verilog
}

if {$VIVADO_IMPL == 1} {
  export_design -flow impl -rtl verilog
}

exit