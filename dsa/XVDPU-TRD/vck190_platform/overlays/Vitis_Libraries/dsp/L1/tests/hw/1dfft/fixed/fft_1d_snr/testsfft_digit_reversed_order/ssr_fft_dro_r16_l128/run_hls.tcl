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

source settings.tcl

set PROJ "prj_ssr_fft_dro_reg_test_r16_l128.prj"
set SOLN "solution1"

if {![info exists CLKP]} {
  set CLKP 3.3
}

open_project -reset $PROJ

add_files "src/main.cpp src/hls_ssr_fft_dro_data_path.hpp src/DEBUG_CONSTANTS.hpp" -cflags "-I${XF_PROJ_ROOT}/L1/include/hw/vitis_fft/fixed  -I${XF_PROJ_ROOT}/L1/tests/common"
add_files -tb "src/main.cpp ${XF_PROJ_ROOT}/L1/tests/hw/1dfft/fixed/commonFix/verif/fftStimulusIn_L128.verif ${XF_PROJ_ROOT}/L1/tests/hw/1dfft/fixed/commonFix/verif/fftGoldenOut_L128.verif" -cflags "-I${XF_PROJ_ROOT}/L1/include/hw/vitis_fft/fixed  -I${XF_PROJ_ROOT}/L1/tests/common"
set_top fft_top

open_solution -reset $SOLN




set_part $XPART
create_clock -period $CLKP

if {$CSIM == 1} {
  csim_design
}

if {$CSYNTH == 1} {
  csynth_design
}

if {$COSIM == 1} {
  cosim_design
}

if {$VIVADO_SYN == 1} {
  export_design -flow syn -rtl verilog
}

if {$VIVADO_IMPL == 1} {
  export_design -flow impl -rtl verilog
}

exit