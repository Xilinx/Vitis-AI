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

set PROJ "forward_RTM_x50_y60_z80_t9_p2_test.prj"
set SOLN "sol"

if {![info exists CLKP]} {
  set CLKP 3.3333
}

open_project -reset $PROJ

add_files "${XF_PROJ_ROOT}/L1/tests/hw/rtm3d/forward_rbc/tests/RTM_x50_y60_z80_t9_p2/../../forward.cpp" -cflags "-I${XF_PROJ_ROOT}/L1/tests/hw/rtm3d/forward_rbc/tests/RTM_x50_y60_z80_t9_p2 -I${XF_PROJ_ROOT}/L1/include/hw -I${XF_PROJ_ROOT}/L2/include/hw/rtm3d -I${XF_PROJ_ROOT}/../blas/L1/include/hw"
add_files -tb "${XF_PROJ_ROOT}/L1/tests/hw/rtm3d/forward_rbc/tests/RTM_x50_y60_z80_t9_p2/../../main.cpp" -cflags "-std=c++14 -I${XF_PROJ_ROOT}/L1/tests/hw/rtm3d/forward_rbc/tests/RTM_x50_y60_z80_t9_p2 -I${XF_PROJ_ROOT}/L1/include/hw -I${XF_PROJ_ROOT}/../blas/L1/include/hw -I${XF_PROJ_ROOT}/../blas/L1/tests/sw/include"
set_top top

open_solution -reset $SOLN




set_part $XPART
create_clock -period $CLKP

if { [ file exists directives.tcl ] == 1 } {
    source directives.tcl
}


if {$CSIM == 1} {
  csim_design -argv "${XF_PROJ_ROOT}/L1/tests/hw/rtm3d/forward_rbc/tests/RTM_x50_y60_z80_t9_p2/data/"
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