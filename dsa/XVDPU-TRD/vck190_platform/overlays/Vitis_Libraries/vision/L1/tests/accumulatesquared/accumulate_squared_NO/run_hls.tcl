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

set PROJ "accumulatesquared.prj"
set SOLN "sol1"

if {![info exists CLKP]} {
  set CLKP 3.3
}

open_project -reset $PROJ

add_files "${XF_PROJ_ROOT}/L1/examples/accumulatesquared/xf_accumulate_squared_accel.cpp" -cflags "-I${XF_PROJ_ROOT}/L1/include -I ./ -D__SDSVHLS__ -std=c++0x" -csimflags "-I${XF_PROJ_ROOT}/L1/include -I ./ -D__SDSVHLS__ -std=c++0x"
add_files -tb "${XF_PROJ_ROOT}/L1/examples/accumulatesquared/xf_accumulate_squared_tb.cpp" -cflags "-I${OPENCV_INCLUDE} -I${XF_PROJ_ROOT}/L1/include -I ./ -D__SDSVHLS__ -std=c++0x" -csimflags "-I${XF_PROJ_ROOT}/L1/include -I ./ -D__SDSVHLS__ -std=c++0x"
set_top accumulate_squared

open_solution -reset $SOLN

set_part $XPART
create_clock -period $CLKP

if {$CSIM == 1} {
  csim_design -ldflags "-L ${OPENCV_LIB} -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_flann -lopencv_features2d" -argv " ${XF_PROJ_ROOT}/data/128x128_1.png ${XF_PROJ_ROOT}/data/128x128_1.png "
}

if {$CSYNTH == 1} {
  csynth_design
}

if {$COSIM == 1} {
  cosim_design -ldflags "-L ${OPENCV_LIB} -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_flann -lopencv_features2d" -argv " ${XF_PROJ_ROOT}/data/128x128_1.png ${XF_PROJ_ROOT}/data/128x128_1.png "
}

if {$VIVADO_SYN == 1} {
  export_design -flow syn -rtl verilog
}

if {$VIVADO_IMPL == 1} {
  export_design -flow impl -rtl verilog
}

exit