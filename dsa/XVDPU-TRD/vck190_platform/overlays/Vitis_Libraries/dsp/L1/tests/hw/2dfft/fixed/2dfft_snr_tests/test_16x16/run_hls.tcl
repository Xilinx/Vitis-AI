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

set PROJ "prj_2dfft_snr_test.prj"
set SOLN "solution1"

if {![info exists CLKP]} {
  set CLKP 3.33
}

open_project -reset $PROJ

add_files "src/top_2d_fft_test.cpp" -cflags "-I${XF_PROJ_ROOT}/L1/include/hw/vitis_2dfft/fixed/"
add_files -tb "${XF_PROJ_ROOT}/L1/tests/common_2dfft/2dFFTVerificationData/d_2dFFTCrandomData/d_2dFFTCrandomData_2/fft2DGoldenOut_L1024.verif ${XF_PROJ_ROOT}/L1/tests/common_2dfft/2dFFTVerificationData/d_2dFFTCrandomData/d_2dFFTCrandomData_2/fft2DGoldenOut_L16.verif ${XF_PROJ_ROOT}/L1/tests/common_2dfft/2dFFTVerificationData/d_2dFFTCrandomData/d_2dFFTCrandomData_2/fft2DGoldenOut_L16384.verif ${XF_PROJ_ROOT}/L1/tests/common_2dfft/2dFFTVerificationData/d_2dFFTCrandomData/d_2dFFTCrandomData_2/fft2DGoldenOut_L256.verif ${XF_PROJ_ROOT}/L1/tests/common_2dfft/2dFFTVerificationData/d_2dFFTCrandomData/d_2dFFTCrandomData_2/fft2DGoldenOut_L4096.verif ${XF_PROJ_ROOT}/L1/tests/common_2dfft/2dFFTVerificationData/d_2dFFTCrandomData/d_2dFFTCrandomData_2/fft2DGoldenOut_L64.verif ${XF_PROJ_ROOT}/L1/tests/common_2dfft/2dFFTVerificationData/d_2dFFTCrandomData/d_2dFFTCrandomData_2/fft2DGoldenOut_L65536.verif ${XF_PROJ_ROOT}/L1/tests/common_2dfft/2dFFTVerificationData/d_2dFFTCrandomData/d_2dFFTCrandomData_2/fft2DStimulusIn_L1024.verif ${XF_PROJ_ROOT}/L1/tests/common_2dfft/2dFFTVerificationData/d_2dFFTCrandomData/d_2dFFTCrandomData_2/fft2DStimulusIn_L16.verif ${XF_PROJ_ROOT}/L1/tests/common_2dfft/2dFFTVerificationData/d_2dFFTCrandomData/d_2dFFTCrandomData_2/fft2DStimulusIn_L16384.verif ${XF_PROJ_ROOT}/L1/tests/common_2dfft/2dFFTVerificationData/d_2dFFTCrandomData/d_2dFFTCrandomData_2/fft2DStimulusIn_L256.verif ${XF_PROJ_ROOT}/L1/tests/common_2dfft/2dFFTVerificationData/d_2dFFTCrandomData/d_2dFFTCrandomData_2/fft2DStimulusIn_L4096.verif ${XF_PROJ_ROOT}/L1/tests/common_2dfft/2dFFTVerificationData/d_2dFFTCrandomData/d_2dFFTCrandomData_2/fft2DStimulusIn_L64.verif ${XF_PROJ_ROOT}/L1/tests/common_2dfft/2dFFTVerificationData/d_2dFFTCrandomData/d_2dFFTCrandomData_2/fft2DStimulusIn_L65536.verif src/main_2d_fft_test.cpp src/top_2d_fft_test.cpp" -cflags "-I${XF_PROJ_ROOT}/L1/include/hw/vitis_2dfft/fixed/ -Wno-unknown-pragmas"
set_top top_fft2d

open_solution -reset $SOLN




set_part $XPART
create_clock -period $CLKP
set_clock_uncertainty 0.10

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