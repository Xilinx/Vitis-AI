#
# Copyright 2019-2021 Xilinx, Inc.
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
set PROJ "snappy_decompress_test.prj"
set SOLN "sol1"
set CLKP 3.3

# Create a project
open_project -reset $PROJ

# Add design and testbench files
add_files snappy_decompress_test.cpp -cflags "-I${XF_PROJ_ROOT}/L1/include/hw"
add_files -tb snappy_decompress_test.cpp -cflags "-I${XF_PROJ_ROOT}/L1/include/hw"

# Set the top-level function
set_top snappyDecompressEngineRun

# Create a solution
open_solution -reset $SOLN

# Define technology and clock rate
set_part {xcu200}
create_clock -period $CLKP

config_compile -pragma_strict_mode

if {$CSIM == 1} {
  csim_design -O -argv "${XF_PROJ_ROOT}/L1/tests/snappy_decompress/sample.txt.snappy ${XF_PROJ_ROOT}/L1/tests/snappy_decompress/sample.txt"
}

if {$CSYNTH == 1} {
  csynth_design  
}

if {$COSIM == 1} {
  cosim_design -disable_dependency_check -O -argv "${XF_PROJ_ROOT}/L1/tests/snappy_decompress/sample.txt.snappy ${XF_PROJ_ROOT}/L1/tests/snappy_decompress/sample.txt"
}

if {$VIVADO_SYN == 1} {
  export_design -flow syn -rtl verilog
}

if {$VIVADO_IMPL == 1} {
  export_design -flow impl -rtl verilog
}

if {$QOR_CHECK == 1} {
  puts "QoR check not implemented yet"
}
exit
