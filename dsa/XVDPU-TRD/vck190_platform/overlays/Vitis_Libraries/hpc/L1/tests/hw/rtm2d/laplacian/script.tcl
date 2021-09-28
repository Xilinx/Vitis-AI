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

set ROOT_DIR "../../../../"

set COMMON_FLAGS "-I${ROOT_DIR}/../xf_spatial/L1/tests/sw/include -I$ROOT_DIR/../xf_blas/L1/include/hw/ -I$ROOT_DIR/L1/include/hw -I${ROOT_DIR}/../xf_blas/L1/tests/sw/include/ -I${ROOT_DIR}/kernel/ -std=c++11"
set CFLAGS_K "${COMMON_FLAGS}"
set CFLAGS_H "${COMMON_FLAGS} -I${ROOT_DIR}/kernel/"

set HOST_ARGS [file normalize "$ROOT_DIR/L1/tests/hw/laplacian/data"]

open_project "hls_prj" -reset
set_top top 
add_files laplacian.cpp -cflags "$CFLAGS_K"
add_files -tb main.cpp -cflags "$CFLAGS_H"
open_solution sol -reset
set_part xcu280-fsvh2892-2L-e
create_clock -period 3.333333 -name default
csim_design -argv "$HOST_ARGS/"
csynth_design
cosim_design -trace_level all -argv "$HOST_ARGS/"
exit

