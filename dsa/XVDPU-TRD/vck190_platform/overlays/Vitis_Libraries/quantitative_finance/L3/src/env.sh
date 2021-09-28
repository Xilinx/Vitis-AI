#!/bin/bash
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



##SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
SCRIPTPATH="$(readlink -f $(dirname ${BASH_SOURCE[0]}))"



L3_DIR="$(dirname $SCRIPTPATH)"
FINTECH_DIR="$(dirname $L3_DIR)"


export XILINX_FINTECH_L3_INC="$FINTECH_DIR/L3/include"
export XILINX_FINTECH_L2_INC="$FINTECH_DIR/L2/include"
export XILINX_FINTECH_LIB_DIR="$L3_DIR/src/output"
export XILINX_XCL2_DIR="$FINTECH_DIR/ext/xcl2"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$XILINX_FINTECH_LIB_DIR"




echo "XILINX_FINTECH_L3_INC   : $XILINX_FINTECH_L3_INC"
echo "XILINX_FINTECH_L2_INC   : $XILINX_FINTECH_L2_INC"
echo "XILINX_FINTECH_LIB_DIR  : $XILINX_FINTECH_LIB_DIR"
echo "XILINX_XCL2_DIR         : $XILINX_XCL2_DIR"
echo "LD_LIBRARY_PATH         : $LD_LIBRARY_PATH"
