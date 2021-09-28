#!/bin/csh -f
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

#set called=($_)
set script_path=""
set fintech_l3_src_dir=""

# revisit if there is a better way than lsof to obtain the script path
# in non-interactive mode.  If lsof is needed, then revisit why
# why sbin need to be prepended looks like some environment issue in
# user shell, e.g. /usr/local/bin/mis_env: No such file or directory.
# is because user path contain bad directories that are searched when
# looking of lsof.
set path=(/usr/sbin $path)
set called=(`\lsof +p $$ |\grep env.csh`)

# look for the right cmd component that contains env.csh
foreach x ($called)
    if ( "$x" =~ *env.csh ) then
        set script_path=`readlink -f $x`
        set fintech_l3_src_dir=`dirname $script_path`
		break
    endif
end


set fintech_l3_dir=`dirname "$fintech_l3_src_dir"`
set fintech_dir=`dirname "$fintech_l3_dir"`



setenv XILINX_FINTECH_L3_INC $fintech_dir/L3/include
setenv XILINX_FINTECH_L2_INC $fintech_dir/L2/include
setenv XILINX_FINTECH_LIB_DIR $fintech_l3_dir/src/output
setenv XILINX_XCL2_DIR $fintech_dir/ext/xcl2



if ( ! $?LD_LIBRARY_PATH ) then
   setenv LD_LIBRARY_PATH $XILINX_FINTECH_LIB_DIR
else
   setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:$XILINX_FINTECH_LIB_DIR
endif



echo "XILINX_FINTECH_L3_INC   : $XILINX_FINTECH_L3_INC"
echo "XILINX_FINTECH_L2_INC   : $XILINX_FINTECH_L2_INC"
echo "XILINX_FINTECH_LIB_DIR  : $XILINX_FINTECH_LIB_DIR"
echo "XILINX_XCL2_DIR         : $XILINX_XCL2_DIR"
echo "LD_LIBRARY_PATH         : $LD_LIBRARY_PATH"

