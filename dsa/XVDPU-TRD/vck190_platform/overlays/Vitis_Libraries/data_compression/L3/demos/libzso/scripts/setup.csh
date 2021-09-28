#!/bin/csh
if ( $#argv != 1 ) then
    echo "Input Argument Missing"
    echo "Usage: source setup.csh <absolute path to xclbin>"
    exit 1
endif

source /opt/xilinx/xrt/setup.csh
setenv XILINX_LIBZ_XCLBIN $1
setenv XRT_INI_PATH $PWD/xrt.ini
setenv LD_LIBRARY_PATH ${PWD}:$LD_LIBRARY_PATH
source /opt/xilinx/xrm/setup.csh
systemctl status xrmd
echo "Run ./scripts/xrmxclbin.sh <no of devices>"
