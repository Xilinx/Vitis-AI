#!/bin/bash
# XRT setting
#SD_ROOT=/media/card
SD_ROOT=/run/media/mmcblk0p1

export XILINX_XRT=/usr # /opt/xilinx/xrt
export LD_LIBRARY_PATH=/usr/lib:${SD_ROOT}/models
export PATH=$XILINX_XRT/bin:$PATH
export PYTHONPATH=$XILINX_XRT/python:$PYTHONPATH
export PATH=$PATH:/media/card/bin


alias ll='ls -lh'
alias c="clear"

cd ${SD_ROOT}
cp ${SD_ROOT}/dnndk/lib*so ${SD_ROOT}/dpu.xclbin  /usr/lib
cp ${SD_ROOT}/models/* /usr/lib


#ifconfig eth0 192.168.1.2 netmask 255.255.255.0
# route add default gw 192.168.1.3

