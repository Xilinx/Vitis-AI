#!/usr/bin/env bash
# Copyright 2021 Xilinx Inc.
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

if [[ "$CONDA_DEFAULT_ENV" = "base" ]]; then
  echo "WARNING: No conda environment has been activated."
fi

if [[ -z $VAI_HOME ]]; then
	export VAI_HOME="$( readlink -f "$( dirname "${BASH_SOURCE[0]}" )/../.." )"
fi

echo "------------------"
echo "VAI_HOME = $VAI_HOME"
echo "------------------"

source /opt/xilinx/xrt/setup.sh
echo "---------------------"
echo "XILINX_XRT = $XILINX_XRT"
echo "---------------------"

source /opt/xilinx/xrm/setup.sh
echo "---------------------"
echo "XILINX_XRM = $XILINX_XRM"
echo "---------------------"

echo "---------------------"
echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
echo "---------------------"

PLATFORMS="u50_ u50lv_ u200_ u250_ u280_" 

for platform in ${PLATFORMS};
do
xbutil scan | grep ${platform}
if [ $? -eq 0 ]; then
  echo "${platform} card detected"
  break
fi
done

case $1 in

  DPUCAHX8H | dpuv3e)
    if [ "${platform}" = "u50_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCAHX8H
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCAHX8H/dpuv3e_6E300_xilinx_u50_gen3x4_xdma_base_2.xclbin
    elif [ "${platform}" = "u50lv_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCAHX8H
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCAHX8H/dpuv3e_10E275_xilinx_u50lv_gen3x4_xdma_base_2.xclbin
    elif [ "${platform}" = "u280_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCAHX8H
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCAHX8H/dpuv3e_14E300_xilinx_u280_xdma_201920_3.xclbin
    else
      export XCLBIN_PATH=
      export XLNX_VART_FIRMWARE=
    fi
    ;;
  
  DPUCAHX8L | dpuv3me)
    if [ "${platform}" = "u50_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCAHX8L
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCAHX8L/dpuv3me_1E333_xilinx_u50_gen3x4_xdma_base_2.xclbin
    elif [ "${platform}" = "u50lv_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCAHX8L
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCAHX8L/dpuv3me_1E250_xilinx_u50lv_gen3x4_xdma_base_2.xclbin
    elif [ "${platform}" = "u280_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCAHX8L
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCAHX8L/dpuv3me_2E250_xilinx_u280_xdma_201920_3.xclbin
    else
      export XCLBIN_PATH=
      export XLNX_VART_FIRMWARE=
    fi
    ;;
  
  DPUCADF8H | dpuv3int8)
    if [ "${platform}" = "u200_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCADF8H
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCADF8H/dpdpuv3_wrapper.hw.xilinx_u200_gen3x16_xdma_1_202110_1.xclbin
    elif [ "${platform}" = "u250_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCADF8H
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCADF8H/dpdpuv3_wrapper.hw.xilinx_u250_gen3x16_xdma_3_1_202020_1.xclbin
    else
      export XCLBIN_PATH=
      export XLNX_VART_FIRMWARE=
    fi
    ;;

  *)
    echo "Invalid argument $1!!!!"
    ;;
esac


echo "---------------------"
echo "XCLBIN_PATH = $XCLBIN_PATH"
echo "XLNX_VART_FIRMWARE = $XLNX_VART_FIRMWARE"
echo "---------------------"

