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

PLATFORMS="u55c_ u50lv_ u200_ u250_ AWS_" 
find_card="false"

for platform in ${PLATFORMS};
do
xbutil examine | grep ${platform}
if [ $? -eq 0 ]; then
  echo "${platform} card detected"
  find_card="true"
  break
fi
done

case $1 in

  DPUCAHX8H | dpuv3e)
    if [ "${platform}" = "u50lv_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCAHX8H/u50lv
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCAHX8H/u50lv/dpu_DPUCAHX8H_10PE275_xilinx_u50lv_gen3x4_xdma_base_2.xclbin
    else
      export XCLBIN_PATH=
      export XLNX_VART_FIRMWARE=
    fi
    ;;
  
  DPUCAHX8H-DWC | dpuv3e-dwc)
    if [ "${platform}" = "u50lv_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCAHX8H/u50lvdwc
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCAHX8H/u50lvdwc/dpu_DPUCAHX8H_DWC_8PE275_xilinx_u50lv_gen3x4_xdma_base_2.xclbin
    elif [ "${platform}" = "u55c_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCAHX8H/u55cdwc
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCAHX8H/u55cdwc/dpu_DPUCAHX8H_DWC_11PE300_xilinx_u55c_gen3x16_xdma_base_2.xclbin
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
    elif [ "${platform}" = "AWS_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCADF8H
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCADF8H/dpu-aws.xclbin
    else
      export XCLBIN_PATH=
      export XLNX_VART_FIRMWARE=
    fi
    ;;

  *)
    echo "Invalid argument $1!!!!"
    echo "Please choose to use the following command:"
    echo "    source ./setup.sh DPUCADF8H"
    echo "    source ./setup.sh DPUCAHX8H"
    echo "    source ./setup.sh DPUCAHX8H-DWC"
    ;;

esac

if [ "${find_card}" = "false" ]; then
  echo "Error: Can't find the supported xilinx Alvio card"
  export XCLBIN_PATH=
  export XLNX_VART_FIRMWARE=
fi

echo "---------------------"
echo "XCLBIN_PATH = $XCLBIN_PATH"
echo "XLNX_VART_FIRMWARE = $XLNX_VART_FIRMWARE"
echo "---------------------"

