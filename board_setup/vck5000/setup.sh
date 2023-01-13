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

PLATFORMS="vck5000_" 
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

  DPUCVDX8H_8pe_normal | dpuv4e-8pe-normal)
    if [ "${platform}" = "vck5000_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCVDX8H/8PE
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCVDX8H/8PE/dpu_DPUCVDX8H_8PE_350M_xilinx_vck5000_gen4x8_qdma_base_2.xclbin
    else
      export XCLBIN_PATH=
      export XLNX_VART_FIRMWARE=
    fi
    ;;

  DPUCVDX8H_6pe_dwc | dpuv4e-6pe-dwc)
    if [ "${platform}" = "vck5000_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCVDX8H/6PE_aieDWC
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCVDX8H/6PE_aieDWC/dpu_DPUCVDX8H_6PE_aieDWC_350M_xilinx_vck5000_gen4x8_qdma_base_2.xclbin
    else
      export XCLBIN_PATH=
      export XLNX_VART_FIRMWARE=
    fi
    ;;

    DPUCVDX8H_6pe_misc | dpuv4e-6pe-misc)
    if [ "${platform}" = "vck5000_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCVDX8H/6PE_aieMISC
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCVDX8H/6PE_aieMISC/dpu_DPUCVDX8H_6PE_aieMISC_350M_xilinx_vck5000_gen4x8_qdma_base_2.xclbin
    else
      export XCLBIN_PATH=
      export XLNX_VART_FIRMWARE=
    fi
    ;;

    DPUCVDX8H_4pe_miscdwc | dpuv4e-4pe-miscdwc)
    if [ "${platform}" = "vck5000_" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCVDX8H/4PE
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCVDX8H/4PE/dpu_DPUCVDX8H_4PE_350M_xilinx_vck5000_gen4x8_qdma_base_2.xclbin
    else
      export XCLBIN_PATH=
      export XLNX_VART_FIRMWARE=
    fi
    ;;

  *)
    echo "Invalid argument $1!!!!"
    echo "Please choose to use the following command:"
    echo "    source ./setup.sh DPUCVDX8H_4pe_miscdwc"
    echo "    source ./setup.sh DPUCVDX8H_6pe_dwc"
    echo "    source ./setup.sh DPUCVDX8H_6pe_misc"
    echo "    source ./setup.sh DPUCVDX8H_8pe_normal"
    ;;
esac

if [ "${find_card}" = "false" ]; then
  echo "Error: Can't find the supported xilinx Visual card"
  export XCLBIN_PATH=
  export XLNX_VART_FIRMWARE=
fi

echo "---------------------"
echo "XCLBIN_PATH = $XCLBIN_PATH"
echo "XLNX_VART_FIRMWARE = $XLNX_VART_FIRMWARE"
echo "---------------------"

