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

PLATFORMS="v70" 
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

  DPUCV2DX8G_v70 | dpuv70)
    if [ "${platform}" = "v70" ]; then
      export XCLBIN_PATH=/opt/xilinx/overlaybins/DPUCV2DX8G
      export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/DPUCV2DX8G/dpu_DPUCV2DX8G_250M_xilinx_v70_gen5x8_qdma_base_2.xclbin
    else
      export XCLBIN_PATH=
      export XLNX_VART_FIRMWARE=
    fi
    ;;

  *)
    echo "Invalid argument $1!!!!"
    echo "Please choose to use the following command:"
    echo "    source ./setup.sh DPUCV2DX8G_v70"
    echo "    source ./setup.sh dpuv70" 
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

