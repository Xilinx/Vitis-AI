#!/usr/bin/env bash

usage() {
  echo -e ""
  echo "Usage:"
  echo "------------------------------------------------"
  echo "    ./run.sh --dir <path-to-directory> --num-threads <num-of-threads> --verbose <AKS_VERBOSE>"
  echo -e ""
  echo "Arguments:"
  echo "-----------------------------------------------------------------------------"
  echo "  -d  DIRECTORY                | --dir          DIRECTORY                Images/videos directory"
  echo "  -a  AKS_ROOT                 | --aks-root     AKS_ROOT                 Path to AKS root directory"
  echo "  -t  MAX_NUM_CONCURRENT_STREAMS   | --num-threads  MAX_NUM_CONCURRENT_STREAMS   Path to AKS root directory"
  echo "  -v  VERBOSE                  | --verbose      VERBOSE                  Verbosity level"
  echo "                                   Possible values: [0 - Only Warnings & Errors]"
  echo "                                   Possible values: [1 - Important Information, warnings & errors]"
  echo "                                   Possible values: [2 - All debug, performance metrics, warnings & errors]"
  echo "  -h                           | --help                                  Print this message."
  echo "-----------------------------------------------------------------------------"
  echo -e ""
}

# Default
DIRECTORY=""
MAX_NUM_CONCURRENT_STREAMS=""
VERBOSE=1
AKS_ROOT=${VAI_HOME}/tools/AKS

# Parse Options
while true
do
  if [[ -z "$1" ]]; then break; fi
  case "$1" in
    -d  |--dir           ) DIRECTORY="$2"                       ; shift 2 ;;
    -t  |--num-threads   ) MAX_NUM_CONCURRENT_STREAMS="$2"      ; shift 2 ;;
    -a  |--aks-root      ) AKS_ROOT="$2"                        ; shift 2 ;;
    -v  |--verbose       ) VERBOSE="$2"                         ; shift 2 ;;
    -h  |--help          ) usage                                ; exit  1 ;;
     *) echo "Unknown argument : $1";
        echo "Try ./run.sh -h to get correct usage. Exiting ...";
        exit 1 ;;
  esac
done

if [ ! -d "${AKS_ROOT}" ]
then
  echo -e "Path '${AKS_ROOT}' doesn't exist. Please specify a valid path to AKS."
  exit 1
fi

export XLNX_VART_FIRMWARE=/opt/xilinx/overlaybins/dpuv3int8/waa/lucas_kanade_of/dpuv3int8_lkof.hw.xilinx_u200_gen3x16_xdma_1_202110_1.xclbin
export AKS_ROOT
# Add verbose level
export AKS_VERBOSE=$VERBOSE

if [ -z ${DIRECTORY} ];
then
  usage;
  exit 1;
fi

AKS_GRAPH_META_URL="https://www.xilinx.com/bin/public/openDownload?filename=fall_detection.tar.gz"
NAME="model/fall_detection.xmodel"
if [[ ! -f "${NAME}" ]]; then
    echo -e "$NAME doesn't exist"
    wget $AKS_GRAPH_META_URL -O fall_detection.tar.gz && tar -xzvf fall_detection.tar.gz && rm fall_detection.tar.gz
    if [[ $? != 0 ]]; then echo "Network download failed. Exiting ..."; exit 1; fi;
fi;

# Add Library Paths
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:libs
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${AKS_ROOT}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

./src/bin/main.exe $DIRECTORY $MAX_NUM_CONCURRENT_STREAMS
