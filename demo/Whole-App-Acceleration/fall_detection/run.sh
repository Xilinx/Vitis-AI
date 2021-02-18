#!/usr/bin/env bash

usage() {
  echo -e ""
  echo "Usage:"
  echo "------------------------------------------------"
  echo "    ./run.sh --nfpga <number-of-fpgas> --dir <path-to-directory>"
  echo -e ""
  echo "Arguments:"
  echo "-----------------------------------------------------------------------------"
  echo "  -n  nFPGA      | --nfpga        nFPGA      Number of FPGAs (Connected on System)"
  echo "  -d  DIRECTORY  | --dir          DIRECTORY  Images/videos directory"
  echo "  -a  AKS_ROOT   | --aks-root     AKS_ROOT   Path to AKS root directory"
  echo "  -v  VERBOSE    | --verbose      VERBOSE Verbosity level"
  echo "                                       Possible values: [0 - Only Warnings & Errors]"
  echo "                                       Possible values: [1 - Important Information, warnings & errors]"
  echo "                                       Possible values: [2 - All debug, performance metrics, warnings & errors]"
  echo "  -h             | --help         Print this message."
  echo "-----------------------------------------------------------------------------"
  echo -e ""
}

# Default
NUM_FPGA=""
DIRECTORY=""
VERBOSE=1
AKS_ROOT=${VAI_HOME}/tools/AKS

# Parse Options
while true
do
  if [[ -z "$1" ]]; then break; fi
  case "$1" in
    -n  |--nfpga         ) NUM_FPGA="$2"         ; shift 2 ;;
    -d  |--dir           ) DIRECTORY="$2"        ; shift 2 ;;
    -a  |--aks-root      ) AKS_ROOT="$2"         ; shift 2 ;;
    -v  |--verbose       ) VERBOSE="$2"          ; shift 2 ;;
    -h  |--help          ) usage                 ; exit  1 ;;
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

export AKS_ROOT
# Add verbose level
export AKS_VERBOSE=$VERBOSE

CARDS_CONNECTED=`xbutil scan | grep "xilinx_u" | wc -l`
if [ ! -z "${NUM_FPGA}" ]
then
  if [ ${NUM_FPGA} -gt ${CARDS_CONNECTED} ] 
  then
    echo -e "[ERROR] Number of FPGAs mentioned are more than installed on System."
    echo -e "[ERROR] Cards Mentioned: ${NUM_FPGA}"
    echo -e "[ERROR] Cards Connected: ${CARDS_CONNECTED}"
    echo -e ""
    exit 1
  else
    # Number of FPGAs to be used
    export NUM_FPGA=$NUM_FPGA
  fi
else
  unset NUM_FPGA
fi

if [ -z ${DIRECTORY} ];
then
  usage;
  exit 1;
fi

AKS_GRAPH_META_URL="https://www.xilinx.com/bin/public/openDownload?filename=meta_vgg_fall_detection.zip"
NAME="graph_zoo/meta_vgg_fall_detection"
if [[ ! -d "${NAME}" ]]; then
    echo -e "$NAME doesn't exist"
    wget $AKS_GRAPH_META_URL -O temp.zip && unzip temp.zip -d graph_zoo/ && rm temp.zip
    if [[ $? != 0 ]]; then echo "Network download failed. Exiting ..."; exit 1; fi;
fi;

# Add Library Paths
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:libs
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${AKS_ROOT}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

./src/bin/main.exe $DIRECTORY
