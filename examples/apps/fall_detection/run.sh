#!/usr/bin/env bash

usage() {
  echo -e ""
  echo "Usage:"
  echo "------------------------------------------------"
  echo "    ./run.sh --nfpga <number-of-fpgas> --dir <path-to-directory>"
  echo -e ""
  echo "Arguments:"
  echo "-----------------------------------------------------------------------------"
  echo "  -n  nFPGA      | --nfpga   nFPGA      Number of FPGAs (Connected on System)"
  echo "  -d  DIRECTORY  | --dir     DIRECTORY  Images/videos directory"
  echo "  -h             | --help               Print this message."
  echo "-----------------------------------------------------------------------------"
  echo -e ""
}

# Default
NUM_FPGA=""
DIRECTORY=""
VERBOSE=1

# Parse Options
while true
do
  if [[ -z "$1" ]]; then break; fi
  case "$1" in
    -n  |--nfpga         ) NUM_FPGA="$2"         ; shift 2 ;;
    -d  |--dir           ) DIRECTORY="$2"        ; shift 2 ;;
    -v  |--verbose       ) VERBOSE="$2"          ; shift 2 ;;
    -h  |--help          ) usage                 ; exit  1 ;;
     *) echo "Unknown argument : $1";
        echo "Try ./run.sh -h to get correct usage. Exiting ...";
        exit 1 ;;
  esac
done

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

# Add Library Paths
if [ -d "${VAI_ALVEO_ROOT}/vai/dpuv1/rt/xdnn_cpp/lib" ]
then
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${VAI_ALVEO_ROOT}/vai/dpuv1/rt/xdnn_cpp/lib
fi
if [ -d "${VAI_ALVEO_ROOT}/vai/dpuv1/utils" ]
then
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${VAI_ALVEO_ROOT}/vai/dpuv1/utils
fi

# Add libs under current folder
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:libs
# Add libs under apps/aks/
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${VAI_ALVEO_ROOT}/apps/aks/libs:${VAI_ALVEO_ROOT}/apps/aks/
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

./build/fall_detection.exe $DIRECTORY
