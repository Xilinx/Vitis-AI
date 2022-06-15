#!/usr/bin/env bash

usage() {
  echo -e ""
  echo "Usage:"
  echo "------------------------------------------------"
  echo "    ./run.sh --dir <path-to-directory>"
  echo -e ""
  echo "Arguments:"
  echo "-----------------------------------------------------------------------------"
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
DIRECTORY=""
VERBOSE=1
AKS_ROOT=${VAI_HOME}/src/AKS

# Parse Options
while true
do
  if [[ -z "$1" ]]; then break; fi
  case "$1" in
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

if [ -z ${DIRECTORY} ];
then
  usage;
  exit 1;
fi

AKS_GRAPH_META_URL="https://www.xilinx.com/bin/public/openDownload?filename=meta_tf_resnet50_dpuczdx8g_zcu104_tvm_v2.5.0.zip"
NAME="graph_zoo/meta_resnet50_dpuczdx8g_zcu104_tvm"
if [[ ! -d "${NAME}" ]]; then
    echo -e "$NAME doesn't exist"
    wget $AKS_GRAPH_META_URL -O temp.zip && unzip temp.zip -d graph_zoo/ && rm temp.zip
    if [[ $? != 0 ]]; then echo "Network download failed. Exiting ..."; exit 1; fi;
fi;

# Add Library Paths
LD_LIBRARY_PATH=${AKS_ROOT}:${LD_LIBRARY_PATH}

export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.7/site-packages/pyxir-0.4.0-py3.7-linux-aarch64.egg
export XLNX_VART_FIRMWARE=/usr/lib/dpu.xclbin
export TVM_NUM_THREADS=1
export TVM_BIND_THREADS=0

./examples/bin/resnet50_dpuczdx8g_zcu014_tvm.exe $DIRECTORY
