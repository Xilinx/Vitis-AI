#!/usr/bin/env bash
# Copyright 2019 Xilinx Inc.
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

usage() {
  echo -e ""
  echo "Usage:"
  echo "------------------------------------------------"
  echo "    ./aks.sh --impl <py/cpp> --model <model-variant> --dir1 <image-dir> --dir2 <image-dir>"

  echo -e ""
  echo "Examples:"
  echo "------------------------------------------------"
  echo "Run GoogleNet with AKS C++: "
  echo "    ./aks.sh --impl cpp --model googlenet --dir1 ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min"
  echo -e ""
  echo "Run ResNet50 with AKS Python: "
  echo "    ./aks.sh --impl py --model resnet50 --dir1 ~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min"
  echo -e ""
  echo "Run Multinet example:"
  echo "    ./aks.sh --impl cpp --model googlenet_yolov2"
  echo "    ./aks.sh --impl cpp --model googlenet_resenet50"

  echo -e ""
  echo "Arguments:"
  echo "------------------------------------------------"
  echo "  -m  MODEL   | --model  MODEL    Model variant"
  echo "                                  Possible values: [googlenet, resnet50]"
  echo "                                  Possible values: [tinyyolov3, stdyolov2]"
  echo "                                  Possible values: [googlenet_tinyyolov3]"
  echo "                                  Possible values: [googlenet_resnet50]"
  echo "  -i  IMPL    | --impl   IMPL     Implemetation"
  echo "                                  Possible values: [cpp, py]"
  echo "  -d1 IMAGES  | --dir1   IMAGES   Classification Network's Image Directory"
  echo "  -d2 IMAGES  | --dir2   IMAGES   Detection Network's Image Directory"
  echo "  -h          | --help            Print this message."
  echo "------------------------------------------------"
  echo -e ""
}

# Default
MODEL="googlenet"
IMPL="cpp"
VERBOSE=1
C_DIRECTORY=~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min
D_DIRECTORY=../yolo/test_image_set
URL_v2="https://www.xilinx.com/bin/public/openDownload?filename=models.caffe.yolov2_2019-08-01.zip"

# Parse Options
while true
do
  if [[ -z "$1" ]]; then break; fi
  case "$1" in
    -m  |--model         ) MODEL="$2"            ; shift 2 ;;
    -d1 |--dir1          ) C_DIRECTORY="$2"      ; shift 2 ;;
    -d2 |--dir2          ) D_DIRECTORY="$2"      ; shift 2 ;;
    -i  |--impl          ) IMPL="$2"             ; shift 2 ;;
    -v  |--verbose       ) VERBOSE="$2"          ; shift 2 ;;
    -h  |--help          ) usage                 ; exit  1 ;;
     *) echo "Unknown argument : $1";
        echo "Try ./aks.sh -h to get correct usage. Exiting ...";
        exit 1 ;;
  esac
done

# Supported Modes & Models
declare -A SUPPORTED_MODELS
for name in "googlenet" "resnet50" "googlenet_resnet50" "tinyyolov3" "googlenet_tinyyolov3" "stdyolov2"
do
    SUPPORTED_MODELS[$name]=1
done

if [[ ${SUPPORTED_MODELS["$MODEL"]} ]]; then 
  # Start Execution
  echo -e ""
  echo -e "[INFO] Running"
  echo -e "[INFO] Model: $MODEL with $IMPL"
  echo -e ""
else
  echo -e ""
  echo -e "[ERROR] $MODEL is an invalid model !"
  echo -e "[ERROR] Check Usage with: ./aks.sh -h "
  echo -e ""
  exit 1
fi

# Add verbose level
export AKS_VERBOSE=$VERBOSE

# Add AKS utils to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:${VAI_ALVEO_ROOT}

# Add Library Paths
if [ -d "${VAI_ALVEO_ROOT}/vai/dpuv1/rt/xdnn_cpp/lib" ]
then
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${VAI_ALVEO_ROOT}/vai/dpuv1/rt/xdnn_cpp/lib
fi
if [ -d "${VAI_ALVEO_ROOT}/vai/dpuv1/utils" ]
then
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${VAI_ALVEO_ROOT}/vai/dpuv1/utils
fi
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${VAI_ALVEO_ROOT}/apps/aks/libs
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

# Create meta.json
META_JSON="{ 
  \"target\": \"xdnn\",
  \"filename\": \"\", \"kernel\": \"xdnn\", \"config_file\": \"\", 
  \"lib\": \"${LIBXDNN_PATH}\", 
  \"xclbin\": \"/opt/xilinx/overlaybins/xdnnv3\", 
  \"publish_id\": \"$BASHPID\"
}"

CPP_EXE=""
PY_EXE=""
NET_DEF=""
NET_WEIGHTS=""

# Model Selection
if [ "$MODEL" == "googlenet" ]; then
  echo $META_JSON > graph_zoo/meta_googlenet/meta.json
  CPP_EXE=examples/bin/googlenet.exe
  PY_EXE=examples/googlenet.py
  D_DIRECTORY=""

elif [ "$MODEL" == "resnet50" ]; then
  echo $META_JSON > graph_zoo/meta_resnet50/meta.json
  CPP_EXE=examples/bin/resnet50.exe
  PY_EXE=examples/resnet50.py
  D_DIRECTORY=""

elif [ "$MODEL" == "tinyyolov3" ]; then
  echo $META_JSON > graph_zoo/meta_tinyyolov3/meta.json
  CPP_EXE=examples/bin/tinyyolov3.exe
  PY_EXE=examples/tinyyolov3.py
  C_DIRECTORY=""

elif [ "$MODEL" == "stdyolov2" ]; then
  echo $META_JSON > graph_zoo/meta_stdyolov2/meta.json
  CPP_EXE=examples/bin/stdyolov2.exe
  PY_EXE=examples/stdyolov2.py
  C_DIRECTORY=""
  NET_DEF=${VAI_ALVEO_ROOT}/models/caffe/yolov2/fp32/yolo_v2_standard_608.prototxt
  NET_WEIGHTS=${VAI_ALVEO_ROOT}/models/caffe/yolov2/fp32/yolo_v2_standard.caffemodel

elif [ "$MODEL" == "googlenet_resnet50" ]; then
  CPP_EXE=examples/bin/googlenet_resnet50.exe
  PY_EXE=examples/googlenet_resnet50.py
  D_DIRECTORY=""

elif [ "$MODEL" == "googlenet_tinyyolov3" ]; then
  CPP_EXE=examples/bin/googlenet_tinyyolov3.exe
  PY_EXE=examples/googlenet_tinyyolov3.py
fi

if [[ ( ! -z $NET_DEF ) && ( ( ! -f $NET_DEF ) || ( ! -f $NET_WEIGHTS ) ) ]]; then
  echo "$MODEL does not exist on disk :("
  echo "Downloading the models ..."
  cd $VAI_ALVEO_ROOT && wget $URL_v2 -O temp.zip && unzip temp.zip && rm -rf temp.zip && cd -
  if [[ $? != 0 ]]; then echo "Network download failed. Exiting ..."; exit 1; fi;
  if [ ! -f "$NET_DEF" ]; then echo "Couldn't find $NET_DEF in models. Please check the filename. Exiting ..."; exit 1; fi;
fi


if [[ "$IMPL" == "cpp" ]]; then
  ${CPP_EXE} ${C_DIRECTORY} ${D_DIRECTORY}
elif [[ "$IMPL" == "py" ]]; then
  python ${PY_EXE} ${C_DIRECTORY} ${D_DIRECTORY}
fi
