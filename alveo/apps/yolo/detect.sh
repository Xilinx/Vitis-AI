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
  echo "Usage: ./detect.sh --test <test> --model <model> [...]"

  echo -e ""
  echo "Examples:"
  echo "Run Tiny YOLO v3 COCO model on CPU: "
  echo "    ./detect.sh --test cpu_detect --model tiny_yolo_v3"
  echo "Quantize, Compile & Run Tiny YOLO v3 COCO model on FPGA: "
  echo "    ./detect.sh --test test_detect --model tiny_yolo_v3"
  echo "Measure mAP on coco dataset for input resolution of 608x608:"
  echo "    ./detect.sh --test test_detect --model tiny_yolo_v3 --neth 608 --netw 608 -g <coco_gt_dir> -d <coco_val_data>"

  echo -e ""
  echo "Arguments"
  echo "  -t TEST, --test TEST          Mode of execution. Possible values: [test_detect, streaming_detect, cpu_detect]"
  echo "  -m MODEL, --model MODEL       Yolo model variant. Possible values: [yolo_v2, yolo_v2_prelu, standard_yolo_v3, "
  echo "                                                                  tiny_yolo_v3, yolo_v3_spp, custom]"
  echo "  -g DIR, --checkaccuracy DIR   Ground truth directory. Only required if mAP need to be calculated."
  echo "  -l FILE, --labels FILE        Label file containing names of each class. Used with -g flag (Default : coco.names)"
  echo "  --results_dir DIR             Directory to write the results. (Default : ./out_labels)."

  echo -e ""
  echo "Other Arguments"
  echo "  -d DIR, --directory DIR       Directory containing test images. (Default : ./test_images)"
  echo "  -s BATCH, --batchsize BATCH   Batch Size. (Default : 4)"
  echo "  -c OPT, --compilerOpt OPT     Compiler mode. (Default : latency)"
  echo "  --neth HEIGHT                 Network input height. (Default : as provided in the prototxt)."
  echo "  --netw WIDTH                  Network input width. (Default : as provided in the prototxt)."
  echo "  -iou NUM, --iouthresh NUM     IOU threshold for NMS overlap. (Default : 0.45)"
  echo "  -st NUM, --scorethresh NUM    Score threshold for the boxes. (Default : 0.24 in normal case, 0.005 if -g provided)"

  echo -e ""
  echo "YOLO Config Arguments for Custom network"
  echo "  -net NET, --network NET       YOLO Caffe prototxt."
  echo "  -wts WTS, --weights WTS       YOLO Caffemodel."
  echo "  -bs TXT, --bias TXT           Text file containing bias values for YOLO network."
  echo "  -ac NUM, --anchorcnt NUM      Number of anchor boxes."
  echo "  -yv VER, --yoloversion VER    Yolo Version. Possible values : [v2, v3]."
  echo "  -nc NUM, --nclasses NUM       Number of classes."

  echo -e ""
  echo "Arguments to skip Quantizer/Compiler"
  echo "  -cn JSON, --customnet JSON    Use pre-compiled compiler.json file."
  echo "  -cq JSON, --customquant JSON  Use pre-compiled quantizer.json file."
  echo "  -cw H5, --customwts H5        Use pre-compiled weights.h5 file."
  echo "  -sq, --skip_quantizer         Skip quantization. If -cn, -cq and -cw, quantizer is automatically skipped."

  echo -e ""
  echo "Config params for asynchronous execution. Valid only for '--test streaming_detect'"
  echo "  -np NUM, --numprepproc NUM    Number of preprocessing threads to feed the data (Default : 4)"
  echo "  -nw NUM, --numworkers NUM     Number of worker threads (Default : 4)"
  echo "  -ns NUM, --numstreams NUM     Number of parallel streams (Default : 16)"

  echo -e ""
  echo "  --profile                     Provides performance related metrics"
  echo "  --visualize                   Draws the boxes on input images and saves them to --results_dir"
  echo "  -h, --help                    Print this message."
}

# Default
TEST="test_detect"
MODEL="tiny_yolo_v3"
KCFG="v3"
BITWIDTH="8"
ACCELERATOR="0"
BATCHSIZE=4
VERBOSE=0
ZELDA=0
PERPETUAL=0
IMG_INPUT_SCALE=1.0
# These variables are used in case there multiple FPGAs running in parallel
NUMDEVICES=1
DEVICEID=0
NUMPREPPROC=4
COMPILEROPT="latency"
RUN_QUANTIZER=1

# YOLO configs
NUM_CLASSES=80
LABELS='./coco.names'
IOU_THRESHOLD=0.45
SCORE_THRESHOLD=0.24
ANCHOR_COUNT=5
YOLO_VERSION='v3'
RESULTS_DIR=out_labels

URL_v2="https://www.xilinx.com/bin/public/openDownload?filename=models.caffe.yolov2_2019-08-01.zip"
URL_v3="https://www.xilinx.com/bin/public/openDownload?filename=models.caffe.yolov3_2019-11-26.zip"

# Parse Options
while true
do
  if [[ -z "$1" ]]; then break; fi
  case "$1" in
    -p |--platform      ) MLSUITE_PLATFORM="$2" ; shift 2 ;;
    -t |--test          ) TEST="$2"             ; shift 2 ;;
    -m |--model         ) MODEL="$2"            ; shift 2 ;;
    -k |--kcfg          ) KCFG="$2"             ; shift 2 ;;
    -b |--bitwidth      ) BITWIDTH="$2"         ; shift 2 ;;
    -d |--directory     ) DIRECTORY="$2"        ; shift 2 ;;
    -s |--batchsize     ) BATCHSIZE="$2"        ; shift 2 ;;
    -a |--accelerator   ) ACCELERATOR="$2"      ; shift 2 ;;
    -n |--numdevices    ) NUMDEVICES="$2"       ; shift 2 ;;
    -i |--deviceid      ) DEVICEID="$2"         ; shift 2 ;;
    -c |--compilerOpt   ) COMPILEROPT="$2"      ; shift 2 ;;
    -g |--checkaccuracy ) GOLDEN="$2"           ; shift 2 ;;
    -v |--verbose       ) VERBOSE=1             ; shift 1 ;;
    -z |--zelda         ) ZELDA=1               ; shift 1 ;;
    -x |--perpetual     ) PERPETUAL=1           ; shift 1 ;;
    -cn|--customnet     ) CUSTOM_NETCFG="$2"    ; shift 2 ;;
    -cq|--customquant   ) CUSTOM_QUANTCFG="$2"  ; shift 2 ;;
    -cw|--customwts     ) CUSTOM_WEIGHTS="$2"   ; shift 2 ;;
    -net|--network      ) CUSTOM_NETWORK="$2"   ; shift 2 ;;
    -wts|--weights      ) CUSTOM_CAFFEMODEL="$2"; shift 2 ;;
    -bs|--bias          ) CUSTOM_BIAS="$2"      ; shift 2 ;;
    -ac|--anchorcnt     ) ANCHOR_COUNT="$2"     ; shift 2 ;;
    -yv|--yoloversion   ) YOLO_VERSION="$2"     ; shift 2 ;;
    -nc|--nclasses      ) NUM_CLASSES="$2"      ; shift 2 ;;
    -iou|--iouthresh    ) QIOU_THRESHOLD="$2"   ; shift 2 ;;
    -st|--scorethresh   ) QSCORE_THRESHOLD="$2" ; shift 2 ;;
    -l|--labels         ) LABELS="$2"           ; shift 2 ;;
    -sq|--skip_quantizer) RUN_QUANTIZER=0       ; shift 1 ;;
    --neth              ) NETWORK_HEIGHT="$2"   ; shift 2 ;;
    --netw              ) NETWORK_WIDTH="$2"    ; shift 2 ;;
    -np|--numprepproc   ) NUMPREPPROC="$2"      ; shift 2 ;;
    -nw|--numworkers    ) NUMWORKERS="$2"       ; shift 2 ;;
    -ns|--numstreams    ) NUMSTREAMS="$2"       ; shift 2 ;;
    --profile           ) PROFILE=1             ; shift 1 ;;
    --dump_results      ) DUMP_RES=1            ; shift 1 ;;
    --visualize         ) VISUALIZE=1           ; shift 1 ;;
    --results_dir       ) RESULTS_DIR="$2"      ; shift 2 ;;
    --gpu               ) GPU="$2"              ; shift 2 ;;
    -h |--help          ) usage                 ; exit  1 ;;
     *) echo "Unknown argument : $1";
        echo "Try ./detect.sh -h to get correct usage. Exiting ...";
        exit 1 ;;
  esac
done

# Supported Modes & Models
SUPPORTED_MODES="cpu_detect | test_detect | streaming_detect | xtreaming_detect"
SUPPORTED_MODELS="yolo_v2 | yolo_v2_prelu | standard_yolo_v3 | yolo_v3_spp | tiny_yolo_v3 | custom"

if [[ "$SUPPORTED_MODES" != *"$TEST"* ]]; then
  echo "$TEST is not a supported test mode."
  echo "Supported Modes : $SUPPORTED_MODES."
  echo "Exiting ..."
  exit 1;
fi

if [[ "$SUPPORTED_MODELS" != *"$MODEL"* ]]; then
  echo "$MODEL is an invalid model."
  echo "Valid Models : $SUPPORTED_MODELS."
  echo "Exiting ..."
  exit 1;
fi

export XBLAS_EMIT_PROFILING_INFO=$VERBOSE
export XDNN_VERBOSE=$VERBOSE

# Set Platform Environment Variables
if [ -f /workspace/alveo/overlaybins/setup.sh ]; then
  . /workspace/alveo/overlaybins/setup.sh
else
  . ../../overlaybins/setup.sh
fi

# Determine XCLBIN
XCLBIN=${VAI_ALVEO_ROOT}/overlaybins/xdnnv3
if [ -d $XCLBIN ]; then
  echo "--- Using Local XCLBIN ---"
else
  echo "--- Using System XCLBIN ---"
  XCLBIN=/opt/xilinx/overlaybins/xdnnv3
fi

# Determine Quantizer
if [[ -f $(which vai_q_caffe) ]]; then
  QUANTIZER=vai_q_caffe
else
  QUANTIZER=decent_q
fi

# Determine Compiler
if [[ -f $(which vai_c_caffe) ]]; then
  COMPILER=vai_c_caffe
elif [[ -f $VAI_ALVEO_ROOT/vai/dpuv1/tools/compile/bin/vai_c_caffe.py ]]; then
  COMPILER=$VAI_ALVEO_ROOT/vai/dpuv1/tools/compile/bin/vai_c_caffe.py
else
  echo "Couldn't find the VAI compiler. Exiting ..."
  exit 1
fi

# Misc environment
mkdir -p work
mkdir -p $RESULTS_DIR


# Model Selection
if [ "$MODEL" == "yolo_v2" ]; then
  NET_DEF=${VAI_ALVEO_ROOT}/models/caffe/yolov2/fp32/yolo_deploy_224.prototxt
  NET_WEIGHTS=${VAI_ALVEO_ROOT}/models/caffe/yolov2/fp32/yolov2.caffemodel
  NET_DEF_FPGA=$NET_DEF
  YOLO_TYPE="xilinx_yolo_v2"
  ANCHOR_COUNT=5
  YOLO_VERSION="v2"
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=224
  INSHAPE_HEIGHT=224
elif [ "$MODEL" == "tiny_yolo_v2_voc" ]; then
  NET_DEF=${VAI_ALVEO_ROOT}/models/caffe/yolov2/fp32/yolo_v2_tiny_voc_224.prototxt
  NET_DEF_FPGA=${VAI_ALVEO_ROOT}/models/caffe/yolov2/fp32/yolo_v2_tiny_voc_224.prototxt
  NET_WEIGHTS=${VAI_ALVEO_ROOT}/models/caffe/yolov2/fp32/yolo_v2_tiny_voc.caffemodel
  YOLO_TYPE="tiny_yolo_v2_voc"
  NUM_CLASSES=20
  ANCHOR_COUNT=5
  YOLO_VERSION="v2"
  LABELS='./voc.names'
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=224
  INSHAPE_HEIGHT=224
elif [ "$MODEL" == "tiny_yolo_v2" ]; then
  NET_DEF=${VAI_ALVEO_ROOT}/models/caffe/yolov2/fp32/yolo_v2_tiny_224.prototxt
  NET_DEF_FPGA=${VAI_ALVEO_ROOT}/models/caffe/yolov2/fp32/yolo_v2_tiny_224_fpga.prototxt
  NET_WEIGHTS=../../models/caffe/yolov2/fp32/yolo_v2_tiny.caffemodel
  YOLO_TYPE="tiny_yolo_v2"
  ANCHOR_COUNT=5
  YOLO_VERSION="v2"
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=224
  INSHAPE_HEIGHT=224
elif [ "$MODEL" == "standard_yolo_v3" ]; then
  NET_DEF=${VAI_ALVEO_ROOT}/models/caffe/yolov3/fp32/yolo_v3_standard_608.prototxt
  NET_DEF_FPGA=${VAI_ALVEO_ROOT}/models/caffe/yolov3/fp32/yolo_v3_standard_608_fpga.prototxt
  NET_WEIGHTS=${VAI_ALVEO_ROOT}/models/caffe/yolov3/fp32/yolo_v3_standard.caffemodel
  YOLO_TYPE="standard_yolo_v3"
  ANCHOR_COUNT=3
  LABEL=coco.names
  YOLO_VERSION="v3"
elif [ "$MODEL" == "yolo_v3_spp" ]; then
  NET_DEF=${VAI_ALVEO_ROOT}/models/caffe/yolov3/fp32/yolov3_spp_608.prototxt
  NET_DEF_FPGA=${VAI_ALVEO_ROOT}/models/caffe/yolov3/fp32/yolov3_spp_608.prototxt
  NET_WEIGHTS=${VAI_ALVEO_ROOT}/models/caffe/yolov3/fp32/yolov3_spp.caffemodel
  YOLO_TYPE="yolo_v3_spp"
  ANCHOR_COUNT=3
  YOLO_VERSION="v3"
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=608
  INSHAPE_HEIGHT=608
elif [ "$MODEL" == "standard_yolo_v2" ]; then
  NET_DEF=${VAI_ALVEO_ROOT}/models/caffe/yolov2/fp32/yolo_v2_standard_224.prototxt
  NET_DEF_FPGA=${VAI_ALVEO_ROOT}/models/caffe/yolov2/fp32/yolo_v2_standard_224_fpga.prototxt
  NET_WEIGHTS=${VAI_ALVEO_ROOT}/models/caffe/yolov2/fp32/yolo_v2_standard.caffemodel
  YOLO_TYPE="standard_yolo_v2"
  YOLO_VERSION="v2"
  ANCHOR_COUNT=5
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=224
  INSHAPE_HEIGHT=224
elif [ "$MODEL" == "yolo_v2_prelu" ]; then
  NET_DEF=${VAI_ALVEO_ROOT}/models/caffe/yolov2/fp32/yolo_v2_prelu_224.prototxt
  NET_WEIGHTS=${VAI_ALVEO_ROOT}/models/caffe/yolov2/fp32/yolo_v2_prelu.caffemodel
  YOLO_TYPE="yolo_v2_prelu"
  NET_DEF_FPGA=$NET_DEF
  YOLO_VERSION="v2"
  ANCHOR_COUNT=5
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=224
  INSHAPE_HEIGHT=224
elif [ "$MODEL" == "tiny_yolo_v3" ]; then
  NET_DEF=${VAI_ALVEO_ROOT}/models/caffe/yolov3/fp32/yolov3_tiny_without_bn.prototxt
  NET_DEF_FPGA=${VAI_ALVEO_ROOT}/models/caffe/yolov3/fp32/yolov3_tiny_without_bn.prototxt
  NET_WEIGHTS=${VAI_ALVEO_ROOT}/models/caffe/yolov3/fp32/yolov3_tiny_without_bn.caffemodel
  YOLO_TYPE="tiny_yolo_v3"
  YOLO_VERSION="v3"
  ANCHOR_COUNT=3
elif [ "$MODEL" == "custom" ]; then
  NET_DEF=${CUSTOM_NETWORK}
  NET_DEF_FPGA=${CUSTOM_NETWORK}
  NET_WEIGHTS=${CUSTOM_CAFFEMODEL}
  YOLO_TYPE="custom"
fi

# If models are not available, download them from xilinx
if [[ ( ! -f "$NET_DEF" ) && ( $YOLO_TYPE != "custom" ) ]]; then
  echo "$NET_DEF does not exist on disk :("
  echo "Downloading the yolo${YOLO_VERSION} models ..."
  eval "URL=\$URL_$YOLO_VERSION"
  cd $VAI_ALVEO_ROOT && wget $URL -O temp.zip && unzip temp.zip && rm -rf temp.zip && cd -
  if [[ $? != 0 ]]; then echo "Network download failed. Exiting ..."; exit 1; fi;
  if [ ! -f "$NET_DEF" ]; then echo "Couldn't find $NET_DEF in models. Please check the filename. Exiting ..."; exit 1; fi;
fi


## Modify the prototxt for new dimension
if [[ ( -v NETWORK_HEIGHT ) && ( -v NETWORK_WIDTH ) ]]; then
  echo "Changing input dimensions in the $NET_DEF ..."
  echo "python modify_network_dims.py --input_deploy_file $NET_DEF --output_deploy_file work/new_dim.prototxt --in_shape 3 $NETWORK_HEIGHT $NETWORK_WIDTH"
  python modify_network_dims.py --input_deploy_file $NET_DEF --output_deploy_file work/new_dim.prototxt --in_shape 3 $NETWORK_HEIGHT $NETWORK_WIDTH
  if [[ $? != 0 ]]; then echo "Network modification failed. Exiting ..."; exit 1; fi;
  NET_DEF=work/new_dim.prototxt

  echo "python modify_network_dims.py --input_deploy_file $NET_DEF_FPGA --output_deploy_file work/new_dim_fpga.prototxt --in_shape 3 $NETWORK_HEIGHT $NETWORK_WIDTH"
  python modify_network_dims.py --input_deploy_file $NET_DEF_FPGA --output_deploy_file work/new_dim_fpga.prototxt --in_shape 3 $NETWORK_HEIGHT $NETWORK_WIDTH
  if [[ $? != 0 ]]; then echo "Network modification failed. Exiting ..."; exit 1; fi;
  NET_DEF_FPGA=work/new_dim_fpga.prototxt
fi

## Run Quantizer
if [[ ( ! -z $CUSTOM_NETCFG ) && ( ! -z $CUSTOM_WEIGHTS ) && ( ! -z $CUSTOM_QUANTCFG ) ||
      ( RUN_QUANTIZER == 0 ) || ( $TEST == "cpu_detect" ) ]]; then
  RUN_QUANTIZER=0
fi

if [[ $RUN_QUANTIZER == 1 ]]; then
  export DECENT_DEBUG=1
  DUMMY_PTXT=work/dummy.prototxt
  IMGLIST="$VAI_ALVEO_ROOT/"apps/yolo/images.txt
  CALIB_DATASET="$VAI_ALVEO_ROOT/"apps/yolo/test_image_set/
  python get_decent_q_prototxt.py $(pwd) $NET_DEF  $DUMMY_PTXT $IMGLIST  $CALIB_DATASET
  if [[ $? != 0 ]]; then echo "Network generation failed. Exiting ..."; exit 1; fi

  echo -e "quantize  -model $DUMMY_PTXT -weights $NET_WEIGHTS --output_dir work/  -calib_iter 5 -weights_bit $BITWIDTH -data_bit $BITWIDTH"
  $QUANTIZER quantize  -model $DUMMY_PTXT -weights $NET_WEIGHTS --output_dir work/  -calib_iter 5 -weights_bit $BITWIDTH -data_bit $BITWIDTH
  if [[ $? != 0 ]]; then echo "Quantization failed. Exiting ..."; exit 1; fi
else
  cp $NET_WEIGHTS work/deploy.caffemodel
fi


## Run Compiler
DSP_WIDTH=96
MEMORY=9
BPP=1
DDR=1024
RUN_COMPILER=0

NETCFG=$CUSTOM_NETCFG
QUANTCFG=$CUSTOM_QUANTCFG
WEIGHTS=$CUSTOM_WEIGHTS

if [[ ( -z $CUSTOM_NETCFG ) && ( -z $CUSTOM_WEIGHTS ) && ( -z $CUSTOM_QUANTCFG ) ]]; then
  RUN_COMPILER=1
  echo "Running Compiler ..."
  echo -n "To skip compilation, please provide precompiled files to "
  echo -n "-cn, -cq and -cw arguments".
fi

if [[ ( $TEST == "cpu_detect") ]]; then
  RUN_COMPILER=0
fi

if [[ $RUN_COMPILER == 1 ]]
then
  export GLOG_minloglevel=2 # Supress Caffe prints

  COMPILER_BASE_OPT=" --prototxt $NET_DEF_FPGA \
      --caffemodel work/deploy.caffemodel \
      --arch arch.json \
      --net_name tmp \
      --output_dir work"

  COMPILER_OTHER_OPT="{"
  COMPILER_OTHER_OPT+=" 'ddr':1024, 'quant_cfgfile': 'work/quantize_info.txt', "

  if [ "$KCFG" == "v3" ] ; then
    if [ $COMPILEROPT == "latency" ] || [ $COMPILEROPT == "throughput" ]; then
       COMPILER_OTHER_OPT+=" 'mixmemorystrategy': True, 'poolingaround': True, "
       COMPILER_OTHER_OPT+=" 'parallism':True, 'parallelread':['bottom','tops'], 'parallelismstrategy':['tops','bottom'], "
       COMPILER_OTHER_OPT+=" 'pipelineconvmaxpool':True, 'fancyreplication':True "
    fi
  fi
  COMPILER_OTHER_OPT+="}"

  ${COMPILER} $COMPILER_BASE_OPT --options "$COMPILER_OTHER_OPT"
  if [[ $? != 0 ]]; then echo "Compilation failed. Exiting ..."; exit 1; fi
  NETCFG=work/compiler.json
  WEIGHTS=work/weights.h5
  QUANTCFG=work/quantizer.json

  if [ $COMPILEROPT == "throughput" ] && [ "$KCFG" == "v3" ]; then
     python $VAI_ALVEO_ROOT/vai/dpuv1/tools/compile/scripts/xfdnn_gen_throughput_json.py --i work/compiler.json --o work/compiler_tput.json
     NETCFG=work/compiler_tput.json
  fi
fi

if [[ -z $QSCORE_THRESHOLD ]]; then
  SCORE_THRESHOLD=0.24
else
  SCORE_THRESHOLD=$QSCORE_THRESHOLD
fi

if [[ -z $QIOU_THRESHOLD ]]; then
  IOU_THRESHOLD=0.45
else
  IOU_THRESHOLD=$QIOU_THRESHOLD
fi

# Simply passing the golden will force app to dump the results and compute mAP
if [ ! -z $GOLDEN ];  then
  echo -e "To check mAP score of the network please note the following"
  echo -e "   mAP score check for VOC and COCO data set is supported provided the data is in darknet style  "
  echo -e "   To get COCO data in darknet format run script https://github.com/pjreddie/darknet/blob/master/scripts/get_coco_dataset.sh  "
  echo -e "   To get VOC data in darknet format run script https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py  "
  echo -e "   All the images in the Val dataset should be provided in one folder and specified by --directory option"
  echo -e "   The corresponding groud truth label .txt files with same name as images should be provided in one folder and specified by --checkaccuracy option"
  echo -e "   The script will generate the corresponding labels in ./out_labels folder "
  BASEOPT+=" --golden $GOLDEN --results_dir $RESULTS_DIR"
  echo "Image Directory : $DIRECTORY"
  echo "for mAP score calculation class score threshold is set to low value of 0.005"
  if [[ -z $QSCORE_THRESHOLD ]]; then
    SCORE_THRESHOLD=0.005
  else
    SCORE_THRESHOLD=$QSCORE_THRESHOLD
  fi
elif [ ! -z $DUMP_RES ]; then
  BASEOPT+=" --results_dir $RESULTS_DIR"
fi

if [ ! -z $VISUALIZE ]; then
  BASEOPT+=" --visualize"
fi

# VITIS Directory
get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}
if [ -z $VITIS_RUNDIR ]; then
  find /tmp/vitis-{$USER} -mtime +3 -exec rm -fr {} \; &> /dev/null
  VITIS_RUNDIR=/tmp/vitis-${USER}/${BASHPID}
  mkdir -p $VITIS_RUNDIR
  rm -f ${VITIS_RUNDIR}/{compiler.json,quantizer.json,weights.h5}
  ln -s $(get_abs_filename $NETCFG) ${VITIS_RUNDIR}/compiler.json
  ln -s $(get_abs_filename $QUANTCFG) ${VITIS_RUNDIR}/quantizer.json
  ln -s $(get_abs_filename $WEIGHTS) ${VITIS_RUNDIR}/weights.h5
  echo "{ \"target\": \"xdnn\", \"filename\": \"\", \"kernel\": \"xdnn\", \"config_file\": \"\", \"lib\": \"${LIBXDNN_PATH}\", \"xclbin\": \"${XCLBIN}\", \"publish_id\": \"${BASHPID}\" }" > ${VITIS_RUNDIR}/meta.json
  # meta.json accepts {env_variables} in paths as well, e.g.:
  #echo "{ \"lib\": \"{VAI_ALVEO_ROOT}/xfdnn/rt/xdnn_cpp/lib/libxfdnn.so\", \"xclbin\": \"{VAI_ALVEO_ROOT}/overlaybins/xdnnv3\" }" > ${VITIS_RUNDIR}/meta.json
  cp -fr $VITIS_RUNDIR ${VITIS_RUNDIR}_worker
  echo "{ \"target\": \"xdnn\", \"filename\": \"\", \"kernel\": \"xdnn\", \"config_file\": \"\", \"lib\": \"${LIBXDNN_PATH}\", \"xclbin\": \"${XCLBIN}\", \"subscribe_id\": \"${BASHPID}\" }" > ${VITIS_RUNDIR}_worker/meta.json
fi

# Start Execution
echo -e "Running:\n Test: $TEST\n Model: $MODEL\n Platform: $MLSUITE_PLATFORM\n \
  Xclbin: $XCLBIN\n Kernel Config: $KCFG\n Precision: $BITWIDTH\n Accelerator: $ACCELERATOR\n"

if [ -z ${DIRECTORY+x} ]; then
  DIRECTORY=${VAI_ALVEO_ROOT}/apps/yolo/test_image_set/
fi

if [[ ( $TEST != "cpu_detect" ) ]]; then
  BASEOPT+=" --xclbin $XCLBIN"
  BASEOPT+=" --netcfg $NETCFG"
  BASEOPT+=" --weights $WEIGHTS"
  BASEOPT+=" --quantizecfg $QUANTCFG"
  BASEOPT+=" --vitis_rundir $VITIS_RUNDIR"
  BASEOPT+=" --batch_sz $BATCHSIZE"
  BASEOPT+=" --dsp $DSP_WIDTH"
  BASEOPT+=" --outsz $NUM_CLASSES"
  BASEOPT+=" --img_input_scale $IMG_INPUT_SCALE"
  BASEOPT+=" --vitis_rundir ${VITIS_RUNDIR}"

  if [[ -n $NUMPREPPROC ]]; then
    BASEOPT+=" --numprepproc ${NUMPREPPROC}"
  fi
  if [[ -n $NUMWORKERS ]]; then
    BASEOPT+=" --numworkers ${NUMWORKERS}"
  fi
  if [[ -n $NUMSTREAMS ]]; then
    BASEOPT+=" --numstream ${NUMSTREAMS}"
  fi
  if [[ -n $PROFILE ]]; then
    BASEOPT+=" --profile"
  fi

fi

BASEOPT+=" --classes $NUM_CLASSES"
BASEOPT+=" --labels $LABELS"
BASEOPT+=" --images $DIRECTORY"
BASEOPT+=" --yolo_model $YOLO_TYPE"
BASEOPT+=" --scorethresh $SCORE_THRESHOLD"
BASEOPT+=" --iouthresh $IOU_THRESHOLD"
BASEOPT+=" --anchorCnt $ANCHOR_COUNT"
BASEOPT+=" --yolo_version ${YOLO_VERSION}"

if [[ "$MODEL" == "custom" ]]; then
  BASEOPT+=" --bias_file $CUSTOM_BIAS"
fi

if [ "$TEST" == "test_detect" ]; then
  TEST=test_detect_vitis.py
elif [ "$TEST" == "xtreaming_detect" ]; then
  TEST=xs_detect_vitis.py
elif [ "$TEST" == "streaming_detect" ]; then
  TEST=mp_detect_vitis.py
elif [ "$TEST" == "cpu_detect" ]; then
  BASEOPT+=" --deploymodel $NET_DEF --caffemodel $NET_WEIGHTS"
  if [[ ! -z $GPU ]]; then BASEOPT+=" --gpu $GPU"; fi
  TEST=cpu_detect.py
else
  echo "ERROR : -t|--test should be one of <test_detect|streaming_detect|xtreaming_detect|cpu_detect>. Exiting ..."
  exit 1
fi


if [ $ZELDA -eq "0" ]; then
  echo -e "python $TEST $BASEOPT" | sed 's/--/\n  --/g' -
  python $TEST $BASEOPT 2>&1 | tee single_img_out.txt
  if [[ $? != 0 ]]; then echo "Execution failed. Exiting ..."; exit 1; fi
else
  gdb --args python $TEST $BASEOPT
fi
