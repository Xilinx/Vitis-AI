## Copyright 2019 Xilinx Inc.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.


usage() {
  echo "Usage:"
  echo "./run.sh --test <test> --model <model>"
  echo "./run.sh  -t    <test>  -m <model>"
  echo "<test>     : test_classify"
  echo "<compilerOpt> : autoAllOpt / latency / throughput"
  echo "Some tests require a directory of images to process."
  echo "To process a directory in a non-standard location use -d <directory> or --directory <directory>"
  echo "Some tests require a batchSize argument to know how many images to load simultaneously."
  echo "To provide batchSize use --batchsize <batchsize>"
  echo "-c allows to choose compiler optimization, for example, latency or throughput or autoAllOptimizations."


}

# Default
TEST="test_classify"
MODEL="googlenet_v1"
ACCELERATOR="0"
BATCHSIZE=1
VERBOSE=0
PERPETUAL=0
IMG_INPUT_SCALE=1.0
# These variables are used in case there multiple FPGAs running in parallel
NUMDEVICES=1
NUMSTREAMS=8
DEVICEID=0
NUMPREPPROC=1
COMPILEROPT="autoAllOpt"
# Parse Options
OPTS=`getopt -o t:m:d:s:a:n:ns:i:c:y:gvxh --long test:,model:,directory:,numdevices:,numstreams:,deviceid:,batchsize:,compilerOpt:,numprepproc,checkaccuracy,verbose,perpetual,help -n "$0" -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; usage; exit 1 ; fi

while true
do
  case "$1" in
    -t |--test          ) TEST="$2"             ; shift 2 ;;
    -m |--model         ) MODEL="$2"            ; shift 2 ;;
    -d |--directory     ) DIRECTORY="$2"        ; shift 2 ;;
    -r |--vitisrundir   ) VITIS_RUNDIR="$2"     ; shift 2 ;;
    -s |--batchsize     ) BATCHSIZE="$2"        ; shift 2 ;;
    -a |--accelerator   ) ACCELERATOR="$2"      ; shift 2 ;;
    -n |--numdevices    ) NUMDEVICES="$2"       ; shift 2 ;;
    -ns|--numstreams    ) NUMSTREAMS="$2"       ; shift 2 ;;
    -i |--deviceid      ) DEVICEID="$2"         ; shift 2 ;;
    -c |--compilerOpt   ) COMPILEROPT="$2"      ; shift 2 ;;
    -y |--numprepproc   ) NUMPREPPROC="$2"      ; shift 2 ;;
    -g |--checkaccuracy ) GOLDEN=gold.txt       ; shift 1 ;;
    -v |--verbose       ) VERBOSE=1             ; shift 1 ;;
    -x |--perpetual     ) PERPETUAL=1           ; shift 1 ;;
    -cn|--customnet     ) CUSTOM_NETCFG="$2"    ; shift 2 ;;
    -cq|--customquant   ) CUSTOM_QUANTCFG="$2"  ; shift 2 ;;
    -cw|--customwts     ) CUSTOM_WEIGHTS="$2"   ; shift 2 ;;
    -h |--help          ) usage                 ; exit  1 ;;
     *) break ;;
  esac
done

# Verbose Debug Profiling Prints
# Note, the VERBOSE mechanic here is not working
# Its always safer to set this manually
export XBLAS_EMIT_PROFILING_INFO=1
# To be fixed
export XBLAS_EMIT_PROFILING_INFO=$VERBOSE
export XDNN_VERBOSE=$VERBOSE

. ../../../overlaybins/setup.sh




XCLBIN=/opt/xilinx/overlaybins/xdnnv3/xplusml/classification/
if [ -d $XCLBIN ]; then
  echo "--- XCLBIN Found---"
else
  echo "--- XCLBIN Not Found---"
  exit 1
fi
WEIGHTS=./data/${MODEL}_data.h5
DSP_WIDTH=96

NETCFG=./data/${MODEL}_8b_${COMPILEROPT}.json
QUANTCFG=./data/${MODEL}_8b_xdnnv3.json

if [ ! -z $CUSTOM_NETCFG ]; then
  NETCFG=$CUSTOM_NETCFG
fi
if [ ! -z $CUSTOM_WEIGHTS ]; then
  WEIGHTS=$CUSTOM_WEIGHTS
fi
if [ ! -z $CUSTOM_QUANTCFG ]; then
  QUANTCFG=$CUSTOM_QUANTCFG
fi

echo -e "Running:\n Test: $TEST\n Model: $MODEL\n Xclbin: $XCLBIN\n Accelerator: $ACCELERATOR\n"

BASEOPT="--xclbin $XCLBIN
         --netcfg $NETCFG
         --weights $WEIGHTS
         --labels ./synset_words.txt
         --quantizecfg $QUANTCFG
         --img_input_scale $IMG_INPUT_SCALE
         --batch_sz $BATCHSIZE"



##############################################
# auto-generate vitis rundir if not provided
##############################################
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
  #echo "{ \"lib\": \"{MLSUITE_ROOT}/vai/dpuv1/rt/xdnn_cpp/lib/libxfdnn.so\", \"xclbin\": \"{MLSUITE_ROOT}/overlaybins/xdnnv3\" }" > ${VITIS_RUNDIR}/meta.json
  cp -fr $VITIS_RUNDIR ${VITIS_RUNDIR}_worker
  echo "{ \"target\": \"xdnn\", \"filename\": \"\", \"kernel\": \"xdnn\", \"config_file\": \"\", \"lib\": \"${LIBXDNN_PATH}\", \"xclbin\": \"${XCLBIN}\", \"subscribe_id\": \"${BASHPID}\" }" > ${VITIS_RUNDIR}_worker/meta.json
  BASEOPT+=" --vitis_rundir ${VITIS_RUNDIR}"
fi

# Build options for appropriate python script

####################
# single image test
####################

if [ "$TEST" == "test_classify" ]; then
  TEST=test_classify.py
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=dog.jpg
  fi
  BASEOPT+=" --images $DIRECTORY --deviceid $DEVICEID"

  echo python $TEST $BASEOPT
  python $TEST $BASEOPT
fi
