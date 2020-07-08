#!/bin/bash
#set -x

# Set Platform Environment Variables
if [ -z $VAI_ALVEO_ROOT ]; then
  echo "Please set VAI_ALVEO_ROOT, see you next time!"
  exit 1
fi

if [ -z $1 ]; then
  echo "Please provide MODEL name, see you next time!"
  exit 1
fi

MODEL=$1
export MODELDIR="$( readlink -f "$( dirname "${BASH_SOURCE[0]}" )" )"
OUTDIR=$MODELDIR/work

# Choose the target
ARCH_JSON="/opt/vitis_ai/compiler/arch/DPUCADX8G/ALVEO/arch.json"
if [ ! -f $ARCH_JSON ]; then
  ARCH_JSON="$VAI_ALVEO_ROOT/arch.json"
fi

vai_c_caffe \
  --net_name $MODEL.compiler \
  --prototxt $MODELDIR/quantize_results/$MODEL.prototxt \
  --caffemodel $MODELDIR/quantize_results/$MODEL.caffemodel \
  --options "{'quant_cfgfile': '$MODELDIR/quantize_results/quantize_info.txt', 'parallelism': True}" \
  --arch $ARCH_JSON \
  -o $OUTDIR


  

python -m vai.dpuv1.rt.scripts.framework.caffe.xfdnn_subgraph \
  --inproto $MODELDIR/quantize_results/$MODEL.prototxt \
  --outproto xfdnn_$MODEL.prototxt \
  --cutAfter input_3 \
  --xclbin /opt/xilinx/overlaybins/xdnnv3 \
  --netcfg $OUTDIR/compiler.json \
  --quantizecfg $OUTDIR/quantizer.json \
  --weights $OUTDIR/weights.h5
