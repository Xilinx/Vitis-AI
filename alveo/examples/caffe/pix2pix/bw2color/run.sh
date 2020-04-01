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


/opt/vitis_ai/compiler/vai_c_caffe \
  --net_name $MODEL.compiler \
  --prototxt $VAI_ALVEO_ROOT/examples/caffe/models/bw2color/$MODEL.prototxt \
  --caffemodel $VAI_ALVEO_ROOT/examples/caffe/models/bw2color/$MODEL.caffemodel \
  --options "{'quant_cfgfile': 'fix_info.txt', 'parallelism': True, 'parallelismgraphalgorithm': 'tfs', 'parallelismstrategy': ['tops', 'bottom'], 'parallelread' : ['bottom', 'tops'],'saveschedule': 'upconv.sched','laodschedule':'upconv.sched'}" \
  --arch /opt/vitis_ai/compiler/arch/dpuv1/ALVEO/ALVEO.json \
  -o $OUTDIR


  

python /opt/vitis_ai/conda/envs/vitis-ai-caffe/lib/python3.6/site-packages/vai/dpuv1/rt/scripts/framework/caffe/xfdnn_subgraph.py \
  --inproto $VAI_ALVEO_ROOT/examples/caffe/models/bw2color/$MODEL.prototxt \
  --outproto xfdnn_$MODEL.prototxt \
  --cutAfter data \
  --xclbin /opt/xilinx/overlaybins/xdnnv3 \
  --netcfg $OUTDIR/compiler.json \
  --quantizecfg $OUTDIR/quantizer.json \
  --weights $OUTDIR/weights.h5 


