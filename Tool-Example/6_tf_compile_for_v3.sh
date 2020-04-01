#!/bin/sh

#DPUv3	    
ALVEO_TARGET=dpuv3e
NET_NAME=resnet50
DEPLOY_MODEL_PATH=vai_q_output

ALVEO_ARCH=/opt/vitis_ai/compiler/arch/${ALVEO_TARGET}/arch.json

vai_c_tensorflow --frozen_pb ${TF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/quantize_eval_model.pb \
                 --arch ${ALVEO_ARCH} \
		 --output_dir ${TF_NETWORK_PATH}/vai_c_output_${ALVEO_TARGET}/ \
		 --net_name ${NET_NAME}


