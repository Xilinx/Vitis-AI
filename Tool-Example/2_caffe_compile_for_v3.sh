#!/bin/sh

ALVEO_TARGET=dpuv3e
NET_NAME=resnet50
DEPLOY_MODEL_PATH_v3=vai_q_output_dpuv3

ALVEO_ARCH=/opt/vitis_ai/compiler/arch/${ALVEO_TARGET}/arch.json

#DPUv3	    
vai_c_caffe --prototxt ${CF_NETWORK_PATH}/${DEPLOY_MODEL_PATH_v3}/deploy.prototxt \
	    --caffemodel ${CF_NETWORK_PATH}/${DEPLOY_MODEL_PATH_v3}/deploy.caffemodel \
    	    --arch ${ALVEO_ARCH} \
    	    --output_dir ${CF_NETWORK_PATH}/vai_c_output_${ALVEO_TARGET} \
    	    --net_name ${NET_NAME} \
