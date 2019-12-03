#!/bin/sh

TARGET=ZCU102
NET_NAME=resnet50
DEPLOY_MODEL_PATH=vai_q_output
ARCH=/opt/vitis_ai/compiler/arch/dpuv2/${TARGET}/${TARGET}.json

vai_c_caffe --prototxt ${CF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/deploy.prototxt \
	    --caffemodel ${CF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/deploy.caffemodel \
    	    --arch ${ARCH} \
    	    --output_dir ${CF_NETWORK_PATH}/vai_c_output_${TARGET}/ \
    	    --net_name ${NET_NAME} \


