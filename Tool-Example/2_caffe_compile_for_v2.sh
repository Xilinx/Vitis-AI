#!/bin/sh

NET_NAME=resnet50
EDGE_TARGET=ZCU102
DEPLOY_MODEL_PATH=vai_q_output_dpuv2

EDGE_ARCH=/opt/vitis_ai/compiler/arch/dpuv2/${EDGE_TARGET}/${EDGE_TARGET}.json

#DPUv2
vai_c_caffe --prototxt ${CF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/deploy.prototxt \
	    --caffemodel ${CF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/deploy.caffemodel \
    	    --arch ${EDGE_ARCH} \
    	    --output_dir ${CF_NETWORK_PATH}/vai_c_output_${EDGE_TARGET}/ \
    	    --net_name ${NET_NAME} \
	    --options "{'save_kernel':''}"
