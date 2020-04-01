#!/bin/sh
#conda activate vitis-ai-caffe

#For DPUv2
vai_q_caffe quantize -model ${CF_NETWORK_PATH}/float/trainval.prototxt \
                     -weights ${CF_NETWORK_PATH}/float/trainval.caffemodel \
	             -output_dir ${CF_NETWORK_PATH}/vai_q_output_dpuv2 \
                     -calib_iter 100 \
	             -test_iter 100 \
	             -auto_test \


