#!/bin/sh

vai_q_caffe quantize -model ${CF_NETWORK_PATH}/float/trainval.prototxt \
                     -weights ${CF_NETWORK_PATH}/float/float.caffemodel \
	             -output_dir ${CF_NETWORK_PATH}/vai_q_output \
                     -calib_iter 100 \
	             -test_iter 100 \
	             -auto_test \
