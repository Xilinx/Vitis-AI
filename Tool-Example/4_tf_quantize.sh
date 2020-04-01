#!/bin/sh

vai_q_tensorflow quantize --input_frozen_graph ${TF_NETWORK_PATH}/float/resnet_v1_50_inference.pb \
			  --input_fn example_file.input_fn.calib_input \
			  --output_dir ${TF_NETWORK_PATH}/vai_q_output \
	                  --input_nodes input \
			  --output_nodes resnet_v1_50/predictions/Reshape_1 \
			  --input_shapes ?,224,224,3 \
			  --calib_iter 100 \
