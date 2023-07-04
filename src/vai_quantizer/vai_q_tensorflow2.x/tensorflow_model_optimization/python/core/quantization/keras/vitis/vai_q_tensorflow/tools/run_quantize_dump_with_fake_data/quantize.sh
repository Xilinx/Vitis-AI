set -ex

export INPUT_HEIGHT=$4
export INPUT_WIDTH=$5
export CHANNEL_NUM=$6

export INPUT_NODES=$2
export OUTPUT_NODES=$3
export PREPROCESS_TYPE=inception
export CALIB_IMAGE_DIR=/group/modelzoo/val_datasets/imagenet/val_dataset
export CALIB_IMAGE_LIST=/group/modelzoo/val_datasets/imagenet/calib_list.txt
export CALIB_BATCH_SIZE=1

export DECENT_DEBUG=1
export TF_CPP_MIN_LOG_LEVEL=0

vai_q_tensorflow quantize \
  --input_frozen_graph $1 \
  --input_nodes ${INPUT_NODES} \
  --input_shapes ?,${INPUT_HEIGHT},${INPUT_WIDTH},$CHANNEL_NUM \
  --output_nodes ${OUTPUT_NODES} \
  --input_fn input_fn.calib_input \
  --method 1 \
  --gpu 0 \
  --calib_iter 10 \
  --output_dir quantize_results
