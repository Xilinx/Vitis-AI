set -ex

export INPUT_HEIGHT=$4
export INPUT_WIDTH=$5
export CHANNEL_NUM=$6

export INPUT_NODES=$2
export OUTPUT_NODES=$3
export PREPROCESS_TYPE=inception
export DUMP_IMAGE_DIR=/group/modelzoo/val_datasets/imagenet/val_dataset
export DUMP_IMAGE_LIST=/group/modelzoo/val_datasets/imagenet/calib_list.txt
export DUMP_BATCH_SIZE=1
export MAX_DUMP_BATCHES=1

vai_q_tensorflow dump \
  --input_frozen_graph $1 \
  --input_fn input_fn.dump_input \
  --gpu -1 \
  --dump_float 1 \
  --max_dump_batches $MAX_DUMP_BATCHES \
  --output_dir quantize_results
