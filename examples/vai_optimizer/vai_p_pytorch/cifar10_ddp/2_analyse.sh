source config.sh

python analyse.py --gpus "0,1" \
  --pretrained ${BASELINE_PATH} \
  --data_dir ${DATA_DIR} \
  --num_workers 1 \
  --batch_size 128 \
