source config.sh

python search.py --gpus "0,1" \
  --num_subnet 40 \
  --sparsity ${SPARSITY} \
  --pretrained ${BASELINE_PATH} \
  --data_dir ${DATA_DIR} \
  --num_workers 1 \
  --batch_size 128 \
