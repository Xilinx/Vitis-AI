source config.sh

METHOD="iterative"
SAVEDIR=${WORKSPACE}/${METHOD}

if [[ ! -d "${SAVEDIR}" ]]; then
  mkdir -p "${SAVEDIR}"
fi

python compare.py --gpus '0,1' \
  --sparsity ${SPARSITY} \
  --save_dir ${SAVEDIR} \
  --data_dir ${DATA_DIR} \
  --num_workers 48 \
  --batch_size 64
