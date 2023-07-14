source config.sh

METHOD="one_step"
SAVEDIR=${WORKSPACE}/${METHOD}

if [[ ! -d "${SAVEDIR}" ]]; then
  mkdir -p "${SAVEDIR}"
fi

python -m torch.distributed.launch \
	--nnodes=1 \
	--nproc_per_node=2 \
  slim_model_train_ddp.py --gpus "0,1" \
  --lr 1e-3 \
  --epochs 5 \
  --sparsity ${SPARSITY} \
  --pretrained ${BASELINE_PATH} \
  --save_dir ${SAVEDIR} \
  --data_dir ${DATA_DIR} \
  --num_workers 48 \
  --batch_size 64 \
  --weight_decay 1e-4 \
  --momentum 0.9
