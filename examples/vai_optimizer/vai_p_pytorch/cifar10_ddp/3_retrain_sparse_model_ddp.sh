source config.sh

METHOD="iterative"
SAVEDIR=${WORKSPACE}/${METHOD}

if [[ ! -d "${SAVEDIR}" ]]; then
  mkdir -p "${SAVEDIR}"
fi

sparsity_ratios=(0.1 0.2 ${SPARSITY})

for i in ${sparsity_ratios[*]}; do
echo "sparsity=${i}" 
python -m torch.distributed.launch \
	--nnodes=1 \
	--nproc_per_node=2 \
  sparse_model_train_ddp.py --gpus "0,1" \
  --lr 1e-3 \
  --epochs 5 \
  --sparsity ${i} \
  --pretrained ${BASELINE_PATH} \
  --save_dir ${SAVEDIR} \
  --data_dir ${DATA_DIR} \
  --num_workers 48 \
  --batch_size 64 \
  --weight_decay 1e-4 \
  --momentum 0.9
done
