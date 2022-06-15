source config.sh

METHOD="iterative"
SAVEDIR=${WORKSPACE}/${METHOD}

if [[ ! -d "${SAVEDIR}" ]]; then
  mkdir -p "${SAVEDIR}"
fi

python prune.py --sparsity ${SPARSITY} \
  --save_dir ${SAVEDIR}

