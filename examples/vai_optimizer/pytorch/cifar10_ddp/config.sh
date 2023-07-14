WORKSPACE="./pruning"
BASELINE_DIR="./baseline"

if [[ ! -d "${WORKSPACE}" ]]; then
  mkdir -p "${WORKSPACE}"
fi

if [[ ! -d "${BASELINE_DIR}" ]]; then
  mkdir -p "${BASELINE_DIR}"
fi

DATA_DIR="./dataset/cifar10"
SPARSITY=0.3
BASELINE_PATH="${BASELINE_DIR}/model.pth"

FT_EPOCHS=5

