#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

env_setup() {
  source /opt/vitis_ai/conda/etc/profile.d/conda.sh
  export PATH=/opt/vitis_ai/conda/bin:/opt/vitis_ai/utility:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/usr/local/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/lib
  export CUDA_HOME=/usr/local/cuda
}

env_setup

conda create -y -n nndct --clone vitis-ai-pytorch
conda activate nndct

WORKSPACE=${WORKSPACE:-${SCRIPT_DIR}/..}
cd ${WORKSPACE}/pytorch_binding
python setup.py install

cd ${WORKSPACE}/model_regression_test/tools
./run.sh
