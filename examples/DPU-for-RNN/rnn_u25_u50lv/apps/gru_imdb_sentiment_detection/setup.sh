#
# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

source "../common/setup.sh"
APP_CONDA_ENV="gru_imdb"
VAI_CONDA_PKG_PATH="/scratch/conda-channel/linux-64"

#  Create conda environment if necessary
if [[ ! `conda info --envs | grep ${APP_CONDA_ENV}` ]]; then
  echo "Creating conda environment ..."
  conda env create -f gru_imdb.yml
  conda install -n ${APP_CONDA_ENV}                          \
    ${VAI_CONDA_PKG_PATH}/xir-2.0.0-py37h893bffd_108.tar.bz2 \
    ${VAI_CONDA_PKG_PATH}/vart-2.0.0-py37h076edd9_144.tar.bz2
fi

MODEL_DIR="../vai-rnn-models-2.0"

if [[ $TARGET_DEVICE != "U50LV" ]]; then
  echo "[ERROR] TARGET_DEVICE should be U50LV"
  return 1;
else
  echo "TARGET_DEVICE = $TARGET_DEVICE"
  device=$(echo $TARGET_DEVICE | awk '{print tolower($0)}')
fi

echo "Get compiled models ..."
if [[ ! -d $MODEL_DIR ]]; then
  wget -nc -O /tmp/vai-rnn-models-2.0.tar.gz https://www.xilinx.com/bin/public/openDownload?filename=vai-rnn-models-2.0.tar.gz
  tar -xvf /tmp/vai-rnn-models-2.0.tar.gz -C ..
fi

echo "Copying the data ..."
mkdir -p data
rm data/*.xmodel 2>/dev/null
cp $MODEL_DIR/$device/gru_sentiment_detection/*.xmodel data/
cp $MODEL_DIR/float/gru_sentiment_detection/*.pth data/

echo "Checking xclbin ..."
src_xclbin="../dpu.xclbin"
dst_xclbin=/usr/lib/dpu.xclbin
xclbin_md5sum=RNN_${TARGET_DEVICE}_XCLBIN_MD5SUM
if [[ ! -f $dst_xclbin || `md5sum $dst_xclbin` != ${!xclbin_md5sum} ]]; then
  if [[ ! -f $src_xclbin || `md5sum $src_xclbin` != ${!xclbin_md5sum} ]]; then
    get_rnn_xclbin ${TARGET_DEVICE} ..
  fi
  sudo cp $src_xclbin $dst_xclbin
fi

echo "Activate the environment ..."
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
conda activate ${APP_CONDA_ENV}
export PYTHONPATH=../common:$PYTHONPATH
