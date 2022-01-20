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
APP_CONDA_ENV="rnn-pytorch-1.7.1"
VAI_CONDA_PKG_PATH="/scratch/conda-channel/linux-64"

MODEL_DIR="../vai-rnn-models-2.0"

if [[ $TARGET_DEVICE != "U50LV" && $TARGET_DEVICE != "U25" ]]; then
  echo "[ERROR] TARGET_DEVICE should be U50LV or U25"
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
cp $MODEL_DIR/$device/openie-new/*.xmodel data/
wget -nc https://allennlp.s3.amazonaws.com/models/openie-model.2018-08-20.tar.gz
mkdir -p weights
tar -xzvf openie-model.2018-08-20.tar.gz -C weights

echo "Get oie-wrapper ..."
if [[ ! -d oie-benchmark ]]; then
  git clone -q https://github.com/gabrielStanovsky/supervised_oie_wrapper
fi
cp supervised_oie_wrapper/src/format_oie.py .
cp supervised_oie_wrapper/src/run_oie.py .

echo "Get oie-benchmark ..."
if [[ ! -d oie-benchmark ]]; then
  git clone -q https://github.com/gabrielStanovsky/oie-benchmark
fi
cp utils/moveConf.py  oie-benchmark/
cp utils/benchmark.py oie-benchmark/
cp utils/tabReader.py oie-benchmark/oie_readers/
cp utils/test.oie     oie-benchmark/oie_corpus/

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

echo "Activating environment, ${APP_CONDA_ENV}"
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
conda activate ${APP_CONDA_ENV}
mkdir -p output
mkdir -p test
python convert.py
