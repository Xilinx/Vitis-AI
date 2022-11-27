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

OMS_PREFIX="https://www.xilinx.com/bin/public/openDownload?filename="

RNN_U25_XCLBIN_FILE="dpu4rnn_u25_xclbin.tar.gz"
RNN_U25_XCLBIN_MD5SUM="c54f1e0c8e074036bb559e6e4fa54109"

RNN_U50LV_XCLBIN_FILE="dpu4rnn_u50lv_xclbin.tar.gz"
RNN_U50LV_XCLBIN_MD5SUM="084485126d0158e00a975461614fe26e"

function get_rnn_xclbin() {
  device=$1
  output_dir=$2
  tar_var_name=RNN_${device}_XCLBIN_FILE
  oms_link=${OMS_PREFIX}${!tar_var_name}
  downloaded_tar="/tmp/${!tar_var_name}"
  extracted_file="${output_dir}/dpu.xclbin"
  if [[ ! -f ${extracted_file} ]]; then
    wget -nc -O ${downloaded_tar} ${oms_link}
    tar -xvf ${downloaded_tar} -C ${output_dir}
  fi
  if [[ ! -f ${extracted_file} ]]; then
    echo "[ERROR] RNN xclbin extraction failed."
    return 1
  fi
}
