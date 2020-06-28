#
# Copyright 2019 Xilinx Inc.
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

cd /workspace

#Download the Model packetes and install them.
wget https://www.xilinx.com/bin/public/openDownload?filename=xilinx_model_zoo_u50_1.2.0_amd64.deb -O xilinx_model_zoo_u50_1.2.0_amd64.deb
sudo dpkg -i xilinx_model_zoo_u50_1.2.0_amd64.deb
wget https://www.xilinx.com/bin/public/openDownload?filename=xilinx_model_zoo_u50_pytorch_1.2.0_amd64.deb -O xilinx_model_zoo_u50_pytorch_1.2.0_amd64.deb
sudo dpkg -i xilinx_model_zoo_u50_pytorch_1.2.0_amd64.deb

#Download the xclbin package and install it.
wget https://www.xilinx.com/bin/public/openDownload?filename=alveo_xclbin-1.2.0.tar.gz -O alveo_xclbin-1.2.0.tar.gz
tar -xzvf alveo_xclbin-1.2.0.tar.gz
cd alveo_xclbin-1.2.0/U50/6E300M
sudo cp dpu.xclbin hbm_address_assignment.txt /usr/lib

echo "Finish setting up the host for U50."