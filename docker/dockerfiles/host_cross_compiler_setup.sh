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
#!/bin/bash


wget https://www.xilinx.com/bin/public/openDownload?filename=sdk-2021.2.0.0.sh -O sdk-2021.2.0.0.sh
wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2021.2-r2.0.0.tar.gz -O vitis_ai_2021.2-r2.0.0.tar.gz

chmod +x sdk-2021.2.0.0.sh

echo "The Cross Compiler will be installed in ~/petalinux_sdk_2021.2 by default"
install_path=~/petalinux_sdk_2021.2

if [ -d $install_path ]
then
echo ""
else
mkdir -p $install_path
fi

echo $install_path|./sdk-2021.2.0.0.sh

rm -r $install_path/sysroots/cortexa72-cortexa53-xilinx-linux/usr/share/cmake/XRT/

tar -xzvf vitis_ai_2021.2-r2.0.0.tar.gz -C $install_path/sysroots/cortexa72-cortexa53-xilinx-linux/

echo "Complete Cross Compiler installation"
echo ""
echo "Please run the following command to enable Cross Compiler"
echo "    source $install_path/environment-setup-cortexa72-cortexa53-xilinx-linux"
echo "If you run the above command failed, run the following commands to enable Cross Compiler"
echo "    unset LD_LIBRARY_PATH"
echo "    source $install_path/environment-setup-cortexa72-cortexa53-xilinx-linux"
echo ""
