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

#wget https://www.xilinx.com/bin/public/openDownload?filename=vitis-ai-runtime-3.0.0.tar.gz -O vitis-ai-runtime-3.0.0.tar.gz
#cd vitis-ai-runtime-3.0.0/2022.2/aarch64/centos

cd 2022.2
rpm -ivh --force libunilog-3.0.*-r*.aarch64.rpm
rpm -ivh --force libxir-3.0.*-r*.aarch64.rpm
rpm -ivh --force libtarget-factory-3.0.*-r*.aarch64.rpm
rpm -ivh --force libvart-3.0.*-r*.aarch64.rpm
rpm -ivh --force libvitis_ai_library-3.0.*-r*.aarch64.rpm

echo "Complete VART packages installation"
