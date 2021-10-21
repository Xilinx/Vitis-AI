#/*
# * Copyright 2020 Xilinx, Inc.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

source /opt/xilinx/xrt/setup.sh
source /opt/xilinx/xrm/setup.sh
source /opt/xilinx/Vitis/2020.1/settings64.sh
export DEVICE=/opt/xilinx/platforms/xilinx_u50_gen3x16_xdma_201920_3/xilinx_u50_gen3x16_xdma_201920_3.xpfm
cd ../../lib/
./build_so.sh
cd ../tests/cosineSimilaritySSDenseIntBench
make build TARGET=hw
