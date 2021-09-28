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
cd ../../lib/
./build_so.sh
cd ../tests/cosineSimilaritySSDenseIntBench
make host TARGET=hw
LD_LIBRARY_PATH=../../lib:/opt/xilinx/xrm/lib:/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH  build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3/host.exe -weight ./data/cosine_dense_weight.csr -golden ./data/cosine_dense.mtx -topK 100
