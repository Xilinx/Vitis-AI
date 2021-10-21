#
# Copyright 2021 Xilinx, Inc.
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

all: rapidcsv fastcluster

rapidcsv:
	git clone https://github.com/d99kris/rapidcsv.git -b v8.43

fastcluster:
	git clone https://github.com/dmuellner/fastcluster.git -b v1.1.28
	cp fastcluster/src/fastcluster.cpp fastcluster/src/fastcluster.hpp

clean:
	rm -rf rapidcsv fastcluster
