#
# Copyright 2019-2020 Xilinx, Inc.
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
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))

all: lib 

lib:
	wget https://github.com/maxmind/libmaxminddb/releases/download/1.4.2/libmaxminddb-1.4.2.tar.gz
	tar zxvf libmaxminddb-1.4.2.tar.gz
	cd libmaxminddb-1.4.2 && ./configure --prefix=$(CUR_DIR)/lib CFLAGS="-O3 -gdwarf-2 -g3"
	cd libmaxminddb-1.4.2 && make && make install

clean:
	rm -rf libmaxminddb* lib
