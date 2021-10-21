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
XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%L2/demos/text/log_analyzer/*}')
EXT_DIR := $(CUR_DIR)

CXXFLAGS += -I$(EXT_DIR) -I$(EXT_DIR)/../re_compile/oniguruma/src -I$(CUR_DIR)/host -I$(EXT_DIR)/../utils_sw
CXXFLAGS += -I$(EXT_DIR)/libmaxminddb/lib/include
CXXFLAGS += -I$(EXT_DIR)/rapidjson/rapidjson/include -std=c++11
CXXFLAGS += -O3 -DBS_TEST

LDFLAGS += -L$(EXT_DIR)/libmaxminddb/lib/lib -lmaxminddb
LDFLAGS += -L$(EXT_DIR)/../re_compile/lib/lib -lonig
all: | lib
	g++ ref_result.cpp -o test $(CXXFLAGS) $(LDFLAGS)
	export LD_LIBRARY_PATH=$(EXT_DIR)/libmaxminddb/lib/lib:$(EXT_DIR)/../re_compile/lib/lib:$(LD_LIBRARY_PATH);\
	./test -msg ../data/access.log -geoip ../data/GeoLite2-City.mmdb -o results.json

lib:
	make -C $(CUR_DIR)/../re_compile
	make -C $(CUR_DIR)/libmaxminddb
	make -C $(CUR_DIR)/rapidjson

build:
	g++ host/main.cpp -o test $(CXXFLAGS) $(LDFLAGS)

clean:
	rm test results.json
	make -C $(CUR_DIR)/../re_compile clean
	make -C $(CUR_DIR)/libmaxminddb clean
	make -C $(CUR_DIR)/rapidjson clean
