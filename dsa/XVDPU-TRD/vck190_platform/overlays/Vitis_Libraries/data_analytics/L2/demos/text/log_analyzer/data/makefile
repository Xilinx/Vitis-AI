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

REF_DIR := $(CUR_DIR)/../ref_result
HOST_SRCS := $(CUR_DIR)/convert_s1.cpp
CXXFLAGS += -I$(REF_DIR)
CXXFLAGS += -I$(REF_DIR)/libmaxminddb/lib/include
CXXFLAGS += -I$(REF_DIR)/rapidjson/rapidjson/include -std=c++11
CXXFLAGS += -O3

LDFLAGS += -L$(REF_DIR)/libmaxminddb/lib/lib
LDFLAGS += -lmaxminddb

EXE_FILE := convert_s1.exe

all: convert_s1 convert_s2

log:
	wget http://www.almhuette-raith.at/apache-log/access.log

lib:
	make -C $(REF_DIR)/libmaxminddb
	make -C $(REF_DIR)/rapidjson

convert_s2:
	g++ convert_s2.cpp -o convert_s2.exe
	./convert_s2.exe geo_tmp.dat geo.dat

convert_s1: lib
	LD_LIBRARY_PATH=$(REF_DIR)/libmaxminddb/lib/lib:$(LD_LIBRARY_PATH) \
	g++ $(HOST_SRCS) -o $(EXE_FILE) $(LDFLAGS) $(CXXFLAGS) -O3
	LD_LIBRARY_PATH=$(REF_DIR)/libmaxminddb/lib/lib:$(LD_LIBRARY_PATH) \
	./$(EXE_FILE) GeoLite2-City.mmdb geo_tmp.dat 

clean:
	rm out.dat convert.dat $(EXE_FILE) convert_s2.exe
