#
# Copyright 2020 Xilinx, Inc.
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
# makefile-generator v1.0.1
#

MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CUR_DIR := $(patsubst %/,%,$(dir $(MK_PATH)))

XF_PROJ_ROOT ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%L1/tests/*}')
XFLIB_DIR := $(abspath $(XF_PROJ_ROOT))

.PHONY: all clean cleanall

LIB_ONIG = $(CUR_DIR)/lib/lib/libonig.so
LIB_XF_RE_C = $(CUR_DIR)/libxfcompile.so

all: $(LIB_XF_RE_C)

oniguruma:
	git clone https://github.com/kkos/oniguruma.git oniguruma

$(LIB_ONIG): | oniguruma
	cd oniguruma && sh ./autogen.sh
	cd oniguruma && ./configure --prefix=$(CUR_DIR)/lib CFLAGS="-O3 -gdwarf-2 -g3"
	cd oniguruma && make
	cd oniguruma && make install

$(LIB_XF_RE_C): $(XFLIB_DIR)/L1/src/sw/xf_re_compile.c $(XFLIB_DIR)/L1/include/sw/xf_data_analytics/text/xf_re_compile.h $(LIB_ONIG)
	gcc -shared -o $@ $(filter %.c,$^) -g -Wall -Werror -fpic -std=c99 -gdwarf-2 \
		-I$(XFLIB_DIR)/L1/include/sw -I$(CUR_DIR)/oniguruma/src -I$(CUR_DIR)/lib/include \
		-L$(CUR_DIR)/lib/lib -lonig

clean:
	rm -rf $(LIB_XF_RE_C)

cleanall:
	rm -rf lib oniguruma $(LIB_XF_RE_C)
