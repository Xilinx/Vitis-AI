## Copyright 2020 Xilinx Inc.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

SHELL=/bin/bash

DESTDIR=

CUR_DIR   =   $(shell pwd)

# Version number parameter
major=1
minor=3
patch=0
ifneq (${patch}, 0)
patch_str=.${patch}
endif
ifeq (${patch_str}, .)
patch_str=
endif
PACKAGE_NAME   =   vitis-ai_v${major}.${minor}${patch_str}_dnndk

define install_dest
	@if [ ! -d $(1)/usr/bin ]; then mkdir -p $(1)/usr/bin; fi
	@if [ ! -d $(1)/usr/lib ]; then mkdir -p $(1)/usr/lib; fi
	@if [ ! -d $(1)/usr/include ]; then mkdir -p $(1)/usr/include; fi
	cp tools/build/ddump $(1)/usr/bin
	cp tools/build/dexplorer $(1)/usr/bin
	cp tools/build/dsight $(1)/usr/bin
	cp -d n2cube/build/libhineon.so* $(1)/usr/lib
	cp -d n2cube/build/libn2cube.so* $(1)/usr/lib
	cp -d n2cube/build/libdpuaol.so* $(1)/usr/lib
	cp tools/lib/echarts.js $(1)/usr/lib
	cp tools/lib/libdsight.pyc $(1)/usr/lib
	cp n2cube/include/dnndk $(1)/usr/include -r
	cp n2cube/include/vai $(1)/usr/include -r
	@chmod -R 755 $(1)/usr/include/dnndk
	@chmod -R 755 $(1)/usr/include/vai
	@chmod 755 $(1)/usr/bin/ddump
	@chmod 755 $(1)/usr/bin/dexplorer
	@chmod 755 $(1)/usr/bin/dsight
endef

include Makefile.cfg

all:
	@if [ -z "$(SYSROOT)" ]; then \
		make -C ./n2cube; \
		make -C ./tools; \
	else \
		make SYSROOT=$(SYSROOT) CROSS_COMPILE=$(CROSS_COMPILE) -C ./n2cube; \
		make SYSROOT=$(SYSROOT) CROSS_COMPILE=$(CROSS_COMPILE) -C ./tools; \
		if [ "$(PYTHON_CFG)" = "yes" ]; then make python; fi; \
		make package; \
	fi

install:
	$(call install_dest, $(DESTDIR))

uninstall:
	rm -rf $(DESTDIR)/usr/include/dnndk
	rm -rf $(DESTDIR)/usr/include/vai
	rm -r $(DESTDIR)/usr/bin/ddump
	rm -r $(DESTDIR)/usr/bin/dexplorer
	rm -r $(DESTDIR)/usr/bin/dsight
	rm -r $(DESTDIR)/usr/lib/libhineon.so*
	rm -r $(DESTDIR)/usr/lib/libn2cube.so*
	rm -r $(DESTDIR)/usr/lib/libdpuaol.so*
	rm -r $(DESTDIR)/usr/lib/echarts.js
	rm -r $(DESTDIR)/usr/lib/libdsight.pyc

package:
	$(call install_dest, $(PACKAGE_NAME)/pkgs)
	cp common/install.sh $(PACKAGE_NAME)
	@chmod 755 $(PACKAGE_NAME)/install.sh
	@if [ -f ./$(wildcard n2cube/python/dist/*.whl) ]; then                                  \
		[ ! -d $(PACKAGE_NAME)/pkgs/python ] && mkdir $(PACKAGE_NAME)/pkgs/python;           \
		cp $(wildcard n2cube/python/dist/*.whl) $(PACKAGE_NAME)/pkgs/python;                 \
	fi
	tar -czf $(PACKAGE_NAME).tar.gz $(PACKAGE_NAME)
	rm -r $(PACKAGE_NAME)

python:
	cd n2cube/python; ./make.sh

clean:
	make clean -C ./n2cube
	make clean -C ./tools
	cd n2cube/python; ./make.sh clean

.PHONY: all clean install uninstall package python

