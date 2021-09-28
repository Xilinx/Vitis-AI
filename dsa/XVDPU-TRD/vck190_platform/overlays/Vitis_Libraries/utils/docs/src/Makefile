#
# Copyright 2019 Xilinx, Inc.
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

.PHONY: all install xml rst html clean cleanhtml cleanall check_tools

all: install

check_tools:
ifeq (, $(shell which doxygen 2>/dev/null))
	@echo "ERROR: No doxygen in PATH" && false
endif
ifeq (, $(shell which doxyrest 2>/dev/null))
	@echo "ERROR: No doxyrest in PATH" && false
endif
ifeq (, $(shell which sphinx-build 2>/dev/null))
	@echo "ERROR: No sphinx-build in PATH" && false
endif

install: .stamp/sphinx
ifeq (, $(HTML_DEST_DIR))
	@echo "ERROR: HTML_DEST_DIR is not set." && false
endif
	rsync -av --delete --exclude .git --exclude .nojekyll --exclude .buildinfo \
		 _build/html/ $(HTML_DEST_DIR) 1>/dev/null

.stamp/doxygen: | check_tools
	rm -rf xml
	doxygen Doxyfile
	mkdir -p .stamp
	touch .stamp/doxygen

DOXYREST_FRAME = $(dir $(shell which doxyrest))/../doxyrest_frame

.stamp/doxyrest: .stamp/doxygen | check_tools
	rm -rf rst
	mkdir rst
	doxyrest xml/index.xml -o rst/index.rst \
		-F $(DOXYREST_FRAME) -f c_index.rst.in -Dg_verbatimToCodeBlock
	touch .stamp/doxyrest

.stamp/sphinx: .stamp/doxyrest Makefile.sphinx | check_tools
	make -f Makefile.sphinx html
	touch .stamp/sphinx

xml:
	rm -f .stamp/doxygen
	make .stamp/doxygen

rst:
	rm -f .stamp/doxyrest
	make .stamp/doxyrest

html:
	rm -rf .stamp/sphinx _build/doctrees _build/html
	make .stamp/sphinx

clean:
	rm -rf .stamp

cleanhtml:
	rm -rf .stamp/sphinx _build/html _build/doctrees

cleanall:
	rm -rf .stamp xml rst _build/html _build/doctrees
