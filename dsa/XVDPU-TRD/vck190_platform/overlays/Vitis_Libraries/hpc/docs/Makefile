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

ifeq (, $(shell which doxygen 2>/dev/null))
$(error "No doxygen in PATH.")
endif
ifeq (, $(shell which doxyrest 2>/dev/null))
$(error "No doxyrest in PATH.")
endif
ifeq (, $(shell which sphinx-build 2>/dev/null))
$(error "No sphinx-build in PATH.")
endif
ifeq (, $(HTML_DEST_DIR))
$(error "HTML_DEST_DIR is not set.")
endif

.PHONY: all install clean cleanhtml cleanall

all: install

install: .stamp/sphinx
	rsync -av --delete --exclude .git --exclude .nojekyll --exclude .buildinfo \
		 _build/html/ $(HTML_DEST_DIR) 1>/dev/null

DOXYREST_FRAME = $(dir $(shell which doxyrest))/../doxyrest_frame
rsts := $(patsubst Doxyfile_%, rst_%, $(wildcard Doxyfile_*))

rst_%: Doxyfile_%
	doxygen $^
	rm -rf rst_$*
	mkdir -p rst_$*
	doxyrest xml_$*/index.xml -o rst_$*/index.rst \
		-F $(DOXYREST_FRAME) -f c_index.rst.in -Dg_verbatimToCodeBlock

.stamp/sphinx: $(rsts) Makefile.sphinx
	mkdir -p $(HTML_DEST_DIR)
	mkdir -p _build
	mkdir -p src/_static
	mkdir -p .stamp
	make -f Makefile.sphinx html
	touch .stamp/sphinx

clean:
	rm -rf .stamp

cleanhtml:
	rm -rf .stamp/sphinx _build/html _build/doctrees

cleanall:
	rm -rf .stamp src/_static _build $(HTML_DEST_DIR)
	rm -rf xml_* rst_*
