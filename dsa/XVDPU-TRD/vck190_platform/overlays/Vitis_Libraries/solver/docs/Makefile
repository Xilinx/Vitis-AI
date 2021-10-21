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

.PHONY: all install xml rst html clean cleanhtml cleanall

all: install

install: .stamp/sphinx
	rm -rf $(HTML_DEST_DIR)/*
#	make cleanall
	rsync -av --delete --exclude .git --exclude .nojekyll --exclude .buildinfo \
		 _build/html/ $(HTML_DEST_DIR) 1>/dev/null

.stamp/doxygen:
	rm -rf xml_*
	doxygen Doxyfile_L2_mat
	doxygen Doxyfile_L2_linearsolver
	doxygen Doxyfile_L2_eigensolver
	doxygen Doxyfile_2
	mkdir -p .stamp
	touch .stamp/doxygen

DOXYREST_FRAME = $(dir $(shell which doxyrest))/../doxyrest_frame

.stamp/doxyrest: .stamp/doxygen
	rm -rf rst_L2_*
	mkdir rst_L2_mat rst_L2_linearsolver rst_L2_eigensolver

	doxyrest xml_L2_mat/index.xml -o rst_L2_mat/index.rst -F $(DOXYREST_FRAME) -f c_index.rst.in -Dg_verbatimToCodeBlock
	doxyrest xml_L2_linearsolver/index.xml -o rst_L2_linearsolver/index.rst -F $(DOXYREST_FRAME) -f c_index.rst.in -Dg_verbatimToCodeBlock
	doxyrest xml_L2_eigensolver/index.xml -o rst_L2_eigensolver/index.rst -F $(DOXYREST_FRAME) -f c_index.rst.in -Dg_verbatimToCodeBlock

	rm -f rst_L2_mat/index.rst rst_L2_mat/global.rst
	rm -f rst_L2_linearsolver/index.rst rst_L2_linearsolver/global.rst
	rm -f rst_L2_eigensolver/index.rst rst_L2_eigensolver/global.rst

	rm -rf rst_2
	mkdir rst_2
	doxyrest xml_L2/index.xml -o rst_2/index.rst -F $(DOXYREST_FRAME) -f c_index.rst.in -Dg_verbatimToCodeBlock
	rm -f rst_2/index.rst rst_2/global.rst
	touch .stamp/doxyrest

.stamp/sphinx: .stamp/doxyrest Makefile.sphinx
	make -f Makefile.sphinx html
	touch .stamp/sphinx

xml:
	rm -f .stamp/doxygen
	make .stamp/doxygen

rst:
	rm -f .stamp/doxyrest
	make .stamp/doxyrest

html:
	rm -rf .stamp/sphinx _build/doctrees
	make .stamp/sphinx

clean:
	rm -rf .stamp

cleanhtml:
	rm -rf .stamp/sphinx _build/html _build/doctrees

cleanall:
	rm -rf .stamp xml_L* rst_L* _build/html _build/doctrees doc_html
