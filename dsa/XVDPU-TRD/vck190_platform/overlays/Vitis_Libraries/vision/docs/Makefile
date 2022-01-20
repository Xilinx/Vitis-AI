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


.PHONY: all install xml rst html clean cleanhtml ultraclean

all: ultraclean install


install: .stamp/sphinx
	rm $(HTML_DEST_DIR)/*.html
	rm $(HTML_DEST_DIR)/*.inv
	rm $(HTML_DEST_DIR)/*.js
	rm -rf $(HTML_DEST_DIR)/source
	rm -rf $(HTML_DEST_DIR)/_sources
	rm -rf $(HTML_DEST_DIR)/_static
	rm -rf $(HTML_DEST_DIR)/doxygen
	rsync -av _build/html/ $(HTML_DEST_DIR)/
	

# different for L1 and L2
.ONESHELL:
.stamp/doxygen:
	rm -rf doxygen/L1/xml doxygen/L2/xml doxygen/L3/xml
	mkdir -p .stamp
	#cd doxygen/L1
	#doxygen ./Doxyfile
	#cd ../../
	touch .stamp/doxygen

.ONESHELL:
.stamp/doxyrest: .stamp/doxygen
	rm -rf doxygen/L1/rst 
	#mkdir doxygen/L1/rst 
	#cd ./doxygen/L1
	#doxyrest xml/index.xml -o rst/api_index.rst \
	#	-F ../doxyrest_frame -f c_index.rst.in -Dg_verbatimToCodeBlock
	#cd ../../
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
	
.ONESHELL:
ultraclean:
	rm -rf .stamp _build/html _build/doctrees
	cd doxygen/
	rm -rf L1/xml L1/rst L1/html 
	rm -rf L1/latex L2/latex L3/latex
	cd ../

