.. 
   Copyright 2019 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. meta::
   :keywords: framework, libxilinxfintech, getting started, setup, environment, building library
   :description: Getting started with Vitis quantitative finance library.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

********************************
Getting Started
********************************

In order to prepare the framework for use, it is first necessary to build it.  
This step compiles all the relevant files into a single shared library (**libxilinxfintech.so**), that may then be linked against by the users application.


Environment Setup
#################
A couple of environment setup scripts are provided. These setup several environment variables that are required by the build process.

CSH:

.. code-block:: sh

	cd xf_fintech/L3/src
	source env.csh


BASH:

.. code-block:: sh

	cd xf_fintech/L3/src
	source env.sh



These scripts set up the following environment variables:

* XILINX_FINTECH_INC
	* This points to *xf_fintech/L3/include*

* XILINX_FINTECH_LIB_DIR
	* This points to *xf_fintech/L3/src/output* (where **libxilinxfintech.so** is placed after building)

* XILINX_XCL2_DIR
	* This points to *xf_fintech/ext/xcl2*

The script also modifies **LD_LIBRARY_PATH** to add the location of **libxilinxfintech.so** to it


Building the library
####################

To build **libxilinxfintech.so**:

.. code-block:: sh

	cd xf_fintech/L3/src
	make
  

After the build is complete, **libxilinxfintech.so** should be available in *xf_fintech/L3/src/output*
