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
   :keywords: python, examples, pybind, PyBind11
   :description: As an alternative to C++ it is possible to run the L3 models from the Python language. The python sub-directory within L3 contains examples for the various financial models.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


********************************
Python
********************************

As an alternative to C++ it is possible to run the L3 models from the Python language. 

The python sub-directory within L3 contains examples for the various financial models.

They utilise a specific python module as described below.

They also require the shared library to be built as described within the Getting Started section.

*Note that these are intended to be executed on real hardware - the software and hardware emulation modes as mentioned elsewhere are not applicable to the python section.*

Python Module Description
#########################
The Python module is created in C++ using the PyBind11 tool. This provides the abstraction layer between the C++ software apis i.e. the **.hpp** files and Python. The one file **module.cpp** contains the code to access all the Python financial models - these are split into clearly delineated sections within the file. 


Dependencies and versions
#########################
To be able to build the module and run the examples there are some dependencies these are listed below alongside the specific releases used during development, which was performed on machines using the CentOS 7.4 operating system. By default it comes with Python 2.7.5.

Python   - version 3.6.8	which can be found at https://www.python.org/ installation details vary based on operating system

Pybind11 - version 2.4.3	which can be found at https://pypi.org/project/pybind11/  to download the code select  https://github.com/pybind/pybind11   note this was tested at latest current version of 2.5.0

python-devel package which contains python3-config. For CentOS the command to install is - yum install python-devel


Building the module
####################
Before attempting to build the module ensure you have followed the contents of L3 Getting Started page - both the environment needs to be setup and the built library are required.

To generate the module run the Makefile - *make* from within your copy of the ../L3/python directory.

This will create a **module.o** file in the output sub-directory, along with **xf_fintech_python.so**. The python examples then import the required functions from this file, similarly to standard python modules.

*Hints: whilst including some checks in the scripts be aware your local environmental settings may vary. pip3 list will show if pybind11 is installed, it's directory path should be seen in the directories shown by running the command python3.6-config --includes . For CentOS that directory is /usr/include/python3.6m*

Running the examples
####################
Once the module has been compiled the simplest way to run is to copy the desired example python script and matching generated xclbin file, process described elsewhere, into the output subdirectory and run from that directory, appending both the xclbin filename and card type (u200 or u250). Again note that these should be hardware xclbin build files and not emulation ones.

For example to run the Dow Jones Monte Carlo European financial python script type the following or similar, depending upon your build filename and card type:
 
.. code-block:: sh

	cd output
	python36 ./dje_test.py dowjones.xclbin u200
 
Note within each example there is a comment describing the expected result. These generally mirror the C++ examples. 

Hint: Ensure the compiled library file **libxilinxfintech.so*** is accessible, as generated in the L3 Getting Started page by appending to the existing existing environmental LD_LIBRARY_PATH - for example - setenv LD_LIBRARY_PATH /<your_local_checkout_directory>/xf_fintech/L3/src/output:$LD_LIBRARY_PATH
