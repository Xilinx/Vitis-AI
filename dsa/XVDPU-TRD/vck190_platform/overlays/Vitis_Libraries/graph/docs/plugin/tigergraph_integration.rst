.. 
   Copyright 2020 Xilinx, Inc.
  
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
   :keywords: TigerGraph, GSQL, Graph L3
   :description: TigerGraph is a complete, distributed, parallel graph computing platform supporting web-scale data analytics in real-time. GSQL is a language designed for TigerGraph Inc.'s property graph database. Users can define their own expression functions in C++ and add them into GSQL. Graph L3 APIs can be easily integrated with TigerGraph.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


    
**********************
TigerGraph Integration
**********************

TigerGraph is a complete, distributed, parallel graph computing platform supporting web-scale data analytics in real-time. GSQL is a language designed for TigerGraph Inc.'s property graph database. Users can define their own expression functions in C++ and add them into GSQL. Graph L3 APIs can be easily integrated with TigerGraph.

Software Requirements
~~~~~~~~~~~~~~~~~~~~~
* Ubuntu 16.04 LTS
* `Xilinx RunTime (XRT) <https://github.com/Xilinx/XRT>`_ 2020.1
* `Xilinx FPGA Resource Manager (XRM) <https://github.com/Xilinx/XRM>`_ 2020.2

TigerGraph integration needs static boost version XRT. Please follow the steps:

* Download `Xilinx RunTime (XRT) <https://github.com/Xilinx/XRT>`_ release version source code 
* sudo apt install libboost-program-options-dev
* sudo apt-get update
* sudo apt install libcurl4-gnutls-dev
* sudo apt-get update
* source PATH_XRT/src/runtime_src/tools/scripts/xrtdeps.sh
* mkdir PATH_XRT/boost
* source PATH_XRT/src/runtime_src/tools/scripts/boost.sh -prefix PATH_XRT/boost
* source PATH_XRT/build/build.sh -clean
* sudo apt intall cmake
* sudo apt-get update
* env XRT_BOOST_INSTALL=PATH_XRT/boost/xrt PATH_XRT/build/build.sh
* cd PATH_XRT/build/Debug and PATH_XRT/build/Release
* make packages

Hardware Requirements
~~~~~~~~~~~~~~~~~~~~~
* `Alveo U50 <https://www.xilinx.com/products/boards-and-kits/alveo/u50.html>`_

Integration Flow
~~~~~~~~~~~~~~~~
In order to simplify the integration of graph L3 and TigerGraph, a shell script is written. Please follow the following steps: 

* Download `TigerGraph <https://www.tigergraph.com/>`_ 2.4.0
* Install TigerGraph
* cd PATH_GRAPH_LIB/plugin
* Change `TigerGraphPath` in install.sh, Makefile and tigergraph/MakeUdf to the path of TigerGraph installed
* Change TigerGraph path related parameters in tigergraph/bash_tigergraph
* ./install.sh

Running Flow
~~~~~~~~~~~~~
In the Graph library, some L3 APIs have been integrated to TigerGraph and the corresponding gsql testcases are offered. Once Users finish integration flow, they can `source run.sh` in each testcase and run the corresponding TigerGraph API.


   
