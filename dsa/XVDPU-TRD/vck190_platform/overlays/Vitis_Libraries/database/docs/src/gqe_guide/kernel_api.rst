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
   :keywords: Vitis Database Library, GQE, kernel, api
   :description: The GQE kernel application programming interface reference.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _gqe_kernel_api:

*******************
GQE Kernel APIs
*******************

These APIs are implemented as OpenCL kernels:

.. toctree::
   :maxdepth: 1

.. include:: ../rst_2/global.rst
   :start-after: FunctionSection

.. NOTE::
   GQE has been tested on Alveo U280 card, and makes use of both HBM and DDR.
   While other cards like U250 and U200 are not supported out-of-box,
   porting and gaining acceleration is surely possible, with tailoring and tuning.

