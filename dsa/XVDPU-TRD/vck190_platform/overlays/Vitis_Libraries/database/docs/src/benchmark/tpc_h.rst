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
   :keywords: Vitis Database Library, GQE, kernel, TPC-H
   :description: TPC-H queries with GQE.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _tpch:


**********************
TPC-H Queries with GQE
**********************
.. NOTE::
   Source code reference: https://github.com/Xilinx/Vitis_Libraries/tree/2019.2/database/L2/demos

   TPC-H queries have been obsolete in GQE 2020.2, because GQE 2020.2 has been re-designed with 
   non-compatible APIs for better integration with SQL engines.

GQE acceleration on TPC-H queries is introduced in Section :ref:`gqe_kernel_demo`.
Current experiment only involes the GQE kernels, and dedicated host C++ code is developed for each query.

