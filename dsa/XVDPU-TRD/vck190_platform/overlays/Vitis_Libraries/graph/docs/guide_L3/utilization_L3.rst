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
   :keywords: Asynchronous, XRM, XRT, graph L3, Graph Library, OpenCL
   :description: The graph L3 layer provides an asynchronous and easy-to-integrate framework. By using the Xilinx FPGA Resource Manager (XRM) and this completely original and innovative L3 asynchronous framework, users can easily call the L3 asynchronous APIs in their pure software (heterochronous or synchronous) codes without the need of considering any hardware related things. The innovative L3 asynchronous framework can automatically acquire current platform's available resources (available FPGA boards, boards' serie numbers, available compute units, kernel names, etc.). In addition, this framework seperates FPGA deployments with L3 API calling, so the graph L3 layer can be easily deployed in the Cloud and can be easily used by pure software developers.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials



***********
User Guide
***********

The graph L3 layer provides an asynchronous and easy-to-integrate framework. By using the Xilinx FPGA Resource Manager (XRM) and this completely original and innovative L3 asynchronous framework, users can easily call the L3 asynchronous APIs in their pure software (heterochronous or synchronous) codes without the need of considering any hardware related things. The innovative L3 asynchronous framework can automatically acquire current platform's available resources (available FPGA boards, boards' serie numbers, available compute units, kernel names, etc.). In addition, this framework seperates FPGA deployments with L3 API calling, so the graph L3 layer can be easily deployed in the Cloud and can be easily used by pure software developers.


.. toctree::
	:maxdepth: 1

	L3_internal/getting_started.rst
	L3_internal/user_model.rst

   
