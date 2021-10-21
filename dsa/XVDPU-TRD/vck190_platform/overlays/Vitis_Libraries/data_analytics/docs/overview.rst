.. 
   Copyright 2019-2020 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. toctree::
   :hidden:

.. _overview:

*****************************
Vitis Data Analytics Library
*****************************

Introduction
------------

Vitis Data Analytics Library is an open-sourced Vitis library written in C++ for accelerating
data analytics applications in a variety of use cases.

Three categories of APIs are provided by this library, namely:

* **Data Mining** APIs, including all most common subgroups:

  * *Classification*: decision tree, random forest, native Bayes and SVM algorithms.
  * *Clustering*: K-means algorithm.
  * *Regression*: linear, gradient and decision tree based algorithms.

* **Text Processing** APIs for unstructured information extraction and transformation. New in 2020.2 release!

  * *Regular expression* with capture groups is a powerful and popular tool of information extraction.
  * *Geo-IP* enables looking up IPv4 address for geographic information.
  * Combining these APIs, a complete demo has been developed to batch transform Apache HTTP server log into structured JSON text.

* **DataFrame** APIs, also new in 2020.2, can be used to store and load multiple types of data with both fixed and variable length into DataFrames.

  * The in-memory format follows the design principles of `Apache Arrow <https://arrow.apache.org/>`_
    with goal of allowing access without per-element transformation.
  * Loaders from common formats are planned to be added in future releases, for example CSV and JSONLine loaders.

Like most other Vitis sub libraries, Data Analytics Library also organize its APIs by levels.

* The bottom level, L1, is mostly hardware modules with its software configuration generators.
* The second level, L2, provides kernels that are ready to be built into FPGA binary and invoked with standard OpenCL calls.
* The top level, L3, is meant for solution integrators as pure software C++ APIs.
Little background knowledge of FPGA or heterogeneous development is required for using L3 APIs.

At each level, the APIs are designed to be as reusable as possible, combined with the corporate-friendly
Apache 2.0 license, advanced users are empowered to easily tailor, optimize and assemble solutions.


License
-------

Licensed using the `Apache 2.0 license <https://www.apache.org/licenses/LICENSE-2.0>`_.

    Copyright 2019-2020 Xilinx, Inc.
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Trademark Notice
----------------

    Xilinx, the Xilinx logo, Artix, ISE, Kintex, Spartan, Virtex, Zynq, and
    other designated brands included herein are trademarks of Xilinx in the
    United States and other countries.
    
    All other trademarks are the property of their respective owners.

