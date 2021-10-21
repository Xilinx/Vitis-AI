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

.. meta::
   :keywords: Vitis, Database, Vitis Database Library, Alveo
   :description: Vitis Database Library is an open-sourced Vitis library written in C++ for accelerating database applications in a variety of use cases.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _brief:

======================
Vitis Database Library
======================

Introduction
------------

Vitis Database Library is an open-sourced Vitis library written in C++ and released under
`Apache 2.0 license <https://www.apache.org/licenses/LICENSE-2.0>`_
for accelerating database applications in a variety of use cases.

The main target audience of this library is SQL engine developers, who want to accelerate
the query execution with FPGA cards.
Currently, this library offers three levels of acceleration:

* At module level, it provides an optimized hardware implementation of most common relational database execution plan steps,
  like hash-join and aggregation.
* In kernel level, the post-bitstream-programmable kernel can be used to map a sequence of execution plan steps,
  without having to compile FPGA binaries for each query.
* The software APIs level wrap the details of offloading acceleration with programmable kernels,
  and allow users to accelerate supported database tasks on Alveo cards without heterogeneous development knowledge.

At each level, this library strives to make modules configurable through documented parameters,
so that advanced users can easily tailor, optimize or combine with property logic for specific needs.
Test cases are provided for all the public APIs, and can be used as examples of usage.


Generic Query Engine
--------------------

This library refers its solution to accelerated key execution step(s) in query plan,
like table JOIN as Generic Query Engine (GQE).
GQE consists of post-bitstream programmable kernel(s) and corresponding software stack.

.. image:: /images/gqe_overview.png
   :alt: General Query Engine Overview
   :scale: 50%
   :align: left

.. NOTE::
   GQE is still under extensive development, so its APIs are subject to future changes.


License
-------

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

