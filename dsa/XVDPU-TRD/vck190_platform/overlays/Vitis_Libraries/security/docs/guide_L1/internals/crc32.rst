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
   :keywords: Vitis, Security, Library, CRC
   :description: A cyclic redundancy check (CRC) is an error-detecting code commonly which encode messages by adding a fixed-length check value, for the purpose of error detection.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


******************
CRC32
******************

.. toctree::
   :maxdepth: 1

Overview
========

A cyclic redundancy check (CRC) is an error-detecting code commonly which encode messages by adding a fixed-length check value, for the purpose of error detection. CRC32 is a 32bit CRC code. More details in `Wiki CRC`_.

.. _`Wiki CRC`: https://en.wikipedia.org/wiki/Cyclic_redundancy_check

Implementation on FPGA
======================

For the CRC32 design, it can be specified as:

- Width : 32
- Poly  : 0xEDB88320
- Init  : 0xFFFFFFFF
- Way   : Lookup table

For the :math:`table` and the CRC32 implementation, please check out `Ref`_.

.. _'Ref`: https://create.stephan-brumme.com/crc32/#slicing-by-16-overview

For more information, please check out source code.
