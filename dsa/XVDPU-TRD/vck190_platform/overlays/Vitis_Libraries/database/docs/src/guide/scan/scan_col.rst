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
   :keywords: scan, scanCol
   :description: Describes the structure and execution of the dynamic evaluation module.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _guide-scan:

********************************************************
Internals of Scan
********************************************************

.. toctree::
   :hidden:
   :maxdepth: 1

This document describes the structure and execution of dynamic evaluation module,implemented as ``scanCol`` function.

Scan is a function to transfer data form external DDR/HBM port to an internal HLS stream. 
As we known, a stream based data interface can be easily processed in FPGA. 
So it provides a bridge to across from external memory to FPGA. 
There are 3 versions of ``scanCol`` in ``xf_database`` now: 

Version1: defined in ``L1/include/hw/xf_database/scan_col.hpp``.
    In this head file, 6 kind of ``scanCol`` are provided to cope with 1-6 column number in DDR/HBM. 
    Each column has its own DDR/HBM port as input, and the data is scanned into 1-6 stream as output. 
    We also provide an example to scan one-column's data into multiple-channel's output stream, 
    and this logic is very easy to be re-designed for other dedicated purposes.

Version2: defined in ``L1/include/hw/xf_database/scan_col_2.hpp``.
    Unlike version1, a data structure is designed in this version, and it provides row number of a column inside the memory header. 
    So there is no need for ``Nrow`` in API of version2. 
    It is suggested to use this version for cases in which the row number of a column is unknown or not decided before calling kernels.

Version3: defined in ``L1/include/hw/xf_database/scan_cmp_str_col.hpp``.
    This version provides an internal logic scan and compare string. 
    The output is a boolean value to indicate whether the input string is equal to the constant string.
    It is more cost efficiency to process the boolean result than directly using the original string on FPGA.


