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

.. _guide-axi_to_multi_stream:

********************************
Internals of axiToMultiStream
********************************

.. toctree::
   :hidden:
   :maxdepth: 2

This document describes the structure and execution flow of axiToMultiStream,
implemented as :ref:`axiToMultiStream <cid-xf::common::utils_hw::axiToMultiStream>` function.

.. image:: /images/axi_to_multi_stream.png
   :alt: axi_to_multi_stream Structure
   :width: 80%
   :align: center

The axiToMultiStream primitive is non-blocking and uses Round Robin to load multiple categories of data from one AXI master into multiple streams.
For example, in the implementation of one AXI port loading three types of data, the data of each type could be tightly packed.
Each type of data's length should be multiple of 8 bits. The three types of data width could be unaligned or aligned,
e.g. three types of data compressed in one binary files. AXI port is assumed to have width as multiple of 8-bit char.

.. CAUTION::
   Applicable condition:

   This module only supports 3 categories of data.

   When input data ptr width is less than AXI port width, the AXI port bandwidth
   will not be fully used. So, AXI port width should be minimized while meeting
   performance requirements of application.


This primitive has two modules working simultaneously in side.

1. ``read_to_vec_stream``: It reads AXI master to three BRAM buffers,
   and loads the buffers in non-blocking and round robin way to ``_WAxi`` width stream.

2. ``split_vec_to_aligned``: It takes the ``_WAxi`` width stream, aligns to the stream width, and splits the _WAxi width data to stream width and output the stream.

