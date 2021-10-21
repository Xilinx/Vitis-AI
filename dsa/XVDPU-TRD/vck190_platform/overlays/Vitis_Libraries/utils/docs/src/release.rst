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

.. _release_note:

Release Note
============

.. toctree::
   :hidden:
   :maxdepth: 1

2021.1
------

In the 2021.1 release, Data-Mover is added into this library. Unlike other C++ based APIs, this addition
is targeting people less experienced in HLS based kernel design and just wants to test their stream-based designs.
The Data-Mover is actually a kernel source code generator, creating a list of common helper kernels to drive
or valdiate designs, like those on AIE devices.

2020.2
------

In the 2020.2 release the following APIs has been added

* **Argument parser**: this (experimental) implementation parses the options and flags passed from command line,
  and offers automatic help information generation. It should help the developers to create unified experience on test
  cases and user applications.
* **FIFO multiplexer**: this module wraps around a FIFO (implemented through hls::stream) to enable passing data
  of different type through the same hardware resource. When the data is too wide, it will automatically be transferred
  using multiple cycles. This module is expected to make the dataflow code more compact and readable.


2020.1
------

This release adds the following API to this library:

* Read-only cache (cache):
  This API stores history data recently loaded from DDR/HBM in the on-chip memory(URAM).
  It aims to reduce DDR/HBM access when the memory is accessed randomly.
* AXI master read without e signal (an overload of axiToStream):
  This API provides buffered read from AXI master into stream.
  It is assumed that the receiver of the stream knows the number of element to process.


2019.2
------

The 2019.2 release provides a range of HLS primitives for:

* Work distribution and result collection in different algorithms.
* Stream manipulation, including combination, duplication, synchronization, shuffle, etc.
* Updating URAM array in higher frequency.
