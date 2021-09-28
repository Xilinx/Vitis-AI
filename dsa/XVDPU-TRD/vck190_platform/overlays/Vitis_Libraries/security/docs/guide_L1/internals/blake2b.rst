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
   :keywords: Vitis, Security, Library, BLAKE2, algorithms
   :description: BLAKE2 is a set of cryptographic hash functions defined in  RFC 7693 : The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC).
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials



*****************
BLAKE2 Algorithms
*****************

.. toctree::
   :maxdepth: 1

Overview
========

BLAKE2 is a set of cryptographic hash functions defined in `RFC 7693`_: The BLAKE2 Cryptographic Hash and Message Authentication Code (MAC).

The BLAKE2 family consists of 2 hash functions, and both of them provide security superior to SHA-2.
The BLAKE2B is optimized for 64-bit platforms, while the BLAKE2S is optimized for 8-bit to 32-bit platforms.

Currently this library supports BLAKE2B algorithm.

.. _`RFC 7693`: https://tools.ietf.org/html/rfc7693

Implementation on FPGA
======================

The internal structure of BLAKE2B algorithm is shown as the figure below:

.. image:: /images/internal_structure_of_blake2b.png
   :alt: Structure of BLAKE2B algorithm
   :width: 100%
   :align: center

As we can see from the figure, the BLAKE2B hash calculation can be partitioned into two parts.

* The generateBlock module pads the input message and the optional input key into fixed sized blocks,
  and informs the digest part that how many blocks do we have in this message.
  The message word size is 64-bit for BLAKE2B, 32-bit for BLAKE2S,
  and each block has a size of 16 message words.
* The disgest part iteratively computes the hash values. Loop-carried dependency
  is enforced by the algorithm, and thus this part cannot reach II=1.

As these two parts can work independently, they are designed into parallel dataflow process,
connected by streams (FIFOs).

Performance
===========

BLAKE2B
-------

A single instance of BLAKE2B function processes input message at the rate of
``1024 bit / 737 cycles`` at 315.95MHz.

The hardware resource utilizations of BLAKE2B is listed in :numref:`tab1BLAKE2B` below:

.. _tab1BLAKE2B:

.. table:: Hardware resources for single BLAKE2B hash calculation
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |     0    |     0    |  20665   |   15635  |   3466    |     0     |      3.053      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+

