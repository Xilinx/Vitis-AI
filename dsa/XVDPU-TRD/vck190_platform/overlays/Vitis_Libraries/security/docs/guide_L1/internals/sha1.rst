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
   :keywords: Vitis, Security, Library, SHA-1, Algorithm
   :description: The SHA-1 secure hash algorithm is a hash-based cryptographic function, it takes a message of arbitrary length as its input, produces a 160-bit digest. 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


***************
SHA-1 Algorithm
***************

.. toctree::
   :maxdepth: 1

Overview
========

The SHA-1 secure hash algorithm is a hash-based cryptographic function, it takes a message of arbitrary length as its input, produces a 160-bit digest. It has a padding and appending process before digest the message of arbitrary length.

The SHA-1 algorithm is defined in `FIPS 180`_.

.. _`FIPS 180`: https://csrc.nist.gov/CSRC/media/Publications/fips/180/4/archive/2012-03-06/documents/Draft-FIPS180-4_Feb2011.pdf

Implementation on FPGA
======================

The internal structure of SHA-1 is shown in the figure below:

.. image:: /images/internal_structure_of_sha1.png
   :alt: Structure of SHA-1
   :width: 100%
   :align: center

As we can see from the figures, the hash calculation can be partitioned into two parts.

* The pre-processing part pads or splits the input message which is comprised by a stream of 32-bit words into fixed sized blocks (512-bit for each).

* The digest part iteratively computes the hash values. Loop-carried dependency is enforced by the algorithm itself, thus this part cannot reach an initiation interval (II) = 1.

As the two parts can work independently, they are designed into parallel dataflow process, connected by streams (FIFOs).

Performance
===========

A single instance of SHA-1 function processes input message at the rate of ``512 bit / 84 cycles`` at 346.62MHz.

The hardware resource utilizations are listed in :numref:`tab1SHA1` below:

.. _tab1SHA1:

.. table:: Hardware resources for single SHA-1 hash calculation
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |     1    |     0    |   7518   |   3633   |    976    |     0     |      3.004      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+

