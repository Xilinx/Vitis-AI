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
   :keywords: Vitis, Security, Library, GMAC, mode
   :description: Galois Message Authentication Code (GMAC) is an specialization of the GCM(Galois/Counter mode) and used for authentication, as defined in NIST800_38D.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

****
GMAC
****

.. toctree::
   :maxdepth: 1

Overview
========

Galois Message Authentication Code (GMAC) is an specialization of the GCM(Galois/Counter mode) and used for authentication, as defined in `NIST800_38D`_.
In this version, we provide Advanced Encryption Standard (AES) based processing ability,
The cipherkey length for AES should be 128/192/256 bits.
Our implementation takes a fix-sized (128 bits per block) data stream, but text in real world has a variety of lengths.
Thus, you need to provide the data length in bits accompany with the data.

.. _`NIST800_38D`: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38d.pdf

Implementation on FPGA
======================

The GMAC algorithm is shown as the figure below:

.. image:: /images/GMAC.png
   :alt: GMAC algorithm flow chart
   :width: 100%
   :align: center

We support GMAC using AES block cipher in this implementation.

.. ATTENTION::
    The bit-width of the interfaces we provide is shown as follows:

    +-----------+----------+-----------+-----------+----+----+
    |           |   data   |  lenData  | cipherkey | IV | tag|
    +-----------+----------+-----------+-----------+----+----+
    |GMAC-AES128|   128    |    64     |    128    | 96 | 128|
    +-----------+----------+-----------+-----------+----+----+
    |GMAC-AES192|   128    |    64     |    192    | 96 | 128|
    +-----------+----------+-----------+-----------+----+----+
    |GMAC-AES256|   128    |    64     |    256    | 96 | 128|
    +-----------+----------+-----------+-----------+----+----+


.. CAUTION::
    Applicable conditions:

    1. The bit-width of initialization vector must be precisely 96 as recommended in the standard 
    to promote interoperability, efficiency, and simplicity of the design.

The internal structure of GMAC is shown in the figure below:

.. image:: /images/internal_structure_of_gmac.png
   :alt: Structure of GMAC
   :width: 100%
   :align: center

As we can see from the chart, the GMAC can be divided into two individual parts: The preGMAC and genGMAC.
These two parts can work independently, so they are designed into parallel dataflow processes,
connected by streams (FIFOs).

There is an overload genGMAC for Galois/Counter Mode (GCM) cipher mode of operation, as it must work with the structure of GCM,
the interface of the overload is a little bit more complex than the overload for GMAC.

Profiling
=========

GMAC-AES128
-----------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 3851    18468   15707    0     2     1445    0     2.915
======= ======= ======= ===== ====== ====== ====== ========


GMAC-AES192
-----------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 4854    24401   17523    0     6     1573    0     2.849
======= ======= ======= ===== ====== ====== ====== ========


GMAC-AES256
-----------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 5217    26430   18900    0     2     1701    0     3.069
======= ======= ======= ===== ====== ====== ====== ========



