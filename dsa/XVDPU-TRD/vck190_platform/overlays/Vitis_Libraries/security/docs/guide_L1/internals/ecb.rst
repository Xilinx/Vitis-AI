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
   :keywords: Vitis, Security, Library, ECB, mode
   :description: The Electronic Codebook (ECB) mode is a typical block cipher mode of operation using block cipher algorithm. 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials



********
ECB Mode
********

.. toctree::
   :maxdepth: 1

Overview
========

The Electronic Codebook (ECB) mode is a typical block cipher mode of operation using block cipher algorithm.
In this version, we provide Data Encryption Standard (DES) and Advanced Encryption Standard (AES) processing ability,
the cipherkey length for DES should be 64 bits, and 128/192/256 bits for AES.
Another limitation is that our working mode works on units of a fixed size (64 or 128 bits for 1 block),
but text in the real world has a variety of lengths.
So, the last block of the text provided to this primitive must be padded to 128 bits before encryption or decryption. 

Implementation on FPGA
======================

We support ECB-DES, ECB-AES128, ECB-AES192, and ECB-AES256 modes in this implementation.

.. ATTENTION::
    The bit-width of the interfaces we provide is shown as follows:

    +-----------+-----------+------------+-----------+----+
    |           | plaintext | ciphertext | cipherkey | IV |
    +-----------+-----------+------------+-----------+----+
    |  ECB-DES  |    64     |    64      |    64     | 64 |
    +-----------+-----------+------------+-----------+----+
    |ECB-AES128 |    128    |    128     |    128    | 128|
    +-----------+-----------+------------+-----------+----+
    |ECB-AES192 |    128    |    128     |    192    | 128|
    +-----------+-----------+------------+-----------+----+
    |ECB-AES256 |    128    |    128     |    256    | 128|
    +-----------+-----------+------------+-----------+----+


The algorithm flow chart is shown as follow:

.. image:: /images/ECB_working_mode.png
   :alt: algorithm flow chart of ECB
   :width: 80%
   :align: center

As we can see from the chart, both encryption and decryption part of ECB mode has no dependencies,
so the input block of each iteration needs no feedback data from its last iteration.
Thus, both encryption and decryption part of ECB mode can achieve an initiation interval (II) = 1.

Profiling
=========

ECB-DES encryption
------------------

====== ====== ====== ===== ====== ===== ====== ========
 CLB    LUT     FF    DSP   BRAM   SRL   URAM   CP(ns)
====== ====== ====== ===== ====== ===== ====== ========
 317    1569   2544    0     0      1     0     2.095
====== ====== ====== ===== ====== ===== ====== ========


ECB-DES decryption
------------------

====== ====== ====== ===== ====== ===== ====== ========
 CLB    LUT     FF    DSP   BRAM   SRL   URAM   CP(ns)
====== ====== ====== ===== ====== ===== ====== ========
 310    1567   2544    0     0      1     0     2.243
====== ====== ====== ===== ====== ===== ====== ========


ECB-AES128 encryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT      FF    DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 2100    8998    6381     0     2     513     0     2.649
======= ======= ======= ===== ====== ====== ====== ========


ECB-AES128 decryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT      FF    DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 5219    27494   10416    0     10    513     0     3.061
======= ======= ======= ===== ====== ====== ====== ========


ECB-AES192 encryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT      FF    DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 2916    14989   8065     0     6     641     0     2.916
======= ======= ======= ===== ====== ====== ====== ========


ECB-AES192 decryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT      FF    DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 6353    33200   12312    0     14    641     0     3.091
======= ======= ======= ===== ====== ====== ====== ========


ECB-AES256 encryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT      FF    DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 3253    16971   9057     0     2     769     0     2.921
======= ======= ======= ===== ====== ====== ====== ========


ECB-AES256 decryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT      FF    DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 7261    38266   13936    0     10    769     0     2.982
======= ======= ======= ===== ====== ====== ====== ========



