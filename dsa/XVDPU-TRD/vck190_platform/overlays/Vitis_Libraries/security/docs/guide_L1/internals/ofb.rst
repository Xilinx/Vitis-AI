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
   :keywords: Vitis, Security, Library, OFB, mode
   :description: The Output Feedback (OFB) mode is a typical block cipher mode of operation using block cipher algorithm.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

********
OFB Mode
********

.. toctree::
   :maxdepth: 1

Overview
========

The Output Feedback (OFB) mode is a typical block cipher mode of operation using block cipher algorithm.
In this version, we provide Data Encryption Standard (DES) and Advanced Encryption Standard (AES) processing ability,
the cipherkey length for DES should be 64 bits, and 128/192/256 bits for AES.
Another limitation is that our working mode works on units of a fixed size (64 or 128 bits for 1 block),
but text in the real world has a variety of lengths.
So, the last block of the text provided to this primitive must be padded to 128 bits before encryption or decryption. 

Implementation on FPGA
======================

We support OFB-DES, OFB-AES128, OFB-AES192, and OFB-AES256 modes in this implementation.

.. ATTENTION::
    The bit-width of the interfaces we provide is shown as follows:

    +-----------+-----------+------------+-----------+----+
    |           | plaintext | ciphertext | cipherkey | IV |
    +-----------+-----------+------------+-----------+----+
    |  OFB-DES  |    64     |    64      |    64     | 64 |
    +-----------+-----------+------------+-----------+----+
    |OFB-AES128 |    128    |    128     |    128    | 128|
    +-----------+-----------+------------+-----------+----+
    |OFB-AES192 |    128    |    128     |    192    | 128|
    +-----------+-----------+------------+-----------+----+
    |OFB-AES256 |    128    |    128     |    256    | 128|
    +-----------+-----------+------------+-----------+----+


The algorithm flow chart is shown as follow:

.. image:: /images/OFB_working_mode.png
   :alt: algorithm flow chart of OFB
   :width: 100%
   :align: center

As we can see from the chart, both encryption and decryption part of OFB mode has dependencies,
so the input block of each iteration (except for iteration 0) needs a feedback data from its last iteration.
Thus, the initiation interval (II) of OFB mode cannot achieve an II = 1.

Profiling
=========

OFB-DES encryption
------------------

====== ====== ====== ===== ====== ===== ====== ========
 CLB    LUT     FF    DSP   BRAM   SRL   URAM   CP(ns)
====== ====== ====== ===== ====== ===== ====== ========
 338    1693   2734    0     0      0     0     2.120
====== ====== ====== ===== ====== ===== ====== ========


OFB-DES decryption
------------------

====== ====== ====== ===== ====== ===== ====== ========
 CLB    LUT     FF    DSP   BRAM   SRL   URAM   CP(ns)
====== ====== ====== ===== ====== ===== ====== ========
 340    1694   2734    0     0      0     0     2.255
====== ====== ====== ===== ====== ===== ====== ========


OFB-AES128 encryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 2192    9313    6633     0     2     512     0     2.817 
======= ======= ======= ===== ====== ====== ====== ========


OFB-AES128 decryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 2089    9316    6633     0     2     512     0     2.668
======= ======= ======= ===== ====== ====== ====== ========


OFB-AES192 encryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 2921    15136   8319     0     6     640     0     2.900
======= ======= ======= ===== ====== ====== ====== ========


OFB-AES192 decryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 2946    15138   8308     0     6     640     0     3.037
======= ======= ======= ===== ====== ====== ====== ========


OFB-AES256 encryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 3177    14874   9309     0     2     768     0     3.001
======= ======= ======= ===== ====== ====== ====== ========


OFB-AES256 decryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 2998    14864   9309     0     2     768     0     3.129
======= ======= ======= ===== ====== ====== ====== ========



