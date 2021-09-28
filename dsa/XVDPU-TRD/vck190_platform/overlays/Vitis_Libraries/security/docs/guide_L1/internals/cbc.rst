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
   :keywords: Vitis, Security, Library, CBC, mode
   :description: The Cipher Block Chaining (CBC) mode is a typical block cipher mode of operation using block cipher algorithm.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials



********
CBC Mode
********

.. toctree::
   :maxdepth: 1

Overview
========

The Cipher Block Chaining (CBC) mode is a typical block cipher mode of operation using block cipher algorithm.
In this version, we provide Data Encryption Standard (DES) and Advanced Encryption Standard (AES) processing ability,
the cipherkey length for DES should be 64 bits, and 128/192/256 bits for AES.
Another limitation is that our working mode works on units of a fixed size (64 or 128 bits for 1 block),
but text in the real world has a variety of lengths.
So, the last block of the text provided to this primitive must be padded to 128 bits before encryption or decryption. 

Implementation on FPGA
======================

We support CBC-DES, CBC-AES128, CBC-AES192, and CBC-AES256 modes in this implementation.

.. ATTENTION::
    The bit-width of the interfaces we provide is shown as follows:

    +-----------+-----------+------------+-----------+-----+
    |           | plaintext | ciphertext | cipherkey | IV  |
    +-----------+-----------+------------+-----------+-----+
    |  CBC-DES  |    64     |    64      |    64     | 64  |
    +-----------+-----------+------------+-----------+-----+
    |CBC-AES128 |    128    |    128     |    128    | 128 |
    +-----------+-----------+------------+-----------+-----+
    |CBC-AES192 |    128    |    128     |    192    | 128 |
    +-----------+-----------+------------+-----------+-----+
    |CBC-AES256 |    128    |    128     |    256    | 128 |
    +-----------+-----------+------------+-----------+-----+

    
The algorithm flow chart is shown as follow:

.. image:: /images/CBC_working_mode.png
   :alt: algorithm flow chart of CBC
   :width: 100%
   :align: center

As we can see from the chart, the encryption part of CBC mode has loop-carried dependency which is enforced by the algorithm,
then the input block of each iteration (except for iteration 0) needs a feedback data from its last iteration.
Thus, the initiation interval (II) of CBC encryption cannot achieve an II = 1.
However, the decryption part of CBC mode has no dependencies, so that it can achieve an II = 1.

Profiling
=========

CBC-DES encryption
------------------

====== ====== ====== ===== ====== ===== ====== ========
 CLB    LUT     FF    DSP   BRAM   SRL   URAM   CP(ns)
====== ====== ====== ===== ====== ===== ====== ========
 332    1642   2670    0     0     33     0     1.854
====== ====== ====== ===== ====== ===== ====== ========


CBC-DES decryption
------------------

====== ====== ====== ===== ====== ===== ====== ========
 CLB    LUT     FF    DSP   BRAM   SRL   URAM   CP(ns)
====== ====== ====== ===== ====== ===== ====== ========
 348    1700   2738    0     0     33     0     1.854
====== ====== ====== ===== ====== ===== ====== ========


CBC-AES128 encryption
---------------------

======= ======= ======= ===== ====== ===== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL   URAM   CP(ns)
======= ======= ======= ===== ====== ===== ====== ========
 2236    9197    6633     0     2     512    0     3.143
======= ======= ======= ===== ====== ===== ====== ========


CBC-AES128 decryption
---------------------

======= ======= ======= ===== ====== ===== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL   URAM   CP(ns)
======= ======= ======= ===== ====== ===== ====== ========
 5245    27690   10805    0     10    577    0     3.121
======= ======= ======= ===== ====== ===== ====== ========


CBC-AES192 encryption
---------------------

======= ======= ======= ===== ====== ===== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL   URAM   CP(ns)
======= ======= ======= ===== ====== ===== ====== ========
 2962    15130   8312     0     6     640    0     2.993
======= ======= ======= ===== ====== ===== ====== ========


CBC-AES192 decryption
---------------------

======= ======= ======= ===== ====== ===== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL   URAM   CP(ns)
======= ======= ======= ===== ====== ===== ====== ========
 6292    33382   12696    0     14    705    0     3.103
======= ======= ======= ===== ====== ===== ====== ========


CBC-AES256 encryption
---------------------

======= ======= ======= ===== ====== ===== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL   URAM   CP(ns)
======= ======= ======= ===== ====== ===== ====== ========
 3135    14865   9309     0     2     768    0     2.734
======= ======= ======= ===== ====== ===== ====== ========


CBC-AES256 decryption
---------------------

======= ======= ======= ===== ====== ===== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL   URAM   CP(ns)
======= ======= ======= ===== ====== ===== ====== ========
 7206    38329   14320    0     10    833    0     2.951
======= ======= ======= ===== ====== ===== ====== ========


