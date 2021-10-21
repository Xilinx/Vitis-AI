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
   :keywords: Vitis, Security, Library, GCM, mode
   :description: The Electronic Codebook (ECB) mode is a typical block cipher mode of operation using block cipher algorithm. 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

********
GCM Mode
********

.. toctree::
   :maxdepth: 1

Overview
========

The Galois/Counter Mode (GCM) is a typical block cipher modes of operation using block cipher algorithm.
In this version, we provide Advanced Encryption Standard (AES) processing ability,
the cipherkey length for AES should be 128/192/256 bits.
Our implementation takes a fix-sized (128 bits per block) payload and additional authenticated data (AAD) streams,
but text may have a variety of length in different situation.
Thus, you need to provide the length in bits accompany with the data or the AAD.
Meanwhile, the ciphers which is the output of the primitive also comply to this pattern.

Implementation on FPGA
======================

We support GCM mode including both encryption and decryption parts in this implementation.

.. ATTENTION::
    The bit-width of the interfaces we provide is shown as follows:

    +-----------+-----------+------------+----------+-----------+----+-----------+--------+--------+----+
    |           |  payload  |   lenPld   |   AAD    |   lenAAD  | IV | cipherkey | cipher | lenCph | tag|
    +-----------+-----------+------------+----------+-----------+----+-----------+--------+--------+----+
    |GCM-AES128 |    128    |     64     |   128    |     64    | 96 |    128    |  128   |   64   | 128|
    +-----------+-----------+------------+----------+-----------+----+-----------+--------+--------+----+
    |GCM-AES192 |    128    |     64     |   128    |     64    | 96 |    192    |  128   |   64   | 128|
    +-----------+-----------+------------+----------+-----------+----+-----------+--------+--------+----+
    |GCM-AES256 |    128    |     64     |   128    |     64    | 96 |    256    |  128   |   64   | 128|
    +-----------+-----------+------------+----------+-----------+----+-----------+--------+--------+----+


.. CAUTION::
    Applicable conditions:

    1. The bit-width of initialization vector must be precisely 96 as recommended in the standard
    to promote interoperablility, efficiency, and simplicity of the design.

    2. We provide the MAC value instead of a FAIL flag in decryption part, therefore, you should take care the MAC
    which is given by the encryption part to judge the authenticity of the data.
    If the data is authentic, then the MACs should be equal.


The algorithm flow chart of encryption part of GCM mode is shown as follow:

.. image:: /images/GCM_encryption.png
   :alt: algorithm flow chart of GCM_encryption
   :width: 100%
   :align: center

As we can see from the chart, the GCM encryption part can be divided into two individual parts: The Counter Mode (CTR) and The Galois Message Authentication Code (GMAC).
GCM is used to encrypt the plaintext to ciphertext, and GMAC is used to generate the MAC.
The algorithm flow chart of decryption part of GCM mode is shown as follow:

.. image:: /images/GCM_decryption.png
   :alt: algorithm flow chart of GCM_decryption
   :width: 100%
   :align: center

The decryption part is very similar with the encryption part of GCM mode.
The only difference is that we decrypt the ciphertext to plaintext in the decryption part.

The internal structure of both encryption and decryption parts of GCM are shown as the figures below:

.. image:: /images/internal_structure_of_gcm_enc.png
   :alt: internal structure of GCM
   :width: 100%
   :align: center

.. image:: /images/internal_structure_of_gcm_dec.png
   :alt: internal structure of GCM
   :width: 100%
   :align: center

In our implementation, the encryption part of GCM mode has two modules, which are aesGctrEncrypt and genGMAC.
Please be noticed that we use the first two iterations in aesGctrEncrypt to calculate the hash subkey (H) and E(K,Y0),
and pass them onto the genGMAC module through streams in dataflow region instead of implementing the AES block cipher in genGMAC to save the resources.
As the two modules can work independently, they are designed into parallel dataflow processes, and connected by streams (FIFOs).
The decryption part can be deduced in the same way, the only difference is that the ciphertext and its length streams, which are feed to genGMAC, are directly taken from the input ports.

Profiling
=========

GCM-AES128 encryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 3836    17765   16537    0     2     516     0     3.165
======= ======= ======= ===== ====== ====== ====== ========


GCM-AES128 decryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 3711    17899   16536    0     2     516     0     2.885
======= ======= ======= ===== ====== ====== ====== ========


GCM-AES192 encryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 4324    21732   18221    0     6     644     0     2.854
======= ======= ======= ===== ====== ====== ====== ========


GCM-AES192 decryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 4504    21814   18219    0     6     644     0     2.899
======= ======= ======= ===== ====== ====== ====== ========


GCM-AES256 encryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 4777    23389   19213    0     2     772     0     2.955
======= ======= ======= ===== ====== ====== ====== ========


GCM-AES256 decryption
---------------------

======= ======= ======= ===== ====== ====== ====== ========
  CLB     LUT     FF     DSP   BRAM   SRL    URAM   CP(ns)
======= ======= ======= ===== ====== ====== ====== ========
 5130    25891   19212    0     2     772     0     3.077
======= ======= ======= ===== ====== ====== ====== ========



