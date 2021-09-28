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
   :keywords: Vitis, Security, Library, CCM, mode
   :description: The Counter with Cipher Block Chaining-Message Authentication Code (CCM) mode is a typical block cipher mode of operation using block cipher algorithm.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials



********
CCM Mode
********

.. toctree::
   :maxdepth: 1

Overview
========

The Counter with Cipher Block Chaining-Message Authentication Code (CCM) mode is a typical block cipher mode of operation using block cipher algorithm.
In this version, we provide Advanced Encryption Standard (AES) processing ability,
the cipherkey length for AES should be 128/192/256 bits.
Unlike other primitives in this library, this primitive takes an arbitrary length of text in bytes and produces the ciphers with the same length.
The MAC is generated simultaneously with the ciphers using the associated data (AD) in arbitrary length.
Thus, you must provide the length of payload text and AD before encrypting/decrypting the text.

Implementation on FPGA
======================

We support CCM mode including both encryption and decryption parts in this implementation.

.. ATTENTION::
    The bit-width of the interfaces we provide is shown as follows:

    +------------+-----------+------------+-----------+--------+-----+--------+--------+--------+-------+
    |            |  payload  |   cipher   | cipherkey | nonce  | AD  |   tag  | lenPld | lenCph | lenAD |
    +------------+-----------+------------+-----------+--------+-----+--------+--------+--------+-------+
    | CCM-AES128 |    128    |    128     |    128    | 56-104 | 128 | 32-128 |   64   |   64   |   64  |
    +------------+-----------+------------+-----------+--------+-----+--------+--------+--------+-------+
    | CCM-AES192 |    128    |    128     |    192    | 56-104 | 128 | 32-128 |   64   |   64   |   64  |
    +------------+-----------+------------+-----------+--------+-----+--------+--------+--------+-------+
    | CCM-AES256 |    128    |    128     |    256    | 56-104 | 128 | 32-128 |   64   |   64   |   64  |
    +------------+-----------+------------+-----------+--------+-----+--------+--------+--------+-------+

    The bit-width for the nonce and the tag is specified by the template parameters _t and _q.
    Please read the API's specification for further information.

The tag (MAC) is used to verify the received data is whether authentic or not.
To maintain the same interface for both input and output ports, we do not provide a bool flag to indicate the received data is authentic or not but a tag for you to verify it outside the primitive.

.. CAUTION::
    Applicable conditions:

    1. To verify the received data, you need to compare the tag which is the output of the decrypting process with the tag from the encrypting process.
    If they are equal, the received data is authentic.

The algorithm flow chart of encryption part of CCM mode is shown as follow:

.. image:: /images/CCM_encryption.png
   :alt: algorithm flow chart of CCM_encryption
   :width: 100%
   :align: center

As we can see from the chart, the CCM encryption part can be divided into two individual parts: The Counter Mode (CTR) and The Cipher Block Chaining-Message Authentication Code (CBC-MAC).
CTR is used to encrypt the plaintext to ciphertext, and CBC-MAC is used to generate the data tag (MAC). 

The algorithm flow chart of decryption part of CCM mode is shown as follow:

.. image:: /images/CCM_decryption.png
   :alt: algorithm flow chart of CCM_decryption
   :width: 100%
   :align: center

The decryption part is very similar with the encryption part of CCM mode.
The only difference is that we decrypt the ciphertext to plaintext in the decryption part.
In decryption part of CCM mode, we don't provide a bool flag to indicate whether the data is authentic or not.
You should compare the tag which the decryption part gives with the tag from CCM encryption part to judge the authenticity of the data.
If the data is authentic, then the tags should be equal.

The internal data flow of both encryption and decryption parts of CCM mode is shown as the figures below:

.. image:: /images/internal_structure_of_ccm_enc.png
   :alt: internal structure of CCM encryption
   :width: 100%
   :align: center

.. image:: /images/internal_structure_of_ccm_dec.png
   :alt: internal structure of CCM decryption
   :width: 100%
   :align: center

In our implementation, the CCM mode has four independent modules which are dupStrm, formatting, aesCtrEncrypt/aesCtrDecrypt, and CBC_MAC.
As the four modules can work independently, they are designed into parallel dataflow processes, and connected by streams (FIFOs).
Loop-carried dependency is enforced by the algorithm to the CBC-MAC, so its initiation internal (II) cannot achieve 1.
On the contrary, the input block for the single block cihper inside the mode can be directly calculated by the counter, it can achieve II = 1 for the CTR part.

Profiling
=========

CCM-AES128 encryption
---------------------

====== ======= ======= ===== ====== ====== ====== ========
 CLB     LUT     FF     DSP   BRAM    SRL   URAM   CP(ns)
====== ======= ======= ===== ====== ====== ====== ========
 5212   26036   17347    0     4     1090    0     2.932
====== ======= ======= ===== ====== ====== ====== ========


CCM-AES128 decryption
---------------------

====== ======= ======= ===== ====== ====== ====== ========
 CLB     LUT     FF     DSP   BRAM    SRL   URAM   CP(ns)
====== ======= ======= ===== ====== ====== ====== ========
 5143   26042   17345    0     4     1090    0     2.951
====== ======= ======= ===== ====== ====== ====== ========


CCM-AES192 encryption
---------------------

====== ======= ======= ===== ====== ====== ====== ========
 CLB     LUT     FF     DSP   BRAM    SRL   URAM   CP(ns)
====== ======= ======= ===== ====== ====== ====== ========
 6885   35165   20926    0     12    1346    0     2.874
====== ======= ======= ===== ====== ====== ====== ========


CCM-AES192 decryption
---------------------

====== ======= ======= ===== ====== ====== ====== ========
 CLB     LUT     FF     DSP   BRAM    SRL   URAM   CP(ns)
====== ======= ======= ===== ====== ====== ====== ========
 6768   35147   20926    0    12     1346    0     2.948
====== ======= ======= ===== ====== ====== ====== ========


CCM-AES256 encryption
---------------------

====== ======= ======= ===== ====== ====== ====== ========
 CLB     LUT     FF     DSP   BRAM    SRL   URAM   CP(ns)
====== ======= ======= ===== ====== ====== ====== ========
 7713   39245   23102    0     4     1602    0     3.140
====== ======= ======= ===== ====== ====== ====== ========


CCM-AES256 decryption
---------------------

====== ======= ======= ===== ====== ====== ====== ========
 CLB     LUT     FF     DSP   BRAM    SRL   URAM   CP(ns)
====== ======= ======= ===== ====== ====== ====== ========
 7532   39177   23101    0     4     1602    0     3.109
====== ======= ======= ===== ====== ====== ====== ========



