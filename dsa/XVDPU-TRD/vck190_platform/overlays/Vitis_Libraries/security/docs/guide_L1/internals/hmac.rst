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
   :keywords: Vitis, Security, Library, HMAC, mode
   :description: HMAC is a message authentication code (MAC) using a hash function. It combines with any cryptographic hash function, for example, md5, sha1, sha256.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


*****************************
HMAC Algorithms
*****************************

.. toctree::
   :maxdepth: 1

HMAC is a message authentication code (MAC) using a hash function. It combines with any cryptographic hash function, for example, md5, sha1, sha256.
Hash function is wrapped to a class as one template parameter in HMAC and the wrapper class only has a static function involving the hash function.
HMAC uses the wrapper's hash function directly inside. The design makes combination HMAC algorithm with hash function more flexible. 
In xf_security lib, the key width (keyW), message width (msgW), key and message length width (lenW), hash value width (hshW) and block size of each hash function are list as below.

Configuration
=================================

============ ============ ============= ============ ============ ==================
    name      keyW(bits)    msgW(bits)   lenW(bits)   hshW(bits)   blockSize(bytes)
------------ ------------ ------------- ------------ ------------ ------------------
    md4         32             32           64           128            64
------------ ------------ ------------- ------------ ------------ ------------------
    md5         32             32           64           128            64
------------ ------------ ------------- ------------ ------------ ------------------
    sha1        32             32           64           160            64
------------ ------------ ------------- ------------ ------------ ------------------
    sha224      32             32           64           224            64
------------ ------------ ------------- ------------ ------------ ------------------
    sha256      32             32           64           256            64
------------ ------------ ------------- ------------ ------------ ------------------
    sha384      64             64           128          384            128
------------ ------------ ------------- ------------ ------------ ------------------
    sha512      64             64           128          512            128
============ ============ ============= ============ ============ ==================



Implementation
=======================

HMAC consists of 3 parts: compute kipad and kopad, `mhsh=hash(kipad+msg)`, `hash(kopad+msh)`.
kipad and kopad are derived from the input key. When the length of key is greater than hash's block size, `K=hash(key)`. kipad is `K XOR kip` while kopad is `K XOR kop`, in which kip is a constant consisting of repeated bytes valued 0x36 block size times and kop is repeating 0x5c.

.. image:: /images/hmac_detail.png
   :alt: hmac
   :width: 100%
   :align: center



Performance (Device: U250)
=================================

============ ====== ======= ======= ===== ====== ===== ====== ========
    name      CLB     LUT     FF     DSP   BRAM   SRL   URAM   CP(ns)
------------ ------ ------- ------- ----- ------ ----- ------ --------
 hmac+md4     2510   11048   13928    0      1     0      0     3.167
------------ ------ ------- ------- ----- ------ ----- ------ --------
 hmac+md5     2460   11890   15646    0      0     0      0     2.992
------------ ------ ------- ------- ----- ------ ----- ------ --------
 hmac+sha1    3422   13750   27992    0      4     0      0     2.988
------------ ------ ------- ------- ----- ------ ----- ------ --------
 hmac+sha224  2861   11960   22434    0      1     0      0     3.049
------------ ------ ------- ------- ----- ------ ----- ------ --------
 hmac+sha256  2880   11835   22596    0      1     0      0     3.214
------------ ------ ------- ------- ----- ------ ----- ------ --------
 hmac+sha384  4655   19515   39299    0      2     0      0     3.245
------------ ------ ------- ------- ----- ------ ----- ------ --------
 hmac+sha512  4810   19695   39933    0      2     0      0     3.498
============ ====== ======= ======= ===== ====== ===== ====== ========
