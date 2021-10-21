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
   :keywords: Vitis, Security, Library, SHA-2, Algorithm
   :description: SHA-2 (Secure Hash Algorithm 2) is a set of cryptographic hash functions defined in RFC 6234: US Secure Hash Algorithms (SHA and SHA-based HMAC and HKDF). 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

****************
SHA-2 Algorithms
****************

.. toctree::
   :maxdepth: 1

Overview
========

SHA-2 (Secure Hash Algorithm 2) is a set of cryptographic hash functions defined in
`RFC 6234`_: US Secure Hash Algorithms (SHA and SHA-based HMAC and HKDF).

The SHA-2 family consists of six hash functions with digests (hash values) that are
224, 256, 384 or 512 bits: SHA-224, SHA-256, SHA-384, SHA-512, SHA-512/224, SHA-512/256.

This library supports all of the algorithms mentioned above.

.. _`RFC 6234`: https://tools.ietf.org/html/rfc6234

Implementation on FPGA
======================

The internal structure of SHA-2 algorithms can be shown as the figure below:

.. image:: /images/internal_structure_of_sha384_sha512_sha512_t.png
   :alt: Structure of SHA-2 algorithms
   :width: 100%
   :align: center

As we can see from the figure, the SHA-2 hash calculation can be partitioned into two main parts.

* The pre-processing part pads or splits the input message into fixed sized blocks,
  and informs the down-stream parts that how many blocks do we have in this message.
  The message word size is 32-bit for SHA-224/SHA-256, 64-bit for the rest 4 algorithms,
  and each block has a size of 16 message words.
* The digest part iteratively computes the hash values. Loop-carried dependency
  is enforced by the algorithm, and thus this part cannot reach II=1.

As these two parts can work independently, they are designed into parallel dataflow process,
connected by streams (FIFOs).

The dup_strm module is used to duplicate the number of block stream,
and generateMsgSchedule module is responsible for generating the message word stream in sequence.

Performance
===========

SHA-224 and SHA-256
-------------------

As SHA-224 is simply truncated SHA-256 with different initialization values, and they share
the same internal structure, as illustrated in the figure above.

A single instance of SHA-256/SHA-224 function processes input message at the rate of
``512 bit / 68 cycles`` at 330.25MHz/314.36MHz respectively.

The hardware resource utilizations of SHA-224 is listed in :numref:`tab1SHA224` below:

.. _tab1SHA224:

.. table:: Hardware resources for single SHA-224 hash calculation
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |     0    |     0    |   7806   |   4976   |   1121    |     0     |      3.028      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+

The hardware resource utilizations of SHA-256 is listed in :numref:`tab1SHA256` below:

.. _tab1SHA256:

.. table:: Hardware resources for single SHA-256 hash calculation
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |     0    |     0    |   7806   |   4973   |   1176    |     0     |      3.181      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+

SHA-384, SHA-512, SHA-512/224, and SHA-512/256
----------------------------------------------

As SHA-384 and SHA-512/t is simply truncated SHA-512 with different initialization values, they share
the same internal structure, as illustrated in the figure above.

A single instance of one of SHA-384/SHA-512/SHA512-224/SHA512-256 processes input message at the rate of
``1024 bit / 84 cycles`` at 313.28MHz/323.31MHz/310.26MHz/313.57MHz.

The hardware resource utilizations of SHA-384 is listed in :numref:`tab1SHA384` below:

.. _tab1SHA384:

.. table:: Hardware resources for single SHA-384 hash calculation
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |     0    |     0    |  15494   |   8317   |   2045    |     0     |      3.192      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+

The hardware resource utilizations of SHA-512 is listed in :numref:`tab1SHA512` below:

.. _tab1SHA512:

.. table:: Hardware resources for single SHA-512 hash calculation
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |     0    |     0    |  15497   |   8318   |   2015    |     0     |      3.093      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+

The hardware resource utilizations of SHA-512/224 is listed in :numref:`tab1SHA512224` below:

.. _tab1SHA512224:

.. table:: Hardware resources for single SHA-512/224 hash calculation
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |     0    |     0    |  15498   |   8318   |   2101    |     0     |      3.223      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+

The hardware resource utilizations of SHA-512/256 is listed in :numref:`tab1SHA512256` below:

.. _tab1SHA512256:

.. table:: Hardware resources for single SHA-512/256 hash calculation
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |     0    |     0    |  15497   |   8322   |   2029    |     0     |      3.189      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+

Clustering
-----------

To boost the throughput of SHA-2 primitives, multiple instance can be organized into a cluster,
and offer message level parallelism.

