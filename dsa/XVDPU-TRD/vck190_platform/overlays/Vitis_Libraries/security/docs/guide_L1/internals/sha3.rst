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
   :keywords: Vitis, Security, Library, SHA-3, Algorithm
   :description: SHA-3 (Secure Hash Algorithm 3) is a set of cryptographic hash functions defined in FIPS 202: SHA-3 Standard: Permutation-Based Hash and Extendable-Output Functions. 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

****************
SHA-3 Algorithms
****************

.. toctree::
   :maxdepth: 1

Overview
========

SHA-3 (Secure Hash Algorithm 3) is a set of cryptographic hash functions defined in
`FIPS 202`_: SHA-3 Standard: Permutation-Based Hash and Extendable-Output Functions.

The SHA-3 family consists of six hash functions with digests (hash values) that are
128, 224, 256, 384 or 512 bits: SHA3-224, SHA3-256, SHA3-384, SHA3-512, SHAKE128, SHAKE256.

Currently, this library supports all of the algorithms mentioned above.

* SHA3-224
* SHA3-256
* SHA3-384
* SHA3-512
* SHAKE-128
* SHAKE-256

.. _`FIPS 202`: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf

Implementation on FPGA
======================

The internal structure of SHA-3 algorithms can be shown as the figures below:

.. image:: /images/internal_structure_of_sha3.png
   :alt: Structure of SHA-3 algorithms
   :width: 100%
   :align: center

.. image:: /images/internal_structure_of_shakeXOF.png
   :alt: Structure of SHAKE algorithms
   :width: 100%
   :align: center


As we can see from the figures, hash calculation in both SHA-3 and SHAKE is much different from SHA-1 and SHA-2.
Since the internal state array is updated iteratively (by the input message) and used in the next permutation,
it cannot be partitioned into block generation part and digest part.

Both the digest parts of SHA-3 and SHAKE pad or split the input message into fixed sized blocks (1600-bit for each),
and XOR it to the state array of the last iteration.

The message word size is 64-bit for both SHA-3 and SHAKE, and each block has a different number of message words 
according to the specific suffix of the algorithm which is selected. The number can be defined as:

.. math::
    NumMsgWord = \frac{200 - \frac{Suffix}{4}}{8}

Loop-carried dependency is enforced by the algorithm, and thus the digest part cannot reach II=1.

Performance
===========

SHA3-224
--------

A single instance of SHA3-224 function processes input message at the rate of
``144 byte / 1105 cycles`` at 303.58MHz.

The hardware resource utilizations of SHA3-224 is listed in :numref:`tab1SHA3224` below:

.. _tab1SHA3224:

.. table:: Hardware resources for single SHA3-224 hash calculation
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |     0    |     0    |  36974   |  45821   |   7819    |    606    |      3.294      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+

SHA3-256
--------

A single instance of SHA3-256 function processes input message at the rate of
``136 byte / 1104 cycles`` at 306.65MHz.

The hardware resource utilizations of SHA3-256 is listed in :numref:`tab1SHA3256` below:

.. _tab1SHA3256:

.. table:: Hardware resources for single SHA3-256 hash calculation
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |     0    |     0    |  36787   |  44975   |   7203    |    606    |      3.261      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+

SHA3-384
--------

A single instance of SHA3-384 function processes input message at the rate of
``104 byte / 1100 cycles`` at 310.75MHz.

The hardware resource utilizations of SHA3-384 is listed in :numref:`tab1SHA3384` below:

.. _tab1SHA3384:

.. table:: Hardware resources for single SHA3-384 hash calculation
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |     0    |     0    |  33782   |  40511   |   7264    |    611    |      3.218      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+

SHA3-512
--------

A single instance of SHA3-512 function processes input message at the rate of
``72 byte / 1096 cycles`` at 316.25MHz.

The hardware resource utilizations of SHA3-512 is listed in :numref:`tab1SHA3512` below:

.. _tab1SHA3512:

.. table:: Hardware resources for single SHA3-512 hash calculation
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |     0    |     0    |  32994   |  39794   |   7264    |    611    |      3.162      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+

SHAKE-128
---------

A single instance of SHAKE-128 function processes input message at the rate of
``168 byte / 1108 cycles`` at 306.37MHz.

The hardware resource utilizations of SHAKE-128 is listed in :numref:`tab1SHAKE128` below:

.. _tab1SHAKE128:

.. table:: Hardware resources for single SHAKE-128 hash calculation
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |     0    |     0    |  37577   |  47898   |   8229    |    610    |      3.264      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+

SHAKE-256
---------

A single instance of SHAKE-256 function processes input message at the rate of
``136 byte / 1104 cycles`` at 302.02MHz.

The hardware resource utilizations of SHAKE-256 is listed in :numref:`tab1SHAKE256` below:

.. _tab1SHAKE256:

.. table:: Hardware resources for single SHAKE-256 hash calculation
    :align: center

    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |   BRAM   |    DSP   |    FF    |    LUT   |    CLB    |    SRL    | clock period(ns)|
    +----------+----------+----------+----------+-----------+-----------+-----------------+
    |     0    |     0    |  36789   |  44909   |   7889    |    606    |      3.311      |
    +----------+----------+----------+----------+-----------+-----------+-----------------+

Clustering
-----------

To boost the throughput of SHA-3 primitives, multiple instance can be organized into a cluster,
and offer message level parallelism.

