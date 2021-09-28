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
   :keywords: Vitis, Security, Library, RSA, Cryptography
   :description: RSA is a public-key cryptosystem. Its encryption key is public and different from decryption key. 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

****************
RSA Cryptography
****************

.. toctree::
   :maxdepth: 1

RSA is a public-key cryptosystem. Its encryption key is public and different from decryption key. 
RSA cryptosystem includes key generation, key distribution, encryption/decryption and padding schemes. In this release we provide the encryption/decryption part.

Implementation
==============

RSA key pair has 3 part: modulus :math:`n` , encryption key exponent :math:`e`, decryption key exponent :math:`d`. 
Plain text is an unsigned integer :math:`m`, not larger than modulus and Cipher text is also an unsigned integer :math:`c`.

In Encryption we have:

.. math::
   c = m^{e} \mod{n}

In Decryption we have:

.. math::
   m = c^{d} \mod{n}

Optimized Implementation on FPGA
=================================

We seperate encryption into two part: updateKey and process. Each time we got an message with a new RSA key, we have to call updateKey before get into encryption/decryption. If we process messages with the same key continuously, updateKey only need be called once at the beginning. 

It should be notice that we provide actually two implementation of function keydateKey. One have two inputs: modulus and exponent. The other one has three inputs: modulus, exponent and rMod. The extract argument "rMod" is actually 2^(2*N) mod modulus. This is a key parameters in the encryption/decryption calculation. If you has pre-calculated this arguments, you could call the second updateKey and directly set it up. If you don't have it, you could call the first one and we will do the calculation on chip, with extract resource.

RSA encryption and decryption are basically the same calculation: big integer modulus exponential calculation.
Instead of doing straight calculation, we convert the big integer into its Montgomery field and do exponential calculation. Finally we convert the result back to normal representation. In such we, we could avoid most integer division and multiplication to save resource and have higher frequency.

Reference
========

Peter Montgomery. "Modular Multiplication Without Trial Division", Mathematics of Computation, vol. 44 no. 170, pp. 519–521, April 1985.

"Efficient architectures for implementing montgomery modular multiplication and RSA modular exponentiation" by Alan Daly, William Marnane
