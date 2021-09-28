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
   :keywords: Vitis, Security, Library, ECC, ECDSA, EdDSA, Secp256k1, Ed25519, Cryptography
   :description: ECC is a public-key cryptosystem. Its encryption key is public and different from decryption key. 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

****************************
Elliptic-curve Cryptography
****************************

.. toctree::
   :maxdepth: 1

ECC(Elliptic-curve Cryptography) is a public-key cryptosystem. Its encryption key is public and different from decryption key. ECC allows smaller keys compared to non-EC cryptography to provide equivalent security.
ECC cryptosystem includes key generation, key distribution, encryption/decryption and padding schemes. In this release we provide the encryption/decryption part.
Current implementation is a templated design which could adapt to various curve parameters.


*******************************************
Elliptic Curve Digital Signature Algorithm
*******************************************

.. toctree::
   :maxdepth: 1

Elliptic Curve Digital Signature Algorithm (ECDSA) is a variant of Digital Signature Algorithm which utilize elliptic curve cryptography. In this release, we provide support for curve Secp256k1.


******************************************
Edwards-curve Digital Signature Algorithm
******************************************

.. toctree::
   :maxdepth: 1

Edwards-curve Digital Signature Algorithm provide digital signature functionalities using a vriant of n twisted Edwards curves. In this release, we provide support for curve ed25519.

Reference
========

Peter Montgomery. "Modular Multiplication Without Trial Division", Mathematics of Computation, vol. 44 no. 170, pp. 519–521, April 1985
RFC 8032 "Edwards-Curve Digital Signature Algorithm (EdDSA)".

