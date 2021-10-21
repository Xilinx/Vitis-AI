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

****************************
Digital Signature Algorithm
****************************

.. toctree::
   :maxdepth: 1

Digital Signature Algorithm (DSA) is a public-key cryptosystem. It's is used to generate and verify digital signature.
Details of DSA could be found in FIPS.186-4, section 4.

Implementation
==============

DSA have two pair of functions: updateSigningParam and sign, updateVerifyingParam and verify. Also we have two implementation for updateSigningParam and updateVerifyingParam, a trivial one and another one with one extra arguments. The extra argument is actually 2^(2*N) mod P. If you have this arguments pre-calculated, you could call the one with this argument. Otherwise you could call the one without this arguments and we will calculate it on chip with extra resource.

Optimized Implementation on FPGA
=================================

Like RSA, DSA also relies on modular exponential calculation. We adopt the same method and do the expensive modular exponential calculaiton on the Montgomery field and then convert it back to normal representation. In this way, we could eliminate most integer division and multiplication to save resource and have higher frequency.

Reference
========

Peter Montgomery. "Modular Multiplication Without Trial Division", Mathematics of Computation, vol. 44 no. 170, pp. 519–521, April 1985.

"Efficient architectures for implementing montgomery modular multiplication and RSA modular exponentiation" by Alan Daly, William Marnane

FIPS.186-4, section 4
