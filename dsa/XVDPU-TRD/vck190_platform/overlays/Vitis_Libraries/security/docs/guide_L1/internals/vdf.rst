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
   :keywords: Vitis, Security, Library, VDF
   :description: Verifiable Delay Function
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


*************************
Verifiable Delay Function
*************************

.. toctree::
   :maxdepth: 1

Overview
========

A Verifiable Delay Function (VDF) is a function whose evaluation requires running a given number of sequentialsteps, yet the result can be efficiently verified. Here implement two VDFs: Efficient verifiable delay functions (Wesolowski) and Simple Verifiable Delay Functions (Pietrzak). Its algorithm is defined in `REF Wesolowski`_ and `REF Pietrzak`_.

.. _`REF Wesolowski`: https://eprint.iacr.org/2018/623.pdf

.. _`REF Pietrzak`: https://eprint.iacr.org/2018/627.pdf


Implementation on FPGA
======================

There are 3 APIs provided that are `evaluate`, `verifyWesolowski`, and `verifyPietrzak`. The APIs are completely implemented according to the algorithm of the above reference paper, except that the multiplication of large-bit-width integers is implemented based on "Montgomery Production". For "Montgomery Production", please refer to `REF monProduct`_ for details.

.. _`REF monProduct`: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.99.1897&rep=rep1&type=pdf
