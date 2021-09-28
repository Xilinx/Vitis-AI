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

.. _release_note:

Release Note
============

.. toctree::
   :hidden:
   :maxdepth: 1

Codec Library is an open-sourced library written in C/C++ accelerating image processing including JPEG decoder and pik encoder. It now covers a level of acceleration: the module level(L1) and the pre-defined kernel level(L2).

2021.1
----

The 2021.1 release provides a range of algorithm, includes:

- JPEG Decoder: This API supports the 'Sequential DCT-based mode' of ISO/IEC 10918-1 standard. It is a high-performance implementation based-on Xilinx HLS design methodolygy. It can process 1 Huffman token and create up to 8 DCT coeffiects within one cycle. It is also an easy-to-use decoder as it can direct parser the JPEG file header without help of software functions.  
- Pik Encoder: This API is based on Google's PIK, which was 'chosen as the base framework for JPEG XL'. The pikEnc is based on the 'fast mode' of PIK which can provide better encoding efficnty than most of other still image encoding methods. The pikEnc is based on Xilinx HLS design methodology and optimized for FPGA arthitecture. It can proved higher throughput and lower latency compared to software-based solutions.  
