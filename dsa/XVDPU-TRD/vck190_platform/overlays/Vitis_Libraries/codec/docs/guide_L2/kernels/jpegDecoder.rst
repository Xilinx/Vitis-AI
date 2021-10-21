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


*************************************************
Internal Design of jpegDecoder
*************************************************


Overview
========
This API is decoder supports the 'Sequential DCT-based mode' of ISO/IEC 10918-1 standard. It is a high-performance implementation based-on Xilinx HLS design methodolygy. It can process 1 Huffman token and create up to 8 DCT coeffiects within one cycle. It is also an easy-to-use decoder as it can direct parser the JPEG file header without help of software functions.

As an independent IP, L1 API is the key circuit of L2 API.  
L2 API runs as a kernel demo, which can also show the overall performance of the circuit.

It can be seen from the benchmark of the API that the decoding speed of huffman decoder(L1 IP) is usually faster than that of iDCT(in L2 kernel). In practical applications, jpeg decoder is often used as the front module of the entire codec board.

Algorithm
============
JPEG Decoder algorithm implementation:

Figure 1 : jpegDecoder kernel work flow

.. _my-figure-jpegDec-1:
.. figure:: /images/jpegDec/jpegL2flow.png
      :alt: Figure 1 jpegDecoder kernel work flow on FPGA
      :width: 20%
      :align: center

The output stream will be recoverd to a .yuv file by the host code and be stored to the folder of the input jpeg.
 
Implemention
============
The input JPEG and output Features:

Table 1 : jpegDecoder Features

.. table:: Table 1 jpegDecoder Features
    :align: center

    +-------------------+-----------------------------------------------------------------------+
    |    jpegDecoder    |                               Status                                  |
    +===================+=======================================================================+
    |       Input       |  support JPEG that scaned by baseline sequential processing           |
    |                   |  8-bit precision                                                      |
    +-------------------+-----------------------------------------------------------------------+
    |      Output       |  YUV with the mcu scan order                                          |
    +-------------------+-----------------------------------------------------------------------+
    |   Output info     | Image width, height, scan format, quantization tables, number of mcu, |
    |                   | other details...   the reason for the decoding error if there is      |
    +-------------------+-----------------------------------------------------------------------+
    |    performance    |  decode one Huffman symbol in 1 cycle                                 |
    |                   |  Output YUV raw data 8 Byte per cycle with the mcu scan order         |
    +-------------------+-----------------------------------------------------------------------+

The algorithm implemention is shown as the figure below:

Figure 2 : jpegDecoder architecture on FPGA

.. _my-figure-jpegDec-2:
.. figure:: /images/jpegDec/jpegL2architecture.png
      :alt: Figure 2 jpegDecoder architecture on FPGA
      :width: 80%
      :align: center

As we can see from the figure:

The design uses the special statistical characteristics of jpeg compression, that is,  
in most cases, the (huffman length + value length) is less than 15,  
and each clock cycle can solve a huffman symbol.

Profiling
=========

The hardware resource utilizations are listed in the following table.
Different tool versions may result slightly different resource.

Table 2 : Hardware resources for kernelJpegDecoder

.. table:: Table 2 Hardware resources for kernelJpegDecoder in U50
    :align: center

    +-------------------+----------+----------+----------+----------+---------+-----------------+
    |    Kernel         |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   | Frequency(MHz)  |
    +-------------------+----------+----------+----------+----------+---------+-----------------+
    | kernelJpegDecoder |    28    |     0    |    39    |  23652   |  24591  |       243       |
    +-------------------+----------+----------+----------+----------+---------+-----------------+

.. toctree::
   :maxdepth: 1

