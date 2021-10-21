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
   :keywords: Vitis, Security, Library, Adler
   :description: The Adler32 is a checksum algorithm.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


******************
Adler32
******************

.. toctree::
   :maxdepth: 1

Overview
========

Adler32 is a checksum algorithm, An Adler-32 checksum is obtained by calculating two 16-bit checksums :math:`s1` and :math:`s2` and concatenating their bits into a 32-bit integer. :math:`s1` is the sum of all bytes in the stream plus one, and :math:`s2` is the sum of the individual values of :math:`s1` from each step. (from wiki)

The Adler32 algorithm is defined in `RFC 1950`_ and `Wiki Adler32`_.

.. _`RFC 8439`: https://tools.ietf.org/html/rfc1952

.. _`Wiki`: https://en.wikipedia.org/wiki/Adler-32

Implementation on FPGA
======================

For :math:`s1`, its calculation process is very simple and will not affect performance, so there is not explanation for implementation, please refer to the code directly for details.

.. math::
    
    s1=(1+B_{0}+B_{1}+\cdots +B_{n-1})%65521

For :math:`s2`, it can be expressed as

.. math::
    
    s2=((1+B_{0})+(1+B_{0}+B_{1})+\cdots +(1+B_{0}+B_{1}+\cdots +B_{n-1}))%65521

Where B is the input date for which the checksum is to be calculated, and n is its size in byte.

In code, the process is as follows:

1. Initialize :math:`s1=1` and :math:`s2=0`, set :math:`i=0`.
2. calcute :math:`tmp[0]=B[i],tmp[i]=B[i+1],\cdots,tmp[W-1]=B[i+W-1]`.
3. calcute :math:`s1+=tmp[0]` and :math:`s2=s1*W+tmp[0]+tmp[i]+\cdots +tmp[W-1]`, and ensure that :math:`s1` and :math:`s2` are less than 65521.
4. set :math:`i+=W`, if :math:`i<size`, go to step 2, otherwise end.

For more information, please check out source code.
