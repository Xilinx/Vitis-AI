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
Internal Design of Convert CSC CSR
*************************************************


Overview
========
ConvertCSCCSR is an algorithm used to transform CSC format input to CSR format input or CSR format input to CSC format input. The algorithm is quite easy, but due to DDR limits, we cannot get one data per clock, so in our design, we use several caches with depth ajustable to get much better performance.

Algorithm
============
ConvertCSCCSR algorithm implementation:

.. code::

    for each edge (u, v) in graph   // calculate du degree
        degree(v) += 1
        offset2(v) += 1
    begin = 0
    for node u in graph   
        end = offset1(u)
        for i in (begin, end)
            index = indice1(i)
            index2 = offset2[index]
            offset2[index] += 1
            indice2[index2] = u
        begin = end


Implemention
============
The input matrix should ensure that the following conditions hold:

1. No duplicate edges
2. compressed sparse column/row (CSC/CSR) format

The algorithm implemention is shown as the figure below:

Figure 1 : convert CSC CSR architecture on FPGA

.. _my-figure-ConvertCSCCSR-1:
.. figure:: /images/convertCsr2Csc.png
      :alt: Figure 1 Convert CSC CSR architecture on FPGA
      :width: 80%
      :align: center

As we can see from the figure:

1. firstly call the Module `calculate degree` to generate the transfer offset array.
2. by using the input offset and indice arrays and also the calculated new offset array, generate the new indice array

Profiling
=========

The hardware resource utilizations are listed in the following table.


Table 1 : Hardware resources for ConvertCSCCSR with cache
  
.. table:: Table 1 Hardware resources for ConvertCSCCSR with cache (depth 1)
    :align: center

    +-------------------+----------+----------+----------+----------+---------+-----------------+
    |    Kernel         |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   | Frequency(MHz)  |
    +-------------------+----------+----------+----------+----------+---------+-----------------+
    | kernel_pagerank_0 |    413   |     0    |     7    |  295330  |  207754 |      300        |
    +-------------------+----------+----------+----------+----------+---------+-----------------+

Figure 2 : Cache depth's influence to ConvertCSCCSR acceleration

.. _my-figure-ConvertCSCCSR-2:
.. figure:: /images/ConvertCsrCsc_AR.png
      :alt: Figure 2 Cache depth's influence to ConvertCSCCSR acceleration
      :width: 50%
      :align: center


.. note::
    | 1. depth 1, depth 32, depth 1k, they use LUTRAM only, in compare with resources of depth 1, only LUT and FF changes.
    | 2. depth 4k, depth 32k, they use URAM, in compare with resources of depth 1, the URAM utilization will be the major difference.
    | 3. HW Frequency: depth 1 (300MHz), depth 32 (300MHz), depth 1k (275.6MHz), depth 4k (300MHz), depth 32k (275.7MHz)

.. toctree::
   :maxdepth: 1

