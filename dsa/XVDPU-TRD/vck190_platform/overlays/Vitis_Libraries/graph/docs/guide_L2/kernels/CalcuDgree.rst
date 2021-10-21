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
Internal Design of CalcuDgree
*************************************************


Overview
========
CalcuDgree is an algorithm used to calculate indegree for CSR format input or outdegree for CSC format input. The algorithm is quite easy, but due to DDR limits, we cannot get one data per clock, so in our design, we use a cache with depth ajustable to get much better performance.

Algorithm
============
CalcuDgree algorithm implementation:

.. code::

    for each edge (u, v) in graph   // calculate du degree
        degree(v) += 1


Implemention
============
The input matrix should ensure that the following conditions hold:

1. No duplicate edges
2. compressed sparse column/row (CSC/CSR) format

The algorithm implemention is shown as the figure below:

Figure 1 : calculate degree architecture on FPGA

.. _my-figure-CalcuDegree-1:
.. figure:: /images/PageRank/Pagerank_kernelcalDgree.png
      :alt: Figure 1 PageRank calculate degree architecture on FPGA
      :width: 80%
      :align: center

As we can see from the figure:

1. Module `calculate degree`: first get the vertex node's outdegree and keep them in one DDR buffer.

Profiling
=========

The hardware resource utilizations are listed in the following table.


Table 1 : Hardware resources for CalcuDegree with cache
  
.. table:: Table 1 Hardware resources for CalcuDegree with cache (depth 1)
    :align: center

    +-------------------+----------+----------+----------+----------+---------+-----------------+
    |    Kernel         |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   | Frequency(MHz)  |
    +-------------------+----------+----------+----------+----------+---------+-----------------+
    | kernel_pagerank_0 |   276    |     0    |     7    |  250477  |  171769 |      300        |
    +-------------------+----------+----------+----------+----------+---------+-----------------+

Figure 2 : Cache depth's influence to CalcDepth acceleration

.. _my-figure-CalcuDegree-2:
.. figure:: /images/CalDgree_cachedepth.png
      :alt: Figure 2 Cache depth's influence to CalcuDegree acceleration
      :width: 50%
      :align: center

.. note::
    | 1. depth 1, depth 32, depth 1k, they use LUTRAM only, in compare with resources of depth 1, only LUT and FF changes.
    | 2. depth 4k, depth 8k, depth 16k, depth 32k, they use URAM, in compare with resources of depth 1, the URAM utilization will be the major difference.
    | 3. HW Frequency: depth 1 (300MHz), depth 32 (300MHz), depth 1k (300MHz), depth 4k (300MHz), depth 8k (300MHz), depth 16k (300MHz), depth 32k (294.8MHz)

.. toctree::
   :maxdepth: 1

