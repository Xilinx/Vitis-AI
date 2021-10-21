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
Internal Design of PageRankMultiChannels
*************************************************


Overview
========
PageRank (PR) is an algorithm used by Google Search to rank web pages in their search engine results. PageRank is a way of measuring the importance of website pages. PageRank works by counting the number and quality of links to a page to determine a rough estimate of how important the website is. The underlying assumption is that more important websites are likely to receive more links from other websites. Currently, PageRank is not the only algorithm used by Google to order search results, but it is the first algorithm that was used by the companies, and it is the best known.

Algorithm
============
PageRank weighted algorithm implementation:

.. math::
    PR(A) = \alpha + (1 - \alpha) (\frac{PR(B)}{Out(B)}+\frac{PR(C)}{Out(C)}+\frac{PR(D)}{Out(D)}+...)

:math:`A, B, C ,D...` are different vertex. :math:`PR` is the pagerank value of vertex, :math:`Out` represents the out degree of vertex and :math:`\alpha` is the damping factor, normally equals to 0.85

The algorithm's pseudocode is as follows

.. code::

    for each edge (u, v) in graph   // calculate du degree
        degree(v) += weight(v)

    for each node u in graph    // initiate DDR
        const(u) := (1- alpha) / degree(u)
        PR_old(u) := 1.0 

    while norm(PR_old - PR_new) > tolerance   // iterative add 
        for each vertex u in graph
            PR_new(u) := alpha
            for each vertex v point to u
                PR_new(u) += const(v)*PR_old(v)*weight(v)

    sum := 0
    for each node u in graph  // calculate sum of vector PR
        sum += PR_new(u)

    for each node u in graph  // normalization of order 1
        PR_new(u) := PR_new(u) / sum

    return PR_new


Implemention
============
The input matrix should ensure that the following conditions hold:

1. Directed graph
2. No self edges
3. No duplicate edges
4. Compressed sparse column (CSC) format
5. Max 64M Vertex with 128M Edge graph for this design, still board-level scalability.

In order to make the API use higher bandwidth on the board of HBM base, this optimized version for HBM is implemented
The algorithm implemention is shown as the figure below:

Figure 1 : PageRank calculate degree architecture on FPGA

.. _my-figure-PageRank-1:
.. figure:: /images/PageRankMultiChannels/Pagerank_kernelcalDgree_opt.png
      :alt: Figure 1 PageRankMultiChannels calculate degree architecture on FPGA
      :width: 80%
      :align: center


Figure 2 : PageRank initiation module architecture on FPGA

.. _my-figure-PageRank-2:
.. figure:: /images/PageRankMultiChannels/Pagerank_kernelInit_opt.png
      :alt: Figure 2 PageRankMultiChannels initiation module architecture on FPGA
      :width: 80%
      :align: center


Figure 3 : PageRank Adder architecture on FPGA

.. _my-figure-PageRank-3:
.. figure:: /images/PageRankMultiChannels/Pagerank_kernelAdder_opt.png
      :alt: Figure 3 PageRankMultiChannels Adder architecture on FPGA
      :width: 80%
      :align: center


Figure 4 : PageRank calConvergence architecture on FPGA

.. _my-figure-PageRank-4:
.. figure:: /images/PageRankMultiChannels/Pagerank_kernelcalConvergence_opt.png
      :alt: Figure 4 PageRankMultiChannels calConvrgence architecture on FPGA
      :width: 80%
      :align: center

As we can see from the figure:

1. Module `calculate degree`: first get the vertex node's outdegree with weighted and keep them in one DDR buffer.
2. Module `initiation`: initiate PR DDR buffers and constant value buffer.
3. Module `Adder`: calculate Sparse matrix multiplification.
4. Module `calConvergence`: calculate convergence of pagerank iteration.

Profiling
=========

The hardware resource utilizations are listed in the following table.

Table 1 : Hardware resources for PageRankMultiChannels with 2 channels

.. table:: Table 1 Hardware resources for PageRankMultiChannels with 2 working channels
    :align: center

    +-------------------+----------+----------+----------+----------+---------+-----------------+
    |    Kernel         |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   | Frequency(MHz)  |
    +-------------------+----------+----------+----------+----------+---------+-----------------+
    | kernel_pagerank_0 |   425    |   224    |    84    |  352693  |  245636 |      243        |
    +-------------------+----------+----------+----------+----------+---------+-----------------+

Table 2 : Comparison between CPU tigergraph and FPGA VITIS_GRAPH

.. _my-figure-PageRank-5:
.. figure:: /images/PageRankMultiChannels/Performance.png
      :alt: Table 2 : Comparison between CPU tigergraph and FPGA VITIS_GRAPH
      :width: 80%
      :align: center

.. note::
    | 1. Tigergraph time is the execution time of funciton "pageRank" Developer Edition 2.4.1 .
    | 2. Tigergraph running on platform with Intel(R) Xeon(R) CPU E5-2640 v3 @2.600GHz, 32 Threads (16 Core(s)).
    | 3. time unit: second.
    | 4. "-" Indicates that the result could not be obtained due to insufficient memory.
    | 5. FPGA time is the kernel runtime by adding data transfer and executed with pagerank_cache

.. toctree::
   :maxdepth: 1

