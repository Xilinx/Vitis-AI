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
Top K Sort
*************************************************


Overview
========
Top K Sort is a sorting algorithm which is used to calculate the maximum or the minimum K number of elements in the input stream. The algorithm is quite easy, and we can only get one data per clock due to the design requirements in L2 API. So in our design, we use a simple insert sort with ajustable maximum sorting number to get much better performance.

Algorithm
=========
Top K Sort algorithm implementation:

.. code::

    cnt = 0
    tmp[K] = {} //desending array
    for each pair(key, pld) in
        if(cnt < K)
            insert_sort(tmp[], pair)
        else
            if(pair.key > tmp[k].key)
                insert_sort(tmp[], pair)


Implemention
============
The input stream should ensure that it have same number of key and pld. The internal design is based on insert sort algorithm.

The algorithm implemention is shown as the figure below:

Figure 1 : Architecture of Top K Sort

.. _my-figure-topKSort:
.. figure:: /images/topKSort.PNG
      :alt: Figure 1 architecture of Top K Sort
      :width: 40%
      :align: center

Profiling
=========

The hardware resource utilizations are listed in the following table.


Table 1 : Hardware resources for Top K Sort with maximum sorting number 64
  
.. table:: Table 1 Hardware resources for Sort Top K (Maximum sortNUM = 64)
    :align: center

    +-------------------+----------+----------+----------+----------+---------+-----------------+
    |      Report       |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   | Frequency(MHz)  |
    +-------------------+----------+----------+----------+----------+---------+-----------------+
    |    top_k_sort     |     0    |     0    |     0    |   5438   |  14061  |      300        |
    +-------------------+----------+----------+----------+----------+---------+-----------------+

    
.. toctree::
   :maxdepth: 1

