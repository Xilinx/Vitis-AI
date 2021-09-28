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
   :keywords: Bitonic, sort, bitonicSort
   :description: Describes the structure and execution of the Bitonic sort.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _guide-bitonic_sort:

********************************************************
Internals of Bitonic Sort
********************************************************

.. toctree::
   :hidden:
   :maxdepth: 2


This document describes the structure and execution of Bitonic Sort,
implemented as :ref:`bitonicSort <cid-xf::database::bitonicSort>` function.
Bitonic sort is a special kind of sorting network, where the sequence of comparisons is not data-dependent. 
This makes sorting networks suitable for implementation in hardware or in parallel processor arrays. The computing complexity of bitonic sort is O(n*log(n)2).


.. image:: /images/bitonic_sort_architecture.png
   :alt: Bitonic Sort Processing Structure
   :align: center


Bitonic sort have giant data throughput and it demands large resource same time. It is well-fitted for the application with high band of data input.
The table shows the resource consumption for an instance of bitonic sort with input bitwidth=32.


                        +-------------------+----------+-----------+-----------+-----------+
                        | BitonicSortNumber | 8        | 16        | 32        | 64        |
                        +-------------------+----------+-----------+-----------+-----------+
                        | Lantency          | 22       | 42        | 79        | 149       |
                        +-------------------+----------+-----------+-----------+-----------+
                        | Interval          | 9        | 17        | 33        | 65        |
                        +-------------------+----------+-----------+-----------+-----------+
                        | LUT               | 2647     | 7912      | 21584     | 58064     |
                        +-------------------+----------+-----------+-----------+-----------+
                        | Register          | 3136     | 9291      | 26011     | 69160     |
                        +-------------------+----------+-----------+-----------+-----------+

If the bitonic sort number grow twice, the resource consumption of bitonic sort will grow around four times theoretically.


.. image:: /images/bitonic_sort_resource_consumption.png
   :alt: Bitonic Sort Resource Consumption in FPGA
   :align: center


.. IMPORTANT::
   The current version of bitonic sort is stream in and stream out.
   The bitonic sort number must be a power of two because of the algorithm restriction. Combine it with Merge Sort primitive can achieve arbitrary sort number, see reference :ref:`guide-merge_sort`.


.. CAUTION::
   The size of bitonic sort number should be set with the consideration of FPGA resource to pass place and route.


This ``bitonicSort`` primitive has one port for key input, one port for payload input, one port for key output, one port for payload output and one boolean sign for indicating ascending sort or descending sort.

