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
   :keywords: Merge, sort, mergeSort
   :description: Describes the structure and execution of the Merge sort.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _guide-merge_sort:

********************************************************
Internals of Merge Sort
********************************************************

.. toctree::
   :hidden:
   :maxdepth: 2


Principle
~~~~~~~~~

This document describes the structure and execution of Merge Sort,
implemented as :ref:`mergeSort <cid-xf::database::mergeSort>` function.

The algorithm works in software as follows:

1.Divide the unsorted list into N sublists, each containing 1 element (a list of 1 element is considered sorted).

2.Repeatedly merge sublists to produce new sorted sublists until there is only 1 sublist remaining. This will be the sorted list.

For FPGA implementation, a hardware oriented design is realized in the Merge Sort primitive.

.. image:: /images/merge_sort_architecture.png
   :alt: Merge Sort Processing Structure
   :align: center

The Merge Sort primitive has an internal comparator to sort two input stream into one output stream.

Steps for descending (vise versa):

1.Read the 1st right value and the 1st left value.

2.Compare the two value and output the larger one.

3.If the output value in step2 is from the right stream and the right stream is not empty, then keep read value form the right stream. Otherwise, read from the left stream.

4.If both stream are empty, break the loop. Otherwise, return to step2.


Synthesis Results
~~~~~~~~~~~~~~~~~

.. image:: /images/merge_sort_synthesis_resource.png
   :alt: Merge Sort Synthesis
   :align: center

Implementation Results
~~~~~~~~~~~~~~~~~~~~~~

.. image:: /images/merge_sort_implementation_resource.png
   :alt: Merge Sort Implementation
   :align: center

.. IMPORTANT::
   The end flag of input stream should be initialized, otherwise it may cause deadlock in output stream.
   The input strteam of Merge Sort primitive should be pre-sorted stream.

.. CAUTION::
   If the two input stream are both empty, the function output will be also empty.

This ``mergeSort`` primitive has two port for key input, two port for payload input, one port for merged key output, one port for merged payload output and one boolean sign for indicating ascending sort or descending sort.

