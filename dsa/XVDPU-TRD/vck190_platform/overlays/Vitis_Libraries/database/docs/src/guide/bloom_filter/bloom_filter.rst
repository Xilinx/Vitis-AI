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
   :keywords: bloom-filter, bloomFilter, bv-update, bv-check
   :description: Describes the structure and execution of the bloom filter module.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


.. _guide-bloom_filter:

********************************************************
Internals of Bloom-Filter
********************************************************

.. toctree::
   :hidden:
   :maxdepth: 1

This document describes the structure and execution of bloom filter module,
implemented as :ref:`bloomFilter <cid-xf::database::bfGen>` function.

The structure of ``bloomFilter`` is described as below. The primitive have two function which are ``bv-update`` and ``bv-check``. 
It takes BRAM or URAM as internal storage for bloom filter vectors. 
The input hash value is the addressing parameter for updating bloom filter vector and checking the built vector. 
Chip select is based on the MSBs of hash value, and width select is based on its LSBs. The total storage size is related to the value of ``1 << BV_W``.
Make sure the storage size is less than the maximum storage size of a single SLR, otherwise it will result in placing failure.

.. image:: /images/bloom_filter.png
   :alt: Bloom Filter Top Structure
   :align: center

The primitive provides an efficient way to filter redundant data. It can be easily applied on Hash Join primitive to eliminate false shoots in Hash Join Probe, as shown below:

.. image:: /images/bloom_filter_performance.png
   :alt: Bloom Filter Performance
   :align: center


