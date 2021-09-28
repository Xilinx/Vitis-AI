.. 
   Copyright 2019-2020 Xilinx, Inc.
  
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
   :keywords: Vitis, Database, Vitis Database Library, release
   :description: Vitis Database library release notes.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _release_note:

Release Note
============

.. toctree::
   :hidden:
   :maxdepth: 1

2020.2
------

The 2020.2 release brings a major update to the GQE kernel design, and brand new L3 APIs for JOIN and GROUP-BY
AGGREGATE.

* The GQE kernels now take each column as an input buffer, which can greatly simplify the data preparation on the
  host-code side.
  Also, allocating multiple buffers on host side turns should cause less out-of-memory issues comparing to a big
  contiguous one, especially when the server is under heavy load.
* The L2 layer now provides command classes to generate the configuration bits for GQE kernels.
  Developers no longer have to dive into the bitmap table to understand which bit(s) to toggle to enable or disable a
  function in GQE pipeline. Thus the host code can be less error-prone and more sustainable.
* The all-new experimental L3 APIs are built with our experiments and insights into scaling the problem size that GQE
  can handle.
  They can breakdown the tables into parts based on hash, and call the GQE kernels multiple rounds in a well-schedule
  fashion.
  The strategy of execution is separated from execution, so database gurus can fine-tune the execution based on table
  statistics, without messing with the OpenCL execution part.


2020.1
------

The 2020.1 release contains:

* Compound sort API (compoundSort): Previously three sort algorithm modules have been provided,
  and this new API combines ``insertSort`` and ``mergeSort``, to provide a more scalable solution for on-chip sorting.
  When working with 32-bit integer keys, URAM resource on one SLR could support the design to scale to 2M entries.

* Better HBM bandwidth usage in hash-join (``hashJoinV3``): In 2019.2 Alveo U280 shell,
  ECC has been enabled. So sub-ECC size write to HBM becomes read-modify-write, and wastes some bandwidth.
  The ``hashJoinV3`` primitive in this release has been modified to use 256-bit port,
  to avoid this problem.

*  Various bug fixes: many small issues has been cleaned up, especially in host code of L2/demos.


2019.2
------

The 2019.2 release introduces GQE (generic query engine) kernels,
which are post-bitstream programmable
and allow different SQL queries to be accelerated with one xclbin.
It is conceptually a big step from per-query design,
and a sound example of Xilinx's acceleration approach.

Each GQE kernel is essentially a programmable pipeline of
execution step primitives, which can be enabled or bypassed via run-time configuration.

Internal Release
----------------

The first release provides a range of HLS primitives for mapping the execution
plan steps in relational database. They cover most of the occurrence in the
plan generated from TPC-H 22 queries.

These modules work in streaming fashion and can work in parallel
when possible.
