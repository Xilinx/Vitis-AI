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
   :keywords: dynamic-filter
   :description: Describes the structure and execution of the dynamic filter module.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _guide-dynamic_filter:

*********************************
Internals of Dynamic-Filter
*********************************

.. toctree::
   :hidden:
   :maxdepth: 1

Internal Structure
==================

The following figure illustrates the internal of the dynamic filter.
On the left a group of range-checker compares each column data with
upper and lower bounds specified by two constants, and two operators
while on the right each pair of columns is assigned to a comparator.
The final condition is yield by looking into a true-table using
address consists of bits from these two parts.

.. image:: /images/dynamic_filter.png
   :alt: Dynamic Filter Structure
   :align: center

Limitations
===========

Currently, up to **four** condition columns of **integral types** are supported.
Wrappers for less input columns are provided, the configuration structure
**remains the same as the four-input version**.

.. CAUTION::
   Filter operator has signed or unsigned version, check ``enum FilterOp`` in ``enums.h``
   for details.

Generating Config Bits
======================

Currently, there is no expression-string to config bits compiler yet.
For generating the raw config bits, see the demo project in
``L1/demos/q6_mod/host/filter_test.cpp``.

The layout of the configuration bits is illustrated in the figure below.
As the figures shows, the intermediates are always aligned to 32bit boundaries,
and comes in *little-endian*. (In the figure, the intermediates are 48bit wide,
and thus occupies one and a half row.)

.. image:: /images/dynamic_filter_config.png
   :alt: Dynamic Filter Configuration Bits Layout
   :align: center


