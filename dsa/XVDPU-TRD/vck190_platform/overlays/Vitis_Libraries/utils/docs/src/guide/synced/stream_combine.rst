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

.. _guide-stream_combine:

*****************************************
Internals of streamCombine
*****************************************

.. toctree::
   :hidden:
   :maxdepth: 3

The :ref:`streamCombine <cid-xf::common::utils_hw::streamCombine>` function
is designed for packing multiple elements of same width into a vector.

This module offers two static configurations: using the data from LSB or MSB.
With LSB option, the element at LSB is obtained from input stream with 0 index,
while with MSB option, the element at MSB is set using input with 0 index.

As some storage structures in FPGA are bounded to fixed width or width of power
of two, paddings may be necessary sometimes in the combined vector.
These padding bits are added with zeros, as illustrated below:

.. image:: /images/stream_combine_lsb.png
   :alt: combination n streams to one from LSB Structure
   :width: 80%
   :align: center

.. image:: /images/stream_combine_msb.png
   :alt: combination n streams to one from MSB Structure
   :width: 80%
   :align: center

Internally, this module is implemented with a simple loop which iteration
interval (II) is equal to 1.
This means that in each cycle, a vector is yielded using a set of elements.

.. ATTENTION::
   This module expects the width of output stream to be no less than total of
   input streams. To perform collection from multiple streams, consider the
   :ref:`streamNToOne <guide-stream_n_to_one>` module.

