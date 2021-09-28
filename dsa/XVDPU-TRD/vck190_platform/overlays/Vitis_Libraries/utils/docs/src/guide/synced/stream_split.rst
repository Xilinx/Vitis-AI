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

.. _guide-stream_split:

*****************************************
Internals of streamSplit
*****************************************

.. toctree::
   :hidden:
   :maxdepth: 3

The :ref:`streamSplit <cid-xf::common::utils_hw::streamSplit>` is designed
for splitting a wide stream into multiple narrow ones, as it is common to
combine several data elements of the same type as a vector and pass them
together in FPGA data paths.

This module offers two static configurations: using the data from LSB or MSB.
With LSB option, the element at LSB is sent to output stream with 0 index,
while with MSB option, the element at MSB is sent to output with 0 index.

As some storage structures in FPGA are bounded to fixed width or width of power
of two, paddings may be necessary sometimes in the combined vector.
These padding bits are discarded during splitting, as illustrated below:

.. image:: /images/stream_split_lsb.png
   :alt: one stream to n distribution on MSB Structure
   :width: 80%
   :align: center

.. image:: /images/stream_split_msb.png
   :alt: one stream to n distribution on MSB Structure
   :width: 80%
   :align: center

Internally, this module is implemented with a simple loop which iteration
interval (II) is equal to 1.
This means that in each cycle, a vector is split into a set of elements.

.. ATTENTION::
   This module expects the width of input stream to be no less than total of
   output streams. To perform distribution from a vectors of elements to
   multiple streams, use the
   :ref:`streamOneToN <guide-stream_one_to_n>` module.

