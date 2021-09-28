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

.. _guide-stream_sync:

*****************************************
Internals of streamSync
*****************************************

.. toctree::
   :hidden:
   :maxdepth: 3

This document describes the structure and execution of streamSync,
implemented as :ref:`streamSync <cid-xf::common::utils_hw::streamSync>` function.

.. image:: /images/stream_sync.png
   :alt: stream sync Structure
   :width: 80%
   :align: center

The streamSync synchronizes all streams by sharing a same end-flag stream. Each input data stream has an end-flag stream. After sync, all output streams share one end-flag stream, and each of them is a duplicate of an input stream. That is to say, before output an end-flag, each ouput stream should output a data from corresponding input stream.

.. CAUTION::
  Applicable conditions.
  1. It assumes the input elements in each input stream have the same number.
  2. The data type of input stream is same as the one of output.

