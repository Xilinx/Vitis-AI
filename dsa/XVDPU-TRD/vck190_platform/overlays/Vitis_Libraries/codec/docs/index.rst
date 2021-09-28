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


.. Project documentation master file, created by
   sphinx-quickstart on Tue Oct 30 18:39:21 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Vitis Codec Library
==========================

Vitis Codec Library is an open-sourced Vitis library written in C++ for accelerating image applications in a variety of use cases. It now covers a level of acceleration: the pre-defined kernel level (L2), and will evolve to offer the module level (L1).

Currently, this includes the following algorithm implementation:

- JPEG Decoder: "JPEG" stands for Joint Photographic Experts Group, the name of the committee that created the JPEG standard and also other still picture coding standards.
- PIK Encoder: PIK is the prototype of JPEG XL, which is a raster-graphics file format supporting both lossy and lossless compression. It is designed to outperform existing raster formats and thus to become their universal replacement. 

.. toctree::
   :caption: Library Overview
   :maxdepth: 1

   overview.rst
   release.rst

.. toctree::
   :caption: L1 User Guide
   :maxdepth: 3

   guide_L1/api.rst

.. toctree::
   :maxdepth: 2

   guide_L1/internals.rst

.. toctree::
   :caption: L2 User Guide
   :maxdepth: 3

   guide_L2/api.rst

.. toctree::
   :maxdepth: 2

   guide_L2/internals.rst

.. toctree::
   :caption: Benchmark 
   :maxdepth: 1 

   benchmark.rst

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
