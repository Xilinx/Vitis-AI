.. 
   Copyright 2021 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


.. _l3_isppipeline:

=======================================
Image Sensor Processing (ISP) Pipeline
=======================================

ISP Pipeline example resides in ``L3/examples/isppipeline`` directory.

This benchmark tests the performance of `isppipeline` function. Image Sensor Processing (ISP) is a pipeline of functions that enhance the overall visual quality of the raw image from the sensor.

The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in README of L3 folder. For getting the design,

.. code-block:: bash

   cd L3/examples/isppipeline

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   export OPENCV_INCLUDE=< path-to-opencv-include-folder >
   export OPENCV_LIB=< path-to-opencv-lib-folder >
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:< path-to-opencv-lib-folder >
   export DEVICE=< path-to-platform-directory >/< platform >.xpfm
   make host xclbin TARGET=hw

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   make run TARGET=hw

* **Example output(Step 4)** 

.. code-block:: bash
   
   -----------ISP Pipeline Design--------------------------------------------------------------------------------
   Found Platform
   Platform Name: Xilinx
   XCLBIN File Name: krnl_ISPPipeline
   INFO: Importing vision/L3/examples/isppipeline/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_ISPPipeline.xclbin
   Loading: 'vision/L3/examples/isppipeline/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_ISPPipeline.xclbin'
   The maximum depth reached by any of the 8 hls::stream() instances in the design is 196608
   ---------------------------------------------------------------------------------------------------------------

Profiling 
=========

The ISP Pipeline design is validated on Alveo U200 board at 300 MHz frequency. 
The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for ISP Pipeline
    :align: center

    +------------------------------------------+-----------------+------------+------------+------------+
    |            Dataset                       |      LUT        |    BRAM    |     FF     |    DSP     |
    +------------+------+----------------------+                 |            |            |            |
    | Resolution | NPPC | other params         |                 |            |            |            |
    +============+======+======================+=================+============+============+============+
    |     4K     |   1  |      Filter - 3x3    |     18987       |     24     |    17713   |     91     |
    +------------+------+----------------------+-----------------+------------+------------+------------+
    |     FHD    |   1  |      Filter - 3x3    |     19254       |     19     |    17534   |     88     |
    +------------+------+----------------------+-----------------+------------+------------+------------+


The performance is shown below

.. table:: Table 2 Performance numbers in terms of FPS (Frames Per Second) for ISP Pipeline
    :align: center
	
    +----------------------+--------------+--------------+
    |       Dataset        |   FPS(CPU)   |   FPS(FPGA)  |
    +======================+==============+==============+
    |     4k (3840x2160)   |     0.11     |     135      |
    +----------------------+--------------+--------------+
    |   Full HD(1920x1080) |     0.44     |     520      |
    +----------------------+--------------+--------------+

.. toctree::
    :maxdepth: 1
