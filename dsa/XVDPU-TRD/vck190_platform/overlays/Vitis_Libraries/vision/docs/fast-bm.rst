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


.. _l2_fast:

======================
FAST Corner Detection
======================

FAST Corner Detection example resides in ``L2/examples/fast`` directory.

This benchmark tests the performance of `fast` function. Features from accelerated segment test (FAST) is a corner detection algorithm, that is faster than most of the other feature detectors.

The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
=================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in README of L2 folder. For getting the design,

.. code-block:: bash

   cd L2/examples/fast

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
   
   -----------FAST Design---------------
   Found Platform
   Platform Name: Xilinx
   INFO: Device found - xilinx_u200_xdma_201830_2
   XCLBIN File Name: krnl_fast
   INFO: Importing vision/L2/examples/fast/Xilinx_Fast_L2_Test_vitis_hw_u200/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_fast.xclbin
   Loading: 'vision/L2/examples/fast/Xilinx_Fast_L2_Test_vitis_hw_u200/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_fast.xclbin'
   ocvpoints:511=
   INFO: Verification results:
       Common = 511
       Success = 100
       Loss = 0
       Gain = 0
   Test Passed
   ------------------------------------------------------------

Profiling 
=========

The fast corner detection design is validated on Alveo U200 board at 300 MHz frequency. 
The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for FAST corner detection
    :align: center

    +------------------------------------------+-----------------+------------+------------+------------+
    |            Dataset                       |      LUT        |    BRAM    |     FF     |    DSP     |
    +------------+------+----------------------+                 |            |            |            |
    | Resolution | NPPC | other params         |                 |            |            |            |
    +============+======+======================+=================+============+============+============+
    |     4K     |   8  |      NA              |     21171       |     10     |    13396   |     0      |
    +------------+------+----------------------+-----------------+------------+------------+------------+
    |     FHD    |   8  |      NA              |     20437       |     10     |    14322   |     0      |
    +------------+------+----------------------+-----------------+------------+------------+------------+


The performance is shown below

.. table:: Table 2 Performance numbers in terms of FPS (Frames Per Second) for FAST corner detection
    :align: center
	
    +----------------------+--------------+--------------+
    |       Dataset        |   FPS(CPU)   |   FPS(FPGA)  |
    +======================+==============+==============+
    |     4k (3840x2160)   |     79       |     289      |
    +----------------------+--------------+--------------+
    |   Full HD(1920x1080) |     186      |     1100     |
    +----------------------+--------------+--------------+



.. toctree::
    :maxdepth: 1
