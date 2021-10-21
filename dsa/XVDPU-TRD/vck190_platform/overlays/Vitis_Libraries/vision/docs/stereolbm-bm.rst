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


.. _l1_stereolbm:

===========================
Stereo Local Block Matching
===========================

Stereo Local Block Matching example resides in ``L2/examples/stereolbm`` directory.

This benchmark tests the performance of `stereolbm` function with a pair of stereo images. Stereo block matching is a method to estimate the motion of the blocks between the consecutive frames, called stereo pair. The postulate behind this idea is that, considering a stereo pair, the foreground objects will have disparities higher than the background. Local block matching uses the information in the neighboring patch based on the window size, for identifying the conjugate point in its stereo pair. While, the techniques under global method, used the information from the whole image for computing the matching pixel, providing much better accuracy than local methods. But, the efficiency in the global methods are obtained with the cost of resources, which is where local methods stands out.

The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in README of L2 folder. For getting the design,

.. code-block:: bash

   cd L2/examples/stereolbm

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
   
   -----------StereoLBM Design---------------
   Platform Name: Xilinx
   INFO: Device found - xilinx_u200_xdma_201830_2
   XCLBIN File Name: krnl_stereolbm
   INFO: Importing vision/L2/examples/stereolbm/Xilinx_Stereolbm_L2_Test_vitis_hw_u200/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_stereolbm.xclbin
   Loading: 'vision/L2/examples/stereolbm/Xilinx_Stereolbm_L2_Test_vitis_hw_u200/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_stereolbm.xclbin'
       Minimum error in intensity = 0
       Maximum error in intensity = 0
       Percentage of pixels above error threshold = 0
   ------------------------------------------------------------

Profiling 
=========

The StereoLBM design is validated on Alveo U200 board at 300 MHz frequency. 
The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for Stereo Local Block Matching
    :align: center

    +------------------------------------------+-----------------+------------+------------+------------+
    |            Dataset                       |      LUT        |    BRAM    |     FF     |    DSP     |
    +------------+------+----------------------+                 |            |            |            |
    | Resolution | NPPC | other params         |                 |            |            |            |
    +============+======+======================+=================+============+============+============+
    |     4K     |   1  | PARALLEL Units - 32, |     19005       |     26     |    21113   |     7      |
    |            |      | Disparity - 32       |                 |            |            |            |
    +------------+------+----------------------+-----------------+------------+------------+------------+
    |     FHD    |   1  | PARALLEL Units - 32, |     19380       |     13     |    20670   |     7      |
    |            |      | Disparity - 32       |                 |            |            |            |
    +------------+------+----------------------+-----------------+------------+------------+------------+


The performance is shown below

.. table:: Table 2 Performance numbers in terms of FPS (Frames Per Second) for Stereo Local Block Matching
    :align: center
	
    +----------------------+--------------+--------------+
    |       Dataset        |   FPS(CPU)   |   FPS(FPGA)  |
    +======================+==============+==============+
    |     4k (3840x2160)   |     13       |    34        |
    +----------------------+--------------+--------------+
    |   Full HD(1920x1080) |     35       |    135       |
    +----------------------+--------------+--------------+


.. toctree::
    :maxdepth: 1
