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


.. _l2_harris:

========================
Harris Corner Detection
========================

harris example resides in ``L2/examples/harris`` directory.

This benchmark tests the performance of `harris` function. The harris function detects corners in the image using harris corner detection and Non-maximum suppression algorithms.

The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in README file of L2 folder. For getting the design,

.. code-block:: bash

   cd L2/example/harris

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
   
   -----------Harris Design---------------
   INFO: Running OpenCL section.
   Found Platform
   Platform Name: Xilinx
   XCLBIN File Name: krnl_harris
   INFO: Importing Vitis_Libraries/vision/L2/examples/harris/Xilinx_Harris_L2_Test_vitis_hw_u200/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_harris.xclbin
   Loading: 'Vitis_Libraries/vision/L2/examples/harris/Xilinx_Harris_L2_Test_vitis_hw_u200/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_harris.xclbin'
   Kernel Created
   Kernel Args set
   Kernel called
 
   Data copied from device to host
   Execution done!
   ocv corner count = 428, Hls corner count = 446
   Commmon = 405	 Success = 90.807175	 Loss = 5.373832	 Gain = 9.192825
   Test Passed 

   ------------------------------------------------------------

Profiling 
=========

The harris design is validated on Alveo u200 board at 300 MHz frequency. 
The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for Harris corner detection
    :align: center

    +------------------------------------------+-----------------+------------+------------+------------+
    |            Dataset                       |      LUT        |    BRAM    |     FF     |    DSP     |
    +------------+------+----------------------+                 |            |            |            |
    | Resolution | NPPC | other params         |                 |            |            |            |
    +============+======+======================+=================+============+============+============+
    |     4K     |   8  | L1 Norm,             |     40460       |    227     |    27236   |     208    |
    |            |      | Filter - 3x3         |                 |            |            |            |
    +------------+------+----------------------+-----------------+------------+------------+------------+
    |     FHD    |   8  | L1 Norm,             |     22478       |    113     |    16021   |     138    |
    |            |      | Filter - 3x3         |                 |            |            |            |
    +------------+------+----------------------+-----------------+------------+------------+------------+


The performance is shown below

.. table:: Table 2 Performance numbers in terms of FPS (Frames Per Second) for Harris corner detection
    :align: center
	
    +----------------------+--------------+--------------+
    |       Dataset        |   FPS(CPU)   |   FPS(FPGA)  |
    +======================+==============+==============+
    |     4k (3840x2160)   |     22       |     289      |
    +----------------------+--------------+--------------+
    |   Full HD(1920x1080) |     62       |     1100     |
    +----------------------+--------------+--------------+


.. toctree::
    :maxdepth: 1
