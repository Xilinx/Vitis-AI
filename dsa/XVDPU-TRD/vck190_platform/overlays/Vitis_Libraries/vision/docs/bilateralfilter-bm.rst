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


.. _l2_bilateralfilter:

================
Bilateral Filter
================

Bilateral Filter example resides in ``L2/examples/bilateralfilter`` directory.

This benchmark tests the performance of `bilateralfilter` function. To preserve the edges while smoothing, a bilateral filter can be used. In an analogous way as the Gaussian filter, the bilateral filter also considers the neighboring pixels with weights assigned to each of them. 

The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in README of L2 folder. For getting the design,

.. code-block:: bash

   cd L2/examples/bilateralfilter

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
   
   -----------bilateral filter Design---------------
   INFO: Running OpenCL section.
   Found Platform
   Platform Name: Xilinx
   INFO: Device found - xilinx_u200_xdma_201830_2
   XCLBIN File Name: krnl_bilateralfilter
   INFO: Importing Vitis_Libraries/vision/L2/examples/bilateralfilter/Xilinx_Bilateralfilter_L2_Test_vitis_hw_u200/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_bilateralfilter.xclbin
   Loading: 'Vitis_Libraries/vision/L2/examples/bilateralfilter/Xilinx_Bilateralfilter_L2_Test_vitis_hw_u200/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_bilateralfilter.xclbin'
	Minimum error in intensity = 0
	Maximum error in intensity = 0
	Percentage of pixels above error threshold = 0
   ------------------------------------------------------------
			
Profiling 
=========

The Bilateral Filter design is validated on Alveo u200 board at 300 MHz frequency. 
The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for Bilateral Filter
    :align: center

    +------------------+----------------------+--------------+-----------+----------+--------+
    |    Name          |   Dataset            |     LUT      |    BRAM   |   FF     |   DSP  |
    +==================+======================+==============+===========+==========+========+
    | bilateralfilter  | Full HD(1920x1080)   |   12513      |    18     |  13451   |   44   | 
    |                  +----------------------+--------------+-----------+----------+--------+
    |                  |    4k (3840x2160)    |   22523      |    35     |  22862   |   65   | 
    +------------------+----------------------+--------------+-----------+----------+--------+
  
The performance is shown below

.. table:: Table 2 Performance numbers for Bilateral Filter
    :align: center
	
    +----------------------+--------------+--------------+
    |   Dataset            |   FPS(CPU)   |   FPS(FPGA)  |
    +======================+==============+==============+
    | Full HD(1920x1080)   |     200      |     1100     |
    +----------------------+--------------+--------------+
    |    4k (3840x2160)    |     72       |     289      |
    +----------------------+--------------+--------------+
  
  
.. toctree::
    :maxdepth: 1
