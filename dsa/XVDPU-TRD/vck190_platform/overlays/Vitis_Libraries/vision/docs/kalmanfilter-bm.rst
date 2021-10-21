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


.. _l2_kalmanfilter:

=============
Kalman Filter
=============

Kalman Filter example resides in ``L2/examples/kalmanfilter`` directory.

This benchmark tests the performance of `kalmanfilter` function. The classic Kalman Filter is proposed for linear system.

The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in README file of L2 folder. For getting the design,

.. code-block:: bash

   cd L2/example/kalmanfilter

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
   
   -----------Kalman Design---------------
   INFO: Init cv::Mat objects.
   INFO: Kalman Filter Verification:
	Number of state variables: 16
	Number of measurements: 16
	Number of control input: 16
   INFO: Running OpenCL section.
   Found Platform
   Platform Name: Xilinx
   INFO: Device found - xilinx_u200_xdma_201830_2
   XCLBIN File Name: krnl_kalmanfilter
   INFO: Importing Vitis_Libraries/vision/L2/examples/kalmanfilter/Xilinx_Kalmanfilter_L2_Test_vitis_hw_u200/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_kalmanfilter.xclbin
   Loading: 'Vitis_Libraries/vision/L2/examples/kalmanfilter/Xilinx_Kalmanfilter_L2_Test_vitis_hw_u200/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_kalmanfilter.xclbin'
   INFO: Test Pass
   ------------------------------------------------------------

Profiling 
=========

The Kalman Filter design is validated on Alveo u200 board at 300 MHz frequency. 
The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for Kalman Filter
    :align: center

    +------------------------------------------+-----------------+------------+------------+------------+
    |            Dataset                       |      LUT        |    BRAM    |     FF     |    DSP     |
    +------------+------+----------------------+                 |            |            |            |
    | Resolution | NPPC |    other params      |                 |            |            |            |
    +============+======+======================+=================+============+============+============+
    |     NA     |   1  |   SV - 16x16x16      |     59342       |     98     |    90762   |     361    |
    +------------+------+----------------------+-----------------+------------+------------+------------+


The performance is shown below

.. table:: Table 2 Performance numbers for Kalman Filter
    :align: center
	
    +----------------------+-------------------+--------------+
    |       Dataset        |   Latency (CPU)   | Latency(FPGA)|
    +======================+===================+==============+
    |   SV - 16x16x16      |     6.75 ms       |     0.55 ms  |
    +----------------------+-------------------+--------------+

.. toctree::
    :maxdepth: 1
