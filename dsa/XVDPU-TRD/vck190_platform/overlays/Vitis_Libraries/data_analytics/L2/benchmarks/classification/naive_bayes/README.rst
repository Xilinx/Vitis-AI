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

.. _l2_naive_baye:

===========
Naive Bayes
===========

Naive Bayes resides in ``L2/benchmarks/classification/naive_bayes`` directory.


Dataset
=======

There are 3 dataset used in the benchmark:

 1 - RCV1 (https://scikit-learn.org/0.18/datasets/rcv1.html)

 2 - webspam (https://chato.cl/webspam/datasets/uk2007/)

 3 - news20 (https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)

+---------+---------+---------+----------+
| Dataset | samples | classes | features |
+=========+=========+=========+==========+
| RCV1    | 697614  |   2     |  47236   |
+---------+---------+---------+----------+
| webspam | 350000  |   2     |  254     |
+---------+---------+---------+----------+
| news20  | 19928   |   20    |  62061   |
+---------+---------+---------+----------+


Executable Usage
===============

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_data_analytics`. For getting the design,

.. code-block:: bash

   cd L2/benchmarks/classification/naive_bayes

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   make run TARGET=hw DEVICE=xilinx_u200_xdma_201830_2 HOST_ARCH=x86

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   ./build_dir.hw.xilinx_u200_xdma_201830_2/test_nb.exe -xclbin build_dir.hw.xilinx_u200_xdma_201830_2/naiveBayesTrain_kernel.xclbin ./data/test.dat -g ./data/test_g.dat -c 10 -t 13107

Naive Bayes Input Arguments:

.. code-block:: bash

   Usage: test_nb.exe -xclbin <xclbin_name> -in <input_data> -g <golden_data> -c <number of class> -t <number of feature>
          -xclbin:      the kernel name
          -in    :      input data
          -g     :      golden data
          -c     :      number of class
          -t     :      number of feature

Note: Default arguments are set in Makefile, you can use other platforms to build and run.

* **Example output(Step 4)** 

.. code-block:: bash

    ---------------------Multinomial Training Test of Naive Bayes-----------------
    Found Platform
    Platform Name: Xilinx
    Found Device=xilinx_u200_xdma_201830_2
    INFO: Importing build_dir.hw.xilinx_u200_xdma_201830_2/naiveBayesTrain_kernel.xclbin
    Loading: 'build_dir.hw.xilinx_u200_xdma_201830_2/naiveBayesTrain_kernel.xclbin'
    kernel has been created
    kernel start------
    kernel end------
    Total Execution time 17.381ms
    
    Start Profiling...
    Write DDR Execution time 0.108582ms
    Kernel Execution time 0.519421ms
    Read DDR Execution time 0.03953ms
    Total Execution time 0.667533ms
    ============================================================
    
    Prior probability:
    -2.34341 -2.38597 -2.30259 -2.43042 -2.20727 -2.36446 -2.22562 -2.30259 
    -2.27303 -2.21641 0 0 0 0 0 0 
    Check pass.
    
    ------------------------------------------------------------
   

Profiling
=========

The naive bayes design is validated on Alveo U200 board at 266 MHz frequency. 
The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for naive bayes
    :align: center
    
    +--------------------------+---------------+-----------+-----------+----------+
    |           Name           |       LUT     |    BRAM   |    URAM   |    DSP   |
    +--------------------------+---------------+-----------+-----------+----------+
    |         Platform         |     185929    |    345    |    0      |    10    |
    +--------------------------+---------------+-----------+-----------+----------+
    |  naiveBayesTrain_kernel  |     70058     |    114    |    256    |    467   |
    +--------------------------+---------------+-----------+-----------+----------+
    |        User Budget       |     996311    |    1815   |    960    |    6830  |
    +--------------------------+---------------+-----------+-----------+----------+
    |        Percentage        |     7.03%     |    6.28%  |    26.67% |    6.84% |
    +--------------------------+---------------+-----------+-----------+----------+

The performance is shown below.
    This benchmark takes 0.519421ms to train 999 samples with 10 features, so it throughput is 73.37MB/s.


.. toctree::
   :maxdepth: 1

