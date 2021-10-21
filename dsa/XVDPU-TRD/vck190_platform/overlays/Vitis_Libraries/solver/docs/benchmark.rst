
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

.. meta::
   :keywords: Vitis, Solver, Library, Vitis Solver Library, quality, performance
   :description: Vitis Solver Library quality and performance results.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _benchmark:

==========
Benchmark 
==========

Datasets
---------

The row number and column number of matrix are assigned as input arguments. The matrix is then generated randomly.

Performance
-----------

For representing the resource utilization in each benchmark, we separate the overall utilization into 2 parts, where P stands for the resource usage in platform, that is those instantiated in static region of the FPGA card, as well as K represents those used in kernels (dynamic region). The input is matrix, and the target device is set to Alveo U250.

.. table:: Performance for processing solver on FPGA
    :align: center

    +----------------+---------------+----------+--------------+----------+----------------+-------------+------------+------------+
    | Architecture   |  Matrix_Size  |  Unroll  |  Latency(s)  |  Timing  |    LUT(P/K)    |  BRAM(P/K)  |  URAM(P/K) |  DSP(P/K)  |
    +================+===============+==========+==============+==========+================+=============+============+============+
    | GESVDJ (U250)  |    512x512    |    16    |    25.94     |  300MHz  |  108.1K/21.1K  |   178/127   |    0/20    |     4/2    |
    +----------------+---------------+----------+--------------+----------+----------------+-------------+------------+------------+
    | GESVJ (U250)   |    512x512    |     8    |    1.811     |  280MHz  |  101.7K/101.5K |   165/387   |    0/112   |     4/3    |
    +----------------+---------------+----------+--------------+----------+----------------+-------------+------------+------------+
    | GTSV (U250)    |    512x512    |    16    |    3.484     |  275MHz  |  101.7K/160.5K |  165/523.5  |    0/110   |     4/6    |
    +----------------+---------------+----------+--------------+----------+----------------+-------------+------------+------------+


These are details for benchmark result and usage steps.

.. toctree::
   :maxdepth: 1

   ../guide_L2/Benchmark/gesvdj.rst
   ../guide_L2/Benchmark/gesvj.rst
   ../guide_L2/Benchmark/gtsv.rst

Test Overview
--------------

Here are benchmarks of the Vitis Solver Library using the Vitis environment. 


.. _l2_vitis_solver:

Vitis Solver Library
~~~~~~~~~~~~~~~~~~~

* **Download code**

These solver benchmarks can be downloaded from `vitis libraries <https://github.com/Xilinx/Vitis_Libraries.git>`_ ``master`` branch.

.. code-block:: bash

   git clone https://github.com/Xilinx/Vitis_Libraries.git 
   cd Vitis_Libraries
   git checkout master
   cd solver 

* **Setup environment**

Specifying the corresponding Vitis, XRT, and path to the platform repository by running following commands.

.. code-block:: bash

   source /opt/xilinx/Vitis/2021.1/settings64.sh
   source /opt/xilinx/xrt/setup.sh
   export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
