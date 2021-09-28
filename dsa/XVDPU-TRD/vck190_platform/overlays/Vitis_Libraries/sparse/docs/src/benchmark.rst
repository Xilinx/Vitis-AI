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


.. Project documentation master file, created by
   sphinx-quickstart on Thu Jun 20 14:04:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========
Benchmark 
==========
    
.. _datasets:

Datasets
-----------

The dataset used in the benchmark can be downloaded from https://sparse.tamu.edu. 


Performance
-----------

For representing the resource utilization in each benchmark, we separate the overall utilization into 2 parts, where P stands for the resource usage in
platform, that is those instantiated in static region of the FPGA card, as well as K represents those used in kernels (dynamic region). The target device is set to Alveo U280.

.. table:: Table 1 Performance on FPGA
    :align: center

    +-----------------+-------------+--------------+----------+---------------------+------------+------------+------------+
    | Architecture    |    Dataset  |  Latency(ms) |  Timing  |   LUT(P/K)          |  BRAM(P/K) |  URAM(P/K) |  DSP(P/K)  |
    +=================+=============+==============+==========+=====================+============+============+============+
    | SPMV (U280)     |  nasa2910   |  0.0512565   |  256MHz  |  165.475K/220.98K   |   323/211  |    64/64   |    4/900   |
    +-----------------+-------------+--------------+----------+---------------------+------------+------------+------------+

These are details for benchmark results and usage steps.

.. toctree::
   :maxdepth: 1

   benchmark/spmv_double.rst



Test Overview
--------------

Here are benchmarks of the Vitis Sparse Library using the Vitis environment. 

.. _l2_vitis_sparse:

Vitis Sparse Library
~~~~~~~~~~~~~~~~~~~

* **Download code**

These sparse benchmarks can be downloaded from `vitis libraries <https://github.com/Xilinx/Vitis_Libraries.git>`_ ``master`` branch.

.. code-block:: bash

   git clone https://github.com/Xilinx/Vitis_Libraries.git 
   cd Vitis_Libraries
   git checkout master
   cd sparse

* **Setup environment**

Specifying the corresponding Vitis, XRT, and path to the platform repository by running following commands.

.. code-block:: bash

   source <intstall_path>/installs/lin64/Vitis/2021.1_released/settings64.sh
   source /opt/xilinx/xrt/setup.sh
   export PLATFORM_REPO_PATHS=/opt/xilinx/platforms

Python3 environment: follow the steps as per https://xilinx.github.io/Vitis_Libraries/blas/2020.2/user_guide/L1/pyenvguide.html to set up Python3 environment.
