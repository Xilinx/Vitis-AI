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

Please reference to `Dataset` in table 1. 


Performance
-----------

For representing the resource utilization in each benchmark, we separate the overall utilization into 2 parts, where P stands for the resource usage in
platform, that is those instantiated in static region of the FPGA card, as well as K represents those used in kernels (dynamic region). The target device is set to Alveo U280.

.. table:: Table 1 Performance on FPGA
    :align: center

    +-------------------------------+-----------------------------------------------------------------------------+--------------+----------+-----------------+------------+------------+------------+
    |     Architecture              |      Dataset                                                                |  Latency(ms) |  Timing  |   LUT(P/K)      |  BRAM(P/K) |  URAM(P/K) |  DSP(P/K)  |
    +===============================+=============================================================================+==============+==========+=================+============+============+============+
    | Naive Bayes (U200)            | 999 samples with 10 features                                                |    0.519     |  266MHz  |  185.9K/70.1K   |   345/114  |    0/256   |    10/467  |
    +-------------------------------+-----------------------------------------------------------------------------+--------------+----------+-----------------+------------+------------+------------+
    | Support Vector Machine (U250) | 999 samples with 66 features                                                |    0.23      |  300MHz  |  178.5K/367.0K  |   403/276  |    0/132   |    13/1232 |
    +-------------------------------+-----------------------------------------------------------------------------+--------------+----------+-----------------+------------+------------+------------+
    | Log Analyzer Demo (U200)      | 1.2G `access log <http://www.almhuette-raith.at/apache-log/access.log>`_    |    990       |  251MHz  |  282.6K/226.8K  |   835/332  |    0/208   |    16/22   |
    +-------------------------------+-----------------------------------------------------------------------------+--------------+----------+-----------------+------------+------------+------------+
    | Duplicate Record Match (U50)  | Randomly generate 10,000,000 lines (about 1GB)                              |    8215560   |  270MHz  |  135.8K/272.0K  |   180/50   |    0/260   |    4/506   |
    +-------------------------------+-----------------------------------------------------------------------------+--------------+----------+-----------------+------------+------------+------------+

These are details for benchmark result and usage steps.

.. toctree::
   :maxdepth: 1

   benchmark/naive_bayes.rst
   benchmark/svm.rst
   benchmark/log_analyzer.rst
   benchmark/dup_match.rst



Test Overview
--------------

Here are benchmarks of the Vitis Data Analytics Library using the Vitis environment. 

.. _l2_vitis_data_analytics:

Vitis Data Analytics Library
~~~~~~~~~~~~~~~~~~~

* **Download code**

These data analytics benchmarks can be downloaded from `vitis libraries <https://github.com/Xilinx/Vitis_Libraries.git>`_ ``master`` branch.

.. code-block:: bash

   git clone https://github.com/Xilinx/Vitis_Libraries.git 
   cd Vitis_Libraries
   git checkout master
   cd data_analytics

* **Setup environment**

Specifying the corresponding Vitis, XRT, and path to the platform repository by running following commands.

.. code-block:: bash

   source <intstall_path>/installs/lin64/Vitis/2021.1_released/settings64.sh
   source /opt/xilinx/xrt/setup.sh
   export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
