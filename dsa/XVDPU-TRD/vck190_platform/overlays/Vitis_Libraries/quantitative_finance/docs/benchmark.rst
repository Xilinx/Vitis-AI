
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
   :keywords: Finance, Quantitative, Vitis Quantitative Finance Library, quantitative_finance, quality, performance
   :description: Vitis Quantitative Finance library quality and performance results.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _benchmark:

==========
Benchmark 
==========

Application Scenario
---------------------

The application scenarios are provided for each case. User could find details in Profiling section under each case's benchmark description page.

Performance
-----------

For representing the resource utilization in each benchmark, we separate the overall utilization into 2 parts, where P stands for the resource usage in platform, that is those instantiated in static region of the FPGA card, as well as K represents those used in kernels (dynamic region). The input is matrix, and the target device is set to Alveo U250.

.. table:: Performance for processing quantitative_finance on FPGA
    :align: center

    +------------------------------------------------+-----------+--------------+----------+----------------+-------------+------------+------------+
    |            Architecture                        |  Kernels  |  Latency(s)  |  Timing  |    LUT(P/K)    |  BRAM(P/K)  |  URAM(P/K) |  DSP(P/K)  |
    +================================================+===========+==============+==========+================+=============+============+============+
    | MCEuropeanEngine(U250)                         |    4      |    25.94     |  300MHz  |  108.1K/21.1K  |   178/127   |    0/20    |     4/2    |
    +------------------------------------------------+-----------+--------------+----------+----------------+-------------+------------+------------+
    | MCAmericanEngineMultiKernel(U250)              |    3      |    1.811     |  280MHz  |  101.7K/101.5K |   165/387   |    0/112   |     4/3    |
    +------------------+-----------------------------+-----------+--------------+----------+----------------+-----+-------+------------+------------+
    |                  | TreeCallableEngineHWModel   |    1      |    3.484     |  275MHz  |  101.7K/160.5K |  165/523.5  |    0/110   |     4/6    |
    |                  +-----------------------------+-----------+--------------+----------+----------------+-----+-------+------------+------------+
    |                  | TreeCapFloorEngineHWModel   |    1      |    3.484     |  275MHz  |  101.7K/160.5K |  165/523.5  |    0/110   |     4/6    |
    |                  +-----------------------------+-----------+--------------+----------+----------------+-----+-------+------------+------------+
    |                  | TreeSwapEngineHWModel       |    1      |    3.484     |  275MHz  |  101.7K/160.5K |  165/523.5  |    0/110   |     4/6    |
    |                  +-----------------------------+-----------+--------------+----------+----------------+-----+-------+------------+------------+
    | TreeEngine(U250) | TreeSwaptionEngineBKModel   |    1      |    3.484     |  275MHz  |  101.7K/160.5K |  165/523.5  |    0/110   |     4/6    |
    |                  +-----------------------------+-----------+--------------+----------+----------------+-----+-------+------------+------------+
    |                  | TreeSwaptionEngineCIRModel  |    1      |    3.484     |  275MHz  |  101.7K/160.5K |  165/523.5  |    0/110   |     4/6    |
    |                  +-----------------------------+-----------+--------------+----------+----------------+-----+-------+------------+------------+
    |                  | TreeSwaptionEngineECIRModel |    1      |    3.484     |  275MHz  |  101.7K/160.5K |  165/523.5  |    0/110   |     4/6    |
    |                  +-----------------------------+-----------+--------------+----------+----------------+-----+-------+------------+------------+
    |                  | TreeSwaptionEngineG2Model   |    1      |    3.484     |  275MHz  |  101.7K/160.5K |  165/523.5  |    0/110   |     4/6    |
    |                  +-----------------------------+-----------+--------------+----------+----------------+-----+-------+------------+------------+
    |                  | TreeSwaptionEngineHWModel   |    1      |    3.484     |  275MHz  |  101.7K/160.5K |  165/523.5  |    0/110   |     4/6    |
    |                  +-----------------------------+-----------+--------------+----------+----------------+-----+-------+------------+------------+
    |                  | TreeSwaptionEngineVModel    |    1      |    3.484     |  275MHz  |  101.7K/160.5K |  165/523.5  |    0/110   |     4/6    |
    +------------------+-----------------------------+-----------+--------------+----------+----------------+-----+-------+------------+------------+
    | SVD(U250)                                      |    1      |    0.000196  |  300MHz  |  101.7K/40.3K  |  165/9      |    0/0     |     4/126  |
    +------------------------------------------------+-----------+--------------+----------+----------------+-----+-------+------------+------------+


These are details for benchmark result and usage steps.

.. toctree::
   :maxdepth: 1

   ../guide_L2/benchmarks/MCAmericanEngine.rst
   ../guide_L2/benchmarks/MCEuropeanEngine.rst
   ../guide_L2/benchmarks/TreeEngine.rst
   ../guide_L1/benchmarks/SVD.rst


Test Overview
--------------

Here are benchmarks of the Vitis Quantitative_Finance Library using the Vitis environment. 


.. _l2_vitis_quantitative_finance:

Vitis Quantitative_Finance Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Download code**

These quantitative_finance benchmarks can be downloaded from `vitis libraries <https://github.com/Xilinx/Vitis_Libraries.git>`_ ``master`` branch.

.. code-block:: bash

   git clone https://github.com/Xilinx/Vitis_Libraries.git 
   cd Vitis_Libraries
   git checkout master
   cd quantitative_finance 

* **Setup environment**

Specifying the corresponding Vitis, XRT, and path to the platform repository by running following commands.

.. code-block:: bash

   source /opt/xilinx/Vitis/2021.1/settings64.sh
   source /opt/xilinx/xrt/setup.sh
   export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
