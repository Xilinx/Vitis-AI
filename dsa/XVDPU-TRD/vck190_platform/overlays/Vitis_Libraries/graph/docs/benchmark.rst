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

The data is coming from https://sparse.tamu.edu/, our commonly used datasets are listed in table 1. 

.. table:: Table 1 Datasets for benchmark
    :align: center

    +--------------------+----------+-----------+-------------+
    |   Datasets         |  Vertex  |   Edges   |   Degree    |
    +====================+==========+===========+=============+
    |  as-Skitter        | 1694616  |  11094209 | 6.546739202 |
    +--------------------+----------+-----------+-------------+
    |  coPapersDBLP      | 540486   |  15245729 | 28.20744478 |
    +--------------------+----------+-----------+-------------+
    |  coPapersCiteseer  | 434102   |  16036720 | 36.94228545 |
    +--------------------+----------+-----------+-------------+
    |  cit-Patents       | 3774768  |  16518948 | 4.37614921  |
    +--------------------+----------+-----------+-------------+
    |  europe_osm        | 50912018 |  54054660 | 1.061726919 |
    +--------------------+----------+-----------+-------------+
    |  hollywood         | 1139905  |  57515616 | 50.45649945 |
    +--------------------+----------+-----------+-------------+
    |  soc-LiveJournal1  | 4847571  |  68993773 | 14.23264827 |
    +--------------------+----------+-----------+-------------+
    |  ljournal-2008     | 5363260  |  79023142 | 14.73416206 |
    +--------------------+----------+-----------+-------------+
    |  patients          | 1250000  |  200      |      -      |
    +--------------------+----------+-----------+-------------+

Performance
-----------

For representing the resource utilization in each benchmark, we separate the overall utilization into 2 parts, where P stands for the resource usage in
platform, that is those instantiated in static region of the FPGA card, as well as K represents those used in kernels (dynamic region). The input is
directed or undirected graph in compressed sparse row (CSR) format, and the target device is set to Alveo U50/U250.

.. table:: Table 2 Performance for processing sparse graph on FPGA
    :align: center

    +-----------------------------------------------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    | Architecture                                              |     Dataset      |  Latency(s)  |  Timing  |    LUT(P/K)    |  BRAM(P/K)  |  URAM(P/K) |  DSP(P/K)  |
    +===========================================================+==================+==============+==========+================+=============+============+============+
    |  Single Source Shortest Path (Directed, U250)             | soc-LiveJournal1 |    25.94     |  300MHz  |  108.1K/21.1K  |   178/127   |    0/20    |     4/2    |
    +-----------------------------------------------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    |  Connected Component (Directed/Undirected, U250)          | coPapersCiteseer |    1.811     |  280MHz  |  101.7K/101.5K |   165/387   |    0/112   |     4/3    |
    +-----------------------------------------------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    | Strongly Connected Component (Directed/Undirected, U250)  |  ljournal-2008   |    3.484     |  275MHz  |  101.7K/160.5K |  165/523.5  |    0/110   |     4/6    |
    +-----------------------------------------------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    | Triangle Counting (Undirected, U250)                      |  europe_osm      |    1.08      |  300MHz  |  150.9K/20.5K  |    338/62   |    0/16    |     7/0    |
    +-----------------------------------------------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    | Label Propagation (Directed, U250)                        |  hollywood       |    107.39    |  292MHz  |  158.2K/71.0K  |   375/100   |     0/0    |     7/0    |
    +-----------------------------------------------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    | PageRank (Directed, U250, Cache 1)                        |  europe_osm      |    47.376    |  300MHz  | 154.6K/252.9K  |   357/546   |     0/0    |    7/52    |
    +-----------------------------------------------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    | PageRank (Directed, U250, Cache 32K)                      |  europe_osm      |    51.919    |  225MHz  | 156.4K/100.4K  |   357/189   |    0/224   |    7/52    |
    +-----------------------------------------------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    | PageRank MultiChannels (Directed, U50)                    |  europe_osm      |    34.26     |  229MHz  | 118.8K/132.0K  |   178/303   |    0/224   |    4/84    |
    +-----------------------------------------------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    | General Similarity Cosine (Undirected, U50)               |  as-Skitter      |    0.0213    |  295MHz  | 121.1K/164.6K  |  180/230.5  |    0/80    |   4/645    |
    +-----------------------------------------------------------+------------------+--------------+----------+----------------+-------------+------------+------------+ 
    | Sparse Similarity Cosine (Directed/Undirected, U50)       |  coPaperDBLP     |    0.0137    |  295MHz  | 132.8K/120.1K  |  180/310.5  |    0/128   |   4/127    |
    +-----------------------------------------------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    | Dense Similarity Cosine (Directed/Undirected, U50)        |  patients        |    0.0112    |  260MHz  | 119.1K/266.1K  |   180/618   |    0/48    |   4/2364   |
    +-----------------------------------------------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    | Two Hop Path Count (Directed, u50)                        | soc-LiveJournal1 |    38.90     |  300MHz  |  145.9K/34.1K  |   180/210   |     0/0    |     4/0    |
    +-----------------------------------------------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    | Louvain modularity fast (Undirected, u50)                 |  europe_osm      |    111.092   | 188.3MHz |  123.4K/127.6K |   180/461   |    0/208   |    4/115   |
    +-----------------------------------------------------------+------------------+--------------+----------+----------------+-------------+------------+------------+

These are details for benchmark result and usage steps.

.. toctree::
   :maxdepth: 1

   guide_L2/manual/connectedComponent.rst
   guide_L2/manual/stronglyConnectedComponent.rst
   guide_L2/manual/triangleCount.rst
   guide_L2/manual/labelPropagation.rst
   guide_L2/manual/pageRank.rst
   guide_L2/manual/pageRankMultichannels.rst
   guide_L2/manual/shortestPath.rst
   guide_L2/manual/twoHop.rst
   guide_L2/manual/louvainFast.rst

Test Overview
--------------

Here are benchmarks of the Vitis Graph Library using the Vitis environment and comparing with Spark(v3.0.0) and Tigergraph(v2.4.0). 


Spark
~~~~~
* Spark 3.0.0 installed and configured.
* Spark running on platform with Intel(R) Xeon(R) CPU E5-2690 v4 @2.600GHz, 56 Threads (2 Sockets, 14 Core(s) per socket, 2 Thread(s) per core). 

Tigergraph
~~~~~~~~~~
* `Tigergraph 2.4.1 installed and configured <https://xilinx.github.io/Vitis_Libraries/graph/2020.2/plugin/tigergraph_integration.html>`_.
* Tigergraph running on platform with Intel(R) Xeon(R) CPU E5-2640 v3 @2.600GHz, 32 Threads (16 Core(s)).

.. _l2_vitis_graph:

Vitis Graph Library
~~~~~~~~~~~~~~~~~~~

* **Download code**

These graph benchmarks can be downloaded from `vitis libraries <https://github.com/Xilinx/Vitis_Libraries.git>`_ ``master`` branch.

.. code-block:: bash

   git clone https://github.com/Xilinx/Vitis_Libraries.git 
   cd Vitis_Libraries
   git checkout master
   cd graph

* **Setup environment**

Specifying the corresponding Vitis, XRT, and path to the platform repository by running following commands.

.. code-block:: bash

   source <intstall_path>/installs/lin64/Vitis/2021.1/settings64.sh
   source /opt/xilinx/xrt/setup.sh
   export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
