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


*************************************************
Internal Design of Sparse Similarity
*************************************************

Interface
===========
The input should be a directed/undirected graph in compressed sparse row (CSR) format.
The result will return a vertex list with each vertex corresponding similarity value.
The config contains several boolean value to control the similarityType (0:Jaccard Similarity, 1:Cosine Similarity), dataType(0:uint, 1:float).

.. image:: /images/sparse_similarity_api.PNG
   :alt: API of Sparse Similarity
   :width: 65%
   :align: center

Implemention
============

The detail algorithm implemention is illustrated as below:

.. image:: /images/sparse_similarity_internal.PNG
   :alt: Diagram of Sparse Similarity
   :width: 70%
   :align: center

As it is shown in the aboved pictures, every PE directly have 3 AXI port for the input of offset, indice and weight (CSR format data) and the data should be partitioned in host side. 
The internal function in the PE perform searching and matching index to find out the similarity between reference vertex and the others.
The overall diagram of sparse similarity kernel have a insert sort module which return the top K number of similarity values.
The maximum number of K is a template number which can be changed by rebuilding the xclbin. The default value of top K is 32.

Profiling and Benchmarks
========================

The Sparse Similarity Kernel is validated on Alveo U50 board at 295MHz frequency. 
The hardware resource utilization and benchmark results are shown in the two table below.

.. table:: Table 1 Hardware resources
    :align: center

    +------------------------+--------------+----------------+----------+----------+--------+
    |          Name          |      LUT     |    Register    |   BRAM   |   URAM   |   DSP  |
    +------------------------+--------------+----------------+----------+----------+--------+
    | sparseSimilarityKernel |    123017    |    190284      |    310.5 |    128   |   127  |
    +------------------------+--------------+----------------+----------+----------+--------+


.. table:: Table 2 Comparison between TigerGraph on CPU and FPGA
    :align: center
    
    +------------------+----------+----------+-----------------+----------------+------------------------------+
    |                  |          |          |                 |                |  TigerGraph (32 core 512 GB) |
    |     Datasets     |  Vertex  |   Edges  | Similarity Type | FPGA Time / ms +----------------+-------------+
    |                  |          |          |                 |                |   Time / ms    |  Speed up   |
    +------------------+----------+----------+-----------------+----------------+----------------+-------------+
    |    as-Skitter    | 1694616  | 11094209 |      Cosine     |    12.3        |    278         |    22.6     |
    |                  |          |          +-----------------+----------------+----------------+-------------+
    |                  |          |          |     Jaccard     |    12.2        |    271         |    22.3     |
    +------------------+----------+----------+-----------------+----------------+----------------+-------------+
    |    coPaperDBLP   | 540486   | 15245729 |      Cosine     |    13.7        |    289         |    21.1     |
    |                  |          |          +-----------------+----------------+----------------+-------------+
    |                  |          |          |     Jaccard     |    13.5        |    271         |    20.1     |
    +------------------+----------+----------+-----------------+----------------+----------------+-------------+
    | coPaperCiteseer  | 434102   | 16036720 |      Cosine     |    17.6        |    282         |    16.0     |
    |                  |          |          +-----------------+----------------+----------------+-------------+
    |                  |          |          |     Jaccard     |    17.7        |    283         |    15.9     |
    +------------------+----------+----------+-----------------+----------------+----------------+-------------+
    |    cit-Patents   | 3774768  | 16518948 |      Cosine     |    24.8        |    268         |    10.8     |
    |                  |          |          +-----------------+----------------+----------------+-------------+
    |                  |          |          |     Jaccard     |    24.7        |    262         |    10.6     |
    +------------------+----------+----------+-----------------+----------------+----------------+-------------+
    |    europe_osm    | 50912018 | 54054660 |      Cosine     |    105.0       |    309         |    2.94     |
    |                  |          |          +-----------------+----------------+----------------+-------------+
    |                  |          |          |     Jaccard     |    104.9       |    315         |     3.0     |
    +------------------+----------+----------+-----------------+----------------+----------------+-------------+
    |     hollywood    | 1139905  | 57515616 |      Cosine     |    68.7        |    280         |    4.07     |
    |                  |          |          +-----------------+----------------+----------------+-------------+
    |                  |          |          |     Jaccard     |    68.8        |    329         |    4.78     |
    +------------------+----------+----------+-----------------+----------------+----------------+-------------+
    | soc-LiveJournal1 | 4847571  | 68993773 |      Cosine     |    75.5        |    293         |    3.88     |
    |                  |          |          +-----------------+----------------+----------------+-------------+
    |                  |          |          |     Jaccard     |    75.4        |    288         |    3.82     |
    +------------------+----------+----------+-----------------+----------------+----------------+-------------+
    |   ljournal-2008  | 5363260  | 79023142 |      Cosine     |    83.6        |    281         |    3.36     |
    |                  |          |          +-----------------+----------------+----------------+-------------+
    |                  |          |          |     Jaccard     |    83.8        |    384         |    4.58     |
    +------------------+----------+----------+-----------------+----------------+----------------+-------------+
    |     GEOMEAN      |          |          |                 |    50.1        |    292.7       |    5.84     |
    +------------------+----------+----------+-----------------+----------------+----------------+-------------+

.. note::
    | 1. Tigergraph running on platform with Intel(R) Xeon(R) CPU E5-2640 v3 @2.600GHz, 32 Threads (16 Core(s)).
    | 2. Time unit: ms.

.. toctree::
    :maxdepth: 1
