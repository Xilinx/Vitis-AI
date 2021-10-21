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

.. _release_note:

Release Note
============

.. toctree::
   :hidden:
   :maxdepth: 1

Vitis Graph Library is an open-sourced Vitis library written in C++ for accelerating graph applications in a variety of use cases. It now covers a level of acceleration: the module level (L1), the pre-defined kernel level (L2), the asynchronous software level (L3) and TigerGraph integration (plugin) APIs.

2021.1
----
The algorithms implemented by Vitis Graph Library include:
 - Similarity analysis: Cosine Similarity, Jaccard Similarity, k-nearest neighbor.
 - Centrality analysis: PageRank.
 - Pathfinding: Single Source Shortest Path (SSSP), Multi-Sources Shortest Path (MSSP).
 - Connectivity analysis: Weekly Connected Components and Strongly Connected Components.
 - Community Detection: Louvain Modularity( New for 2021.1 ), Label Propagation and Triangle Count.
 - Search: Breadth First Search, 2-Hop Search( New for 2021.1 ) 
 - Graph Format: Calculate Degree and Format Convert between CSR and CSC.


2020.2
----

The 1.0 release provides a range of algorithm, includes:

- Centrality analysis: PageRank.
- Pathfinding:  Single Source Shortest Path.
- Connectivity analysis: Weekly Connected Components and Strongly Connected Components.
- Community Detection:  Label Propagation and Triangle Count.
- Search: Breadth First Search.
- Graph Format: Calculate Degree and Format Convert between CSR and CSC.

