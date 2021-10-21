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
Internal Design of two hop path count 
*************************************************


Overview
========
The two hop path count calculate how many 2 hop paths between a given set of source and destianation pairs.

Implementation
==============
The implemention is shown in the figure below:

.. image:: /images/twoHop_design.png
   :alt: two hop path count design
   :width: 80%
   :align: center

The kernel will do the following steps:

1. Load the ID of the source and destination vertices. Pass the source vertices ID to the 1st hop path finder through hls::stream. Pass the destination vertices to the filter and counter.

2. The 1st hop path finder will find all the 1st hop vertices and pass them to the 2nd hop path finder through hls::stream.

3. The 2nd hop path finder will find all the 2nd hop vertices and pass them to the final filter and counter through hls::stream.

4. The filter and finder will drop all the 2nd hop vertices that is not the target destination vertex and count all the 2nd hop vertices that is the targeted destination vertex. And then stored it into the result. 

Interface
=========
The input should be a directed graph in compressed sparse row (CSR) format.

The result is an array which shows the number of 2 hop paths. The order of the result is the same as the order of input pairs.

