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
Internal Design of Breadth-first Search
*************************************************


Overview
========
Bread-first Search is an algorithm that traverse graph data structures. It starts from a source vertex and explores its neighbor vertices at the present depth level before moving on to the next depth level.

Algorithm
=========
The implemented Breadth-first Search is based on a First-In-First-Out execution queue. Below is the pseudo-code of the algorithm:

.. code::

    procedure BFS(graph, source)
    for each vertex v in graph
        discover(v) := positive infinity

    discover(source) := 0
    dtime := 0
    cnt_bfr := 1
    cnt_cur := 0
    cnt_nxt := 0
    cnt_lvl := 0
    push source into Q
    while Q is not empty
        u := pop Q
        cnt_cur++
        for each edge(u,v) in graph
            if discover(v) == positive infinity then
                discover(v) := dtime
                level(v) := cnt_lvl
                parent(v) := u
                push v into Q
                dtime++
                cnt_nxt++
            end if
        end for
        finish(u) := dtime
        dtime++
        if cnt_cur == cur_bfr then
            cnt_bfr := cnt_nxt
            cnt_cur := 0
            cnt_nxt := 0
            cnt_lvl ++
        end if
    end while

    return (level, parent, discover, finish)

Here, graph is a graph with a list of vertices and a list of edges. source is the start vertex of the algorithm. Q is a first-in-first-out queue. 

Interface
=========
The input should be a directed graph in compressed sparse row (CSR) format.
The result include level, parent, discover and finish. Level is a list which shows the final distance level to the source vertex. Parent is a list which shows the parent vertex of every vertex in the BFS. Discovery shows when a vertex is discovered in the BFS. Finish shows when all one-hop children vertices of a vertex are processed.

Implementation
==============
The algorithm implemention is shown in the figure below:

.. image:: /images/BFS.png
   :alt: Figure 1 Breadth-first Search design
   :width: 80%
   :align: center

There are 4 functional blocks as shown in the figure:

1. GetVertex is responsible to load the next vertex in the queue and pass it to the ReadGraph.

2. ReadGraph collects all next hop vertices and pass them to the next module.

3. ReadColor check each next hop vertex whether it has already been discovered in ealier stages of the BFS. This module only passes first discovered vertices to the next block.

4. When the 3rd functional block ends, WriteRes update the discovery, finish, level and parent value accordingly. And also this block push all the vertices collected from block 3 into Queue.

This system starts from pushing the source vertex into the queue and iterate until the queue is empty.

Profiling
=========
The hardware resource utilizations are listed in the following table. The BFS kernel is validated on Alveo U250 board at 300MHz frqeuency.

.. table:: Table 1 Hardware resources
    :align: center

    +------------+--------------+-----------+---------+--------+
    |    Name    |      LUT     |    BRAM   |   URAM  |   DSP  |
    +------------+--------------+-----------+---------+--------+
    |  Platform  |    104112    |    165    |    0    |    4   |
    +------------+--------------+-----------+---------+--------+
    | bfs_kernel |     67284    |    245    |    10   |    3   |
    +------------+--------------+-----------+---------+--------+
    |    Total   | 171396 (10%) | 410 (15%) | 10 (1%) | 7 (0%) |
    +------------+--------------+-----------+---------+--------+
