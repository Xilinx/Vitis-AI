

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
   :keywords: fintech, trapezoidal, Simpson, Romberg
   :description: Three Numerical Integration methods are included: the Adaptive Trapezoidal method, the Adaptive Simpson method and the Romberg method.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


***************************
Pentadiagonal Matrix Solver
***************************

Overview
========

The Pentadiagonal Matrix Solver solves a Pentadiagonal linear system using parallel cyclic reduction (also known as odd-even elimination). More details about this algorithm can be found in the paper: `Penta Solver`_.

.. _`Penta Solver`: https://www.academia.edu/8031041/Parallel_Solution_of_Pentadiagonal_Systems_Using_Generalized_Odd-Even_Elimination

Implementation
==============

The solver works on a row-based scheme. For each row of diagonals, it applies a reduction procedure. 
Each row is processed :math:`\log_2N -1` times, which leads to a complete reduction of the upper and lower diagonals. Because many experiments show that the algorithm fails for the number of steps greater than 8. In that case, it is recommended to limit the number of steps to 8.
The input matrix is stored as five vectors, one for each diagonal.

Since the algorithm needs random memory access in every iteration, 3 copies of the whole matrix are stored internally in the solver to allow full pipelining of the implementation. 

.. caution::
    Please note that the solver is very sensitive to zeros in **any** of the diagonals on input data. Due to the nature of the algorithm, any zeros on the three inner diagonals will lead to an attempt to divide-by-zero and the algorithm will fail.


