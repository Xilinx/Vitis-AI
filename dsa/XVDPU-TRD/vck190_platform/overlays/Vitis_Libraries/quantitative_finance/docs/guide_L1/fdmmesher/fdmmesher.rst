
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
   :keywords: fintech, meshers
   :description: The conception of Mesher is used in the finite-difference method. Each mesher stores the discretization of one dimension.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

*******************
Meshers 
*******************

Overview
=========
The conception of Mesher is used in the finite-difference (FD) method. Each mesher stores the discretization of one dimension. It has a array :math:`locations` that stores the discretization at points :math:`x_{0},x_{1},..,x_{n-1}`. It also has two other arrays :math:`dplus` and :math:`dminus` that store the `i-th` elements :math:`(x_{i+1}-x_{i})` and :math:`(x_{i}-x_{i-1})` respectively.     

The multi-dimensional mesh for a finite-difference model is represented by multi 1-D meshers, which build the full mesh by composing a 1-D mesh for every dimension.

Implementation
==============
Mesher creates an equally-spaced grid between two given boundaries and with the given number of points with a specific processor. 
1. locations: The coordinates of location are evolved based a normal random number.
2. dplus: The argument is calculated at point :math:`(x_{i+1}-x_{i})`.
3. dminus: The argument is calculated at point :math:`(x_{i}-x_{i-1})`.

