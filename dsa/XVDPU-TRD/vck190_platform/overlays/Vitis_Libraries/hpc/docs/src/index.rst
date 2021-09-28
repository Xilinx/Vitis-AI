.. 
   Copyright 2019 - 2021 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

=====================
Vitis HPC Library
=====================

Vitis HPC Library provides an acceleration libray for applications with high
computation workload, e.g.  seismic imaging and inversion, high-precision simulations, 
genomics and etc. Three types of components are provided in this library, 
namely L1 primitives, L2 kernels and L3 software APIs. These implementations are organized in their
corresponding directories L1, L2 and L3. The L1 primitives' implementations can be leveraged
by FPGA hardware developers. The L2 kernels' implementations provide examples for 
Vitis kernel developers. The L3 APIs provide C/C++ functions for software developers to offload HPC workloads.
This library depends on the **Xilinx BLAS and SPARSE** library to implement some components.

Because HPC applications normally have high precision requirements, the current supported data
type are mainly single precision floating point type (FP32 type) and double precision floating point type (FP64 type). 
Although most components can be configured
to support other data types, some of the architectures are specifically optimized to address
FP32 operations, e.g. accumulations.

In the current release, three types of applications have been addressed by this library, namely
RTM (Reverse Time Migration), CG (Conjugate Gradient) method and MLP-based high precesion seismic inversion.  
RTM is an important seismic imaging technique used for producing an accurate representation of the subsurface.
The basic computation unit of an RTM application is a stencil module, which is the essential 
step for explicit **FDTD (Finite Difference Time Domain)** solutions. Seismic inversion is a procedure
used to reconstruct subsurface properties via the seismic reflection data. 

Many engineering problems, such as FEM, are eventually transformed to a group of linear systems. 
Conjugate Gradient method, an iterative method, is widely adopted to solve linear systems, 
especially those with highly sparse and large-dimention matrices.
Preconditioner matrix is necessary for most of the problems in order to achieve convergent results and reduce dramatically the
number of iterations, hence improves the entire performance. 

Modern technology uses high precision MLP (Multilayer perceptron) based neural network to speed up this process.
The basic unit of a MLP application normally includes a fully connected neural (**FCN**) network and an activation
function, e.g. sigmoid function.


In this library, you will find the implementations of stencil module, 2D and 3D RTM forward propogation path,
2D RTM application, CG solvers with Jacobi preconditioner, high-precision fully connected neural network and sigmoid activation function.


Since all the kernel code is developed with the permissive Apache 2.0 license,
advanced users can easily tailor, optimize or combine them for their own need.
Demos and usage examples of different implementation level are also provided
for reference. 

.. toctree::
   :caption: Library Overview
   :maxdepth: 1

   overview.rst
   release.rst
 
.. toctree::
   :caption: User Guide
   :maxdepth: 2 

   pyenvguide.rst
   user_guide/L1/L1.rst
   user_guide/L2/L2.rst
   user_guide/L3/L3.rst

.. toctree::
  :caption: Benchmark 
  :maxdepth: 1 

  benchmark.rst


Index
-----

* :ref:`genindex` 
