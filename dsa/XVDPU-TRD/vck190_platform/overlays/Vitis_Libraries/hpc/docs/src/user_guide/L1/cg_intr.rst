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

.. _cg_introduction:

**********************************************
Conjugate Gradient Solver Introduction 
**********************************************

Linear solvers are super important as they are used in all major industries.
Most engineering problems can be turned into one or more linear equation
systems. Typically, the matrix formed in those systems is large in dimension and
highly sparse in data pattern. Iteration methods such as the preconditioned
conjugate gradient solver are a type of indirect solutions to these linear
systems with very high efficiency. 

Conjugate Gradient Algorithm
======================================
For linear system :math:`Ax=b` with given preconditioner matrix :math:`M`, the preconditioned conjugate gradient method is shown in the
following equations. 

.. math::
    :label: eq_init

    x_0 &= 0 \\
    r_0 &= b-Ax_0 \\
    z_0 &= M^{-1}r_0 \\
    \rho_0 &= r_0^Tz_0 \\
    \beta_k &= 0 \\


.. math::
    :label: eq_init

    while\ k<maxIter\ &AND\ ||r_{k}|| > tol*||b|| \\
        p_{k} &= z_{k} + \beta_{k-1}p_{k-1} \\
        \alpha_k&=\frac{\rho_k}{p_k^TAp_k} \\
        x_{k+1} &= x_k+\alpha_kp_k  \\
        r_{k+1} &= r_k+\alpha_kAp_k \\
        z_{k+1} &= M^{-1}r_{k+1} \\ 
        \rho_{k+1} &= r_{k+1}^Tz_{k+1} \\
        \beta_k &= \frac{\rho_{k+1}}{\rho_k}  \\
        k &= k+ 1 \\
