
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
   :keywords: fintech, singular value decomposition, SVD, Jacobi, Profiling
   :description: The singular value decomposition (SVD) is a very useful technique for dealing with general dense matrix problems.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials


**********************************
Singular Value Decomposition (SVD)
**********************************

Overview
========

The `singular value decomposition` (SVD) is a very useful technique for dealing with general dense matrix problems. Recent years, SVD has become a computationally viable tool for solving a wide variety of problems raised in many practical applications, such as least-squares data fitting, image compression, facial recognition, principal component analysis, latent semantic analysis, and computing the 2-norm, condition number, and numerical rank of a matrix. 

For more information, please refer to `SVD`_.

.. _`SVD`: http://www.netlib.org/utk/people/JackDongarra/PAPERS/svd-sirev-M111773R.pdf

Theory
========

The SVD of an m-by-n matrix A is given by

.. math::
            A = U \Sigma V^T (A = U \Sigma V^H \, in \, the \, complex \, case)

where :math:`U` and :math:`V` are orthogonal (unitary) matrix and :math:`\Sigma` is an m-by-n matrix with real diagonal elements.

Theoretically, the SVD can be characterized by the fact that the singular values are the square roots of eigenvalues of :math:`A^TA`, the columns of :math:`V` are the corresponding eigenvectors, and the columns of :math:`U` are the eigenvectors of :math:`AA^T`, assuming distinct singular values. The approximation can simplify the general m-by-n matrix SVD problem to a general symmetric matrix SVD problem. 
Due to the roundoff errors in the formulation of :math:`AA^T` and :math:`A^TA`, the accuracy is influenced slightly, but if we don't need a high-accuracy, the approximation can largely reduce the complexity of calculation.

There are two dominant categories of SVD algorithms for dense matrix: bidiagonalization methods and Jacobi methods. The classical bidiagonalization method is a long sequential calculation, FPGA has no advantage in that case. In contrast, Jacobi methods apply plane rotations to the entire matrix A. Two-sided Jacobi methods iteratively apply rotations on both sides of matrix A to bring it to diagonal form, while one-sided Hestenes Jacobi methods apply rotations on one side to orthogonalize the columns of matrix A and bring :math:`A^TA` to diagonal form. While Jacobi methods are often slower than bidiagonalization methods, they have better potential in unrolling and pipelining. 

Jacobi Methods
--------------
Jacobi uses a sequence of plane rotations to reduce a symmetric matrix A to a diagonal matrix

.. math::
            A_{0} = A,   \> A_{k+1} = J_{k}^{T}A_{k}J_{k},   \> A_{k} \rightarrow \Sigma \> as \> k \rightarrow \infty
            


Each plane rotation, :math:`J_{k} = J_{k}(i, j, \theta)`, now called a Jacobi or Givens rotation

.. math::
            \begin{equation}
                J_{k}(i, j, \theta)=\begin{vmatrix}
                    I &\,  &  &  & \\
                    \, &\,c &  &s & \\
                    \, &\,  &I &  & \\
                    \, &-s  &  &c & \\
                    & &\,  &  &  &I
                \end{vmatrix}
            \end{equation}
    :label: Jacobi_rotation

where :math:`c=cos \theta` and :math:`s=sin \theta`. The angle :math:`\theta` is chosen to eliminate the pair :math:`a_{ij}`, :math:`a_{ji}` by applying :math:`J(i,j, \theta )` on the left and right of :math:`A`, which can be viewed as the 2x2 eigenvalue problem

.. math::
         \begin{equation}
          \hat{J}_{(k)}^{T} \hat{A}_{(k)} \hat{J}_{(k)}= \begin{vmatrix}
            \, c &s \\
              -s &c
              \end{vmatrix}^{T} \begin{vmatrix}
              a_{ii} &a_{ij} \\
              a_{ji} &a_{jj}
              \end{vmatrix} \begin{vmatrix}
              d_{ii}  &0 \\
                  0   &d_{jj}
                  \end{vmatrix}= \hat{A}_{(k+1)}
         \end{equation}

where :math:`\hat{A}` is a 2X2 submatrix of matrix A. After the Givens rotations of the whole matrix A, the off-diagonal value of A will be reduced after 5-10 times iteration of the process.
            

Implementation
=================

SVD workflow:
-------------

.. _my-figure-SVD:
.. figure:: /images/SVD/SVD.png
    :alt: SVD workflow in FPGA
    :width: 80%
    :align: center
    
The input parameters for the 4x4 SVD function is the 4x4 matrix :math:`A`, and the output matrices are :math:`U`, :math:`V`, and :math:`\Sigma` respectively. As shown in the above figure, the SVD process has 4 main steps:

1. Find the max value of matrix :math:`A`;
2. Divide all member of A by :math:`max(A)` and initiate :math:`U`, :math:`\Sigma`, :math:`V`;
3. The iterative process of Jacobi SVD;
4. Sort matrix :math:`S`, and change :math:`U` and :math:`V`;

The iterative process of Jacobi SVD is the core function. Firstly, the matrix :math:`A` will be divided into 2x3 (= :math:`C_{4}^{2}`) sub-blocks of 2x2 through its diagonal elements, since for a 4x4 matrix, at the same moment, it has only 2 pairs independent 2x2 matrix, and 3 rounds of nonredundant full permutation. 

Once we get the 2x2 Submatrix, the Jacobi methods or Givens rotation (module SVD 2x2) can be applied. Here we use pipelining to bind the two 2x2 SVD process. The output of 2x2 SVD is the rotation matrix Equation :eq:`Jacobi_rotation`. 

The next step is to decompose the rotation matrix from original matrix :math:`A` and add it to matrix :math:`U` and :math:`V`. After that, a convergence determination is performed to reduce the off-diagonal value of matrix :math:`A`. When the matrix :math:`A` is reduced to a diagonal matrix, step 3 will be finished.


.. note::
    The SVD function in this library is a customized function designated to solve the decomposition for a 3X3 or 4X4 symmetric matrix. It has some tradeoffs between resources and latency. A general SVD solver can be found in Vitis Solver Library.


Profiling
=========

The hardware resources for 4x4 SVD are listed in :numref:`tabSVD`. (Vivado result)

.. _tabSVD:

.. table:: Hardware resources for single 4x4 SVD
    :align: center

    +--------------------------+----------+----------+----------+----------+----------+-----------------+
    |          Engines         |   BRAM   |    DSP   | Register |    LUT   |  Latency | clock period(ns)|
    +--------------------------+----------+----------+----------+----------+----------+-----------------+
    |           SVD            |    12    |    174   |   57380  |   38076  |    3051  |       3.029     |
    +--------------------------+----------+----------+----------+----------+----------+-----------------+

The accuracy of SVD implementation has been verified with Lapack dgesvd (QR based SVD) and dgesvj (Jacobi SVD) functions. For a 2545-by-4 matrix, the relative error between our SVD and the two Lapack functions (dgesvd and dgesvj) is about :math:`1e^{-9}`

.. caution::
    The profiling resources differ a lot when choosing different chips. Here we use xcu250-figd2104-2L-e, with clock frequency 300MHz and the margin for clock uncertainty is set to 12.5%.

.. toctree::
   :maxdepth: 1
