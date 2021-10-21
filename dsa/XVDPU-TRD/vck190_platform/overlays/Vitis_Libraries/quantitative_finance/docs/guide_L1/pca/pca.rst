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
****************************
Principal Component Analysis
****************************
Overview
========

`Principal Component Analysis` (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of linearly uncorrelated variables called *Principal Components*.

In quantitative finance, `PCA` can be directly applied to risk management of interest rate derivative portfolios.
It helps reducing the complexity of swap tradings from a function of 30-500 market instruments to, usually, just 3 or 4, which can represent the interest rate paths on a macro basis.

Implementation
======

The PCA of `N` components of an m-by-n matrix A is given by the following process:

- Calculate the covariance matrix of A
.. math::
    \Sigma = \frac{1}{n-1}((A-\bar{A})^T(A-\bar{A}))

.. math::
    \bar{A} = \frac{1}{n}\sum_{k=1}^{n}A_i

- Solve n-by-n covariance matrix for its n-by-n eigen-vectors (:math:`V`) and n eigen-values (:math:`D`)
- Sort the eigen-values from largest to smallest and then select the top :math:`N` eigen-values and their corresponding eigen-vectors.

Once the process is completed there are several outputs available from the library:

- **ExplainedVariance**: This is a vector `N` wide which corresponds to the selected sorted eigen-values.
- **Components**: These are the `N` eigen-vectors associated with the selected eigen-values of the original matrix.
- **LoadingsMatrix**: The loadings matrix represent the weigths associated to each original variable when calculating the principal components. It can be computed as follows:

.. math::
    Loadings=Components*\sqrt{ExplainedVariance^T}

.. note::
    Due to the arbitrary sign of eigen-vectors, them being implementation dependent, calculations of the loadings matrix could return inverted values in a non-deterministic way.
    To avoid that, we use the same convention as matlab, where the sign for the first element of each eigen-vector must be positive, multiplying the whole vector by :math:`-1` otherwise.

Below is a diagram of the internal implementation of PCA:

.. image:: /images/pca/PCA_Architecture.png
    :alt: Architectural diagram of Principal Component Analysis implementation
    :align: center

Profiling
=========

The hardware resources for `N = 3`, `MAX_VARS = 54`, `MAX_OBS = 1280` PCA are listed in :numref:`tabPCA`. (Vivado result)

.. _tabPCA:

.. table:: Hardware resources for PCA
    :align: center

    +--------------------------+----------+----------+----------+----------+----------+
    |          Engines         |   BRAM   |    DSP   | Register |    LUT   |  URAM    |
    +--------------------------+----------+----------+----------+----------+----------+
    |           PCA            |    69    |    113   |   39750  |   37867  |  2       |
    +--------------------------+----------+----------+----------+----------+----------+