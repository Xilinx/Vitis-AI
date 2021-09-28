
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

************************************
Covariance Matrix and Regularizaiton
************************************

Overview
========

In probability theory and statistics, a covariance matrix, (aka, variance-covariance matrix) is a :math: `N\times N` square matrix that contains the variances and covariances associated with N observed variables. The diagonal elements of the matrix contain the variances of the variables, and the off-diagonal elements contain the covariances between all possible pairs of variables. At the same time, in order to solve an ill-posed problem or to prevent overfitting, there are four ways to regularize the covariance matrix, including hard-thresholding. soft-thresholding, banding, and tapering.

Algorithm
=========


Covariance Matrix
-----------------

For a Matrix :math: `X=\left ( a_{n,m} \right )_{N\times M}` with N variables from M observations, its covariance matrix can be expressed as:

.. math::

   C\left ( i,j \right )=\frac{1}{M-1}\sum_{m=0}^{M-1}\left ( a_{i,m}-\bar{a_{i}} \right )\left ( a_{jm}-\bar{a_{j}} \right )

where :math: `i,j\in \left [ 0,N-1 \right ]`, :math: `a_{i,m}` denotes m-th element of the i-th row, :math: `\bar{a_{i}}` denotes the expected value (mean) of the all observations on the i-th row.

The variance-covariance matrix is symmetric because the element :math: `c\left ( i,j \right )` the same as the element :math: `c\left ( j,i \right )`. Therefore, for the implementation of the covariance matrix, it only needs to calculate a lower triangular matrix.


Covariance Regularizaiton
-------------------------

For the algorithm of covariance regularization, please refer to "High-Dimensional Covariance Estimation" by Mohsen Pourahmadi.


Implementation
================

The implementation of the covariance matrix and covariance regularization are very common and simple. There is not elaborated here. Here,  the key optimization based on the design of FPGA of covariance matrix is introduced. According to the formula or implementation code of the covariance matrix, the core design requires 3 layers of loops, which will cause the biggest latency. Therefore, it needs to  be optimized to improve throughput.

Firstly, the loop of the bottom and middle layers is unrolled to increase throughput. However, due to the self-addition operation in the loop, the effect of `unroll` operator is not particularly obvious to reduce latency. So, the core calculation part and the self-addition part in the underlying loop is split into two processes, passing the intermediate result through the stream by using pragma `dataflow` to improve throughput. See the function `covCoreWrapper` for details.

Profiling
=========

The hardware resources utilization for the covariance matrix are listed in :numref:`tabCov`. (Vivado result)

.. _tabCov:

.. table:: Hardware resources for covariance matrix
    :align: center

    +--------------------------+----------+----------+----------+-----------------+
    |        Primitives        |   BRAM   |    DSP   |    LUT   | clock period(ns)|
    +--------------------------+----------+----------+----------+-----------------+
    |       covCoreStrm        |   448    |    730   |   92318  |       3.962     |
    +--------------------------+----------+----------+----------+-----------------+
    |     covReHardThreshold   |    0     |     4    |    312   |       2.341     |
    +--------------------------+----------+----------+----------+-----------------+
    |     covReSoftThreshold   |    0     |    15    |   5006   |       3.016     |
    +--------------------------+----------+----------+----------+-----------------+
    |         covReBand        |    0     |     4    |    403   |       2.797     |
    +--------------------------+----------+----------+----------+-----------------+
    |         covReTaper       |    0     |     7    |    1168  |       2.729     |
    +--------------------------+----------+----------+----------+-----------------+

.. toctree::
   :maxdepth: 1
