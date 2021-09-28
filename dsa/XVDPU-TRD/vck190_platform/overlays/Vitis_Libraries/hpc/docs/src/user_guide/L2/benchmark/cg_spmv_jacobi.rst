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

*****************************************************************
SPMV-based Conjugate Gradient Solver with Jacobi Preconditioner
*****************************************************************

Introduction
###################

CG solver is widely adopted to solve linear system Ax=b, where the matrix A is symmetric and positive definite. 
Here is the benchmark for SPMV-based CG solver with the Jacobi preconditioner on Xilinx FPGA Alveo U280. 

Benchmark on Hardware
#######################

Environment Setup (Step 1)
******************************
Please follow the page :doc:`Benchmark Overview <../../../benchmark>` to correctly setup the environment first.  

Hardware Build (Step 2)
*************************

With the following commands, kernel bitstream *cgSolver.xclbin * is built under the directory *./build_dir.hw.xilinx_u280_xdma_201920_3*

.. code-block:: bash

    $ make build TARGET=hw DEVICE=xilinx_u280_xdma_201920_3

Prepare Data (Step 3)
***********************

Here is a list of the URLs of **SPD** sparse matrices in the file *test.txt*. All these sparse matrices are from `SuiteSparse Matrix Collection<https://sparse.tamu.edu/>`. Users could add more links or trim the existing links in the file. With the following command, these matrices listed in the *test.txt* file are download from the given links and then are preprocessed. It may take some time to finish the downloading and preprocessing. 

.. code-block:: bash

    $ make data_gen TARGET=hw DEVICE=xilinx_u280_xdma_201920_3

Run on FPGA (Step 4)
********************

Check Device
====================

If you followed the guide and correctly setup the environment, you are able to run the following command line. You could check whether the target device is prepared and find out the device ID. 

.. code-block:: bash

    $ xbutil scan

Benchmark
=============

With the following command, users could benchmark the CG solver with a given matrix. 

.. code-block:: bash

    $ make run TARGET=hw DEVICE=xilinx_u280_xdma_201920_3 mtxName=ted_B

Here lists the configurable parameters with the *make* command for the benchmark. 


.. table:: Parameters with make command 
    :align: center

    +----------------+---------------+----------------------------------------------+
    | Parameter Name | Default Value | Notes                                        |
    +================+===============+==============================================+
    | mtxName        | ted_B         | Vector size, must be multiple of 16          |
    +----------------+---------------+----------------------------------------------+
    | maxIter        | 5000          | Maximum No. iterations for the solver        |
    +----------------+---------------+----------------------------------------------+
    | tol            | 1e-12         | Fault Tolerance                              |
    +----------------+---------------+----------------------------------------------+
    | deviceID       | 0             | Alveo U50 Card ID                            |
    +----------------+---------------+----------------------------------------------+

Usage
**************

.. code-block:: bash

    Usage: host.exe <XCLBIN File> <Max Iteration> <signature path> <vector path> <mtx_name> [--debug] [device id]
                <XCLBIN File>       path to the xclbin file
                <Max Iteration>     maximum number of iterations
                <Tolerence>         Fault tolerence
                <signature path>    path to the signature files
                <vector path>       path to the vector binary files
                <mtx_name>          sparse matrix name 
                <device id>         Device id given by *xbutil scan*

Resource Utilization on Alveo U280
##################################

The following table lists the resource utilization for SPMV-based CG kernel. 

.. table:: Resource Utilization on U280
    :align: center

    +----------------------------+-------------------+------------------+-------------------+----------------+---------------+----------------+
    | Name                       |  LUT              | LUTAsMem         | REG               | BRAM           | URAM          | DSP            |
    +============================+===================+==================+===================+================+===============+================+
    | User Budget                | 1104369 [100.00%] | 552814 [100.00%] | 2217989 [100.00%] | 1693 [100.00%] | 896 [100.00%] | 9020 [100.00%] |
    +----------------------------+-------------------+------------------+-------------------+----------------+---------------+----------------+
    |    Used Resources          |  285372 [ 25.84%] |  36605 [  6.62%] |  442368 [ 19.94%] |  267 [ 15.77%] |  64 [  7.14%] | 1192 [ 13.22%] |
    +----------------------------+-------------------+------------------+-------------------+----------------+---------------+----------------+

Benchmark Results on Alveo U280 FPGA
#########################################

CPU Hardware information

*   Model name: Intel(R) Xeon(R) CPU E5-2667 v4 @ 3.20GHz
*   Total threads: 32, Threads/Core: 2, Cores/Socket: 8, Total sockets: 2, Total Cores:16

FPGA Hardware Information

* Device name:  Xilinx Alveo U280
* Fmax: 243MHz

.. table:: Benchmark Results on U280
    :align: center


    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | Matrix Name    | Rows/Cols | NNZs    | Padded Rows/Cols | Padded NNZs | Padding Ratio | No. iterations | Time per Iter [ms] | Time per Iter on CPU [ms] | Acceleration Ratio |
    +================+===========+=========+==================+=============+===============+================+====================+===========================+====================+
    | nasa2910       |   2910    | 174296  |   2912           |   297952    |   1.70946     |   1777         |   0.0511172        |   0.0692836               |    1.36            |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | ex9            |   3363    | 99471   |   3364           |   199328    |   2.00388     |   5000         |   0.0497677        |   0.0559332               |    1.12            |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | bcsstk24       |   3562    | 159910  |   3564           |   222656    |   1.39238     |   5000         |   0.0598962        |   0.0581827               |    0.97            |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | bcsstk15       |   3948    | 117816  |   3948           |   267488    |   2.27039     |   658          |   0.0927269        |   0.125615                |    1.35            |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | bcsstk28       |   4410    | 219024  |   4412           |   319264    |   1.45767     |   4878         |   0.0586356        |   6.92198                 |    118.05          |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | s3rmt3m3       |   5357    | 207695  |   5360           |   330624    |   1.59187     |   5000         |   0.0744822        |   6.55229                 |    87.97           |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | s2rmq4m1       |   5489    | 281111  |   5492           |   427648    |   1.52128     |   1779         |   0.084562         |   6.75384                 |    79.87           |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | nd3k           |   9000    | 3279690 |   9000           |   4277792   |   1.30433     |   5000         |   0.363479         |   4.66861                 |    12.84           |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | ted_B          |   10605   | 144579  |   10608          |   548416    |   3.79319     |   30           |   0.984467         |   6.53108                 |    6.63            |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | ted_B_unscaled |   10605   | 144579  |   10608          |   548416    |   3.79319     |   16           |   1.75354          |   8.59891                 |    4.90            |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | msc10848       |   10848   | 1229778 |   10848          |   2050720   |   1.66755     |   5000         |   0.230942         |   5.43921                 |    23.55           |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | cbuckle        |   13681   | 676515  |   13684          |   924832    |   1.36705     |   1282         |   0.16427          |   5.48588                 |    33.40           |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | olafu          |   16146   | 1015156 |   16148          |   1452320   |   1.43064     |   5000         |   0.169174         |   5.05108                 |    29.86           |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | gyro_k         |   17361   | 1021159 |   17364          |   1932384   |   1.89234     |   5000         |   0.254172         |   4.85938                 |    19.12           |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | bodyy4         |   17546   | 121938  |   17548          |   710112    |   5.82355     |   230          |   0.174435         |   4.73164                 |    27.13           |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | nd6k           |   18000   | 6897316 |   18000          |   9415552   |   1.3651      |   5000         |   0.809868         |   4.25772                 |    5.26            |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | raefsky4       |   19779   | 1328611 |   19780          |   2268704   |   1.70758     |   5000         |   0.268956         |   4.22843                 |    15.72           |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+
    | bcsstk36       |   23052   | 1143140 |   23052          |   1833056   |   1.60353     |   5000         |   0.253049         |   3.9882                  |    15.76           |
    +----------------+-----------+---------+------------------+-------------+---------------+----------------+--------------------+---------------------------+--------------------+


Convergence
******************

Conjugate gradient method may suffer convergent issue for matrices with large condition number. 
Jacobi preconditioner, adopted in this kernel, is widely used and dramatically reduces the overall
number of iterations to solve the linear system.
For some matrices, however, the solver with Jacobi preconditioner is not able to converge. 
For instance, the number of iterations for some matrices in the above table reached the upper limit
5000 with the preset relative tolerance **10e-12**. 

Although the solver, for some other matrices e.g. *ted_B_unscaled*, 
meets the preset tolerance within the preset number of iteration limit, there might still
be some mismatches in the result vector compared to the golden reference `x`.
The solution to this issue is to further reduce the tolerance value to such as **10e-15**. 
