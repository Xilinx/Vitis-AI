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

.. _guide-svm_train:

********************************************************
Internals of svm_train
********************************************************

.. toctree::
   :hidden:
   :maxdepth: 1

Overview
========

SVM (Support Vector Machine) is a model to predict sample's classification.
This document describes the structure and execution of svm train, implemented as ``SVM`` function.


Basic algorithm
================

Svm_train is a function to trains a batch of samples using support vector machine model and output a weight vecotor for classification. 
This function used SGD(Stochastic Gradient Descent) method to get convergence.


Implementation
==============

The structure of ``SVM`` is described as below. It has a svm_dataflow region and control logic region.
There is a iteration loop out of these two regions mentioned above. Its break condition is determined by control logic and maximum iteration config read from DDR.


.. image:: /images/svm_train.png
   :alt: svm_train Top Structure
   :align: center


Config description
=======================

The sample and config input share one 512-bit port. config is stored in first 512bit.
The weight vector is output by a 512-bit port, which will be aligned to 512bit boundaries.
Config's details are documented in the following table:

+----------+---------+-----------+----------+-----------+---------------+---------------+-----------------+
| 511-448  | 447-384 | 383-320   | 319-256  | 255-192   | 191-128       | 127-64        | 63-0            |
+==========+=========+===========+==========+===========+===============+===============+=================+
| columns  | offset  | tolerence | reg_para | step_size | sample_number | max_iteration | features_number |
+----------+---------+-----------+----------+-----------+---------------+---------------+-----------------+


Resource Utilization
====================

The hardware resource utilization of svm_train(4 kernels in 4 SLRs) is shown in the table below (work as 276MHz).

+--------+---------------+--------------+----------+--------+------+------+
|  LUT   | LUT as memory | LUT as logic | Register | BRAM36 | URAM | DSP  |
+--------+---------------+--------------+----------+--------+------+------+
| 617382 |    60868      |    556514    |  853134  |  1007  |  32  | 1256 |
+--------+---------------+--------------+----------+--------+------+------+


Benchmark Result on Board
=========================

Meanwhile, benchmark results at 276MHz frequency on Alveo U250 board with 2019.2 shell are shown as below:

+---------+---------+---------+----------+------------+-------------------+-------------------+--------------------+--------------------+--------------------+------------+
| Dataset | samples | classes | features | iterations | Spark (4 threads) | Spark (8 threads) | Spark (16 threads) | Spark (32 threads) | Spark (56 threads) | FPGA (:ms) |
+---------+---------+---------+----------+------------+-------------------+-------------------+--------------------+--------------------+--------------------+------------+
|  PUF    | 2000000 |   2     |    64    |     20     | 192548 (61.9X)    | 112903 (36.1X)    | 69806 (22.2X)      | 68548 (21.9X)      | 68080 (21.9X)      | 3078       |
+---------+---------+---------+----------+------------+-------------------+-------------------+--------------------+--------------------+--------------------+------------+
| HIGGS   | 5000000 |   2     |    28    |     100    | 2224074 (147.6X)  | 1401467 (92.9X)   | 873412 (57.9X)     | 601885 (39.8X)     | 590843 (39.1X)     | 15067      |
+---------+---------+---------+----------+------------+-------------------+-------------------+--------------------+--------------------+--------------------+------------+
