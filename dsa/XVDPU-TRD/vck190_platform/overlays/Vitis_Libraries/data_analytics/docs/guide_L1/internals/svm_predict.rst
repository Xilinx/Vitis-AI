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

.. _guide-svm_predict:

********************************************************
Internals of svm_predict
********************************************************

.. toctree::
   :hidden:
   :maxdepth: 1

This document describes the structure and execution of svm prediction, implemented as ``svmPredict`` function.

Svm_predict is a function to predict sample's classification by trained weight vector and sample feature vector. 
This function provide a stream in, stream out module to easily get prediction class of sample.

The structure of ``svmPredict`` is described as below. The primitive have two function which are ``getWeight`` and ``getPredictions``.
The input of ``getWeight`` is stream and this part stores weight vector in ram.
``getPredictions`` has a dataflow region including two functions. one is dot_multiply, the other is tag type transform.

.. image:: /images/svm_prediction.png
   :alt: svm_prediction Top Structure
   :align: center

The hardware resource utilization of svm_prediction is shown in the table below (work as 300MHz). 

+--------+----------+--------+------+-----+
|  LUT   |    FF    |  BRAM  | URAM | DSP |
+--------+----------+--------+------+-----+
| 27114  |  37335   |   26   |  0   | 133 |
+--------+----------+--------+------+-----+
