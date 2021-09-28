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

.. _guide-kMeansPredict:

********************************
K-Means (Predict)
********************************

.. toctree::
   :hidden:
   :maxdepth: 2

This document describes the structure and execution of kMeansPredict.

.. image:: /images/kMeanPredict.png
   :alt: k-means prediction Structure
   :width: 80%
   :align: center

kMeansPredict provides prediction the cluster index for each sample, in which the centers are stored in an array
 whose 1st dimension should partition in its definition. 
In order to achieve to accelertion prediction, DV elements in a sample are input at the same time and used for computing distance with KU centers.
The static configures are set by template parameters and dynamic by arguments of the API in which dynamic ones should not greater than static ones. 

There are Applicable conditions:

1.All centers are stored in local buffer, so Dim*Kcluster should less than a fixed value. For example, Dim*Kcluster<=1024*1024 for centers with float stored in URAM and 1024*512 for double on U250.

2.KU and DV should be configured properly due to limitation to local memory. For example,KU*DV=128 when centers are stored in URAMon U250.

3.The dynamic confugures should close to static ones in order to void unuseful computing inside.

