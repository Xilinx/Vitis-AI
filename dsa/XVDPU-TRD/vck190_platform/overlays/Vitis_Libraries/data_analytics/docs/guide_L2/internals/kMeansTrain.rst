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

.. _guide-kMeansTain:

********************************
Internals of kMeansTaim
********************************

.. toctree::
      :hidden:
   :maxdepth: 2

This document describes the structure and execution of kMeansTrain,
implemented as :ref:`kMeansPredict <cid-xf::ml::kMeansTrain>` function.

.. image:: /images/kMeansTrain.png
      :alt: k-means Tainning Structure
   :width: 80%
   :align: center

kMeansTrain fits new centers based native k-means using the existed samples and initial centers provied by user.
In order to achieve to accelertion training, DV elements in a sample are input at the same time and used for computing distance with KU centers and updating centers.
The static configures are set by template parameters and dynamic by arguments of the API in which dynamic ones should not greater than static ones. 

There are Applicable conditions:

1.Dim*Kcluster should less than a fixed value. For example, Dim*Kcluster<=1024*1024 for centers with float stored in URAM and 1024*512 for double on U250.

2.KU and DV should be configured properly due to limitation to URAM. For example,KU*DV=128 when centers are stored in URAM on U250.

3.The dynamic confugures should close to static ones in order to void unuseful computing inside.

.. CAUTION::
      These Applicable conditions.

Benchmark

The below results are based on:
     1) dataset from UCI;
         a) http://archive.ics.uci.edu/ml/datasets/NIPS+Conference+Papers+1987-2015
         b) http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
         c) http://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29
     2) all data as double are processed;
     3) unroll factors DV=8 and KU=16;
     4) results compared to Spark 2.4.4 and initial centers from Spark to ensure same input;
     5) Spark 2.4.4 is deployed in a server which has 56 processers(Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz)

   
Training Resources(Device: U250)
============================================

====== ======= ======== ========== ======== ======== ====== ======= 
  D       K      LUT     LUTAsMem    REG       BRAM    URAM    DSP  
====== ======= ======== ========== ======== ======== ====== ======= 
 5811    80     295110   50378      371542    339     248     420   
------ ------- -------- ---------- -------- -------- ------ ------- 
 561     144    260716   26016      371344    323     152     420   
------ ------- -------- ---------- -------- -------- ------ ------- 
 68     2000    255119   24295      372487    309     168     425   
====== ======= ======== ========== ======== ======== ====== ======= 

Training Performance(Device: U250)
============================================

====== ======= ======== ============= ============== ============== =============== =============== ============ ===========
  D       K    samples    1 thread      8 threads      16 threads     32 threads      48 threads      fpga         fpga  
                         on spark(s)   on spark(s)     on spark(s)    on spark(s)     on spark(s)    execute(s)   freq(MHz)
====== ======= ======== ============= ============== ============== =============== =============== ============ ===========
 5811    80      11463    93.489         49.857          49.860         48.001          50.875         29.410       202
                          (3.17X)        (1.69X)         (1.63X)        (1.89X)         (1.72X)         (1X)    
------ ------- -------- ------------- -------------- -------------- --------------- --------------- ------------ -----------
 561     144     7352      10.781         6.557           6.546          6.216          6.190          2.136        269
                          (5.04X)        (3.06X)         (3.06X)        (2.91X)         (2.89X)         (1X)
------ ------- -------- ------------- -------------- -------------- --------------- --------------- ------------ -----------
 68     2000    857765    547.001        173.116         170.217       161.169          166.214        158.903      239
                          (3.44X)        (1.08X)         (1.07X)       (1.01X)          (1.04X)         (1X)
====== ======= ======== ============= ============== ============== =============== =============== ============ ===========
