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

.. result:

*****************
Benchmark Result
*****************

Text Performance
================

Log Analyzer
~~~~~~~~~~~~

The case `L2/demos/text/log_analyzer` is an integration frame included 3 part: Grok, GeoIP and JsonWriter. 
It supports software and hardware emulation as well as running hardware accelerators on the Alveo U200.

- Input log: http://www.almhuette-raith.at/apache-log/access.log (1.2GB)
- logAnalyzer Demo execute time: 0.99 s, throughput: 1.2 GB/s
- Baseline `ref_result/ref_result.cpp` execute time: 53.1 s, throughput: 22.6 MB/s
- Accelaration Ratio: 53X

.. note::
    | 1. The baseline version run on Intel(R) Xeon(R) CPU E5-2690 v4, clocked at 2.60GHz.
    | 2. The baseline version is a single thread program.


Duplicate Record Match
~~~~~~~~~~~~~~~~~~~~~~

The case `L2/demos/text/dup_match` is to achieve the function of duplicate recoed matching, which includes modules such as Index, Predicate, Pair, Score, Cluster, etc.
It supports software and hardware emulation as well as running hardware accelerators on the Alveo U50.

- Input file: Randomly generate 10,000,000 lines (about 1GB) of csv file similar to `L2/demos/text/dup_match/data/test.csv` as test input file.
- The Demo execute time 8,215.56 s.
- Baseline (Dedupe Python: `https://github.com/dedupeio/dedupe`) execute time 35,030.751 s
- Accelaration Ratio: 5.1X

.. note::
   | 1. The baseline version run on Intel(R) Xeon(R) CPU E5-2690 v4, clocked at 2.60GHz.
   | 2. The training result of Baseline includes `self.predicate=((TfidfNGramCanopyPredicate: (0.8, Site name), TfidfTextCanopyPredicate: (0.8, Address)), (SimplePredicate: (alphaNumericPredicate, Site name), TfidfTextCanopyPredicate: (0.8, Site name)), (SimplePredicate: (wholeFieldPredicate, Site name), SimplePredicate: (wholeFieldPredicate, Zip)))`.

ML Performance
==============

The performance of FPGA accelerated query execution is compared against Spark running time in local mode.
The result is summarized in the table below.

For both FPGA and C++, time is measured assuming the data is already loaded into CPU main memory.
And Spark times don' include the time of loading data from svm files.

In the table below, Spark numbre is collected with version 2.4.4 on
Intel(R) Xeon(R) CPU E5-2690 v4, clocked at 2.60GHz, 256G RAM memory.


Naive Bayes Classification Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset:

 1 - RCV1 (https://scikit-learn.org/0.18/datasets/rcv1.html)

 2 - webspam (https://chato.cl/webspam/datasets/uk2007/)

 3 - news20 (https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)

+---------+---------+---------+----------+-----------------+------------+-------------+------------+
| Dataset | samples | classes | features | End to End (ms) | Speedupd   | Thread num  | Spark (ms) |
+=========+=========+=========+==========+=================+============+=============+============+
| RCV1    | 697614  |   2     |  47236   | 371             | 12.2       | 56          | 5425       |
+---------+---------+---------+----------+-----------------+------------+-------------+------------+
| webspam | 350000  |   2     |  254     | 214             | 35.3       | 56          | 5848       |
+---------+---------+---------+----------+-----------------+------------+-------------+------------+
| news20  | 19928   |   20    |  62061   | 12              | 391        | 56          | 4308       |
+---------+---------+---------+----------+-----------------+------------+-------------+------------+


Linear SVM Classification Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset:

 1 - PUF (https://archive.ics.uci.edu/ml/datasets/Physical+Unclonable+Functions)

 2 - HIGGS   (https://archive.ics.uci.edu/ml/datasets/HIGGS)

+---------+---------+----------+------------+-----------------+----------+------------+------------+
| Dataset | samples | features | iterations | End to End (ms) | Speedup  | Thread num | Spark (ms) |
+=========+=========+==========+============+=================+==========+============+============+
| 1       | 2000000 |    64    |     20     | 3078            | 21.9     | 56         | 68080      |
+---------+---------+----------+------------+-----------------+----------+------------+------------+
| 2       | 5000000 |    28    |     100    | 15067           | 39.1     | 56         | 590843     |
+---------+---------+----------+------------+-----------------+----------+------------+------------+


Decision Tree Classification Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset:

 1 - HEPMASS (https://archive.ics.uci.edu/ml/datasets/HEPMASS) 

 2 - HIGGS   (https://archive.ics.uci.edu/ml/datasets/HIGGS)

+---------+------------+------------+-----------------+----------+------------+------------+
| Dataset | Sample Num | Tree Depth | End-to-End (ms) | Speedup  | Thread num | Spark (ms) |
+=========+============+============+=================+==========+============+============+
| 1       | 7000000    | 9          | 2561.3          | 9        | 56         | 22862      |
+---------+------------+------------+-----------------+----------+------------+------------+
| 2       | 8000000    | 9          | 2908.3          | 9        | 56         | 25196      |
+---------+------------+------------+-----------------+----------+------------+------------+


Decision Tree Regression Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 1 - HEPMASS (https://archive.ics.uci.edu/ml/datasets/HEPMASS) 

 2 - HIGGS   (https://archive.ics.uci.edu/ml/datasets/HIGGS)

 3 - SUSY (https://archive.ics.uci.edu/ml/datasets/SUSY)

 4 - PUF (https://archive.ics.uci.edu/ml/datasets/Physical+Unclonable+Functions)

+---------+------------+------------+-----------------+----------+------------+------------+
| Dataset | Sample Num | Tree Depth | End-to-End (ms) | Speedup  | Thread num | Spark (ms) |
+=========+============+============+=================+==========+============+============+
| 1       | 7000000    | 9          | 22805           | 2.6      | 56         | 8555       |
+---------+------------+------------+-----------------+----------+------------+------------+
| 2       | 8000000    | 9          | 28125           | 2.5      | 56         | 11172      |
+---------+------------+------------+-----------------+----------+------------+------------+
| 3       | 4000000    | 9          | 9717            | 3.2      | 56         | 3018       |
+---------+------------+------------+-----------------+----------+------------+------------+
| 4       | 5000000    | 10         | 16188           | 1.4      | 56         | 11155      |
+---------+------------+------------+-----------------+----------+------------+------------+


Random Forest Classification Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset:

 1 - HEPMASS (https://archive.ics.uci.edu/ml/datasets/HEPMASS) 

 2 - HIGGS   (https://archive.ics.uci.edu/ml/datasets/HIGGS)

+---------+------------+------------+------------+----------------+----------+------------+-----------+
| Dataset | Sample Num | Tree Depth | Tree Num   | End-to-End (s) | Speedup  | Thread num | Spark (s) |
+=========+============+============+============+================+==========+============+===========+
| 1       | 7000000    | 5          | 512        | 61.20          | 10.2     | 28         | 622.30    |
+---------+------------+------------+------------+----------------+----------+------------+-----------+
| 1       | 7000000    | 5          | 1024       | 121.20         | 15.3     | 16         | 1849.724  |
+---------+------------+------------+------------+----------------+----------+------------+-----------+
| 2       | 8000000    | 5          | 512        | 70.30          | 13.3     | 28         | 933.83    |
+---------+------------+------------+------------+----------------+----------+------------+-----------+
| 2       | 8000000    | 5          | 1024       | 138.84         | 15.5     | 16         | 2154      |
+---------+------------+------------+------------+----------------+----------+------------+-----------+


K-Means Clustering Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dataset:

 1 - NIPS Conference Papers (http://archive.ics.uci.edu/ml/datasets/NIPS+Conference+Papers+1987-2015)

 2 - Human Activity Recognition Using Smartphones Data Set (http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)

 3 - US Census Data (1990) Data Set (http://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29)

+---------+--------------+------------+------------+----------------+----------+------------+-----------+
| Dataset | Feature Num  | Sample Num | Center Num | End-to-End (s) | Speedup  | Thread num | Spark (s) |
+=========+==============+============+============+================+==========+============+===========+
| 1       | 5811         | 11463      | 80         | 29.41          | 1.72     | 48         | 50.875    |
+---------+--------------+------------+------------+----------------+----------+------------+-----------+
| 2       | 561          | 7352       | 144        | 2.136          | 2.89     | 48         | 6.19      |
+---------+--------------+------------+------------+----------------+----------+------------+-----------+
| 3       | 68           | 857765     | 2000       | 158.903        | 1.04     | 48         | 166.214   |
+---------+--------------+------------+------------+----------------+----------+------------+-----------+

