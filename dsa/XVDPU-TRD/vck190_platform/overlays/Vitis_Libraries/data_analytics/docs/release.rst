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

.. _release_note:

Release Note
============

.. toctree::
   :hidden:
   :maxdepth: 1


2021.1
------

The 2021.1 release provide Two-Gram text analytics:

* Two Gram Predicate (TGP) is a search of the inverted index with a term of 2 characters. 
  For a dataset that established an inverted index, it can find the matching id in each record in the inverted index.


2020.2
------

The Data Analytics Library has the following addition in the 2020.2 release:

* **Text Processing APIs.** Two major APIs in this family has been included: the *regular expression match* and *geo-IP lookup*.
  The former API can be used to extract content from unstructured data like logs,
  while the later is often used in processing web logs, to annotate with geographic information by IP address.
  A demo tool that converts Apache HTTP server log in batch into JSON file is provided with the library.
* **DataFrame APIs.** DataFrame is widely popular in-memory data abstraction in data analytics domain,
  the DataFrame write and read APIs should enable data analytics kernel developers to store temporal data
  or interact with open-source software using `Apache Arrow`__ DataFrame more easily.
* **Tree Ensemble Method.** *Random forest* is extended to include regression.
  *Gradient boost tree*, based on boosting method, is added to support both classification and regression.
  Support for *XGBoost on classification and regression* is also included to exploit 2nd order derivative of loss function and regularization.

__ http://arrow.apache.org/


2020.1
------

The 2020.1 release provides a range of HLS primitives for:

* Decision Tree
* Random Forest
* Logistic Regression
* Linear SVM
* Naive Bayes
* Linear Least Square Regression
* LASSO Regression
* Ridge Regression
* K-Means
* Stochastic Gradient Descent Optimizer
* L-BFGS Optimizer
