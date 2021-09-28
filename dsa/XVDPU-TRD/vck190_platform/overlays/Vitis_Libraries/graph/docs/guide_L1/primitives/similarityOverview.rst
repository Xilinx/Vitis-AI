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


*************************************************
Similarity Primitives
*************************************************

Overview
========

In graph theory, a similarity measure or similarity function is an real-valued function which indicates whether two vertices are similar to each other (from wikipedia).
The API will provide the two commonly used similarity function which are Jaccard Similarity and Cosine Similarity.

Jaccard Similarity Algorithm
============================
Jaccard similarity is defined as the size of the intersection divided by the size of the union of the sample sets:

.. image:: /images/jaccard_similarity_formula.PNG
   :alt: Jaccard Similarity Formula
   :width: 30%
   :align: center

If both `A` and `B` are empty, the `J(A, B)` is defined as 1. Care must be taken if the union of `A` and `B` is zero or infinite. In that case, `J(A, B)` is not defined. 
Jaccard Similarity is widely used in object detection as a judegment of similarity between the detection rectangular and the ground truth.

Cosine Similarity Algorithm
===========================
Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space. 
It is defined to equal the cosine of the angle between them, which is also the same as the inner product of the same vectors normalized to both have length 1. The result of cosine value is between 0 and 1.
If the result is 1, it indiceate the two vector are exactly the same.
The formula of cosine similarity is:

.. image:: /images/cosine_similarity_formula.PNG
   :alt: Cosine Similarity Formula
   :width: 30%
   :align: center

Cosine Similarity is mainly used for measuring the distance between different text file. It is advantageous because even if the two similar documents are far apart by the Euclidean distance because of the size they could still have a smaller angle between them.

Implemention
============

.. toctree::
    :maxdepth: 1

   generalSimilarity.rst
   sparseSimilarity.rst
   denseSimilarity.rst

