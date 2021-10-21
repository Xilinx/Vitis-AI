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

.. meta::
   :keywords: fintech, Brownian, Brownian Bridge
   :description: A Brownian bridge is a continuous-time stochastic process whose probability distribution is the conditional probability distribution of a Wiener process.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

*************************
Brownian Bridge Transform
*************************

Overview
========

A Brownian bridge is a continuous-time stochastic process :math:`B(t)` whose probability distribution is the conditional probability distribution of a Wiener process :math:`W(t)` subject to the condition (when standardized) that :math:`W(T) = 0`. More precisely: 

.. math::
  B_{t} := (W_{t} | W_{T} = 0), t \in [0, T]

Generally, a Brownian bridge can be defined as:

Suppose :math:`\{W_{t}\}_{t \in [0,T]}` is an 1-dimensional Brownian motion, :math:`a, b \in \mathbb{R}`, then the process

.. math::
  B^{a,b}_{t} = a\dfrac{T-t}{T} + b\dfrac{t}{T} + (W_{t} - \dfrac{t}{T}W_{T}), t \in [0, T]

is a Brownian bridge from :math:`a` to :math:`b`.

It satisfies

.. math::
   B^{a,b}_{t} \sim \mathcal N (a + \dfrac{t}{T}(b-a), t - \dfrac{t^2}{T})

where :math:`\mathcal{N}` is normal probability distribution.


Theory
------

Suppose :math:`W` is an 1-dimensional Brownian motion, :math:`a,b \in \mathbb{R}, 0 < s < t < u`, then with known :math:`(W_{u}, W_{s})`, the conditional probability distribution is as follows:

.. math::
   W_{t} | (W_{u}=b, W_{s}=a) \sim \mathcal N (\dfrac{(u-t)a+(t-s)b}{u-s},\dfrac{(u-t)(t-s)}{u-s})


The backward simulation method for Brownian bridge is to generate a sequence between :math:`a` and :math:`b`. Suppose we have constructed :math:`k` points :math:`W_{0}, W_{t_{1}}, ..., W_{t_{k-2}}, W_{T}`, we want to generate another point at time :math:`s`, :math:`t_{i} < s < t_{i+1}`. According to Markov and independent increments property of Brownian motion, we have 

.. math::
   W_{s} | (W_{0}=a,...,W_{t_{i}}=x,W_{t_{i+1}}=y,...,W_{T}=b) \sim W_{s} | (W_{t_i} = x, W_{t_{i+1}}=y)

where

.. math::
    W_{s} | (W_{t_i} = x, W_{t_{i+1}}=y) \sim \mathcal N (\dfrac{(t_{i+1}-s)x+(s-t_{i})y}{t_{i+1}-t_{i}},\dfrac{(t_{i+1}-s)(t_{i+}-t_{i})}{t_{i+1}-s})


Generation Algorithm
--------------------

We can design an algorithm for generating Brownian bridge according to the theory above. The backward generation algorithm for Brownian bridge is to generate a sequence between :math:`a` and :math:`b`. A practical strategy is called binary partitioning on :math:`[0, T]`. It is based on a procedure of gradually reducing the grid size to half. Specifically, we start from :math:`T` (level 0), then :math:`T/4, 3T/4`, then :math:`T/8, 3T/8, 5T/8, 7T/8`, etc.

The detailed algorithm:

Input: a sequence of random variate :math:`Z` with standardized normal distribution and length :math:`N=2^{K}`

Output: a Brownian bridge sequence :math:`W` in time range :math:`[0,T]` and length :math:`N`

1. :math:`W_{T} = \sqrt{T}Z`, :math:`W_{0}=0`, :math:`h=T`

2. For k from 1 to :math:`K`

   * :math:`h = h/2`

   * for :math:`j` from 1 to :math:`2^{k-1}`

       :math:`W_{(2j-1)h}=\dfrac{1}{2}(W_{2(j-1)h} + W_{2jh}) + \sqrt{h}Z`


A generalized algorithm for any sequence length :math:`N`:

Input: a sequence of random variate :math:`Z` with standardized normal distribution with length :math:`N`

Output: a Brownian bridge sequence :math:`W` in time range :math:`[0,T]` and length :math:`N`

1. :math:`W_{T} = \sqrt{T}Z_{0}`, :math:`j = 0`

2. Intialize array map[:math:`0..N-1`] as all 0, except for map[:math:`N-1`] = 1

3. For :math:`i` from 1 to :math:`N-1`

   * Find the first unpopulated entry in the map from current position of :math:`j`

   * Find the next populated entry in the map from there, noted as :math:`k`

   * Find the middle position of :math:`j` and :math:`k`, noted as :math:`l`

   * bridgeIndex[:math:`i`] = :math:`l`, leftIndex[:math:`i`] = :math:`j`, rightIndex[:math:`i`] = :math:`k`

   * Move :math:`j` to the right position of :math:`k`, if it is out of map boundary, set :math:`j` to 0

4. For :math:`i` from 1 to :math:`N-1`

   * :math:`l` = bridgeIndex[:math:`i`], :math:`j` = leftIndex[:math:`i`], :math:`k` = rightIndex[:math:`i`]

   * :math:`W_{l}=\dfrac{t_{k}-t_{l}}{t_{k}-t_{j-1}} W_{j-1} + \dfrac{t_{l}-t_{j-1}}{t_{k}-t_{j-1}} W_{k} + \sqrt{\dfrac{(t_{l}-t_{j-1})(t_{k}-t_{l})}{t_{k}-t_{j-1}}}Z_{i}`. (:math:`t_{j-1}` is treated as 0 when j is 0.)


Implementation
==============

The transform function in class `BrownianBridge` transform an input sequence with number `size` to an output sequence. The output data applies to Brownian bridge process.
Based on the algorithm, each point in output sequence is generated by previously calculated point. That is to say, each loop dependents on the output of previous loop.
To eliminate the loop-carried dependency, we divide the loop into 7 rounds. The loop body only depends on previous round, and the loop in each round has no dependence so that the initiation interval (II) could achieve
1.


The detailed steps of our implementation is as follows:

1. Init: calculate the value of begin and end point.
2. Round-1: divide the sequence into two parts, calculate value of the mid-point. Now the number of sub-sequence is 2.
3. Round-2: divide each sub-sequence into two parts, calculate the value of mid-point. Then total number of sub-sequence is 4.
4. Round-3: divide each sub-sequence into two parts, calculate the value of mid-point. Then total number of sub-sequence is 8.
5. :math:`\ldots\ldots`
6. Round-6: divide each sub-sequence into two parts, calculate the value of mid-point. Then total number of sub-sequence is 64.
7. Round-7: in this round, the distance of each dependency is greater than the loop latency. So the II could achieve 1.

A more discrete description is provided by the following figures.
In each round, the order of generated data is from left to right.

.. image:: /images/bb_div.PNG
   :alt: order of generated data
   :width: 80%
   :align: center

Each round shares the same hardware logic named trans_body. It gets begin and end value from the result buffer, then aggregate it with `left_weight` and `right_weight` to get the value of mid-point.
It is present as follows:


.. image:: /images/bb_imp.PNG
   :alt: detailed impl
   :width: 80%
   :align: center


Profiling
---------

The hardware resources for Brownian bridge with sequence length 128:

    +--------------------------+----------+----------+----------+----------+-----------------+
    |          Engines         |   BRAM   |    DSP   | Register |    LUT   | clock period(ns)|
    +--------------------------+----------+----------+----------+----------+-----------------+
    |      Brownian bridge     |    15    |    42    |  16297   |   10246  |       3.080     |
    +--------------------------+----------+----------+----------+----------+-----------------+

The hardware resources for Brownian bridge with sequence length 1024: 

    +--------------------------+----------+----------+----------+----------+-----------------+
    |          Engines         |   BRAM   |    DSP   | Register |    LUT   | clock period(ns)|
    +--------------------------+----------+----------+----------+----------+-----------------+
    |      Brownian bridge     |    19    |    33    |  16219   |   13056  |       3.351     |
    +--------------------------+----------+----------+----------+----------+-----------------+

The correctness of Brownian bridge generation is verified by comparing results with QuantLib Brownian bridge using the same input sequence. The results are identical.
