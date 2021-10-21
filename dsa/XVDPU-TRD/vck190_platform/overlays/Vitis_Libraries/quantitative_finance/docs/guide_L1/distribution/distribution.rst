
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

************************
Probability Distribution
************************

Overview
========

In probability theory and statistics, a probability distribution is a mathematical function that provides the probabilities of occurrence of different possible outcomes in an experiment. In more technical terms, the probability distribution is a description of a random phenomenon in terms of the probabilities of events (from wiki). Here, probability distributions implement probability density function (PDF), probability mass funciton (PMF), cumulative distribution function (CDF), and inverse cumulative distribution function (ICDF). The detail are listed below.


.. _tabDist:

.. table:: the detail about the distribution
    :align: center

    +-------------------+-------------+----------+----------+-----------------------+
    |   Distribution    |   PDF/PMF   |    CDF   |   ICDF   |  Reference Algorithm  |
    +-------------------+-------------+----------+----------+-----------------------+
    |     Bernoulli     |      Y      |     Y    |          |    `Bernoulli ref`_   |
    +-------------------+-------------+----------+----------+-----------------------+
    |     Binomial      |      Y      |     Y    |          |    `Binomial ref`_    | 
    +-------------------+-------------+----------+----------+-----------------------+
    |     Normal        |      Y      |     Y    |     Y    |    `Normal ref`_      |
    +-------------------+-------------+----------+----------+-----------------------+
    |     Lognormal     |      Y      |     Y    |     Y    |    `Lognormal ref`_   |
    +-------------------+-------------+----------+----------+-----------------------+
    |     Poisson       |      Y      |     Y    |     Y    |    `Poisson ref`_     |
    +-------------------+-------------+----------+----------+-----------------------+
    |     Gamma         |             |     Y    |          |    `Gamma ref`_       |
    +-------------------+-------------+----------+----------+-----------------------+
    note: "Y" indicates that the function is implemented.

.._`Bernoulli ref`: https://en.wikipedia.org/wiki/Bernoulli_distribution

.._`Binomial ref`: https://en.wikipedia.org/wiki/Binomial_distribution

.._`Normal ref`: https://en.wikipedia.org/wiki/Log-normal_distribution

.._`Lognormal ref`: https://en.wikipedia.org/wiki/Log-normal_distribution

.._`Poisson ref` https://en.wikipedia.org/wiki/Poisson_distribution

.._`Gamma ref`: https://en.wikipedia.org/wiki/Gamma_distribution



.. toctree::
   :maxdepth: 1
