
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

*************
Interpolation
*************

Overview
========

In the mathematics, interpolation can be viewed as an estimate, a method of constructing new data points within a discrete set of known data points. Here a variety of interpolation methods are implemented, including linear interpolation, cubic interpolation, and bicubic interpolation.

Algorithm & Implementation
==========================

Linear interpolation
--------------------

There are two points :math:`A\left ( x_{A},y_{A} \right )` and :math:`B\left ( x_{B},y_{B} \right )` in the two-dimensional coordinate system. The linear interpolation calculates the value of :math: `x\epsilon \left [ x_{B}, x_{B} \right ]` on the AB line. Its expression is:

.. math::

   y=y_{A}+\frac{y_{B}-y_{A}}{x_{B}-x_{A}}\times \left ( x-x_{A} \right )

The implementation is very simple and not elaborated here. For more information, please check out source code and `linear`_.

.._`linear`: https://en.wikipedia.org/wiki/Linear_interpolation

Cubic interpolation
-------------------

The cubic interpolation algorithm is in `cubic`_, and The cubic interpolation implementation is in source code.

.._`cubic`: https://www.paulinternet.nl/?page=bicubic

Bicubic spline interpolation
----------------------------

The bicubic spline interpolation is an extension of cubic interpolation for interpolating data points on a two-dimensional regular grid. For more algorithm details, please see `bicubic`_. Its implementation is essential to perform interpolation according to the x-direction, and then to interpolate in the y-direction. Then the desired result can be obtained.

.._`bicubic`: https://en.wikipedia.org/wiki/Bicubic_interpolation

As shown in the figure below, the interpolation implementation includes 2 parts: `init` and `calcu`.

1. The `init` mainly implements parameter pre-processing and some functions of cubic spline interpolation in x-direction.
2. The `calcu` completes cubic spline interpolation in the x-direction according to the input x and results of the `init`, and then completes cubic spline interpolation in the y-direction according to the obtained result and the input y to obtain an output result.

.. figure:: /images/bicubic.png
    :alt: Bicubic spline interpolation
    :width: 80%
    :align: center
    

.. toctree::
   :maxdepth: 1
