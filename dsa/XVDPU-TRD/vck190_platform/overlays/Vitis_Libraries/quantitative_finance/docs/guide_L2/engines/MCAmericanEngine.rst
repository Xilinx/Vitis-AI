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
   :keywords: American, pricing, engine, MCAmericanEnginePricing
   :description: American option can be exercised at any time up until its maturity date.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials



*************************************************
Internal Design of American Option Pricing Engine
*************************************************


Overview
========
The American option can be exercised at any time up until its maturity date. This early exercise feature of American Option gives the investors more freedom to exercise, compared to the European options. 


Theory
========
Because the American option can be exercised anytime prior to the stock's expire date, exercise price at each time step is required in the process of valuing the optimal exercise. More precisely, the pricing process is as follows:

1. Generate independent stock paths
2. Start at maturity :math:`T` and calculate the exercise price.
3. Calculate the exercise price for previous time steps :math:`T-1`, compare it with the last exercise price, select the larger one.
4. Keep rolling back to previous time steps until :math:`t` is :math:`0` and obtain the max exercise price :math:`V_t` as the optimal exercise price.

In the process above, for each time step :math:`t`, conditional expectation :math:`E_t[Y_t|S_t]` of the payoff is computed according to Least-Squares Monte Carlo (LSMC) approach, proposed by Longstaff and Schwartz. Mathematically, these conditional expectations can be expressed as:

.. math::
        E_t[Y_t|S_t] = \sum_{i=0}^{n}a_iB_i(S_t)

where :math:`t` is the time prior to maturity, :math:`B_i(S_t)` is the basis function of the values of actual stock prices :math:`S_t` at time :math:`t`. The constant coefficients of basis functions are written as :math:`a_i` , and acts as weights to the basis functions. The discounted subsequent realized cash flows from continuation called :math:`Y_t`. 

Here we employed polynomial function with weights as the basis functions, so the conditional expectations can be re-written as:

.. math::
        E_t[Y_t|S_t] = a + bS_t + cS_t^2 

where 3 functions: 1, :math:`S_t` and :math:`S_t^2` are employed. The constant coefficients :math:`a`, :math:`b` and :math:`c` are the weights. 

By adding immediate exercise value :math:`E_t(S_t)` to the Equation above, and exchanging the side of elements, the equation is changed to 

.. math::
        a + bS_t + cS_t^2 + dE_t(S_t) = E_t[Y_t|S_t]
    :label: expect_calc 

:math:`a`, :math:`b`, :math:`c`, :math:`d` are constant coefficients. These coefficients need to be known while calculating the optimal exercise price in mcSimulation model. More details of American-style optimal algorithm used in our library refer to **"da Silva, J. N., & Fernández, L. A Monte Carlo Approach to Price American-Bermudan-Style Derivatives."**

.. note::
    Theoretically, the basis functions :math:`B_i(S_t)` can be any complex functions and the number of basis function may be any number. The American Option implemented in our library employs three polynomial functions, namely, 1, :math:`S_t` and :math:`S_t^2`, which proved by Longstaff and Schwartz that works well, and is a typical setup in real implementations.

Implementation
=================
In the Theory Section, the pricing process is described. However, this process can only be executed in the condition of Equation :eq:`expect_calc` is given. This means that the coefficients in Equation :eq:`expect_calc` must be calculated first. This calculation process is called **Calibration** and executes prior to the **Pricing** process. Therefore, in the implementation, the American Monte Carlo Engine contains two processes: **Calibration** and **Pricing**, as illustrated in :numref:`Figure %s <my-figure_overall>`. 

.. _my-figure_overall:
.. figure:: /images/AM/overall_arch.png
        :alt: The McAmericanEngine structure
        :width: 50%
th: 50%
        :align: center

.. hint:: Why two processes are required in MCAmericanEngine and only one process, pricing, is enough for European Option Monte Carlo engine?

    | For European Option, the exercise price of the stork at the last time step :math:`T` needs to be and only need to be calculated. However, for American Option, the exercise price of the stock at all time steps :math:`t, 0 \leq t \leq T` are required to be computed. Therefore, the mathematical model of American Option Monte Carlo employs several basis functions, refer to Equation :eq:`expect_calc`. 
    | In the aspect of basis functions, the European Option employed only one basis function: :math:`S_T`, where :math:`T` is the maturity date. By introducing multiple basis functions, the coefficients are also introduced. Thus, the calibration process is deployed to compute these coefficients. 


Calibration Process
---------------------
Calibration process aims to calculate the coefficients that will be used in the pricing process. 

The detailed calculation process is as follows:

1. Generate uniform random numbers with Mersenne Twister UNiform MT19937 Random Number Generator (RNG) followed by Inverse Cumulative Normal (ICN) uniform random numbers. Thereafter, generate independent stock paths with the uniform random numbers and Black-Sholes path generator. The default paths (samples) number used in the calibration process is 4096. Thus, 4096 random numbers are generated for each time step :math:`t`.  
2. Refer to Equation :eq:`expect_calc`, :math:`a`, :math:`b`, :math:`c`, :math:`d` are the unknown coefficients that should be calculated. We denote these coefficients as :math:`x`, so

.. math::
        x = \begin{bmatrix}
              a \\
              b \\
              c \\
              d
            \end{bmatrix}

The expressions derived from path data :math:`S_t`, :math:`S_t^2` and :math:`E_t` can be obtained for each time step :math:`t`. Here we denote them as :math:`A`. 
   
.. math::
       A = \begin{bmatrix}
              1 \ S_t \  S_t^2 \  E_t(S_t)
           \end{bmatrix}

Equation :eq:`expect_calc` can be re-written as:
  
.. math::
        y = A x
    :label: y=ax 

where :math:`y` is the conditional expectation :math:`E_t(Y_t|S_t)` in Equation :eq:`expect_calc`. To simplify the expression in deduction, it is denoted as :math:`y` in this section. By backward process, this conditional expectation :math:`y` for each time step can be obtained. Which is to say, :math:`y` is actually the optimal exercise price in the period of :math:`t` to :math:`T`. 
Therefore, the problem of computing coefficients is changes to find the solution for Equation :eq:`y=ax`. 

In practical, in this step, we calculate the value of 4 elements in vector :math:`A_t` for each time step :math:`t`. Which is to say, 4 outputs of this stage are :math:`1`, :math:`S_t`, :math:`S_t^2` and :math:`E_t`. The default size of ::math:`A_t` for each time step is 4096 * 4.  

To simplify the process of solving Equation :eq:`y=ax`, matrix data :math:`A` is multiplied with its transform :math:`A^T`. A new 4*4 matrix :math:`B` can be derived:

.. math:: 
    A^T A = B.

Corresponding to the two steps described above, the hardware architecture is shown in :numref:`Figure %s <my-figure_presamples>`.

.. _my-figure_presamples:
.. figure:: /images/AM/presamples.png
        :alt: The McAmericanEngine structure
        :width: 60%
        :align: center

We denote the process from RNG to generate matrix data :math:`B` and exercise price for each timestep :math:`t` as a Monte-Carlo-Model (MCM) in American Option Calibration Process. Each MCM process 1024 data. With a template parameter UN_PATH, more pieces of MCM can be instanced when hardware resources available. 

To connect multiple MCM data, two merger blocks are created: one for merge price data, one for merge matrix :math:`B`. Meanwhile, to guarantee all calibration path data can be executed in a loop when there is not enough MCM available, a soft-layer Merger that accumulates all elements of :math:`B` data is employed. Since these intermediate data need to be accumulated multiple times, a BRAM is used to save and load them. 

3. Once we get the matrix :math:`B`, the singular matrix :math:`\Sigma` of :math:`B` could be obtained by SVD (Singular Value Decomposition).

.. math::
        B = U \Sigma V.

4. Thereafter, use Least-Squares to calculated the coefficients by modifying Equation :eq:`y=ax` to:

.. math::
        y &= A x \\
        A^T y &= A^TA x \\
        A^T y &= Bx 

Until now, matrix :math:`B` and vector :math:`y` are known, coefficients :math:`x` could be calculated. 

.. _my-figure_calib:
.. figure:: /images/AM/calibrate.png
        :alt: The McAmericanEngine structure
        :width: 60%
        :align: center

The implementation of step 3 and steps 4 is shown in :numref:`Figure %s <my-figure_calib>`. Because step 2 generates matrix :math:`B` and price data :math:`y` from timesteps :math:`0` to :math:`T`. Step 3 processes these data in the backward direction, from timesteps :math:`T` to :math:`0`. Considering the amount of data, 4096*timesteps*8*sizeof(DT) and 9*timesteps*8*sizeof(DT), it is impossible to store all the data on FPGA. In the implementation, DDR/HBM memory is used to save these data. Correspondingly, DDR data read/write modules are added to the design.

Besides, notice that since SVD is purely computing dependent, which is pretty slow in the design. Therefore, a template parameter UN_STEP is added to speed up the SVD calculation process.

.. note::
  It is worth mentioning that 4096 is only the default calibrate sample/path size. This number may change by customers' demands. However, the size must be a multiple of 1024.


Pricing Process
-----------------

MCAmericanEnginePricing
```````````````````````````
The theory of the pricing process is actually already introduced in the Theory Section. Similar to the European option engine, mcSimulation framework is employed. Compared to the European option engine, the difference is that it needs to calculate the optimal exercise at all time steps. The detailed implementation process of American engine is drawn as Figure :numref:`Figure %s <my-figure_pricing>` shows.

.. _my-figure_pricing:
.. figure:: /images/AM/pricing.png
        :alt: The McAmericanEngine structure pricing
        :width: 60%
        :align: center

1. Generate uniform random numbers with Mersenne Twister UNiform MT19937 Random Number Generator (RNG) followed by Inverse Cumulative Normal (ICN) uniform random numbers. Thereafter, generate independent stock paths with the uniform random numbers and Black-Sholes path generator. 

2. Refer to Equation :eq:`expect_calc`, calculate the exercise price at time step :math:`T` by utilizing the calculated coefficients :math:`x`.

3. Calculate the exercise price for previous time steps :math:`t`, and take the max exercise price by comparing the immediate exercise price with the held price value (maximum value for time steps :math:`t+1`). 

4. Continue the process, until time step :math:`t` equals 0, optimal exercise price and the standard deviation is obtained.

5. Check if the standard deviation is smaller than the required tolerance defined by customers. If not, repeat step 1-4, until the final optimal exercise price is obtained. 

.. caution::
  Notice that the pricing module also supports another approach of ending pricing process, which utilizes the input parameter *requiredSamples*. When the *requiredSamples* are processed, we assume the output mean result is the final optimal exercise price. This mode is also supported in the American Option Pricing Engine. And the figure above only illustrates the ending approach with the required tolerance.

.. note::
  In the figure, BRAM is used to load coefficients data that calculated from the calibration process. Here may use BRAM or DDR, it depends on the amount of data that need to be stored beforehand. The reason that coefficients data cannot be streamed is that the multi-Monte-Carlo process in pricing always needs to execute multiple times. Each time execution, these need to be loaded. Thus, it is impossible to use *hls::stream*.


MCAmericanEngine APIs 
-------------------------------------------------------
In our library, the MCAmerican Option Pricing with Monte Carlo simulation is provided as an API MCAmericanEngine(). However, due to external memory usage on DDR/HBM and avoiding the designed hardware cross-SLR placed and routed. The American engine option supports two modes:

- **single API version**: use one API to run the whole American option
- **three APIs version**: three APIs/kernels are provided, connecting them on the host side to compose the overall design.
 
The boundary between them is external memory access. For the calibration process, two APIs are provided. Calibration step 1 and 2 are wrapped as one kernel, namely, **MCAmericanEnginePreSamples**. Step 3 and step 4 compose another kernel **MCAmericanEngineCalibrate**. And pricing process as another kernel **MCAmericanEnginePricing** in this library. Because the pricing process is separated as a kernel, the data exchange between the calibration and pricing process may not through the BRAM any more. Thus, in the implementation, DDR/HBM is used as the coefficients data storage memory.


With the three kernels, the kernel level pipeline by shortening the overall execution time could be achieved. However, employing kernel level pipeline requires a complex schedule from the host code side. An illustration of connection 3 kernels as a complete system is given in this part, which can be seen in :numref:`Figure %s <my-figure_am>`. Price data and :math:`B` matrix data are the outputs from kernel MCAmericanEnginePreSamples. For each timestep, path number (default 4096) price data B and x matrix data (a number of 9) need to be saved to DDR or HBM memory. 

.. _my-figure_am:
.. figure:: /images/AM/AM_structure.png
        :alt: McAmericanEngine Vitis project architecture on FPGA
        :width: 60%
        :align: center

Kernel 1 MCAmericanEngineCalibrate reads price data :math:`y` and matrix data :math:`B` from external memory and outputs coefficients to DDR/HBM. The last kernel MCAmericanEnginePricing reads coefficients data from DDR/HBM and saves the final output optimal exercise price to DDR/HBM.

.. hint:: Why the number of matrix B is 9 in DDR/HBM?
   
  The matrix :math:`A` is 4096 * 4 for each timestep when the path number is 4096 (default). The size of its transform matrix :math:`A^T` is 4 * 4096. So, the size of matrix :math:`B` is 4 * 4. However, some elements in :math:`B` are the same, and 9 can represent all 16 data. More precisely, assuming 

  .. math::
 
      A^T = \begin{bmatrix}
            &1\      1\      ...\  1\     ...\  1    \\ 
            &S_0\    S_1\    ...\  S_t\   ...\  S_T  \\
            &S_0^2\  S_1^2\  ...\  S_t^2\ ...\  S_T^2\\
            &E_0\    E_1\    ...\  E_t\   ...\  E_T  
          \end{bmatrix}, \ \  \ \ 
    A = \begin{bmatrix}
            1\  S_0\  S_0^2\  E_0 \\ 
            1\  S_1\  S_1^2\  E_1 \\ 
            ...                   \\
            1\  S_t\  S_t^2\  E_t \\ 
            ...                    \\
            1\  S_T\  S_T^2\  E_T
          \end{bmatrix} \\
      \\ 
      ==> 
      B = A^T \ A = \begin{bmatrix}
                         \sum(1)\     \sum(S_i)\    \sum(S_i^2)\   \sum(E_i) \\
                         \sum(S_i)\   \sum(S_i^2)\  \sum(S_i^3)\   \sum(S_iE_i) \\
                         \sum(S_i^2)\ \sum(S_i^3)\  \sum(S_i^4)\   \sum(S_i^2E_i) \\
                         \sum(E_i)\   \sum(S_iE_i)\ \sum(S_i^2E_i)\ \sum(E_i^2) \\
                        \end{bmatrix} 

  It is evident that some elements are the same. After removing duplicated elements, the following 9 elements of :math:`B` are stored to DDR/HBM each timestep:

  .. math:: 
      B_{save} = \begin{bmatrix}
                 &\sum(1)\    \\
                 &\sum(S_i)\    \\
                 &\sum(S_i^2)\ \sum(S_i^3)\  \sum(S_i^4)\   \\
                 &\sum(E_i)\   \sum(S_iE_i)\ \sum(S_i^2E_i)\ \sum(E_i^2) 
               \end{bmatrix} 




.. caution:: 
  The architecture illustrated above is only an example design. In fact, multiple numbers of kernels, each with a different unroll number (UN) may be deployed. The number of kernels that can be instanced in design depends on the resource/size of the FPGA.

.. toctree::
   :maxdepth: 1
