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
   :keywords: Vitis Quantitative Finance Library, RNG, SobolRsg, BrownianBridge, TrinomialTree, TreeLattice, 1DMesher, OrnsteinUhlenbeckProcess, StochasticProcess1D, HWModel, G2Model, ECIRModel, CIRModel, VModel, HestonModel, BKModel, BSModel, Interpolation, Distribution, XoShiRo128, Covariance
   :description: L1 module application programming interface reference. 
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials



RNG
-----------------------

Defined in <xf_fintech/rng.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`MT19937<doxid-classxf_1_1fintech_1_1_m_t19937>`
     class :ref:`MT2203<doxid-classxf_1_1fintech_1_1_m_t2203>`
     class :ref:`MT19937IcnRng<doxid-classxf_1_1fintech_1_1_m_t19937_icn_rng>`
     class :ref:`MT19937BoxMullerNomralRng<doxid-classxf_1_1fintech_1_1_m_t19937_box_muller_normal_rng>`
     class :ref:`MT2203IcnRng<doxid-classxf_1_1fintech_1_1_m_t2203_icn_rng>`
     class :ref:`MultiVariateNormalRng<doxid-classxf_1_1fintech_1_1_multi_variate_normal_rng>`

XoShiRo128
----------

Defined in <xf_fintech/xoshiro128.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`XoShiRo128PlusPlus <doxid-classxf_1_1fintech_1_1_xo_shi_ro128_plus_plus>`
     class :ref:`XoShiRo128Plus <doxid-classxf_1_1fintech_1_1_xo_shi_ro128_plus>`
     class :ref:`XoShiRo128StarStar <doxid-classxf_1_1fintech_1_1_xo_shi_ro128_star_star>`


SobolRsg
-----------------------
                                                                                                                                                       
Defined in <xf_fintech/sobol_rsg.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`SobolRsg <doxid-classxf_1_1fintech_1_1_sobol_rsg>`
     class :ref:`SobolRsg1D <doxid-classxf_1_1fintech_1_1_sobol_rsg1_d>`


BrownianBridge
-------------------------
                                                                                                                                                       
Defined in <xf_fintech/brownian_bridge.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`BrownianBridge <doxid-classxf_1_1fintech_1_1_brownian_bridge>`


TrinomialTree
-----------------------
                                                                                                                                                       
Defined in <xf_fintech/trinomial_tree.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`TrinomialTree <doxid-classxf_1_1fintech_1_1_trinomial_tree>`


TreeLattice
-----------------------
                                                                                                                                                       
Defined in <xf_fintech/tree_lattice.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`TreeLattice <doxid-classxf_1_1fintech_1_1_tree_lattice>`


1DMesher
-----------------------
                                                                                                                                                       
Defined in <xf_fintech/fdmmesher.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`1DMesher <doxid-classxf_1_1fintech_1_1_fdm1d_mesher>`


OrnsteinUhlenbeckProcess
----------------------------
                                                                                                                                                       
Defined in <xf_fintech/ornstein_uhlenbeck_process.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`OrnsteinUhlenbeckProcess <doxid-classxf_1_1fintech_1_1_ornstein_uhlenbeck_process>`


StochasticProcess1D
----------------------------
                                                                                                                                                       
Defined in <xf_fintech/stochastic_process.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`StochasticProcess1D <doxid-classxf_1_1fintech_1_1_stochastic_process1_d>`


HWModel
-----------------------
                                                                                                                                                       
Defined in <xf_fintech/hw_model.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`HWModel <doxid-classxf_1_1fintech_1_1_h_w_model>`


G2Model
-----------------------
                                                                                                                                                       
Defined in <xf_fintech/g2_model.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`G2Model <doxid-classxf_1_1fintech_1_1_g2_model>`


ECIRModel
-----------------------
                                                                                                                                                       
Defined in <xf_fintech/ecir_model.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`ECIRModel <doxid-classxf_1_1fintech_1_1_e_c_i_r_model>`


CIRModel
-----------------------
                                                                                                                                                       
Defined in <xf_fintech/cir_model.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`CIRModel <doxid-classxf_1_1fintech_1_1_c_i_r_model>`


VModel
-----------------------
                                                                                                                                                       
Defined in <xf_fintech/v_model.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`VModel <doxid-classxf_1_1fintech_1_1_v_model>`


HestonModel
-----------------------
                                                                                                                                                       
Defined in <xf_fintech/heston_model.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`HestonModel <doxid-classxf_1_1fintech_1_1_heston_model>`


BKModel
-----------------------
                                                                                                                                                       
Defined in <xf_fintech/bk_model.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`BKModel <doxid-classxf_1_1fintech_1_1_b_k_model>`


BSModel
--------------------------------
                                                                                                                                                       
Defined in <xf_fintech/bs_model.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`BSModel <doxid-classxf_1_1fintech_1_1_b_s_model>`

PCA
--------------------------------

Defined in <xf_fintech/pca.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`PCA <doxid-classxf_1_1fintech_1_1_p_c_a>`


BicubicSplineInterpolation
--------------------------------

Defined in <xf_fintech/bicubic_spline_interpolation.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`BicubicSplineInterpolation <doxid-classxf_1_1fintech_1_1_bicubic_spline_interpolation>`

CubicInterpolation
------------------

Defined in <xf_fintech/cubic_interpolation.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`CubicInterpolation <doxid-classxf_1_1fintech_1_1_cubic_interpolation>`

BinomialDistribution
--------------------

Defined in <xf_fintech/binomial_distribution.hpp>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. ref-code-block:: cpp
     :class: overview-code-block

     class :ref:`BinomialDistribution <doxid-classxf_1_1fintech_1_1_binomial_distribution>`

.. toctree::
   :maxdepth: 2

.. include:: ../rst_L1/namespace_xf_fintech.rst
    :start-after: FunctionSection

