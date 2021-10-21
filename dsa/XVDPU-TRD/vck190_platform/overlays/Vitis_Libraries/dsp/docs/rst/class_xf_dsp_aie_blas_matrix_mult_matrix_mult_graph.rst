.. index:: pair: class; xf::dsp::aie::blas::matrix_mult::matrix_mult_graph
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph:

template class xf::dsp::aie::blas::matrix_mult::matrix_mult_graph
=================================================================

.. toctree::
	:hidden:

	struct_xf_dsp_aie_blas_matrix_mult_matrix_mult_graph_no_kernel.rst

.. code-block:: cpp
	:class: overview-code-block

	#include "matrix_mult_graph.hpp"


Overview
~~~~~~~~

:ref:`matrix_mult <doxid-namespacexf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult>` performs a GEneral Matrix Multiply (GEMM), taking two input matrices of configurable dimensions and data type.

These are the templates to configure the Matrix Multiply graph class.



.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - TT_DATA_A

        - describes the type of individual data samples input of Matrix A to the gemm function. This is a typename and must be one of the following: int16, cint16, int32, cint32, float, cfloat.

    *
        - TT_DATA_B

        - 
          describes the type of individual data samples input of Matrix B to the gemm function. This is a typename and must be one of the following: int16, cint16, int32, cint32, float, cfloat. The following rules apply:
          
          * must be an integer type if TT_DATA_A is an integer type
          
          * must be a float type if TT_DATA_A is a float type.

    *
        - TP_DIM_A

        - is an unsigned integer which describes the number of elements along the unique dimension (rows) of Matrix A.

    *
        - TP_DIM_AB

        - is an unsigned integer which describes the number of elements along the common dimension of Matrix A (columns) and Matrix B (rows).

    *
        - TP_DIM_B

        - is an unsigned integer which describes the number of elements along the unique dimension (columns) of Matrix B.

    *
        - TP_SHIFT

        - is describes power of 2 shift down applied to the accumulation of product terms before each output. TP_SHIFT must be in the range 0 to 61.

    *
        - TP_RND

        - describes the selection of rounding to be applied during the shift down stage of processing. TP_RND must be in the range 0 to 7 where 0 = floor (truncate) eg. 3.8 Would become 3. 1 = ceiling e.g. 3.2 would become 4. 2 = round to positive infinity. 3 = round to negative infinity. 4 = round symmetrical to infinity. 5 = round symmetrical to zero. 6 = round convergent to even. 7 = round convergent to odd. Modes 2 to 7 round to the nearest integer. They differ only in how they round for values of 0.5.

    *
        - TP_DIM_A_LEADING

        - describes the scheme in which the data should be stored in memory. ROW_MAJOR = 0, COL_MAJOR = 1. Note, a COL_MAJOR matrix can be transposed to become a ROW_MAJOR matrix.

    *
        - TP_DIM_B_LEADING

        - describes the scheme in which the data should be stored in memory. ROW_MAJOR = 0, COL_MAJOR = 1.

    *
        - TP_DIM_OUT_LEADING

        - describes the scheme in which the data should be stored in memory. ROW_MAJOR = 0, COL_MAJOR = 1.

    *
        - TP_ADD_TILING_A

        - describes wether or not to add an additional kernel to rearrange the matrix samples into their required position. Setting this option to 0 indicates that the re-arrangement will be done externally to the AIE matrix multiply graph.

    *
        - TP_ADD_TILING_B

        - describes wether or not to add an additional kernel to rearrange the matrix samples into their required position. Setting this option to 0 indicates that the re-arrangement will be done externally to the AIE matrix multiply graph.

    *
        - TP_ADD_DETILING_OUT

        - describes wether or not to add an additional kernel to rearrange the matrix samples into their required position. Setting this option to 0 indicates that the re-arrangement will be done externally to the AIE matrix multiply graph.

    *
        - TP_INPUT_WINDOW_VSIZE_A

        - describes the number of samples in the window API used for input to Matrix A. It must be of size TP_DIM_A*TP_DIM_AB*N. Typical use has N=1, however N>1 can be utilised to minimise overhead of window API. This parameter is optional and has a default value of TP_DIM_A*TP_DIM_AB (N=1).

    *
        - TP_INPUT_WINDOW_VSIZE_B

        - describes the number of samples in the window API used for input to Matrix B. It must be of size TP_DIM_B*TP_DIM_AB*M. Typical use has M=1, however M>1 can be utilised to minimise overhead of window API. This parameter is optional and has a default value of TP_DIM_B*TP_DIM_AB (M=1). Note, the output window will be of size: (TP_INPUT_WINDOW_VSIZE_A/TP_DIM_AB * TP_INPUT_WINDOW_VSIZE_B/TP_DIM_AB). When N and M is 1, output window size will be TP_DIM_A * TP_DIM_B.

    *
        - TP_CASC_LEN

        - describes the number of AIE Tiles to split the GEMM operation into. TP_CASC_LEN splits the operation over TP_DIM_AB, where each kernel utilises the cascade stream to pass partial accumulation results to the next kernel. In effect, dot(A,B) + C. Note, it is also possible to tile the operation over multiple AIE tiles by instantiating multiple GEMM graphs with smaller dimensions.

.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1abccf1ef712e4b6b3bfe23ed8b1873737:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::matmultcasc:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1af3cdb62777313eda4c88210a7dd4fdb9:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::onlymatmult:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a8cb20a693747894fa05dfdc4378afa95:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::firstmatmult:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1aa965c789e1b229e55b48ffd265c8263a:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::lastmatmult:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a22c2eb185811969092e6f7cc1fef7fa0:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::middlematmult:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a1d674f3554c6b913055fc14c2db1a8ef:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::tilerclassa:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a5c7d5c145fecd27b4f20269b75a5bfc5:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::tilerclassb:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a9f074207d5bff9b927df07044ce8f7fd:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::detilerclassout:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1af0ecaba2f97c178b441b887a999471c9:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::tileaconditional:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a43eb9c071ffdeda49d025bc462b949c7:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::tilebconditional:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1aee63d336b5fcbc92501ca3caa05bf7dc:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::detileoutconditional:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a30eab2299711ef1ad84e9587e2dd4663:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::inb:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1aa360b383ac83de6b03dd65b3ff10b098:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::tilingscheme:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1ae7da2aa624a49c357ab567cbec73e5d9:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::dimaperkernel:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1ac1bdda607e0c624aae9ab1788d671c69:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::dimbperkernel:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1aff0746e03f6bf938b88640c8409251c2:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::isredundanttilera:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a9fd601c3b58a82857be86e77702a46cb:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::isredundanttilerb:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a3a3d744e5729ca9e30483f89380125a3:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::isredundanttilerout:
.. ref-code-block:: cpp
	:class: overview-code-block

	template <
	    typename TT_DATA_A,
	    typename TT_DATA_B,
	    unsigned int TP_DIM_A,
	    unsigned int TP_DIM_AB,
	    unsigned int TP_DIM_B,
	    unsigned int TP_SHIFT,
	    unsigned int TP_RND,
	    unsigned int TP_DIM_A_LEADING = ROW_MAJOR,
	    unsigned int TP_DIM_B_LEADING = COL_MAJOR,
	    unsigned int TP_DIM_OUT_LEADING = ROW_MAJOR,
	    unsigned int TP_ADD_TILING_A = 1,
	    unsigned int TP_ADD_TILING_B = 1,
	    unsigned int TP_ADD_DETILING_OUT = 1,
	    unsigned int TP_INPUT_WINDOW_VSIZE_A = TP_DIM_A* TP_DIM_AB,
	    unsigned int TP_INPUT_WINDOW_VSIZE_B = TP_DIM_B* TP_DIM_AB,
	    unsigned int TP_CASC_LEN = 1
	    >
	class matrix_mult_graph: public graph

	// typedefs

	typedef matrix_mult <TT_DATA_A, TT_DATA_B, TP_DIM_A, (TP_DIM_AB/TP_CASC_LEN), TP_DIM_B, TP_SHIFT, TP_RND, TP_DIM_A_LEADING, TP_DIM_B_LEADING, TP_DIM_OUT_LEADING, (TP_INPUT_WINDOW_VSIZE_A/TP_CASC_LEN), (TP_INPUT_WINDOW_VSIZE_B/TP_CASC_LEN), cascIn, cascOut> matMultCasc
	typedef typename std::conditional < (TP_CASC_LEN==1), :ref:`matMultCasc<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1abccf1ef712e4b6b3bfe23ed8b1873737>` <false, false>, :ref:`no_kernel<doxid-structxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1_1no__kernel>`>::type onlyMatMult
	typedef typename std::conditional < (TP_CASC_LEN> 1), :ref:`matMultCasc<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1abccf1ef712e4b6b3bfe23ed8b1873737>` <false, true>, :ref:`onlyMatMult<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1af3cdb62777313eda4c88210a7dd4fdb9>`>::type firstMatMult
	typedef typename std::conditional < (TP_CASC_LEN> 1), :ref:`matMultCasc<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1abccf1ef712e4b6b3bfe23ed8b1873737>` <true, false>, :ref:`firstMatMult<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a8cb20a693747894fa05dfdc4378afa95>`>::type lastMatMult
	typedef typename std::conditional < (TP_CASC_LEN> 2), :ref:`matMultCasc<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1abccf1ef712e4b6b3bfe23ed8b1873737>` <true, true>, :ref:`lastMatMult<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1aa965c789e1b229e55b48ffd265c8263a>`>::type middleMatMult
	typedef tilerKernelClass <tilingScheme.Atile, tilingScheme.ABtile, :ref:`dimAPerKernel<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1ae7da2aa624a49c357ab567cbec73e5d9>`, (TP_DIM_AB/TP_CASC_LEN), TP_DIM_A_LEADING, TT_DATA_A> TilerClassA
	typedef tilerKernelClass <tilingScheme.ABtile, tilingScheme.Btile, (TP_DIM_AB/TP_CASC_LEN), :ref:`dimBPerKernel<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1ac1bdda607e0c624aae9ab1788d671c69>`, TP_DIM_B_LEADING, TT_DATA_B> TilerClassB
	typedef untilerKernelClass <tilingScheme.Atile, tilingScheme.Btile, :ref:`dimAPerKernel<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1ae7da2aa624a49c357ab567cbec73e5d9>`, :ref:`dimBPerKernel<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1ac1bdda607e0c624aae9ab1788d671c69>`, TP_DIM_OUT_LEADING, outType_t <TT_DATA_A, TT_DATA_B>> DetilerClassOut
	typedef :ref:`ConditionalWidget<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1_conditional_widget>` <:ref:`isRedundantTilerA<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1aff0746e03f6bf938b88640c8409251c2>`?0:TP_ADD_TILING_A, (TP_INPUT_WINDOW_VSIZE_A/TP_CASC_LEN)*sizeof (TT_DATA_A), :ref:`TilerClassA<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a1d674f3554c6b913055fc14c2db1a8ef>`> TileAConditional
	typedef :ref:`ConditionalWidget<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1_conditional_widget>` <:ref:`isRedundantTilerB<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a9fd601c3b58a82857be86e77702a46cb>`?0:TP_ADD_TILING_B, (TP_INPUT_WINDOW_VSIZE_B/TP_CASC_LEN)*sizeof (TT_DATA_B), :ref:`TilerClassB<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a5c7d5c145fecd27b4f20269b75a5bfc5>`> TileBConditional
	typedef :ref:`ConditionalWidget<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1_conditional_widget>` <:ref:`isRedundantTilerOut<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a3a3d744e5729ca9e30483f89380125a3>`?0:TP_ADD_DETILING_OUT, :ref:`dimAPerKernel<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1ae7da2aa624a49c357ab567cbec73e5d9>`*:ref:`dimBPerKernel<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1ac1bdda607e0c624aae9ab1788d671c69>`*sizeof (outType_t <TT_DATA_A, TT_DATA_B>), :ref:`DetilerClassOut<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a9f074207d5bff9b927df07044ce8f7fd>`> DetileOutConditional

	// structs

	struct :ref:`no_kernel<doxid-structxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1_1no__kernel>` 

	// fields

	port <input> :ref:`inA<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1ae43dd310980567ce29f898d045d0e69d>`[TP_CASC_LEN]
	port <input> inB[TP_CASC_LEN]
	port <output> :ref:`out<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1af639b0a2cba2f9a18a63c46ea8bd2224>`
	static constexpr middleMatMult::tilingStruct tilingScheme
	static constexpr unsigned int dimAPerKernel
	static constexpr unsigned int dimBPerKernel
	static constexpr bool isRedundantTilerA
	static constexpr bool isRedundantTilerB
	static constexpr bool isRedundantTilerOut

Fields
------

.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1ae43dd310980567ce29f898d045d0e69d:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::ina:
.. ref-code-block:: cpp
	:class: title-code-block

	port <input> inA [TP_CASC_LEN]

The input data to the function. This input is two windows of samples of TT_DATA_A and TT_DATA_B type. The number of samples in the window is described by TP_INPUT_WINDOW_VSIZE_A and TP_INPUT_WINDOW_VSIZE_B, which are derived from TP_DIM_A, TP_DIM_AB and TP_DIM_B.

.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1af639b0a2cba2f9a18a63c46ea8bd2224:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::out:
.. ref-code-block:: cpp
	:class: title-code-block

	port <output> out

A window API of TP_INPUT_WINDOW_VSIZE_A/TP_DIM_AB * TP_INPUT_WINDOW_VSIZE_B/TP_DIM_AB samples, or simply TP_DIM_A * TP_DIM_B samples of a derived output type.


Methods
~~~~~~~

.. FunctionSection

.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a74a4e6c15db6f041f93d18b4ffad1b4b:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::getkernels:

getKernels
----------


.. ref-code-block:: cpp
	:class: title-code-block

	kernel* getKernels ()

Access function to get pointer to kernel (or first kernel in a chained configuration).

.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph_1a0991e2fd9b81964e95fede06ef00001a:
.. _cid-xf::dsp::aie::blas::matrix_mult::matrix_mult_graph::matrix_mult_graph:

matrix_mult_graph
-----------------


.. ref-code-block:: cpp
	:class: title-code-block

	matrix_mult_graph ()

This is the constructor function for the Matric Multiply graph.

