.. index:: pair: namespace; matrix_mult
.. _doxid-namespacexf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult:
.. _cid-xf::dsp::aie::blas::matrix_mult:

namespace matrix_mult
=====================

.. toctree::
	:hidden:

	class_xf_dsp_aie_blas_matrix_mult_ConditionalWidget.rst
	class_xf_dsp_aie_blas_matrix_mult_matrix_mult_graph.rst





.. ref-code-block:: cpp
	:class: overview-code-block

	// classes

	template <
	    unsigned int addWidget,
	    unsigned int windowSize,
	    class widgetClass
	    >
	class :ref:`ConditionalWidget<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1_conditional_widget>` 

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
	class :ref:`matrix_mult_graph<doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1matrix__mult__graph>` 

