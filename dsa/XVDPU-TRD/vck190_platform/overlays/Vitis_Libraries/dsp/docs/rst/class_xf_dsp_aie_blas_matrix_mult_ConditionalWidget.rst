.. index:: pair: class; xf::dsp::aie::blas::matrix_mult::ConditionalWidget
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1_conditional_widget:
.. _cid-xf::dsp::aie::blas::matrix_mult::conditionalwidget:

template class xf::dsp::aie::blas::matrix_mult::ConditionalWidget
=================================================================

.. toctree::
	:hidden:

.. code-block:: cpp
	:class: overview-code-block

	#include "matrix_mult_tile_widget.hpp"




.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1_conditional_widget_1a223016f1b343bf46e575aa0c4892dd68:
.. _cid-xf::dsp::aie::blas::matrix_mult::conditionalwidget::portconnect:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1_conditional_widget_1a792d7954dc89be5f0a8450d70e909a59:
.. _cid-xf::dsp::aie::blas::matrix_mult::conditionalwidget::conditionalwidget:
.. _doxid-classxf_1_1dsp_1_1aie_1_1blas_1_1matrix__mult_1_1_conditional_widget_1a8703426172403ccc228cfafad0e2686a:
.. _cid-xf::dsp::aie::blas::matrix_mult::conditionalwidget::create:
.. ref-code-block:: cpp
	:class: overview-code-block

	template <
	    unsigned int addWidget,
	    unsigned int windowSize,
	    class widgetClass
	    >
	class ConditionalWidget

	// typedefs

	typedef connect <window <windowSize>> portConnect

