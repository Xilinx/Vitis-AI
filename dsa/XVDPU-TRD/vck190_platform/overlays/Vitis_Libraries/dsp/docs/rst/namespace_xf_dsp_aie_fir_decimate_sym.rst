.. index:: pair: namespace; decimate_sym
.. _doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__sym:
.. _cid-xf::dsp::aie::fir::decimate_sym:

namespace decimate_sym
======================

.. toctree::
	:hidden:

	class_xf_dsp_aie_fir_decimate_sym_fir_decimate_sym_graph.rst



.. ref-code-block:: cpp
	:class: overview-code-block

	// classes

	template <
	    typename TT_DATA,
	    typename TT_COEFF,
	    unsigned int TP_FIR_LEN,
	    unsigned int TP_DECIMATE_FACTOR,
	    unsigned int TP_SHIFT,
	    unsigned int TP_RND,
	    unsigned int TP_INPUT_WINDOW_VSIZE,
	    unsigned int TP_CASC_LEN = 1,
	    unsigned int TP_DUAL_IP = 0,
	    unsigned int TP_USE_COEFF_RELOAD = 0,
	    unsigned int TP_NUM_OUTPUTS = 1
	    >
	class :ref:`fir_decimate_sym_graph<doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1decimate__sym_1_1fir__decimate__sym__graph>` 

	// global variables

	port <output> :ref:`out2<doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__sym_1ac6decd383139eedd8dcb0dabc47d0a64>`
	port <input> :ref:`coeff<doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__sym_1a0449266a4c8ca6ec0160a0095295fdab>`

Global Variables
----------------

.. _doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__sym_1ac6decd383139eedd8dcb0dabc47d0a64:
.. _cid-xf::dsp::aie::fir::decimate_sym::out2:
.. ref-code-block:: cpp
	:class: title-code-block

	port <output> out2

A second output window API of TP_INPUT_WINDOW_VSIZE/TP_DECIMATE_FACTOR samples of TT_DATA type.

.. _doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__sym_1a0449266a4c8ca6ec0160a0095295fdab:
.. _cid-xf::dsp::aie::fir::decimate_sym::coeff:
.. ref-code-block:: cpp
	:class: title-code-block

	port <input> coeff

A Run-time Parameter API containing the set of coefficient values. A change to these values will be detected and will cause a reload of the coefficients within the kernel or kernels to be used on the next data window. This port is present only when TP_USE_COEFF_RELOAD is set to 1.

