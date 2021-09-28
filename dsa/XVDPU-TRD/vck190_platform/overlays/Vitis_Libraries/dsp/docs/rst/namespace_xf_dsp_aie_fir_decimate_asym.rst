.. index:: pair: namespace; decimate_asym
.. _doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__asym:
.. _cid-xf::dsp::aie::fir::decimate_asym:

namespace decimate_asym
=======================

.. toctree::
	:hidden:

	class_xf_dsp_aie_fir_decimate_asym_fir_decimate_asym_graph.rst



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
	    unsigned int TP_USE_COEFF_RELOAD = 0,
	    unsigned int TP_NUM_OUTPUTS = 1
	    >
	class :ref:`fir_decimate_asym_graph<doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1decimate__asym_1_1fir__decimate__asym__graph>` 

	// global variables

	port <output> :ref:`out2<doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__asym_1a36911de54cad8d47c4650ed1a7545b2b>`

Global Variables
----------------

.. _doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__asym_1a36911de54cad8d47c4650ed1a7545b2b:
.. _cid-xf::dsp::aie::fir::decimate_asym::out2:
.. ref-code-block:: cpp
	:class: title-code-block

	port <output> out2

A second window API of TP_INPUT_WINDOW_VSIZE/TP_DECIMATE_FACTOR samples of TT_DATA type.

