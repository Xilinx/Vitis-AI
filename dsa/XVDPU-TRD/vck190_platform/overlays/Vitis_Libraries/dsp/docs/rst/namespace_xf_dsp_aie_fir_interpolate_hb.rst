.. index:: pair: namespace; interpolate_hb
.. _doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1interpolate__hb:
.. _cid-xf::dsp::aie::fir::interpolate_hb:

namespace interpolate_hb
========================

.. toctree::
	:hidden:

	class_xf_dsp_aie_fir_interpolate_hb_fir_interpolate_hb_graph.rst



.. ref-code-block:: cpp
	:class: overview-code-block

	// classes

	template <
	    typename TT_DATA,
	    typename TT_COEFF,
	    unsigned int TP_FIR_LEN,
	    unsigned int TP_SHIFT,
	    unsigned int TP_RND,
	    unsigned int TP_INPUT_WINDOW_VSIZE,
	    unsigned int TP_CASC_LEN = 1,
	    unsigned int TP_DUAL_IP = 0,
	    unsigned int TP_USE_COEFF_RELOAD = 0,
	    unsigned int TP_NUM_OUTPUTS = 1,
	    unsigned int TP_UPSHIFT_CT = 0
	    >
	class :ref:`fir_interpolate_hb_graph<doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__hb_1_1fir__interpolate__hb__graph>` 

	// global variables

	port <input> :ref:`coeff<doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1interpolate__hb_1a3178e49c89f8253ed2532f7f70e80fe5>`

Global Variables
----------------

.. _doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1interpolate__hb_1a3178e49c89f8253ed2532f7f70e80fe5:
.. _cid-xf::dsp::aie::fir::interpolate_hb::coeff:
.. ref-code-block:: cpp
	:class: title-code-block

	port <input> coeff

A Run-time Parameter API containing the set of coefficient values. A change to these values will be detected and will cause a reload of the coefficients within the kernel or kernels to be used on the next data window. This port is present only when TP_USE_COEFF_RELOAD is set to 1.

