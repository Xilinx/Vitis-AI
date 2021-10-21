.. index:: pair: namespace; decimate_hb
.. _doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__hb:
.. _cid-xf::dsp::aie::fir::decimate_hb:

namespace decimate_hb
=====================

.. toctree::
	:hidden:

	class_xf_dsp_aie_fir_decimate_hb_fir_decimate_hb_graph.rst



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
	    unsigned int TP_NUM_OUTPUTS = 1
	    >
	class :ref:`fir_decimate_hb_graph<doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1decimate__hb_1_1fir__decimate__hb__graph>` 

	// global variables

	port <output> :ref:`out2<doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__hb_1a7032e8bd9fc654c437ecf826a6a9bf56>`
	port <input> :ref:`in2<doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__hb_1a7be6a3e0e1930451baf4a78af27edf61>`
	port <input> :ref:`coeff<doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__hb_1a6185d284258e4ffc1788449e16d603f7>`

Global Variables
----------------

.. _doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__hb_1a7032e8bd9fc654c437ecf826a6a9bf56:
.. _cid-xf::dsp::aie::fir::decimate_hb::out2:
.. ref-code-block:: cpp
	:class: title-code-block

	port <output> out2

A window API of TP_INPUT_WINDOW_VSIZE/2 samples of TT_DATA type.

.. _doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__hb_1a7be6a3e0e1930451baf4a78af27edf61:
.. _cid-xf::dsp::aie::fir::decimate_hb::in2:
.. ref-code-block:: cpp
	:class: title-code-block

	port <input> in2

A second input window API of TP_INPUT_WINDOW_VSIZE samples of TT_DATA type. This window should be a clone of the first input window, holding the same data, but in a different RAM banks so as to eliminate wait states through read contentions.

.. _doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1decimate__hb_1a6185d284258e4ffc1788449e16d603f7:
.. _cid-xf::dsp::aie::fir::decimate_hb::coeff:
.. ref-code-block:: cpp
	:class: title-code-block

	port <input> coeff

A Run-time Parameter API containing the set of coefficient values. A change to these values will be detected and will cause a reload of the coefficients within the kernel or kernels to be used on the next data window. This port is present only when TP_USE_COEFF_RELOAD is set to 1.

