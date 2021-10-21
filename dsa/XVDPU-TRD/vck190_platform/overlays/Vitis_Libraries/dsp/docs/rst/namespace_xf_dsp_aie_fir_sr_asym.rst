.. index:: pair: namespace; sr_asym
.. _doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1sr__asym:
.. _cid-xf::dsp::aie::fir::sr_asym:

namespace sr_asym
=================

.. toctree::
	:hidden:

	class_xf_dsp_aie_fir_sr_asym_fir_sr_asym_graph.rst



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
	    unsigned int TP_USE_COEFF_RELOAD = 0,
	    unsigned int TP_NUM_OUTPUTS = 1
	    >
	class :ref:`fir_sr_asym_graph<doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1sr__asym_1_1fir__sr__asym__graph>` 

	// global variables

	port <input> :ref:`coeff<doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1sr__asym_1a591e9a22f0837b134f63b7556996aa4e>`
	port <output> :ref:`out2<doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1sr__asym_1a513170149e290c7b42505e94443792ae>`

Global Variables
----------------

.. _doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1sr__asym_1a591e9a22f0837b134f63b7556996aa4e:
.. _cid-xf::dsp::aie::fir::sr_asym::coeff:
.. ref-code-block:: cpp
	:class: title-code-block

	port <input> coeff

A Run-time Parameter API containing the set of coefficient values. A change to these values will be detected and will cause a reload of the coefficients within the kernel or kernels to be used on the next data window. This port is present only when TP_USE_COEFF_RELOAD is set to 1.

.. _doxid-namespacexf_1_1dsp_1_1aie_1_1fir_1_1sr__asym_1a513170149e290c7b42505e94443792ae:
.. _cid-xf::dsp::aie::fir::sr_asym::out2:
.. ref-code-block:: cpp
	:class: title-code-block

	port <output> out2

A second output port. This features exists because windows may not be broadcast and because copying a window after creation is less efficient than creating a second output at source.

