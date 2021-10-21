.. index:: pair: class; xf::dsp::aie::fir::interpolate_hb::fir_interpolate_hb_graph
.. _doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__hb_1_1fir__interpolate__hb__graph:
.. _cid-xf::dsp::aie::fir::interpolate_hb::fir_interpolate_hb_graph:

template class xf::dsp::aie::fir::interpolate_hb::fir_interpolate_hb_graph
==========================================================================

.. toctree::
	:hidden:

.. code-block:: cpp
	:class: overview-code-block

	#include "fir_interpolate_hb_graph.hpp"


Overview
~~~~~~~~

fir_interpolate_hb is a Halfband Interpolation FIR filter

These are the templates to configure the halfband interpolator FIR class.



.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - TT_DATA

        - describes the type of individual data samples input to and output from the filter function. This is a typename and must be one of the following: int16, cint16, int32, cint32, float, cfloat.

    *
        - TT_COEFF

        - 
          describes the type of individual coefficients of the filter taps. It must be one of the same set of types listed for TT_DATA and must also satisfy the following rules:
          
          * Complex types are only supported when TT_DATA is also complex.
          
          * 32 bit types are only supported when TT_DATA is also a 32 bit type,
          
          * TT_COEFF must be an integer type if TT_DATA is an integer type
          
          * TT_COEFF must be a float type if TT_DATA is a float type.

    *
        - TP_FIR_LEN

        - is an unsigned integer which describes the number of taps in the filter. TP_FIR_LEN must be in the range 4 to 240 inclusive and must satisfy (TP_FIR_LEN +1)/4 = N where N is a positive integer.

    *
        - TP_SHIFT

        - is describes power of 2 shift down applied to the accumulation of FIR terms before output. TP_SHIFT must be in the range 0 to 61.

    *
        - TP_RND

        - describes the selection of rounding to be applied during the shift down stage of processing. TP_RND must be in the range 0 to 7 where 0 = floor (truncate) eg. 3.8 Would become 3. 1 = ceiling e.g. 3.2 would become 4. 2 = round to positive infinity. 3 = round to negative infinity. 4 = round symmetrical to infinity. 5 = round symmetrical to zero. 6 = round convergent to even. 7 = round convergent to odd. Modes 2 to 7 round to the nearest integer. They differ only in how they round for values of 0.5.

    *
        - TP_INPUT_WINDOW_VSIZE

        - describes the number of samples in the window API used for input to the filter function. The number of values in the output window will be TP_INPUT_WINDOW_VSIZE multiplied by 2 by virtue the halfband interpolation factor. Note: Margin size should not be included in TP_INPUT_WINDOW_VSIZE.

    *
        - TP_CASC_LEN

        - describes the number of AIE processors to split the operation over. This allows resource to be traded for higher performance. TP_CASC_LEN must be in the range 1 (default) to 9.

    *
        - TP_DUAL_IP

        - is an implementation trade-off between performance and data bank resource. When set to 0, the FIR performance may be limited by load contention. When set to 1, two ram banks are used for input.

    *
        - TP_USE_COEFF_RELOAD

        - 
          allows the user to select if runtime coefficient reloading should be used. This currently is only available for single kernel filters. When defining the parameter:
          
          * 0 = static coefficients, defined in filter constructor
          
          * 1 = reloadable coefficients, passed as argument to runtime function This is the class for the halfband interpolator FIR graph

.. ref-code-block:: cpp
	:class: overview-code-block

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
	class fir_interpolate_hb_graph: public graph

	// fields

	port <input> :ref:`in<doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__hb_1_1fir__interpolate__hb__graph_1a405823c8d8ea0cb1bcb011d9703ba850>`
	port <output> :ref:`out<doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__hb_1_1fir__interpolate__hb__graph_1ab27c57884c25a5305df92157e300807e>`

Fields
------

.. _doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__hb_1_1fir__interpolate__hb__graph_1a405823c8d8ea0cb1bcb011d9703ba850:
.. _cid-xf::dsp::aie::fir::interpolate_hb::fir_interpolate_hb_graph::in:
.. ref-code-block:: cpp
	:class: title-code-block

	port <input> in

The input data to the function. This input is a window API of samples of TT_DATA type. The number of samples in the window is described by TP_INPUT_WINDOW_VSIZE. Note: Margin is added internally to the graph, when connecting input port with kernel port. Therefore, margin should not be added when connecting graph to a higher level design unit. Margin size (in Bytes) equals to TP_FIR_LEN rounded up to a nearest multiple of 32 bytes.

.. _doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__hb_1_1fir__interpolate__hb__graph_1ab27c57884c25a5305df92157e300807e:
.. _cid-xf::dsp::aie::fir::interpolate_hb::fir_interpolate_hb_graph::out:
.. ref-code-block:: cpp
	:class: title-code-block

	port <output> out

A window API of TP_INPUT_WINDOW_VSIZE*2 samples of TT_DATA type.


Methods
~~~~~~~

.. FunctionSection

.. _doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__hb_1_1fir__interpolate__hb__graph_1a5fd8b7b442d95e3aa53a87804bce5f8b:
.. _cid-xf::dsp::aie::fir::interpolate_hb::fir_interpolate_hb_graph::getkernels:

getKernels
----------


.. ref-code-block:: cpp
	:class: title-code-block

	kernel* getKernels ()

Access function to get pointer to kernel (or first kernel in a chained configuration).

.. _doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__hb_1_1fir__interpolate__hb__graph_1a322623376990401b7391a43fdb7485bb:
.. _cid-xf::dsp::aie::fir::interpolate_hb::fir_interpolate_hb_graph::fir_interpolate_hb_graph:

fir_interpolate_hb_graph
------------------------


.. ref-code-block:: cpp
	:class: title-code-block

	fir_interpolate_hb_graph (const std::vector <TT_COEFF>& taps)

This is the constructor function for the halfband interpolator FIR graph.



.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - taps

        - - a pointer to the array of taps values of type TT_COEFF. The taps array must be supplied in a compressed form for this halfband application, i.e. taps[] = {c0, c2, c4, ..., cN, cCT} where N = (TP_FIR_LEN+1)/4 and cCT is the center tap. For example, a 7-tap halfband interpolator might use coeffs (1, 0, 2, 5, 2, 0, 1). This would be input as taps[]= {1,2,5} since the context of halfband interpolation allows the remaining coefficients to be inferred.

