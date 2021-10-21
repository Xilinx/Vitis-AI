.. index:: pair: class; xf::dsp::aie::fir::interpolate_asym::fir_interpolate_asym_graph
.. _doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__asym_1_1fir__interpolate__asym__graph:
.. _cid-xf::dsp::aie::fir::interpolate_asym::fir_interpolate_asym_graph:

template class xf::dsp::aie::fir::interpolate_asym::fir_interpolate_asym_graph
==============================================================================

.. toctree::
	:hidden:

.. code-block:: cpp
	:class: overview-code-block

	#include "fir_interpolate_asym_graph.hpp"


Overview
~~~~~~~~

fir_interpolate_asym is an Asymmetric Interpolation FIR filter

These are the templates to configure the Asymmetric Interpolation FIR class.



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

        - is an unsigned integer which describes the number of taps in the filter.

    *
        - TP_INTERPOLATE_FACTOR

        - is an unsigned integer which describes the interpolation factor of the filter. TP_INTERPOLATE_FACTOR must be in the range 1 to 16.

    *
        - TP_SHIFT

        - is describes power of 2 shift down applied to the accumulation of FIR terms before output. TP_SHIFT must be in the range 0 to 61.

    *
        - TP_RND

        - describes the selection of rounding to be applied during the shift down stage of processing. TP_RND must be in the range 0 to 7 where 0 = floor (truncate) eg. 3.8 Would become 3. 1 = ceiling e.g. 3.2 would become 4. 2 = round to positive infinity. 3 = round to negative infinity. 4 = round symmetrical to infinity. 5 = round symmetrical to zero. 6 = round convergent to even. 7 = round convergent to odd. Modes 2 to 7 round to the nearest integer. They differ only in how they round for values of 0.5.

    *
        - TP_INPUT_WINDOW_VSIZE

        - describes the number of samples in the window API used for input to the filter function. The number of values in the output window will be TP_INPUT_WINDOW_VSIZE multipled by TP_INTERPOLATE_FACTOR. Note: Margin size should not be included in TP_INPUT_WINDOW_VSIZE.

    *
        - TP_CASC_LEN

        - describes the number of AIE processors to split the operation over. This allows resource to be traded for higher performance. TP_CASC_LEN must be in the range 1 (default) to 9.

    *
        - TP_USE_COEFF_RELOAD

        - 
          allows the user to select if runtime coefficient reloading should be used. This currently is only available for single kernel filters. When defining the parameter:
          
          * 0 = static coefficients, defined in filter constructor
          
          * 1 = reloadable coefficients, passed as argument to runtime function

.. ref-code-block:: cpp
	:class: overview-code-block

	template <
	    typename TT_DATA,
	    typename TT_COEFF,
	    unsigned int TP_FIR_LEN,
	    unsigned int TP_INTERPOLATE_FACTOR,
	    unsigned int TP_SHIFT,
	    unsigned int TP_RND,
	    unsigned int TP_INPUT_WINDOW_VSIZE,
	    unsigned int TP_CASC_LEN = 1,
	    unsigned int TP_USE_COEFF_RELOAD = 0,
	    unsigned int TP_NUM_OUTPUTS = 1
	    >
	class fir_interpolate_asym_graph: public graph

	// fields

	port <input> :ref:`in<doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__asym_1_1fir__interpolate__asym__graph_1a9f75406de17ad6a0ad636dff938336a2>`
	port <output> :ref:`out<doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__asym_1_1fir__interpolate__asym__graph_1a9da512f2686d42bae8ca8123bbc7d893>`

Fields
------

.. _doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__asym_1_1fir__interpolate__asym__graph_1a9f75406de17ad6a0ad636dff938336a2:
.. _cid-xf::dsp::aie::fir::interpolate_asym::fir_interpolate_asym_graph::in:
.. ref-code-block:: cpp
	:class: title-code-block

	port <input> in

The input data to the function. This input is a window API of samples of TT_DATA type. The number of samples in the window is described by TP_INPUT_WINDOW_VSIZE. Note: Margin is added internally to the graph, when connecting input port with kernel port. Therefore, margin should not be added when connecting graph to a higher level design unit. Margin size (in Bytes) equals to TP_FIR_LEN rounded up to a nearest multiple of 32 bytes.

.. _doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__asym_1_1fir__interpolate__asym__graph_1a9da512f2686d42bae8ca8123bbc7d893:
.. _cid-xf::dsp::aie::fir::interpolate_asym::fir_interpolate_asym_graph::out:
.. ref-code-block:: cpp
	:class: title-code-block

	port <output> out

A window API of TP_INPUT_WINDOW_VSIZE*TP_INTERPOLATE_FACTOR samples of TT_DATA type.


Methods
~~~~~~~

.. FunctionSection

.. _doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__asym_1_1fir__interpolate__asym__graph_1ae16ca566636eb46c872c5b6b554f1cee:
.. _cid-xf::dsp::aie::fir::interpolate_asym::fir_interpolate_asym_graph::getkernels:

getKernels
----------


.. ref-code-block:: cpp
	:class: title-code-block

	kernel* getKernels ()

Access function to get pointer to kernel (or first kernel in a chained configuration).

.. _doxid-classxf_1_1dsp_1_1aie_1_1fir_1_1interpolate__asym_1_1fir__interpolate__asym__graph_1ae4c57192763cb63d9a62eac519684a61:
.. _cid-xf::dsp::aie::fir::interpolate_asym::fir_interpolate_asym_graph::fir_interpolate_asym_graph:

fir_interpolate_asym_graph
--------------------------


.. ref-code-block:: cpp
	:class: title-code-block

	fir_interpolate_asym_graph (const std::vector <TT_COEFF>& taps)

This is the constructor function for the Asymmetric Interpolation FIR graph.



.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - taps

        - - a pointer to the array of taps values of type TT_COEFF.

