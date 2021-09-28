.. index:: pair: class; xf::dsp::aie::widget::api_cast::widget_api_cast_graph
.. _doxid-classxf_1_1dsp_1_1aie_1_1widget_1_1api__cast_1_1widget__api__cast__graph:
.. _cid-xf::dsp::aie::widget::api_cast::widget_api_cast_graph:

template class xf::dsp::aie::widget::api_cast::widget_api_cast_graph
====================================================================

.. toctree::
	:hidden:

.. code-block:: cpp
	:class: overview-code-block

	#include "widget_api_cast_graph.hpp"


Overview
~~~~~~~~

widget_api_cast is a design to change the interface between connected components. This component is able to change the stream interface to window interface and vice-versa. In addition, multiple input stream ports may be defined, as well as multiple copies of the window output.

These are the templates to configure the Widget API Cast class.



.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - TT_DATA

        - describes the type of individual data samples input to and output from the function. This is a typename and must be one of the following: int16, cint16, int32, cint32, float, cfloat.

    *
        - TP_IN_API

        - defines the input interface type. 0 = Window, 1 = Stream

    *
        - TP_OUT_API

        - defines the output interface type. 0 = Window, 1 = Stream

    *
        - TP_NUM_INPUTS

        - describes the number of input stream interfaces to be processed. When 2 inputs are configured, whe data will be read sequentially from each.

    *
        - TP_WINDOW_VSIZE

        - describes the number of samples in the window API used if either input or output is a window. Note: Margin size should not be included in TP_INPUT_WINDOW_VSIZE.

    *
        - TP_NUM_OUTPUT_CLONES

        - sets the number of output ports to write the input data to. Note that while input data from multiple ports is independent, data out is not. This is the class for the Widget API Cast graph

.. ref-code-block:: cpp
	:class: overview-code-block

	template <
	    typename TT_DATA,
	    unsigned int TP_IN_API,
	    unsigned int TP_OUT_API,
	    unsigned int TP_NUM_INPUTS,
	    unsigned int TP_WINDOW_VSIZE,
	    unsigned int TP_NUM_OUTPUT_CLONES = 1
	    >
	class widget_api_cast_graph: public graph

	// fields

	port <input> :ref:`in<doxid-classxf_1_1dsp_1_1aie_1_1widget_1_1api__cast_1_1widget__api__cast__graph_1a9317e12e575b6a20ffd942c92cbc086e>`[TP_NUM_INPUTS]
	port <output> :ref:`out<doxid-classxf_1_1dsp_1_1aie_1_1widget_1_1api__cast_1_1widget__api__cast__graph_1a1e868a00b412933606f080c022d6f22b>`[TP_NUM_OUTPUT_CLONES]

Fields
------

.. _doxid-classxf_1_1dsp_1_1aie_1_1widget_1_1api__cast_1_1widget__api__cast__graph_1a9317e12e575b6a20ffd942c92cbc086e:
.. _cid-xf::dsp::aie::widget::api_cast::widget_api_cast_graph::in:
.. ref-code-block:: cpp
	:class: title-code-block

	port <input> in [TP_NUM_INPUTS]

The input data to the function. This input may be stream or window. Data is read from here and written directly to output. When there are multiple input streams, a read from each will occur of the maximum size supported (32 bits) with these 2 being concatenated, before being written to the output(s). Multiple input windows is not supported

.. _doxid-classxf_1_1dsp_1_1aie_1_1widget_1_1api__cast_1_1widget__api__cast__graph_1a1e868a00b412933606f080c022d6f22b:
.. _cid-xf::dsp::aie::widget::api_cast::widget_api_cast_graph::out:
.. ref-code-block:: cpp
	:class: title-code-block

	port <output> out [TP_NUM_OUTPUT_CLONES]

An API of TT_DATA type.


Methods
~~~~~~~

.. FunctionSection

.. _doxid-classxf_1_1dsp_1_1aie_1_1widget_1_1api__cast_1_1widget__api__cast__graph_1ac9c791f1e1e95ff60b41ae97c98fe94e:
.. _cid-xf::dsp::aie::widget::api_cast::widget_api_cast_graph::getkernels:

getKernels
----------


.. ref-code-block:: cpp
	:class: title-code-block

	kernel* getKernels ()

Access function to get pointer to kernel (or first kernel in a chained configuration).

.. _doxid-classxf_1_1dsp_1_1aie_1_1widget_1_1api__cast_1_1widget__api__cast__graph_1a7c9b45a6e3a1bd33bfa7ae2bc57e0107:
.. _cid-xf::dsp::aie::widget::api_cast::widget_api_cast_graph::widget_api_cast_graph:

widget_api_cast_graph
---------------------


.. ref-code-block:: cpp
	:class: title-code-block

	widget_api_cast_graph ()

This is the constructor function for the Widget API Cast graph.

