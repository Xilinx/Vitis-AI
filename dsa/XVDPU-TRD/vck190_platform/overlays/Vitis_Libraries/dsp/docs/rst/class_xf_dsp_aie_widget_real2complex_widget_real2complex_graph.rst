.. index:: pair: class; xf::dsp::aie::widget::real2complex::widget_real2complex_graph
.. _doxid-classxf_1_1dsp_1_1aie_1_1widget_1_1real2complex_1_1widget__real2complex__graph:
.. _cid-xf::dsp::aie::widget::real2complex::widget_real2complex_graph:

template class xf::dsp::aie::widget::real2complex::widget_real2complex_graph
============================================================================

.. toctree::
	:hidden:

.. code-block:: cpp
	:class: overview-code-block

	#include "widget_real2complex_graph.hpp"


Overview
~~~~~~~~

widget_real2complex is utility to convert real data to complex or vice versa

These are the templates to configure the function.



.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - TT_DATA

        - describes the type of individual data samples input to the function. This is a typename and must be one of the following: int16, cint16, int32, cint32, float, cfloat.

    *
        - TT_OUT_DATA

        - describes the type of individual data samples output from the function. This is a typename and must be one of the following: int16, cint16, int32, cint32, float, cfloat. TT_OUT_DATA must also be the real or complex counterpart of TT_DATA, e.g. TT_DATA = int16 and TT_OUT_DATA = cint16 is valid, TT_DATA = cint16 and TT_OUT_DATA = int16 is valid, but TT_DATA = int16 and TT_OUT_DATA = cint32 is not valid.

    *
        - TP_WINDOW_VSIZE

        - describes the number of samples in the window API used if either input or output is a window. Note: Margin size should not be included in TP_INPUT_WINDOW_VSIZE. This is the class for the Widget API Cast graph

.. ref-code-block:: cpp
	:class: overview-code-block

	template <
	    typename TT_DATA,
	    typename TT_OUT_DATA,
	    unsigned int TP_WINDOW_VSIZE
	    >
	class widget_real2complex_graph: public graph

	// fields

	port <input> :ref:`in<doxid-classxf_1_1dsp_1_1aie_1_1widget_1_1real2complex_1_1widget__real2complex__graph_1ae0f5416c01d9c454f2273b86a89a3ef3>`
	port <output> :ref:`out<doxid-classxf_1_1dsp_1_1aie_1_1widget_1_1real2complex_1_1widget__real2complex__graph_1a0a67564777a57677dc0dedb2a3d074e1>`

Fields
------

.. _doxid-classxf_1_1dsp_1_1aie_1_1widget_1_1real2complex_1_1widget__real2complex__graph_1ae0f5416c01d9c454f2273b86a89a3ef3:
.. _cid-xf::dsp::aie::widget::real2complex::widget_real2complex_graph::in:
.. ref-code-block:: cpp
	:class: title-code-block

	port <input> in

The input data to the function. Window API is expected. Data is read from here, converted and written to output.

.. _doxid-classxf_1_1dsp_1_1aie_1_1widget_1_1real2complex_1_1widget__real2complex__graph_1a0a67564777a57677dc0dedb2a3d074e1:
.. _cid-xf::dsp::aie::widget::real2complex::widget_real2complex_graph::out:
.. ref-code-block:: cpp
	:class: title-code-block

	port <output> out

An API of TT_DATA type.


Methods
~~~~~~~

.. FunctionSection

.. _doxid-classxf_1_1dsp_1_1aie_1_1widget_1_1real2complex_1_1widget__real2complex__graph_1a80216f5c453cc2d3cd4e921098c796da:
.. _cid-xf::dsp::aie::widget::real2complex::widget_real2complex_graph::getkernels:

getKernels
----------


.. ref-code-block:: cpp
	:class: title-code-block

	kernel* getKernels ()

Access function to get pointer to kernel (or first kernel in a chained configuration).

.. _doxid-classxf_1_1dsp_1_1aie_1_1widget_1_1real2complex_1_1widget__real2complex__graph_1ae696c0352b415fcc14f6accd01bf1116:
.. _cid-xf::dsp::aie::widget::real2complex::widget_real2complex_graph::widget_real2complex_graph:

widget_real2complex_graph
-------------------------


.. ref-code-block:: cpp
	:class: title-code-block

	widget_real2complex_graph ()

This is the constructor function for the Widget API Cast graph.

