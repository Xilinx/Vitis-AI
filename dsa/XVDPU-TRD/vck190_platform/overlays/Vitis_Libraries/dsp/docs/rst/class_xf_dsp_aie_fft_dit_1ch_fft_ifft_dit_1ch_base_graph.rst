.. index:: pair: class; xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph:

template class xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph
======================================================================

.. toctree::
	:hidden:

.. code-block:: cpp
	:class: overview-code-block

	#include "fft_ifft_dit_1ch_graph.hpp"


Overview
~~~~~~~~

fft_dit_1ch is a single-channel, decimation-in-time, fixed point size FFT

These are the templates to configure the single-channel decimation-in-time class.



.. rubric:: Parameters:

.. list-table::
    :widths: 20 80

    *
        - TT_DATA

        - describes the type of individual data samples input to and output from the transform function. This is a typename and must be one of the following: int16, cint16, int32, cint32, float, cfloat.

    *
        - TT_TWIDDLE

        - 
          describes the type of twiddle factors of the transform. It must be one of the following: cint16, cint32, cfloat and must also satisfy the following rules:
          
          * 32 bit types are only supported when TT_DATA is also a 32 bit type,
          
          * TT_TWIDDLE must be an integer type if TT_DATA is an integer type
          
          * TT_TWIDDLE must be cfloat type if TT_DATA is a float type.

    *
        - TP_POINT_SIZE

        - is an unsigned integer which describes the number of point size of the transform. This must be 2^N where N is an integer in the range 4 to 16 inclusive. When TP_DYN_PT_SIZE is set, TP_POINT_SIZE describes the maximum point size possible.

    *
        - TP_FFT_NIFFT

        - selects whether the transform to perform is an FFT (1) or IFFT (0).

    *
        - TP_SHIFT

        - selects the power of 2 to scale the result by prior to output.

    *
        - TP_CASC_LEN

        - selects the number of kernels the FFT will be divided over in series to improve throughput

    *
        - TP_DYN_PT_SIZE

        - selects whether (1) or not (0) to use run-time point size determination. When set, each frame of data must be preceeded, in the window, by a 256 bit header. The output frame will also be preceeded by a 256 bit vector which is a copy of the input vector, but for the top byte, which is 0 to indicate a legal frame or 1 to indicate an illegal frame. The lowest significance byte of the input header field describes forward (non-zero) or inverse(0) direction. The second least significant byte 8 bits of this field describe the Radix 2 power of the following frame. e.g. for a 512 point size, this field would hold 9, as 2^9 = 512. Any value below 4 or greater than log2(TP_POINT_SIZE) is considered illegal. When this occurs the top byte of the output header will be set to 1 and the output samples will be set to 0 for a frame of TP_POINT_SIZE

    *
        - TP_WINDOW_VSIZE

        - is an unsigned intered which describes the number of samples in the input window. By default, TP_WINDOW_SIZE is set ot match TP_POINT_SIZE. TP_WINDOW_SIZE may be set to be an integer multiple of the TP_POINT_SIZE, in which case multiple FFT iterations will be performed on a given input window, resulting in multiple iterations of output samples, reducing the numer of times the kernel needs to be triggered to process a given number of input data samples. As a result, the overheads inferred during kernel triggering are reduced and overall performance is increased. This is the base class for the Single channel DIT FFT graph - one or many cascaded kernels for higher throughput

.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a68748f70c4d83099b860eb27cdb56c82:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::m_fftkernels:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a5c85b103b182e51cc4b4515c29b2ac11:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_buf1:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a8ae5b5298669cac261c3c70266023541:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_lut1:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a23f8f76bd60635d88f40d4d9af2c0229:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_lut1a:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1aa079bc20e581d39353b679f726b3cf78:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_lut1b:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a63baf93daf4f97ba6eab72cd005fec72:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_lut2:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a03357396b978a2e70a315af6d4d7d7eb:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_lut2a:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a32a35f36821ad3908aa6776a3d7e60fe:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_lut2b:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1ad2e282ada4c399f960c31109b915db04:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_lut3:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a0f48bd80b951864db1400caf43222206:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_lut3a:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a1ee79c64fbaed6fdff25d211f04dca3b:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_lut3b:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1ac51e5afe267a8160a567df6e4671e907:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_lut4:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1ab8ff3940692a28f1307aa5f60746b766:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_lut4a:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1ac61e70552a6c6918c20a85016be8c2c8:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_lut4b:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a71e0e38a0136721dbdeca6461e78ccc8:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_buf4096:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1ac780c079212dd99ab2dbab40690a2b3c:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_buf2048:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a6597e3b116b018f9fe1e54387c477eb5:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_buf1024:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a91846ab117e25c60cfe5edbd2a51dc8f:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_buf512:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a7061c8aeb5aa92548863be78fb494624:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_buf256:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a6716c91b6d928394f90f572379b13550:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_buf128:
.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a97faf07c0d854527428628929066dc96:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::getkernels:
.. ref-code-block:: cpp
	:class: overview-code-block

	template <
	    typename TT_DATA,
	    typename TT_TWIDDLE,
	    unsigned int TP_POINT_SIZE,
	    unsigned int TP_FFT_NIFFT,
	    unsigned int TP_SHIFT,
	    unsigned int TP_CASC_LEN = 1,
	    unsigned int TP_DYN_PT_SIZE = 0,
	    unsigned int TP_WINDOW_VSIZE = TP_POINT_SIZE
	    >
	class fft_ifft_dit_1ch_base_graph: public graph

	// fields

	port <input> :ref:`in<doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1adda3ba8550dd9a01f1d10fbf31c01f00>`
	port <output> :ref:`out<doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1af711ddd4d3a5e2a7bca96ce54dd94c8d>`
	kernel m_fftKernels[TP_CASC_LEN]
	parameter fft_buf1
	parameter fft_lut1
	parameter fft_lut1a
	parameter fft_lut1b
	parameter fft_lut2
	parameter fft_lut2a
	parameter fft_lut2b
	parameter fft_lut3
	parameter fft_lut3a
	parameter fft_lut3b
	parameter fft_lut4
	parameter fft_lut4a
	parameter fft_lut4b
	parameter fft_buf4096
	parameter fft_buf2048
	parameter fft_buf1024
	parameter fft_buf512
	parameter fft_buf256
	parameter fft_buf128

Fields
------

.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1adda3ba8550dd9a01f1d10fbf31c01f00:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::in:
.. ref-code-block:: cpp
	:class: title-code-block

	port <input> in

The input data to the function. This input is a window API of samples of TT_DATA type. The number of samples in the window is described by TP_POINT_SIZE.

.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1af711ddd4d3a5e2a7bca96ce54dd94c8d:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::out:
.. ref-code-block:: cpp
	:class: title-code-block

	port <output> out

A window API of TP_POINT_SIZE samples of TT_DATA type.


Methods
~~~~~~~

.. FunctionSection

.. _doxid-classxf_1_1dsp_1_1aie_1_1fft_1_1dit__1ch_1_1fft__ifft__dit__1ch__base__graph_1a7d1dbdd99ffda84c9e4c8cc306c70ff1:
.. _cid-xf::dsp::aie::fft::dit_1ch::fft_ifft_dit_1ch_base_graph::fft_ifft_dit_1ch_base_graph:

fft_ifft_dit_1ch_base_graph
---------------------------


.. ref-code-block:: cpp
	:class: title-code-block

	fft_ifft_dit_1ch_base_graph ()

This is the constructor function for the Single channel DIT FFT graph.

