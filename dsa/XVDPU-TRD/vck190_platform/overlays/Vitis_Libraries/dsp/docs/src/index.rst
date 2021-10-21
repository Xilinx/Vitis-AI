..
   Copyright 2021 Xilinx, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

========================
Vitis DSP Library
========================

The Vitis |trade| digital signal processing library (DSPLib) provides an implementation of different L1/L2/L3 primitives for digital signal processing.

The DSPLib contains PL and AI Engine solutions. For documentation on AI Engine solutions, see :ref:`1_INTRODUCTION`.

The current PL library consists of an implementation of a Discrete Fourier Transform using a Fast
Fourier Transform algorithm for acceleration on Xilinx |reg| FPGAs. The library is planned
to provide three types of implementations namely L1 primitives, L2 kernels, and L3 software APIs. Those
implementations are organized in their corresponding directories L1, L2, and L3.

The L1 primitives can be leveraged by developers working on harware design
implementation or designing hardware kernels for acceleration. They are particularly
suitable for hardware designers. The L2 kernels are HLS-based predesigned kernels
that can be directly used for FPGA acceleration of different applications on integration with
the Xilinx Runtime (XRT). The L3 provides software APIs in C, C++, and Python which
allow software developers to offload FFT calculation to FPGAs for acceleration. Before
an FPGA can perform the FFT computation, the FPGA needs to be configured with a particular image
called an overlay.

Since all the kernel code is developed with the permissive Apache 2.0 license,
advanced users can easily tailor, optimize, or combine them for their own needs.
Demos and usage examples of different level implementations are also provided
for reference.


.. toctree::
   :caption: Library Overview
   :maxdepth: 4

   overview.rst
   release.rst

.. toctree::
   :caption: L1 PL DSP Library User Guide
   :maxdepth: 4

   user_guide/L1.rst
   user_guide/L1_2dfft.rst

.. toctree::
   :caption: L2 DSP Library User Guide
   :maxdepth: 4

   user_guide/L2/1-introduction.rst
   user_guide/L2/2-dsp-lib-func.rst
   user_guide/L2/3-using-examples.rst
   user_guide/L2/5-benchmark.rst


.. toctree::
   :maxdepth: 4
   :caption: API Reference
   :hidden:

   API Reference Overview <user_guide/L2/4-api-reference>
   fir_sr_asym_graph <rst/class_xf_dsp_aie_fir_sr_asym_fir_sr_asym_graph>
   fir_sr_sym_graph <rst/class_xf_dsp_aie_fir_sr_sym_fir_sr_sym_graph>
   fir_interpolate_asym_graph <rst/class_xf_dsp_aie_fir_interpolate_asym_fir_interpolate_asym_graph>
   fir_decimate_hb_graph <rst/class_xf_dsp_aie_fir_decimate_hb_fir_decimate_hb_graph>
   fir_interpolate_hb_graph <rst/class_xf_dsp_aie_fir_interpolate_hb_fir_interpolate_hb_graph>
   fir_decimate_asym_graph <rst/class_xf_dsp_aie_fir_decimate_asym_fir_decimate_asym_graph>
   fir_interpolate_fract_asym_graph <rst/class_xf_dsp_aie_fir_interpolate_fract_asym_fir_interpolate_fract_asym_graph>
   fir_decimate_sym_graph <rst/class_xf_dsp_aie_fir_decimate_sym_fir_decimate_sym_graph>
   matrix_mult_graph <rst/class_xf_dsp_aie_blas_matrix_mult_matrix_mult_graph>
   fft_ifft_dit_1ch_graph <rst/class_xf_dsp_aie_fft_dit_1ch_fft_ifft_dit_1ch_base_graph>
   widget_api_cast_graph <rst/class_xf_dsp_aie_widget_api_cast_widget_api_cast_graph>
   widget_real2complex_graph <rst/class_xf_dsp_aie_widget_real2complex_widget_real2complex_graph>

.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:



