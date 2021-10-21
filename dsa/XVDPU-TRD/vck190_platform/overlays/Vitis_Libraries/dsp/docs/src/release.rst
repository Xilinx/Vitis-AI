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

.. _release_note:

Release Note
============

.. toctree::
   :hidden:
   :maxdepth: 1

***********************
2021.1
***********************

The AI Engine DSP Library contains common parameterizable DSP functions used in many advanced signal processing applications. All functions currently support window interfaces with streaming interface support planned for future releases.

.. note:: Namespace aliasing can be utilized to shorten instantiations: ``namespace dsplib = xf::dsp::aie;``


*  **FIR Filters**

The DSPLib contains several variants of Finite Impulse Response (FIR) filters.

+---------------------------------------+-----------------------------------------------------------------------------+
| **Function**                          | **Namespace**                                                               |
+=======================================+=============================================================================+
| Single rate, asymmetrical             |  dsplib::fir::sr_asym::fir_sr_asym_graph                                    |
+---------------------------------------+-----------------------------------------------------------------------------+
| Single rate, symmetrical              |  dsplib::fir::sr_sym::fir_sr_sym_graph                                      |
+---------------------------------------+-----------------------------------------------------------------------------+
| Interpolation asymmetrical            |  dsplib::fir::interpolate_asym::fir_interpolate_asym_graph                  |
+---------------------------------------+-----------------------------------------------------------------------------+
| Decimation, halfband                  |  dsplib::fir::decimate_hb::fir_decimate_hb_graph                            |
+---------------------------------------+-----------------------------------------------------------------------------+
| Interpolation, halfband               |  dsplib::fir::interpolate_hb::fir_interpolate_hb_graph                      |
+---------------------------------------+-----------------------------------------------------------------------------+
| Decimation, asymmetric                |  dsplib::fir::decimate_asym::fir_decimate_asym_graph                        |
+---------------------------------------+-----------------------------------------------------------------------------+
| Interpolation, fractional, asymmetric |  dsplib::fir::interpolate_fract_asym:: fir_interpolate_fract_asym_graph     |
+---------------------------------------+-----------------------------------------------------------------------------+
| Decimation, symmetric                 |  dsplib::fir::decimate_sym::fir_decimate_sym_graph                          |
+---------------------------------------+-----------------------------------------------------------------------------+

All FIR filters can be configured for various types of data and coefficients. These types can be int16, int32, or float and also real or complex.
Both FIR length and cascade length can also be configured for all FIR variants.

*  **FFT/iFFT**

The DSPLib contains one FFT/iFFT solution. This is a single channel, single kernel decimation in time, (DIT), implementation with configurable point size, complex data types, cascade length and FFT/iFFT function.

+---------------------------------------+-----------------------------------------------------------------------------+
| **Function**                          | **Namespace**                                                               |
+=======================================+=============================================================================+
| Single Channel FFT/iFFT               |  dsplib::fft::fft_ifft_dit_1ch_graph                                        |
+---------------------------------------+-----------------------------------------------------------------------------+


*  **Matrix Multiply (GeMM)**

The DSPLib contains one Matrix Multiply/GEMM (GEneral Matrix Multiply) solution. This supports the Matrix Multiplication of 2 Matrices A and B with configurable input data types resulting in a derived output data type.


+---------------------------------------+-----------------------------------------------------------------------------+
| **Function**                          | **Namespace**                                                               |
+=======================================+=============================================================================+
| Matrix Mult / GeMM                    |  dsplib::blas::matrix_mult::matrix_mult_graph                               |
+---------------------------------------+-----------------------------------------------------------------------------+

*  **Widget Utilities**

These widgets support converting between window and streams on the input to the DSPLib function and between streams to windows on the output of the DSPLib function where desired and additional widget for converting between real and complex data-types.

+---------------------------------------+-----------------------------------------------------------------------------+
| **Function**                          | **Namespace**                                                               |
+=======================================+=============================================================================+
| Stream to Window / Window to Stream   |  dsplib::widget::api_cast::widget_api_cast_graph                            |
+---------------------------------------+-----------------------------------------------------------------------------+
| Real to Complex / Complex to Real     |  dsplib:widget::real2complex::widget_real2complex_graph                     |
+---------------------------------------+-----------------------------------------------------------------------------+


*  **AIE DSP in Model Composer**

DSP Library functions are supported in Vitis Model Composer, enabling users to easily plug these functions into the Matlab/Simulink environment to ease AI Engine DSP Library evaluation and overall AI Engine ADF graph development.



***********************
2020.2
***********************

Revised the APIs to fully support Vitis HLS.



***********************
2020.1
***********************

The 1.0 release introduces L1 HLS primitives for Discrete Fourier Transform for 1-D and 2-D input data.
