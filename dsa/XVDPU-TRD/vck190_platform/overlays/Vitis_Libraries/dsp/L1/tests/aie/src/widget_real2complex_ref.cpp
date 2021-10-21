/*
 * Copyright 2021 Xilinx, Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
Widget API Cast reference model
*/

#include "widget_real2complex_ref.hpp"
#include "fir_ref_utils.hpp"

namespace xf {
namespace dsp {
namespace aie {
namespace widget {
namespace real2complex {

// Widget real2complex - default/base 'specialization'
// int16 to cint16
template <typename TT_DATA, typename TT_OUT_DATA, unsigned int TP_WINDOW_VSIZE>
void widget_real2complex_ref<TT_DATA, TT_OUT_DATA, TP_WINDOW_VSIZE>::convertData(
    input_window<TT_DATA>* inWindow0, output_window<TT_OUT_DATA>* outWindow0) {
    TT_DATA d_in;
    TT_OUT_DATA d_out;
    d_out.imag = 0;

#ifdef _DSPLIB_WIDGET_REAL2COMPLEX_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(inWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_REAL2COMPLEX_REF_DEBUG_

    for (unsigned int i = 0; i < TP_WINDOW_VSIZE; i++) {
        d_in = window_readincr(inWindow0); // read input data
        d_out.real = d_in;
        window_writeincr(outWindow0, d_out);
    }
};

// int32 to cint32
template <unsigned int TP_WINDOW_VSIZE>
void widget_real2complex_ref<int32, cint32, TP_WINDOW_VSIZE>::convertData(input_window<int32>* inWindow0,
                                                                          output_window<cint32>* outWindow0) {
    int32 d_in;
    cint32 d_out;

#ifdef _DSPLIB_WIDGET_REAL2COMPLEX_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(inWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_REAL2COMPLEX_REF_DEBUG_

    for (unsigned int i = 0; i < TP_WINDOW_VSIZE; i++) {
        d_in = window_readincr(inWindow0); // read input data
        d_out.real = d_in;
        window_writeincr(outWindow0, d_out);
    }
};

// float to cfloat
template <unsigned int TP_WINDOW_VSIZE>
void widget_real2complex_ref<float, cfloat, TP_WINDOW_VSIZE>::convertData(input_window<float>* inWindow0,
                                                                          output_window<cfloat>* outWindow0) {
    float d_in;
    cfloat d_out;

#ifdef _DSPLIB_WIDGET_REAL2COMPLEX_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(inWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_REAL2COMPLEX_REF_DEBUG_

    for (unsigned int i = 0; i < TP_WINDOW_VSIZE; i++) {
        d_in = window_readincr(inWindow0); // read input data
        d_out.real = d_in;
        window_writeincr(outWindow0, d_out);
    }
};

// cint16 to int16
template <unsigned int TP_WINDOW_VSIZE>
void widget_real2complex_ref<cint16, int16, TP_WINDOW_VSIZE>::convertData(input_window<cint16>* inWindow0,
                                                                          output_window<int16>* outWindow0) {
    cint16 d_in;
    int16 d_out;

#ifdef _DSPLIB_WIDGET_REAL2COMPLEX_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(inWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_REAL2COMPLEX_REF_DEBUG_

    for (unsigned int i = 0; i < TP_WINDOW_VSIZE; i++) {
        d_in = window_readincr(inWindow0); // read input data
        d_out = d_in.real;
        window_writeincr(outWindow0, d_out);
    }
};

// cint32 to int32
template <unsigned int TP_WINDOW_VSIZE>
void widget_real2complex_ref<cint32, int32, TP_WINDOW_VSIZE>::convertData(input_window<cint32>* inWindow0,
                                                                          output_window<int32>* outWindow0) {
    cint32 d_in;
    int32 d_out;

#ifdef _DSPLIB_WIDGET_REAL2COMPLEX_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(inWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_REAL2COMPLEX_REF_DEBUG_

    for (unsigned int i = 0; i < TP_WINDOW_VSIZE; i++) {
        d_in = window_readincr(inWindow0); // read input data
        d_out = d_in.real;
        window_writeincr(outWindow0, d_out);
    }
};

// cfloat to float
template <unsigned int TP_WINDOW_VSIZE>
void widget_real2complex_ref<cfloat, float, TP_WINDOW_VSIZE>::convertData(input_window<cfloat>* inWindow0,
                                                                          output_window<float>* outWindow0) {
    cfloat d_in;
    float d_out;

#ifdef _DSPLIB_WIDGET_REAL2COMPLEX_REF_DEBUG_
    const unsigned int kSamplesInWindow = window_size(inWindow0); // number of samples in window
    if (kSamplesInWindow != TP_WINDOW_VSIZE) {
        printf("Error: mismatch of samples in window versus template parameter");
    }
#endif //_DSPLIB_WIDGET_REAL2COMPLEX_REF_DEBUG_

    for (unsigned int i = 0; i < TP_WINDOW_VSIZE; i++) {
        d_in = window_readincr(inWindow0); // read input data
        d_out = d_in.real;
        window_writeincr(outWindow0, d_out);
    }
};
}
}
}
}
}
