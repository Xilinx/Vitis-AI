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
#ifndef _DSPLIB_L1_UTILS_HPP_
#define _DSPLIB_L1_UTILS_HPP_

/*
L1 testbench utilities.
This file contains sets of overloaded functions for use by the L1 testbench.
*/

#include <stdio.h>
#include <adf.h>

namespace xf {
namespace dsp {
namespace aie {
// Update int16 window with samples starting from offset value
void update_window(
    int offset, int16_t* samples, input_window_int16* input, int firLen, int firMargin, int SAMPLES_IN_WINDOW) {
    int16 unit;
    int real, imag;
    window_incr(input, firMargin);
    for (int i = offset; i < offset + SAMPLES_IN_WINDOW; i++) {
        unit = samples[i + firMargin];
        window_writeincr((output_window_int16*)input, unit);
    }
}
// Update int32 window with samples starting from offset value
void update_window(
    int offset, int32_t* samples, input_window_int32* input, int firLen, int firMargin, int SAMPLES_IN_WINDOW) {
    int32 unit;
    int real, imag;
    window_incr(input, firMargin);
    for (int i = offset; i < offset + SAMPLES_IN_WINDOW; i++) {
        unit = samples[i + firMargin];
        window_writeincr((output_window_int32*)input, unit);
    }
}
// Update cint16 window with samples starting from offset value
void update_window(
    int offset, cint16_t* samples, input_window_cint16* cinput, int firLen, int firMargin, int SAMPLES_IN_WINDOW) {
    cint16 cunit;
    int real, imag;
    window_incr(cinput, firMargin);
    for (int i = offset; i < offset + SAMPLES_IN_WINDOW; i++) {
        cunit.real = samples[i + firMargin].real;
        cunit.imag = samples[i + firMargin].imag;
        window_writeincr((output_window_cint16*)cinput, cunit);
    }
}
// Update cint32 window with samples starting from offset value
void update_window(
    int offset, cint32_t* samples, input_window_cint32* cinput, int firLen, int firMargin, int SAMPLES_IN_WINDOW) {
    cint32 cunit;
    int real, imag;
    window_incr(cinput, firMargin);
    for (int i = offset; i < offset + SAMPLES_IN_WINDOW; i++) {
        cunit.real = samples[i + firMargin].real;
        cunit.imag = samples[i + firMargin].imag;
        window_writeincr((output_window_cint32*)cinput, cunit);
    }
}
// Update float window with samples starting from offset value
void update_window(
    int offset, float* samples, input_window_float* input, int firLen, int firMargin, int SAMPLES_IN_WINDOW) {
    float unit;
    window_incr(input, firMargin);
    for (int i = offset; i < offset + SAMPLES_IN_WINDOW; i++) {
        unit = samples[i + firMargin];
        window_writeincr((output_window_float*)input, unit);
    }
}
// Update cfloat window with samples starting from offset value
void update_window(
    int offset, cfloat* samples, input_window_cfloat* cinput, int firLen, int firMargin, int SAMPLES_IN_WINDOW) {
    cfloat cunit;
    float real, imag;
    window_incr(cinput, firMargin);
    for (int i = offset; i < offset + SAMPLES_IN_WINDOW; i++) {
        cunit.real = samples[i + firMargin].real;
        cunit.imag = samples[i + firMargin].imag;
        window_writeincr((output_window_cfloat*)cinput, cunit);
    }
}
}
}
}
#endif // _DSPLIB_L1_UTILS_HPP_
