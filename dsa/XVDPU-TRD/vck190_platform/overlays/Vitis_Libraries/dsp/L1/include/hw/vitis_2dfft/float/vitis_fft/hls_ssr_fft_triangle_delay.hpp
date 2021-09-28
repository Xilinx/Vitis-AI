/*
 * Copyright 2019 Xilinx, Inc.
 *
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

// Filename : hls_ssr_fft_triangle_delay.hpp
#ifndef HLS_SSR_FFT_TRIANGLE_DELAY_H_
#define HLS_SSR_FFT_TRIANGLE_DELAY_H_
#include <ap_shift_reg.h>

namespace xf {
namespace dsp {
namespace fft {

template <int t_instanceID,
          int t_stage,
          int t_subStage,
          int t_forkNumber,
          int t_PF,
          bool t_delayOnLowIndexFirst,
          unsigned int t_regTriangleHeight>
struct TriangleDelay {
    template <typename T_dtype>
    void process(T_dtype* p_in, T_dtype* p_out);
};

template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_PF, bool t_delayOnLowIndexFirst>
struct TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, t_delayOnLowIndexFirst, 1> {
    template <typename T_dtype>
    void process(T_dtype* p_in, T_dtype* p_out);
};

// recursive funtion
template <int t_instanceID,
          int t_stage,
          int t_subStage,
          int t_forkNumber,
          int t_PF,
          bool t_delayOnLowIndexFirst,
          unsigned int t_regTriangleHeight>
template <typename T_dtype>
void TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, t_delayOnLowIndexFirst, t_regTriangleHeight>::
    process(T_dtype* p_in, T_dtype* p_out) {
    static ap_shift_reg<T_dtype, (t_regTriangleHeight - 1) * t_PF> delayline;
    if (t_delayOnLowIndexFirst) {
        p_out[0] = delayline.shift(p_in[0]);
        TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber - 1, t_PF, t_delayOnLowIndexFirst,
                      t_regTriangleHeight - 1>
            next1;
        next1.template process<T_dtype>(&p_in[1], &p_out[1]);
    } else { // delay on LSB
        p_out[t_regTriangleHeight - 1] = delayline.shift(p_in[t_regTriangleHeight - 1]);
        TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber - 1, t_PF, t_delayOnLowIndexFirst,
                      t_regTriangleHeight - 1>
            next0;
        next0.template process<T_dtype>(&p_in[0], &p_out[0]);
    }
}

// tail
template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_PF, bool t_delayOnLowIndexFirst>
template <typename T_dtype>
void TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, t_delayOnLowIndexFirst, 1>::process(
    T_dtype* p_in, T_dtype* p_out) {
    p_out[0] = p_in[0];
}
} // end namespace fft
} // end namespace dsp
} // end namespace xf
#endif // HLS_SSR_FFT_TRIANGLE_DELAY_H_
