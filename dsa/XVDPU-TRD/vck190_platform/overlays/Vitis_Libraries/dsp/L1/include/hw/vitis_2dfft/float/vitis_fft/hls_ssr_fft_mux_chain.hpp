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

// Filen Name: hls_ssr_fft_mux_chain.hpp
/*
 =========================================================================================
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-

 26-sep-2018-: This file defines a MuxChain class that can be used to create
 chain of muxes for construction input swap transposes for SSR FFT

 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 -_-                                                                                   -_-
 ========================================================================================
 */
#ifndef HLS_SSR_FFT_MUX_CHAIN_H_
#define HLS_SSR_FFT_MUX_CHAIN_H_
#include <ap_shift_reg.h>

namespace xf {
namespace dsp {
namespace fft {

template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_PF, unsigned int t_regTriangleHeight>
struct MuxChain {
    template <int t_R, typename T_controlWordType, typename T_dtype>
    void genChain(T_controlWordType control_bits, T_dtype* p_in, T_dtype* p_out);
};

template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_PF>
struct MuxChain<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, 1> {
    template <int t_R, typename T_controlWordType, typename T_dtype>
    void genChain(T_controlWordType control_bits, T_dtype* p_in, T_dtype* p_out);
};

template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_PF, unsigned int t_regTriangleHeight>
template <int t_R, typename T_controlWordType, typename T_dtype>
void MuxChain<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, t_regTriangleHeight>::genChain(
    T_controlWordType control_bits, T_dtype* p_in, T_dtype* p_out) {
    static ap_shift_reg<unsigned int, t_PF> control_delayline;
    unsigned int next_stage_control_bits = control_delayline.shift(control_bits);

    p_out[t_R - t_regTriangleHeight] = p_in[control_bits];

    MuxChain<t_instanceID, t_stage, t_subStage, t_forkNumber - 1, t_PF, t_regTriangleHeight - 1> nextMux;
    nextMux.template genChain<t_R, T_controlWordType, T_dtype>(next_stage_control_bits, &p_in[0], &p_out[0]);
}

template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_PF>
template <int t_R, typename T_controlWordType, typename T_dtype>
void MuxChain<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, 1>::genChain(T_controlWordType control_bits,
                                                                                  T_dtype* p_in,
                                                                                  T_dtype* p_out) {
    p_out[t_R - 1] = p_in[control_bits];
}
} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif // HLS_SSR_FFT_MUX_CHAIN_H_
