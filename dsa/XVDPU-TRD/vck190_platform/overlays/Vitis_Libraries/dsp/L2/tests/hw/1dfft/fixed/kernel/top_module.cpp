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
//================================== End Lic =================================================
#include "top_module.hpp"

extern "C" void fft1DKernel(ap_uint<512> p_inData[FFT_LEN * N_FFT / SSR],
                            ap_uint<512> p_outData[FFT_LEN * N_FFT / SSR],
                            int n_frames) {
#pragma HLS interface m_axi port p_inData = gmem0 offset = slave
#pragma HLS interface s_axilite port = p_inData bundle = control

#pragma HLS interface m_axi port p_outData = gmem0 offset = slave
#pragma HLS interface s_axilite port = p_outData bundle = control

#pragma HLS interface s_axilite port = n_frames bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    xf::dsp::fft::fftKernel<fftParams, IID, std::complex<ap_fixed<IN_WL, IN_IL> > >(p_inData, p_outData, n_frames);
}
