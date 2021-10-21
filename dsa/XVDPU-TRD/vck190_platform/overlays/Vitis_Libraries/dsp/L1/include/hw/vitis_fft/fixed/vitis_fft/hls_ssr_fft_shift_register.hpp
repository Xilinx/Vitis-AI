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
#ifndef __HLS_SSR_FFT_SHIFT_REGISTER__
#define __HLS_SSR_FFT_SHIFT_REGISTER__

#ifndef __SYNTHESIS__
#include <assert.h>
#include <ap_shift_reg.h>
#endif
namespace xf {
namespace dsp {
namespace fft {

template <typename T_dtype, unsigned int t_len>
struct ssr_shift_reg {
    // T_dtype m_regChain[t_len];
    ap_shift_reg<T_dtype, t_len> m_shifReg;
    T_dtype shift(T_dtype p_sample) {
#pragma HLS INLINE
/* #ifndef __SYNTHESIS__
 assert(t_len > 0);
 #endif
 T_dtype output_sample = m_regChain[t_len-1];
 for(int i=0;i<t_len-1;i++)
 {
 #pragma HLS UNROLL
     m_regChain[t_len-1-i] =  m_regChain[t_len-1-i-1];
 }
 m_regChain[0] =p_sample;
 return output_sample;*/
#ifndef __SYNTHESIS__
// std::cout<<"Using Generic Shift reg implementation"<<std::endl;
#endif

        return m_shifReg.shift(p_sample);
    }
};

template <typename T_dtype>
struct ssr_shift_reg<T_dtype, 1> {
    T_dtype m_regChain;
    T_dtype shift(T_dtype p_sample) {
#pragma HLS INLINE
        T_dtype temp = m_regChain;
        m_regChain = p_sample;
#ifndef __SYNTHESIS__
// std::cout<<"Using reg=1 size specialization."<<std::endl;
#endif
        return temp;
    }
};
} // end namespace fft
} // end namespace dsp
} // end namespace xf
#endif //__HLS_SSR_FFT_SHIFT_REGISTER__
