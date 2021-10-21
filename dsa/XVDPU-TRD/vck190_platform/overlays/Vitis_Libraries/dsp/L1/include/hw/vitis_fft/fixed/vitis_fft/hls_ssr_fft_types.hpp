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
#ifndef __HLS_SSR_FFT_TYPES_H__
#define __HLS_SSR_FFT_TYPES_H__
//#include "vitis_fft/hls_ssr_fft_enums.hpp"

#include "vitis_fft/hls_ssr_fft_enums.hpp"
#include "vitis_fft/hls_ssr_fft_input_traits.hpp"
#include "vitis_fft/hls_ssr_fft_output_traits.hpp"

#define HLS_SSR_FFT_DEFAULT_INSTANCE_ID 999999

namespace xf {
namespace dsp {
namespace fft {

template <typename ssr_fft_param_struct, typename T_in>
struct FFTIOTypes {
    typedef T_in T_inType;
    typedef typename FFTOutputTraits<ssr_fft_param_struct::N,
                                     ssr_fft_param_struct::R,
                                     ssr_fft_param_struct::scaling_mode,
                                     ssr_fft_param_struct::transform_direction,
                                     ssr_fft_param_struct::butterfly_rnd_mode,
                                     typename FFTInputTraits<T_in>::T_castedType>::T_FFTOutType T_outType;
};

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif //__HLS_SSR_FFT_TYPES_H__
