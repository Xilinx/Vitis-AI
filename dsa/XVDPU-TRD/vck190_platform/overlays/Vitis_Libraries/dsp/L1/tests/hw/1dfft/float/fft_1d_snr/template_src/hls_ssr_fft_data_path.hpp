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
// File Name : hls_ssr_fft_data_path.hpp
#ifndef HLS_SSR_FFT_DATA_PATH_
#define HLS_SSR_FFT_DATA_PATH_
#include "vt_fft.hpp"
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * include header which defines the size of ssr fft, radix and length
 **+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */
#include "fft_size.hpp"

using namespace xf::dsp::fft;

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Define double type that is used for creating a reference floating point model which is used for verification and
 *calculating SNR in dbs.
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */
typedef double double_type;
typedef double tip_fftInType;
typedef double tip_fftOutType;
typedef double tip_fftTwiddleType;
typedef double tip_complexExpTableType;
typedef double tip_complexMulOutType;

typedef float T_INNER_SSR_FFT_IN;
typedef float T_INNER_SSR_TWIDDLE_TABLE;
typedef T_INNER_SSR_TWIDDLE_TABLE T_INNER_SSR_EXP_TABLE;

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Define FLOAT complex type for input samples and the complex sin/cos table storage
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */

typedef complex_wrapper<float> T_SSR_FFT_IN;
typedef complex_wrapper<float> T_SSR_TWIDDLE_TABLE;
typedef T_SSR_TWIDDLE_TABLE T_SSR_EXP_TABLE;

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Define a ssr fft parameter structure that extends a predefine structure with defaul values, redefine only the members
 *whose default values are to be changes , the structure which is extended here is called ssr_fft_default_params which
 *defines def- -ault values
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */

struct ssr_fft_params : ssr_fft_default_params {
    static const int N = SSR_FFT_L;
    static const int R = SSR_FFT_R;
};

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Define a type for output storage which is pre-defined in a structure ssr_fft_output_type : this structure returns
 *proper ty- -pe provided the constant parameter structure as defined above and the ssr fft input type also defined
 *above
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */
typedef xf::dsp::fft::ssr_fft_output_type<ssr_fft_params, T_SSR_FFT_IN>::t_ssr_fft_out T_SSR_FFT_OUT;

#endif // HLS_SSR_FFT_DATA_PATH_
