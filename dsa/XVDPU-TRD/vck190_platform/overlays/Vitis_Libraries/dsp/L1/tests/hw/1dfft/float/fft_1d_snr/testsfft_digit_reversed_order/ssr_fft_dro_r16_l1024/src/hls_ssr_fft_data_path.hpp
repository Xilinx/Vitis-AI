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
#include <ap_fixed.h>
#include "vt_fft.hpp"
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *  Inlcude file that declares FFT Radix or SSR factor and length for the test
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */

#include "fft_size.hpp"
using namespace xf::dsp::fft;

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 *  Set appropriate bit-width for the storage of sine/cosine or exponential tables and also define input bit widths
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */

#define SSR_FFT_IN_WL 27
#define SSR_FFT_IN_IL 10
#define SSR_FFT_TW_WL 18
#define SSR_FFT_TW_IL 2

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Define double type that is used for creating a reference floating point model which is used for verification and
 *calculating fixed point model SNR in DBs.
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */
typedef double T_innerDB;
typedef double T_innerDBOut;

typedef float T_INNER_SSR_FFT_IN;
typedef float T_INNER_SSR_FFT_OUT;
typedef float T_INNER_SSR_TWIDDLE_TABLE;
typedef T_INNER_SSR_TWIDDLE_TABLE T_INNER_SSR_EXP_TABLE;

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Define ap_fixed complex type for input samples and the complex sin/cosine table storage
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */

typedef complex_wrapper<T_INNER_SSR_FFT_IN> T_SSR_FFT_IN;
typedef complex_wrapper<T_INNER_SSR_TWIDDLE_TABLE> T_SSR_TWIDDLE_TABLE;

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Define double precision complex types
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */

typedef complex_wrapper<T_innerDB> T_ComplexDouble;

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Define a SSR FFT parameter structure that extends a predefine structure with default values, redefine only the
 *members
 *whose default values are to be changes , the structure which is extended here is called ssr_fft_default_params which
 *defines default values
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */

struct ssr_fft_params : ssr_fft_default_params {
    static const int N = SSR_FFT_L;
    static const int R = SSR_FFT_R;

    /******************************
     * Allowed scaling modes :
     * - SSR_FFT_NO_SCALING
     * - SSR_FFT_GROW_TO_MAX_WIDTH
     * -SSR_FFT_SCALE
     *
     ******************************/
    static const scaling_mode_enum scaling_mode = SSR_FFT_GROW_TO_MAX_WIDTH;

    // sine/cosine storage resolution
    static const int twiddle_table_word_length = SSR_FFT_TW_WL;
    static const int twiddle_table_intger_part_length = SSR_FFT_TW_IL;
    static const fft_output_order_enum output_data_order = SSR_FFT_DIGIT_REVERSED_TRANSPOSED;

    /******************************
     * The default instance ID
     * needs to be unique for
     * each instance if multiple
     * instance of the SSR FFT are
     * used in single design
     ******************************/
    static const int default_t_instanceID = 0;
};

/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Define a type for output storage which is predefined in a structure ssr_fft_output_type : this structure returns
 *proper type provided the constant parameter structure as defined above and the SSR FFT input type also defined
 *above
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */
typedef xf::dsp::fft::ssr_fft_output_type<ssr_fft_params, T_SSR_FFT_IN>::t_ssr_fft_out T_SSR_FFT_OUT;

#endif // HLS_SSR_FFT_DATA_PATH_
