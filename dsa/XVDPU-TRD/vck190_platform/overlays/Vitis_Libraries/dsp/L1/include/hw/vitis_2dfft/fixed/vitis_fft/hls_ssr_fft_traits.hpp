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

// File Name : hls_FFTTraits.h
#ifndef HLS_SSR_FFT_TRAITS_H_
#define HLS_SSR_FFT_TRAITS_H_
//#include <hls_dsp.h>

#include <ap_fixed.h>
//#include <complex>

#include "vitis_fft/hls_ssr_fft_utilities.hpp"
#include "vitis_fft/hls_ssr_fft_enums.hpp"
// Include other files that define traits
#include "vitis_fft/hls_ssr_fft_exp_table_traits.hpp"
#include "vitis_fft/hls_ssr_fft_twiddle_table_traits.hpp"
#include "vitis_fft/hls_ssr_fft_output_traits.hpp"
#include "vitis_fft/hls_ssr_fft_butterfly_traits.hpp"
#include "vitis_fft/fft_complex.hpp"

namespace xf {
namespace dsp {
namespace fft {

template <scaling_mode_enum t_scalingMode,
          int t_L,
          int t_R,
          int t_stage,
          typename T_twiddleTab,
          typename T_expTab,
          typename T_in,
          typename T_out>
struct FFTTraits {};
/*===================================================================================================================
 * COMPLEX float TYPE TRAITS
 * For float type the scaling mode has no effect so all the traits stay the same
 * ==================================================================================================================
 **/
template <scaling_mode_enum t_scalingMode, int t_L, int t_R, int t_stage>
struct FFTTraits<t_scalingMode,
                 t_L,
                 t_R,
                 t_stage,
                 std::complex<float>,
                 std::complex<float>,
                 std::complex<float>,
                 std::complex<float> > {
    typedef std::complex<float> T_stageOutType;
    typedef std::complex<float> T_stageInType;
    typedef std::complex<float> T_twiddleType;
    typedef std::complex<float> T_expTabType;
    static const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
    static const int tp_log2R = ssrFFTLog2<t_R>::val;
    static const unsigned int s = NO_OF_FFT_STAGES - t_stage;

    typedef float T_innerStageOutType;
    typedef float T_innerFFTOutType;
    typedef float T_innerStageInType;
    typedef float T_innerTwiddleType;
    typedef float T_innerExpTabType;
};
/*===================================================================================================================
 * COMPLEX DOUBLE TYPE TRAITS
 * For double type the scaling mode has no effect so all the traits stay the same
 * ==================================================================================================================
 **/
template <scaling_mode_enum t_scalingMode, int t_L, int t_R, int t_stage>
struct FFTTraits<t_scalingMode,
                 t_L,
                 t_R,
                 t_stage,
                 std::complex<double>,
                 std::complex<double>,
                 std::complex<double>,
                 std::complex<double> > {
    typedef std::complex<double> T_stageOutType;
    typedef std::complex<double> T_stageInType;
    typedef std::complex<double> T_twiddleType;
    typedef std::complex<double> T_expTabType;
    static const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
    static const int tp_log2R = ssrFFTLog2<t_R>::val;
    static const unsigned int s = NO_OF_FFT_STAGES - t_stage;

    typedef double T_innerStageOutType;
    typedef double T_innerFFTOutType;
    typedef double T_innerStageInType;
    typedef double T_innerTwiddleType;
    typedef double T_innerExpTabType;

    // const int nextStage_WL = tp_IWL+tp_log2R;
    // const int nextStage_IL= tp_IIL+tp_log2R;
};
/*===================================================================================================================
 * COMPLEX                  "ap_fixed"                TYPE TRAITS
 * ==================================================================================================================
 **/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// (((1))) : Fixed point type with Word length and Integer length specs only

//////////////////////////Generic Template without any scaling type specification ///////////////////////////////////
//                               The next three template will be specializations for SCALING MODE
template <scaling_mode_enum t_scalingMode,
          int t_L,
          int t_R,
          int t_stage,
          int tw_WL,
          int tw_IL,
          int expTab_WL,
          int expTab_IL,
          int in_WL,
          int in_IL,
          int out_WL,
          int out_IL

          >
struct FFTTraits<t_scalingMode,
                 t_L,
                 t_R,
                 t_stage,
                 std::complex<ap_fixed<tw_WL, tw_IL> >,
                 std::complex<ap_fixed<expTab_WL, expTab_IL> >,
                 std::complex<ap_fixed<in_WL, in_IL> >,
                 std::complex<ap_fixed<out_WL, out_IL> > > {
    static const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
    static const int tp_log2R = ssrFFTLog2<t_R>::val;
    static const unsigned int s = NO_OF_FFT_STAGES - t_stage;
    static const int nextStage_WL = (NO_OF_FFT_STAGES == t_stage) ? in_WL + tp_log2R + 1 : in_WL + tp_log2R;
    static const int nextStage_IL = (NO_OF_FFT_STAGES == t_stage) ? in_IL + tp_log2R + 1 : in_IL + tp_log2R;

    typedef std::complex<ap_fixed<nextStage_WL, nextStage_IL, AP_TRN, AP_WRAP, 0> > T_stageOutType;
    typedef std::complex<ap_fixed<in_WL, in_IL, AP_TRN, AP_WRAP, 0> > T_stageInType;
    typedef std::complex<ap_fixed<tw_WL, tw_IL, AP_TRN, AP_WRAP, 0> > T_twiddleType;
    typedef std::complex<ap_fixed<expTab_WL, expTab_IL, AP_TRN, AP_WRAP, 0> > T_expTabType;

    typedef ap_fixed<nextStage_WL, nextStage_IL, AP_TRN, AP_WRAP, 0> T_innerStageOutType;
    typedef ap_fixed<out_WL, out_IL, AP_TRN, AP_WRAP, 0> T_innerFFTOutType;
    typedef ap_fixed<in_WL, in_IL, AP_TRN, AP_WRAP, 0> T_innerStageInType;
    typedef ap_fixed<tw_WL, tw_IL, AP_TRN, AP_WRAP, 0> T_innerTwiddleType;
    typedef ap_fixed<expTab_WL, expTab_IL, AP_TRN, AP_WRAP, 0> T_innerExpTabType;
};

/*===================================================================================================================
 * COMPLEX                  "ap_fixed"                TYPE TRAITS with Stage = 1 specialization
 * ==================================================================================================================
 **/

/////////////////////////////////////////////////////////////////////////////////////////////////

//(((1.1)))

// Specialization for ap_fixed with SSR_FFT_NO_SCALING : in this mode the ssr fft t_stage
// level output will grow by log2(t_R) For every t_stage.

/////////////////////////////////////////////////////////////////////////////////////////////////
template <int t_L,
          int t_R,
          int t_stage,
          int tw_WL,
          int tw_IL,
          int expTab_WL,
          int expTab_IL,
          int in_WL,
          int in_IL,
          int out_WL,
          int out_IL

          >
struct FFTTraits<SSR_FFT_NO_SCALING,
                 t_L,
                 t_R,
                 t_stage,
                 std::complex<ap_fixed<tw_WL, tw_IL> >,
                 std::complex<ap_fixed<expTab_WL, expTab_IL> >,
                 std::complex<ap_fixed<in_WL, in_IL> >,
                 std::complex<ap_fixed<out_WL, out_IL> > > {
    static const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
    static const int tp_log2R = ssrFFTLog2<t_R>::val;
    static const unsigned int s = NO_OF_FFT_STAGES - t_stage;
    static const int nextStage_WL = (NO_OF_FFT_STAGES == t_stage) ? in_WL + tp_log2R + 1 : in_WL + tp_log2R;
    static const int nextStage_IL = (NO_OF_FFT_STAGES == t_stage) ? in_IL + tp_log2R + 1 : in_IL + tp_log2R;

    typedef std::complex<ap_fixed<nextStage_WL, nextStage_IL, AP_TRN, AP_WRAP, 0> > T_stageOutType;
    typedef std::complex<ap_fixed<in_WL, in_IL, AP_TRN, AP_WRAP, 0> > T_stageInType;
    typedef std::complex<ap_fixed<tw_WL, tw_IL, AP_TRN, AP_WRAP, 0> > T_twiddleType;
    typedef std::complex<ap_fixed<expTab_WL, expTab_IL, AP_TRN, AP_WRAP, 0> > T_expTabType;

    typedef ap_fixed<nextStage_WL, nextStage_IL, AP_TRN, AP_WRAP, 0> T_innerStageOutType;
    typedef ap_fixed<out_WL, out_IL, AP_TRN, AP_WRAP, 0> T_innerFFTOutType;
    typedef ap_fixed<in_WL, in_IL, AP_TRN, AP_WRAP, 0> T_innerStageInType;
    typedef ap_fixed<tw_WL, tw_IL, AP_TRN, AP_WRAP, 0> T_innerTwiddleType;
    typedef ap_fixed<expTab_WL, expTab_IL, AP_TRN, AP_WRAP, 0> T_innerExpTabType;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

//(((1.2)))

// Specialization for ap_fixed with SSR_FFT_SCALE : in this mode the ssr fft t_stage
// level output will not grow but output will get scaled, every t_stage will loose one bit resolution

/////////////////////////////////////////////////////////////////////////////////////////////////
template <int t_L,
          int t_R,
          int t_stage,
          int tw_WL,
          int tw_IL,
          int expTab_WL,
          int expTab_IL,
          int in_WL,
          int in_IL,
          int out_WL,
          int out_IL

          >
struct FFTTraits<SSR_FFT_SCALE,
                 t_L,
                 t_R,
                 t_stage,
                 std::complex<ap_fixed<tw_WL, tw_IL> >,
                 std::complex<ap_fixed<expTab_WL, expTab_IL> >,
                 std::complex<ap_fixed<in_WL, in_IL> >,
                 std::complex<ap_fixed<out_WL, out_IL> > > {
    static const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
    static const int tp_log2R = ssrFFTLog2<t_R>::val;
    static const unsigned int s = NO_OF_FFT_STAGES - t_stage;
    static const int nextStage_WL = in_WL;
    static const int nextStage_IL = (NO_OF_FFT_STAGES == t_stage) ? in_IL + tp_log2R + 1 : in_IL + tp_log2R;

    typedef std::complex<ap_fixed<nextStage_WL, nextStage_IL, AP_TRN, AP_WRAP, 0> > T_stageOutType;
    typedef std::complex<ap_fixed<in_WL, in_IL, AP_TRN, AP_WRAP, 0> > T_stageInType;
    typedef std::complex<ap_fixed<tw_WL, tw_IL, AP_TRN, AP_WRAP, 0> > T_twiddleType;
    typedef std::complex<ap_fixed<expTab_WL, expTab_IL, AP_TRN, AP_WRAP, 0> > T_expTabType;

    typedef ap_fixed<nextStage_WL, nextStage_IL, AP_TRN, AP_WRAP, 0> T_innerStageOutType;
    typedef ap_fixed<out_WL, out_IL, AP_TRN, AP_WRAP, 0> T_innerFFTOutType;
    typedef ap_fixed<in_WL, in_IL, AP_TRN, AP_WRAP, 0> T_innerStageInType;
    typedef ap_fixed<tw_WL, tw_IL, AP_TRN, AP_WRAP, 0> T_innerTwiddleType;
    typedef ap_fixed<expTab_WL, expTab_IL, AP_TRN, AP_WRAP, 0> T_innerExpTabType;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

//(((1.3)))

// Specialization for ap_fixed with SSR_FFT_GROW_TO_MAX_WIDTH : in this mode the ssr fft t_stage
// level output will grow but output and finally it will saturate to max width decided based
// on DSP48 multiplier input bit width

/////////////////////////////////////////////////////////////////////////////////////////////////
template <int t_L,
          int t_R,
          int t_stage,
          int tw_WL,
          int tw_IL,
          int expTab_WL,
          int expTab_IL,
          int in_WL,
          int in_IL,
          int out_WL,
          int out_IL

          >
struct FFTTraits<SSR_FFT_GROW_TO_MAX_WIDTH, // specialization for max growth scaling mode
                 t_L,
                 t_R,
                 t_stage,
                 std::complex<ap_fixed<tw_WL, tw_IL> >,
                 std::complex<ap_fixed<expTab_WL, expTab_IL> >,
                 std::complex<ap_fixed<in_WL, in_IL> >,
                 std::complex<ap_fixed<out_WL, out_IL> > > {
    static const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
    static const int tp_log2R = ssrFFTLog2<t_R>::val;
    static const unsigned int s = NO_OF_FFT_STAGES - t_stage;

    static const int EXPECTED_NEXT_STAGE_WL = (NO_OF_FFT_STAGES == t_stage) ? in_WL + tp_log2R + 1 : in_WL + tp_log2R;
    static const int EXPECTED_NEXT_STAGE_IL = (NO_OF_FFT_STAGES == t_stage) ? in_IL + tp_log2R + 1 : in_IL + tp_log2R;

    static const int MAX_ALLOWED_BIT_WIDTH_MARGIN = DSP48_OP2_BIT_WIDTH - in_WL;

    // If the expected grown output has bit width larger then max allowed saturate it.
    static const int nextStage_WL =
        (EXPECTED_NEXT_STAGE_WL <= DSP48_OP2_BIT_WIDTH) ? EXPECTED_NEXT_STAGE_WL : DSP48_OP2_BIT_WIDTH;
    static const int nextStage_IL =
        (EXPECTED_NEXT_STAGE_IL <= DSP48_OP2_BIT_WIDTH) ? EXPECTED_NEXT_STAGE_IL : DSP48_OP2_BIT_WIDTH;
    typedef std::complex<ap_fixed<nextStage_WL, nextStage_IL, AP_TRN, AP_WRAP, 0> > T_stageOutType;
    typedef std::complex<ap_fixed<in_WL, in_IL, AP_TRN, AP_WRAP, 0> > T_stageInType;
    typedef std::complex<ap_fixed<tw_WL, tw_IL, AP_TRN, AP_WRAP, 0> > T_twiddleType;
    typedef std::complex<ap_fixed<expTab_WL, expTab_IL, AP_TRN, AP_WRAP, 0> > T_expTabType;
    typedef ap_fixed<nextStage_WL, nextStage_IL, AP_TRN, AP_WRAP, 0> T_innerStageOutType;
    typedef ap_fixed<out_WL, out_IL, AP_TRN, AP_WRAP, 0> T_innerFFTOutType;
    typedef ap_fixed<in_WL, in_IL, AP_TRN, AP_WRAP, 0> T_innerStageInType;
    typedef ap_fixed<tw_WL, tw_IL, AP_TRN, AP_WRAP, 0> T_innerTwiddleType;
    typedef ap_fixed<expTab_WL, expTab_IL, AP_TRN, AP_WRAP, 0> T_innerExpTabType;
};

/*===================================================================================================================
 * COMPLEX                  "ap_fixed"                TYPE TRAITS
 * ==================================================================================================================
 **/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// (((2))) : Fixed point type with Word length, Integer length , quantization mode, overflow mode and sat bit specs

//////////////////////////Generic Template without any scaling type specification ///////////////////////////////////
//                               The next three template will be specializations for SCALING MODE
template <scaling_mode_enum t_scalingMode,
          int t_L,
          int t_R,
          int t_stage,
          int tw_WL,
          int tw_IL,
          ap_q_mode tw_q_mode,
          ap_o_mode tw_o_mode,
          int tw_sat_bits,
          int expTab_WL,
          int expTab_IL,
          ap_q_mode expTab_q_mode,
          ap_o_mode expTab_o_mode,
          int expTab_sat_bits,
          int in_WL,
          int in_IL,
          ap_q_mode in_q_mode,
          ap_o_mode in_o_mode,
          int in_sat_bits,
          int out_WL,
          int out_IL,
          ap_q_mode out_q_mode,
          ap_o_mode out_o_mode,
          int out_sat_bits>
struct FFTTraits<t_scalingMode,
                 t_L,
                 t_R,
                 t_stage,
                 std::complex<ap_fixed<tw_WL, tw_IL, tw_q_mode, tw_o_mode, tw_sat_bits> >,
                 std::complex<ap_fixed<expTab_WL, expTab_IL, expTab_q_mode, expTab_o_mode, expTab_sat_bits> >,
                 std::complex<ap_fixed<in_WL, in_IL, in_q_mode, in_o_mode, in_sat_bits> >,
                 std::complex<ap_fixed<out_WL, out_IL, out_q_mode, out_o_mode, out_sat_bits> > > {
    static const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
    static const int tp_log2R = ssrFFTLog2<t_R>::val;
    static const unsigned int s = NO_OF_FFT_STAGES - t_stage;

    static const int nextStage_WL = (NO_OF_FFT_STAGES == t_stage) ? in_WL + tp_log2R + 1 : in_WL + tp_log2R;
    static const int nextStage_IL = (NO_OF_FFT_STAGES == t_stage) ? in_IL + tp_log2R + 1 : in_IL + tp_log2R;

    typedef std::complex<ap_fixed<nextStage_WL, nextStage_IL, out_q_mode, out_o_mode, out_sat_bits> > T_stageOutType;

    typedef std::complex<ap_fixed<in_WL, in_IL, in_q_mode, in_o_mode, in_sat_bits> > T_stageInType;
    typedef std::complex<ap_fixed<tw_WL, tw_IL, tw_q_mode, tw_o_mode, tw_sat_bits> > T_twiddleType;
    typedef std::complex<ap_fixed<expTab_WL, expTab_IL, expTab_q_mode, expTab_o_mode, expTab_sat_bits> > T_expTabType;

    typedef ap_fixed<nextStage_WL, nextStage_IL, out_q_mode, out_o_mode, out_sat_bits> T_innerStageOutType;

    typedef ap_fixed<out_WL, out_IL, out_q_mode, out_o_mode, out_sat_bits> T_innerFFTOutType;
    typedef ap_fixed<in_WL, in_IL, in_q_mode, in_o_mode, in_sat_bits> T_innerStageInType;
    typedef ap_fixed<tw_WL, tw_IL, tw_q_mode, tw_o_mode, tw_sat_bits> T_innerTwiddleType;
    typedef ap_fixed<expTab_WL, expTab_IL, expTab_q_mode, expTab_o_mode, expTab_sat_bits> T_innerExpTabType;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

//(((2.1))) std::complex < ap_fixed < WL,IL,ap_q_mode, ap_ovf_mode, int sat_bits > >

// Specialization for ap_fixed with ::SSR_FFT_NO_SCALING:: in this mode the ssr fft t_stage
// level output will grow by log2(t_R) For every t_stage.

/////////////////////////////////////////////////////////////////////////////////////////////////
template <int t_L,
          int t_R,
          int t_stage,
          int tw_WL,
          int tw_IL,
          ap_q_mode tw_q_mode,
          ap_o_mode tw_o_mode,
          int tw_sat_bits,
          int expTab_WL,
          int expTab_IL,
          ap_q_mode expTab_q_mode,
          ap_o_mode expTab_o_mode,
          int expTab_sat_bits,
          int in_WL,
          int in_IL,
          ap_q_mode in_q_mode,
          ap_o_mode in_o_mode,
          int in_sat_bits,
          int out_WL,
          int out_IL,
          ap_q_mode out_q_mode,
          ap_o_mode out_o_mode,
          int out_sat_bits>
struct FFTTraits<SSR_FFT_NO_SCALING,
                 t_L,
                 t_R,
                 t_stage,
                 std::complex<ap_fixed<tw_WL, tw_IL, tw_q_mode, tw_o_mode, tw_sat_bits> >,
                 std::complex<ap_fixed<expTab_WL, expTab_IL, expTab_q_mode, expTab_o_mode, expTab_sat_bits> >,
                 std::complex<ap_fixed<in_WL, in_IL, in_q_mode, in_o_mode, in_sat_bits> >,
                 std::complex<ap_fixed<out_WL, out_IL, out_q_mode, out_o_mode, out_sat_bits> > > {
    static const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
    static const int tp_log2R = ssrFFTLog2<t_R>::val;
    static const unsigned int s = NO_OF_FFT_STAGES - t_stage;

    static const int nextStage_WL = (NO_OF_FFT_STAGES == t_stage) ? in_WL + tp_log2R + 1 : in_WL + tp_log2R;
    static const int nextStage_IL = (NO_OF_FFT_STAGES == t_stage) ? in_IL + tp_log2R + 1 : in_IL + tp_log2R;

    typedef std::complex<ap_fixed<nextStage_WL, nextStage_IL, out_q_mode, out_o_mode, out_sat_bits> > T_stageOutType;

    typedef std::complex<ap_fixed<in_WL, in_IL, in_q_mode, in_o_mode, in_sat_bits> > T_stageInType;
    typedef std::complex<ap_fixed<tw_WL, tw_IL, tw_q_mode, tw_o_mode, tw_sat_bits> > T_twiddleType;
    typedef std::complex<ap_fixed<expTab_WL, expTab_IL, expTab_q_mode, expTab_o_mode, expTab_sat_bits> > T_expTabType;

    typedef ap_fixed<nextStage_WL, nextStage_IL, out_q_mode, out_o_mode, out_sat_bits> T_innerStageOutType;

    typedef ap_fixed<out_WL, out_IL, out_q_mode, out_o_mode, out_sat_bits> T_innerFFTOutType;
    typedef ap_fixed<in_WL, in_IL, in_q_mode, in_o_mode, in_sat_bits> T_innerStageInType;
    typedef ap_fixed<tw_WL, tw_IL, tw_q_mode, tw_o_mode, tw_sat_bits> T_innerTwiddleType;
    typedef ap_fixed<expTab_WL, expTab_IL, expTab_q_mode, expTab_o_mode, expTab_sat_bits> T_innerExpTabType;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

//(((2.2))) std::complex < ap_fixed < WL,IL,ap_q_mode, ap_ovf_mode, int sat_bits > >

// Specialization for ap_fixed with ::SSR_FFT_SCALE:: in this mode the ssr fft t_stage
// level output will grow by log2(t_R) For every t_stage.

/////////////////////////////////////////////////////////////////////////////////////////////////
template <int t_L,
          int t_R,
          int t_stage,
          int tw_WL,
          int tw_IL,
          ap_q_mode tw_q_mode,
          ap_o_mode tw_o_mode,
          int tw_sat_bits,
          int expTab_WL,
          int expTab_IL,
          ap_q_mode expTab_q_mode,
          ap_o_mode expTab_o_mode,
          int expTab_sat_bits,
          int in_WL,
          int in_IL,
          ap_q_mode in_q_mode,
          ap_o_mode in_o_mode,
          int in_sat_bits,
          int out_WL,
          int out_IL,
          ap_q_mode out_q_mode,
          ap_o_mode out_o_mode,
          int out_sat_bits>
struct FFTTraits<SSR_FFT_SCALE,
                 t_L,
                 t_R,
                 t_stage,
                 std::complex<ap_fixed<tw_WL, tw_IL, tw_q_mode, tw_o_mode, tw_sat_bits> >,
                 std::complex<ap_fixed<expTab_WL, expTab_IL, expTab_q_mode, expTab_o_mode, expTab_sat_bits> >,
                 std::complex<ap_fixed<in_WL, in_IL, in_q_mode, in_o_mode, in_sat_bits> >,
                 std::complex<ap_fixed<out_WL, out_IL, out_q_mode, out_o_mode, out_sat_bits> > > {
    static const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
    static const int tp_log2R = ssrFFTLog2<t_R>::val;
    static const unsigned int s = NO_OF_FFT_STAGES - t_stage;
    static const int nextStage_WL = in_WL;
    static const int nextStage_IL = (NO_OF_FFT_STAGES == t_stage) ? in_IL + tp_log2R + 1 : in_IL + tp_log2R;

    typedef std::complex<ap_fixed<nextStage_WL, nextStage_IL, out_q_mode, out_o_mode, out_sat_bits> > T_stageOutType;

    typedef std::complex<ap_fixed<in_WL, in_IL, in_q_mode, in_o_mode, in_sat_bits> > T_stageInType;
    typedef std::complex<ap_fixed<tw_WL, tw_IL, tw_q_mode, tw_o_mode, tw_sat_bits> > T_twiddleType;
    typedef std::complex<ap_fixed<expTab_WL, expTab_IL, expTab_q_mode, expTab_o_mode, expTab_sat_bits> > T_expTabType;

    typedef ap_fixed<nextStage_WL, nextStage_IL, out_q_mode, out_o_mode, out_sat_bits> T_innerStageOutType;

    typedef ap_fixed<out_WL, out_IL, out_q_mode, out_o_mode, out_sat_bits> T_innerFFTOutType;
    typedef ap_fixed<in_WL, in_IL, in_q_mode, in_o_mode, in_sat_bits> T_innerStageInType;
    typedef ap_fixed<tw_WL, tw_IL, tw_q_mode, tw_o_mode, tw_sat_bits> T_innerTwiddleType;
    typedef ap_fixed<expTab_WL, expTab_IL, expTab_q_mode, expTab_o_mode, expTab_sat_bits> T_innerExpTabType;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

//(((2.3))) std::complex < ap_fixed < WL,IL,ap_q_mode, ap_ovf_mode, int sat_bits > >

// Specialization for ap_fixed with ::SSR_FFT_GROW_TO_MAX_WIDTH:: in this mode the ssr fft t_stage
// level output will grow by log2(t_R) For every t_stage and then finally saturate to max allowed
// bit width

/////////////////////////////////////////////////////////////////////////////////////////////////
template <int t_L,
          int t_R,
          int t_stage,
          int tw_WL,
          int tw_IL,
          ap_q_mode tw_q_mode,
          ap_o_mode tw_o_mode,
          int tw_sat_bits,
          int expTab_WL,
          int expTab_IL,
          ap_q_mode expTab_q_mode,
          ap_o_mode expTab_o_mode,
          int expTab_sat_bits,
          int in_WL,
          int in_IL,
          ap_q_mode in_q_mode,
          ap_o_mode in_o_mode,
          int in_sat_bits,
          int out_WL,
          int out_IL,
          ap_q_mode out_q_mode,
          ap_o_mode out_o_mode,
          int out_sat_bits>
struct FFTTraits<SSR_FFT_GROW_TO_MAX_WIDTH,
                 t_L,
                 t_R,
                 t_stage,
                 std::complex<ap_fixed<tw_WL, tw_IL, tw_q_mode, tw_o_mode, tw_sat_bits> >,
                 std::complex<ap_fixed<expTab_WL, expTab_IL, expTab_q_mode, expTab_o_mode, expTab_sat_bits> >,
                 std::complex<ap_fixed<in_WL, in_IL, in_q_mode, in_o_mode, in_sat_bits> >,
                 std::complex<ap_fixed<out_WL, out_IL, out_q_mode, out_o_mode, out_sat_bits> > > {
    static const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
    static const int tp_log2R = ssrFFTLog2<t_R>::val;
    static const unsigned int s = NO_OF_FFT_STAGES - t_stage;

    static const int EXPECTED_NEXT_STAGE_WL = (NO_OF_FFT_STAGES == t_stage) ? in_WL + tp_log2R + 1 : in_WL + tp_log2R;
    static const int EXPECTED_NEXT_STAGE_IL = (NO_OF_FFT_STAGES == t_stage) ? in_IL + tp_log2R + 1 : in_IL + tp_log2R;

    static const int MAX_ALLOWED_BIT_WIDTH_MARGIN = DSP48_OP2_BIT_WIDTH - in_WL;

    // If the expected grown output has bit width larger then max allowed saturate it.
    static const int nextStage_WL =
        (EXPECTED_NEXT_STAGE_WL <= DSP48_OP2_BIT_WIDTH) ? EXPECTED_NEXT_STAGE_WL : DSP48_OP2_BIT_WIDTH;

    static const int nextStage_IL =
        (EXPECTED_NEXT_STAGE_IL <= DSP48_OP2_BIT_WIDTH) ? EXPECTED_NEXT_STAGE_IL : DSP48_OP2_BIT_WIDTH;

    typedef std::complex<ap_fixed<nextStage_WL, nextStage_IL, out_q_mode, out_o_mode, out_sat_bits> > T_stageOutType;

    typedef std::complex<ap_fixed<in_WL, in_IL, in_q_mode, in_o_mode, in_sat_bits> > T_stageInType;
    typedef std::complex<ap_fixed<tw_WL, tw_IL, tw_q_mode, tw_o_mode, tw_sat_bits> > T_twiddleType;
    typedef std::complex<ap_fixed<expTab_WL, expTab_IL, expTab_q_mode, expTab_o_mode, expTab_sat_bits> > T_expTabType;

    typedef ap_fixed<nextStage_WL, nextStage_IL, out_q_mode, out_o_mode, out_sat_bits> T_innerStageOutType;

    typedef ap_fixed<out_WL, out_IL, out_q_mode, out_o_mode, out_sat_bits> T_innerFFTOutType;
    typedef ap_fixed<in_WL, in_IL, in_q_mode, in_o_mode, in_sat_bits> T_innerStageInType;
    typedef ap_fixed<tw_WL, tw_IL, tw_q_mode, tw_o_mode, tw_sat_bits> T_innerTwiddleType;
    typedef ap_fixed<expTab_WL, expTab_IL, expTab_q_mode, expTab_o_mode, expTab_sat_bits> T_innerExpTabType;
};

} // end namespace fft
} // end namespace dsp
} // end namespace xf

///////////////////////////////////////////////////////////////////////

#endif // HLS_SSR_FFT_TRAITS_H_
