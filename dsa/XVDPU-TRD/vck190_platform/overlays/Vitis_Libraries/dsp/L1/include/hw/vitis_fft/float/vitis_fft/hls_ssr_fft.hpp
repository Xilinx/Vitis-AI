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

#ifndef HLS_SSR_FFT_H_
#define HLS_SSR_FFT_H_
//#define __HLS_SSR_FFT_USE_FULL_WAVE_TWIDDLE_TABLE__
//#define HLS_SSR_FFT_DISABLE_NATURAL_ORDER_172836475866778896
#ifndef __SYNTHESIS__
#include <assert.h>
#endif
//#include <complex>
#include <hls_stream.h>

#include "vitis_fft/hls_ssr_fft_fork_merge_utils.hpp"
#include "vitis_fft/hls_ssr_fft_pragma_controls.hpp"
#include "vitis_fft/hls_ssr_fft_enums.hpp"
#include "vitis_fft/hls_ssr_fft_data_reorder.hpp"
#include "vitis_fft/hls_ssr_fft_traits.hpp"
#include "vitis_fft/hls_ssr_fft_twiddle_table.hpp"
#include "vitis_fft/hls_ssr_fft_complex_exp_table.hpp"
#include "vitis_fft/hls_ssr_fft_complex_exp_table_last_stage.hpp"
#include "vitis_fft/hls_ssr_fft_streaming_transposer.hpp"
#include "vitis_fft/hls_ssr_fft_streaming_data_commutor.hpp"
#include "vitis_fft/hls_ssr_fft_adder_tree.hpp"
#include "vitis_fft/hls_ssr_fft_input_traits.hpp"
#include "vitis_fft/hls_ssr_fft_parallel_fft_kernel.hpp"
#include "vitis_fft/hls_ssr_fft_multiplication_traits.hpp"
#include "vitis_fft/fft_complex.hpp"
#include "vitis_fft/hls_ssr_fft_types.hpp"

namespace xf {
namespace dsp {
namespace fft {

/*
 * Convert a 2-D array interface to hls::streaming interface for connection with next
 * stages which are all stream based.
 */
// SSR_FFT_VIVADO_BEGIN
template <int t_L, int t_R, typename T_in, typename T_out>
void castArrayS2Streaming(T_in p_inData[t_R][t_L / t_R], hls::stream<SuperSampleContainer<t_R, T_out> >& p_outData) {
#pragma HLS INLINE off
CONVERT_ARRAY_TO_STREAM_LOOP:
    for (int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 rewind
        SuperSampleContainer<t_R, T_out> temp;

        for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            temp.superSample[r] = p_inData[r][t];
        }
        p_outData.write(temp);
    }
}
// SSR_FFT_VIVADO_END
template <int t_L, int t_R, typename T_in, typename T_out>
void castArrayS2Streaming(hls::stream<T_in> p_inData[t_R], hls::stream<SuperSampleContainer<t_R, T_out> >& p_outData) {
#pragma HLS INLINE off
CONVERT_ARRAY_TO_STREAM_LOOP:
    for (int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 rewind
        SuperSampleContainer<t_R, T_out> temp;

        for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
            temp.superSample[r] = p_inData[r].read();
        }
        p_outData.write(temp);
    }
}

template <int t_L,
          int t_R,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum buterfly_rnd_mode,
          typename T_exp,
          typename T_twiddleInType,
          typename T_twiddleOutType>
void twiddleFactorMulS2S(T_twiddleInType p_inData[t_R],
                         T_twiddleOutType p_outData[t_R],
                         T_exp p_twiddleTable[],
                         int p_k) {
    //#pragma HLS INLINE
    typedef T_twiddleInType inType;
    typedef T_twiddleOutType outType;
//#pragma HLS data_pack variable = p_outData
//#pragma HLS data_pack variable = p_inData

L_TWIDDLE_FACTOR_MUL:
    for (int n = 0; n < t_R; n++) {
#pragma HLS UNROLL /// Unroll Twiddle Factor Multiplication loop, required for proper implementation of SSR FFT

        ap_uint<ssrFFTLog2<t_L>::val> index = n * p_k;
        T_exp exp_factor;
        if (transform_direction == FORWARD_TRANSFORM) {
#ifdef __HLS_SSR_FFT_USE_FULL_WAVE_TWIDDLE_TABLE__
            exp_factor = readTwiddleTable<t_L, t_R, ssrFFTLog2<t_L>::val, T_exp>(index, p_twiddleTable);
#else
            exp_factor = readQuaterTwiddleTable<t_L, t_R, ssrFFTLog2<t_L>::val, T_exp>(index, p_twiddleTable);
#endif
        }

        if (transform_direction == REVERSE_TRANSFORM) {
            exp_factor = readQuaterTwiddleTableReverse<t_L, t_R, ssrFFTLog2<t_L>::val, T_exp>(index, p_twiddleTable);
        }

        complexMultiply(p_inData[n], exp_factor, p_outData[n]);
    }
}

template <int t_L,
          int t_R,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int stage,
          typename T_complexMulOutType,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_in,
          typename T_out>
void fftStageKernelS2S(T_complexExpTableType p_complexExpTable[],
                       T_fftTwiddleType p_twiddleTable[TwiddleTableLENTraits<t_L, t_R>::EXTENDED_TWIDDLE_TALBE_LENGTH],
                       hls::stream<SuperSampleContainer<t_R, T_in> >& p_fftReOrderedInput,
                       hls::stream<SuperSampleContainer<t_R, T_out> >& p_fftOutData_local) {
    //#pragma HLS INLINE recursive

    const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
    const unsigned int s = NO_OF_FFT_STAGES - stage;
    static const bool isFirstStage = (NO_OF_FFT_STAGES == stage);
    const int no_of_ffts_in_stage = ssrFFTPow<t_R, s>::val;
    const int current_fft_length = t_L / ssrFFTPow<t_R, s>::val;
    const int no_bflys_per_fft = current_fft_length / t_R;

L_FFTs_LOOP:
    for (int f = 0; f < no_of_ffts_in_stage; f++) {
    L_BFLYs_LOOP:
        for (int k = 0; k < no_bflys_per_fft; k++) {
#pragma HLS PIPELINE II = 1 rewind
            if (p_fftReOrderedInput.empty()) {
                k--;
            } else {
                T_in X_of_ns[t_R];
                T_complexMulOutType complexExpMulOut[t_R];

                //#pragma HLS data_pack variable = complexExpMulOut

                T_out bflyOutData[t_R];

                //#pragma HLS data_pack variable = bflyOutData

                SuperSampleContainer<t_R, T_in> temp_super_sample_in = p_fftReOrderedInput.read();
            // The following loop should be unrolled for implementation
            L_READ_R_IN_SAMPLES:
                for (int n = 0; n < t_R; n++) {
#pragma HLS UNROLL
                    X_of_ns[n] = temp_super_sample_in.superSample[n];
                }
                Butterfly<t_R> Butterfly_obj;
                Butterfly_obj
                    .template calcButterFly<t_L, isFirstStage, t_scalingMode, transform_direction, butterfly_rnd_mode>(
                        X_of_ns, bflyOutData, p_complexExpTable);
                twiddleFactorMulS2S<t_L, t_R, transform_direction, butterfly_rnd_mode>(
                    bflyOutData, complexExpMulOut, p_twiddleTable, (k << (ssrFFTLog2<t_L / current_fft_length>::val)));

                SuperSampleContainer<t_R, T_out> temp_super_sample_out;
            L_WRITE_R_BUTTERFLY_OUT_SAMPLES:
                for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
                    temp_super_sample_out.superSample[r] = complexExpMulOut[r];
                }
                p_fftOutData_local.write(temp_super_sample_out);
            }
        } // butterflies loop
    }     // sub ffts loop
}

template <int t_L,
          int t_R,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int stage,
          typename T_complexMulOutType,
          typename T_complexExpTableType,
          typename T_in,
          typename T_out>
void fftStageKernelLastStageS2S(T_complexExpTableType p_complexExpTable[],
                                hls::stream<SuperSampleContainer<t_R, T_in> >& p_fftReOrderedInput,
                                hls::stream<SuperSampleContainer<t_R, T_out> >& p_fftOutData_local) {
    //#pragma HLS INLINE recursive

    const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
    const unsigned int s = NO_OF_FFT_STAGES - stage;
    static const bool isFirstStage = (NO_OF_FFT_STAGES == stage);
    const int no_of_ffts_in_stage = ssrFFTPow<t_R, s>::val;
    const int current_fft_length = t_L / ssrFFTPow<t_R, s>::val;
    const int no_bflys_per_fft = current_fft_length / t_R;
    const int iter_max_count = no_of_ffts_in_stage * no_bflys_per_fft;
    int k = 0;
L_FFTs_LOOP:
    for (int iter = 0; iter < iter_max_count; iter++) {
#pragma HLS PIPELINE II = 1 rewind
        if (p_fftReOrderedInput.empty()) {
            iter--;
        } else {
            T_in X_of_ns[t_R];
            //#pragma HLS data_pack variable = X_of_ns
            T_out bflyOutData[t_R];
            //#pragma HLS data_pack variable = bflyOutData
            SuperSampleContainer<t_R, T_in> temp_super_sample_in;
            temp_super_sample_in = p_fftReOrderedInput.read();
        L_READ_R_IN_SAMPLES:
            for (int n = 0; n < t_R; n++) {
#pragma HLS UNROLL
                X_of_ns[n] = temp_super_sample_in.superSample[n];
            }

            Butterfly<t_R> Butterfly_obj;

            Butterfly_obj
                .template calcButterFly<t_L, isFirstStage, t_scalingMode, transform_direction, butterfly_rnd_mode>(
                    X_of_ns, bflyOutData, p_complexExpTable);
            //*******************************************************************************/
            // For last stage there is no need to multiply with twiddles since they are unity.
            // twiddleFactorMul<t_L, t_R>
            //(bflyOutData, complexExpMulOut, p_twiddleTable, k*(t_L / current_fft_length));
            // Should be unrolled
            //*******************************************************************************/
            SuperSampleContainer<t_R, T_out> temp_super_sample_out;
        L_WRITE_R_BUTTERFLY_OUT_SAMPLES:
            for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
                temp_super_sample_out.superSample[r] = bflyOutData[r];
            }
            p_fftOutData_local.write(temp_super_sample_out);
        }
    } // butterflies loop // sub ffts loop
}

// stage kernel is for generalized length FFT , where Length of not power of radix
template <int t_L,
          int t_R,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int stage,
          typename T_complexMulOutType,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_in,
          typename T_out>
void fftStageKernelFullForForkingS2S(
    T_complexExpTableType p_complexExpTable[],
    T_fftTwiddleType p_twiddleTable[TwiddleTableLENTraits<t_L, t_R>::EXTENDED_TWIDDLE_TALBE_LENGTH],
    hls::stream<SuperSampleContainer<t_R, T_in> >& p_fftReOrderedInput,
    hls::stream<SuperSampleContainer<t_R, T_out> >& p_fftOutDataLocal

    ) {
    //#pragma HLS INLINE recursive

    const int NO_OF_FFT_STAGES = (ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val) + 1;
    const unsigned int s = NO_OF_FFT_STAGES - stage;
    static const bool isFirstStage = (NO_OF_FFT_STAGES == stage);
    const int no_of_ffts_in_stage = ssrFFTPow<t_R, s>::val; //((int)pow(t_R, (s)));  // t_L / ((int)pow(t_R, (s + 1)));
    const int current_fft_length = t_L / ssrFFTPow<t_R, s>::val; // t_L / (int)pow(t_R, s); ///(int)pow(t_R, (s + 1));
    /*
     * This is special modification done to generalize SSR FFT : The Number of Butterflies per fft combiner in a stage
     * Go Down by forking_factor
     */
    const int no_bflys_per_fft = current_fft_length / (t_R);

L_FFTs_LOOP:
    for (int f = 0; f < no_of_ffts_in_stage; f++) {
    L_BFLYs_LOOP:
        for (int k = 0; k < no_bflys_per_fft; k++) {
#pragma HLS PIPELINE II = 1 rewind
            if (p_fftReOrderedInput.empty()) {
                k--;
            } else {
                T_in X_of_ns[t_R];
                T_complexMulOutType complexExpMulOut[t_R];
                //#pragma HLS data_pack variable = complexExpMulOut
                T_out bflyOutData[t_R];
                //#pragma HLS data_pack variable = bflyOutData

                // The following loop should be unrolled for implementation
                SuperSampleContainer<t_R, T_in> temp_super_sample_in = p_fftReOrderedInput.read();
            L_READ_R_IN_SAMPLES:
                for (int n = 0; n < t_R; n++) {
#pragma HLS UNROLL
                    X_of_ns[n] = temp_super_sample_in.superSample[n];
                }
                Butterfly<t_R> Butterfly_obj;
                Butterfly_obj
                    .template calcButterFly<t_L, isFirstStage, t_scalingMode, transform_direction, butterfly_rnd_mode>(
                        X_of_ns, bflyOutData, p_complexExpTable);

                twiddleFactorMulS2S<t_L, t_R, transform_direction, butterfly_rnd_mode>

                    (bflyOutData, complexExpMulOut, p_twiddleTable, k << (ssrFFTLog2<t_L / current_fft_length>::val));
                SuperSampleContainer<t_R, T_out> temp_super_sample_out;
            L_WRITE_R_BUTTERFLY_OUT_SAMPLES:
                for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
                    temp_super_sample_out.superSample[r] = complexExpMulOut[r];
                }
                p_fftOutDataLocal.write(temp_super_sample_out);
            }
        } // butterflies loop
    }     // sub ffts loop
}

////////////////////////////////////////////////KERNEL Forking Class////////////////////////////////////START
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* KernelProcessFork : This class will take a [L/R][R] type stream
 *  with R-size sample stream and break it down to R/F streams creating
 *  F new streams of size R/F. Functionally it will take [L/R][R] 2
 *  dimensional array and break it down to F new 2-dimensional arrays
 *  of size [L/R][R/F] to be used by F dataflow processes : here the
 *  processes are FFT Kernel processor
 */

/// Declaration Only
template <int t_forkNumber, int t_forkingFactor, int t_instanceID>
struct KernelProcessForkS2S {
    template <int t_L,
              int t_R,
              int iid,
              scaling_mode_enum t_scalingMode,
              transform_direction_enum transform_direction,
              butterfly_rnd_mode_enum butterfly_rnd_mode,
              int stage,
              typename T_complexMulOutType,
              typename T_complexExpTableType,
              typename T_in,
              typename T_out>
    void fftStageKernelLastStageFork(T_complexExpTableType p_complexExpTable[],
                                     hls::stream<SuperSampleContainer<t_R, T_in> >& p_fftReOrderedInput,
                                     hls::stream<SuperSampleContainer<t_R, T_out> >& p_fftOutDataLocal);
};

template <int t_forkNumber, int t_forkingFactor, int t_instanceID>
template <int t_L,
          int t_R,
          int iid,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int stage,
          typename T_complexMulOutType,
          typename T_complexExpTableType,
          typename T_in,
          typename T_out>
void KernelProcessForkS2S<t_forkNumber, t_forkingFactor, t_instanceID>::fftStageKernelLastStageFork(
    T_complexExpTableType p_complexExpTable[],
    hls::stream<SuperSampleContainer<t_R, T_in> >& p_fftReOrderedInput,
    hls::stream<SuperSampleContainer<t_R, T_out> >& p_fftOutDataLocal)

{
#pragma HLS INLINE off
#pragma HLS dataflow

    hls::stream<SuperSampleContainer<t_R / t_forkingFactor, T_in> > superSampleStreamArray_in[t_forkingFactor];
//#pragma HLS data_pack variable = superSampleStreamArray_in
#pragma HLS RESOURCE variable = superSampleStreamArray_in core = FIFO_LUTRAM
//#pragma HLS ARRAY_PARTITION variable = superSampleStreamArray_in complete dim = 1
#pragma HLS STREAM variable = superSampleStreamArray_in depth = 8

    hls::stream<SuperSampleContainer<t_R / t_forkingFactor, T_out> > superSampleStreamArray_out[t_forkingFactor];
//#pragma HLS data_pack variable = superSampleStreamArray_out
#pragma HLS RESOURCE variable = superSampleStreamArray_out core = FIFO_LUTRAM
//#pragma HLS ARRAY_PARTITION variable = superSampleStreamArray_out complete dim = 1
#pragma HLS STREAM variable = superSampleStreamArray_out depth = 8

    forkSuperSampleStream<t_L, t_R, t_forkingFactor, t_forkingFactor, T_in>(p_fftReOrderedInput,
                                                                            superSampleStreamArray_in);

    for (int fork_no = 0; fork_no < t_forkingFactor; fork_no++) {
#pragma HLS UNROLL
        fftStageKernelLastStageS2S<t_L / t_forkingFactor, t_R / t_forkingFactor, t_scalingMode, transform_direction,
                                   butterfly_rnd_mode, 1, T_complexMulOutType>(
            p_complexExpTable, superSampleStreamArray_in[fork_no], superSampleStreamArray_out[fork_no]);
    }

    mergeSuperSampleStream<t_L, t_R, t_forkingFactor, t_forkingFactor, T_out>(superSampleStreamArray_out,
                                                                              p_fftOutDataLocal);
}

/*
 * ==============================================================================================
 * fft_stage_class : Generates SSR FFT stages: This is a specialized version that is useful for
 * generating FFT stages in the case when Length of the FFT, t_L is an integer power of radix
 * i.e.  t_L = pow ( radix,m), where m is integer
 * ==============================================================================================
 */

template <int t_L,
          int t_R,
          int t_instanceID,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          fft_output_order_enum tp_output_data_order,
          int stage,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_fftIn,
          typename T_fftOut>
class FFTStageClassS2S {
    typedef typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType t_complexExpTableType;
    typedef typename TwiddleTraits<T_fftTwiddleType>::T_twiddleType T_twiddleType;

    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_stageInType T_stageInType;

    static const int TWIDDLE_TALBE_LENGTH = TwiddleTableLENTraits<t_L, t_R>::TWIDDLE_TALBE_LENGTH;

    /// Typedef for next stage:
    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_expTabType T2_expTabType;

    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_twiddleType T2_twiddleType;
    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_stageOutType T2_stageOutType;
    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_stageOutType T_stageOutType;

   public:
    // SSR_FFT_VIVADO_BEGIN
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         T_twiddleType p_twiddleTable[TWIDDLE_TALBE_LENGTH],

                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,

                         T_fftOut p_fftOutData[t_R][t_L / t_R]) {
/// Stage Level inline pragma required for
/// proper implementation of SSR FFT
#pragma HLS INLINE
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
        const int this_stage_pf = ssrFFTPow<t_R, stage - 2>::val;
    /****************************  Function call : Dataflow Pipeline Part 1 ****************************************/
    L_FFT_DATAFLOW_PIPELINE_FUNC1:

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local depth = 8

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local2;

//#pragma HLS data_pack variable = fftOutData_local2
#pragma HLS RESOURCE variable = fftOutData_local2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local2 depth = 8

        fftStageKernelS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage, T_stageOutType>(
            p_complexExpTable, p_twiddleTable, p_fftInData, fftOutData_local);

        static const int t_isLargeMemFlag =
            ((this_stage_pf * t_R > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

        DataCommutationsS2Streaming<t_instanceID, stage, 10000, t_R, t_L, t_R, this_stage_pf, t_isLargeMemFlag> dcomObj;

        dcomObj.template streamingDataCommutor<T_stageOutType>(fftOutData_local, fftOutData_local2);
        FFTStageClassS2S<t_L, t_R, t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode,
                         tp_output_data_order, stage - 1,
                         t_complexExpTableType, // T2_expTabType,
                         T_twiddleType,         // T2_twiddleType,
                         T_stageOutType, T_fftOut>::fftStage(p_complexExpTable, p_twiddleTable, fftOutData_local2,
                                                             p_fftOutData);
    } // end ssr_fft_model function
      // SSR_FFT_VIVADO_END
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         T_twiddleType p_twiddleTable[TWIDDLE_TALBE_LENGTH],

                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,

                         hls::stream<T_fftOut> p_fftOutData[t_R]) {
/// Stage Level inline pragma required for
/// proper implementation of SSR FFT
//#pragma HLS INLINE
#pragma HLS dataflow disable_start_propagation
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
        const int this_stage_pf = ssrFFTPow<t_R, stage - 2>::val;
    /****************************  Function call : Dataflow Pipeline Part 1 ****************************************/
    L_FFT_DATAFLOW_PIPELINE_FUNC1:

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local depth = 8

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local2;

//#pragma HLS data_pack variable = fftOutData_local2
#pragma HLS RESOURCE variable = fftOutData_local2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local2 depth = 8

        fftStageKernelS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage, T_stageOutType>(
            p_complexExpTable, p_twiddleTable, p_fftInData, fftOutData_local);

        static const int t_isLargeMemFlag =
            ((this_stage_pf * t_R > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

        DataCommutationsS2Streaming<t_instanceID, stage, 10000, t_R, t_L, t_R, this_stage_pf, t_isLargeMemFlag> dcomObj;

        dcomObj.template streamingDataCommutor<T_stageOutType>(fftOutData_local, fftOutData_local2);
        FFTStageClassS2S<t_L, t_R, t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode,
                         tp_output_data_order, stage - 1,
                         t_complexExpTableType, // T2_expTabType,
                         T_twiddleType,         // T2_twiddleType,
                         T_stageOutType, T_fftOut>::fftStage(p_complexExpTable, p_twiddleTable, fftOutData_local2,
                                                             p_fftOutData);
    } // end ssr_fft_model function
};
template <int t_L,
          int t_R,
          int t_instanceID,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_fftIn,
          typename T_fftOut>
class FFTStageClassS2S<t_L,
                       t_R,
                       t_instanceID,
                       t_scalingMode,
                       transform_direction,
                       butterfly_rnd_mode,
                       SSR_FFT_DIGIT_REVERSED_TRANSPOSED,
                       1,
                       T_complexExpTableType,
                       T_fftTwiddleType,
                       T_fftIn,
                       T_fftOut> {
    typedef typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType t_complexExpTableType;

    typedef typename TwiddleTraits<T_fftTwiddleType>::T_twiddleType T_twiddleType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageInType T_stageInType;

    static const int TWIDDLE_TALBE_LENGTH = TwiddleTableLENTraits<t_L, t_R>::TWIDDLE_TALBE_LENGTH;

    /// Typedef for next stage:
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_expTabType T2_expTabType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_twiddleType T2_twiddleType;
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageOutType T_stageOutType;
    typedef
        typename FFTScaledOutput<t_L, transform_direction, T_stageOutType>::T_scaledFFTOutputType T_scaledFFTOutputType;

   public:
    // SSR_FFT_VIVADO_BEGIN
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         T_twiddleType p_twiddleTable[TWIDDLE_TALBE_LENGTH],

                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,

                         T_fftOut p_fftOutData[t_R][t_L / t_R]) {
/// Stage Level inline pragma required for
/// proper implementation of SSR FFT
#pragma HLS INLINE
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
    /****************************  Function call : Dataflow Pipeline Part 1 ****************************************/
    L_FFT_DATAFLOW_PIPELINE_FUNC1:

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local depth = 8

        fftStageKernelLastStageS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage,
                                   T_stageOutType

                                   >(p_complexExpTable, p_fftInData, fftOutData_local);
        convertSuperStreamToArrayNScale<stage, transform_direction, 50000, t_L, t_R, T_stageOutType>(fftOutData_local,
                                                                                                     p_fftOutData);

    } // end ssr_fft_model function
      // SSR_FFT_VIVADO_END
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         T_twiddleType p_twiddleTable[TWIDDLE_TALBE_LENGTH],

                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,

                         hls::stream<T_fftOut> p_fftOutData[t_R]) {
/// Stage Level inline pragma required for
/// proper implementation of SSR FFT
//#pragma HLS INLINE
#pragma HLS dataflow
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
    /****************************  Function call : Dataflow Pipeline Part 1 ****************************************/
    L_FFT_DATAFLOW_PIPELINE_FUNC1:

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local depth = 8

        fftStageKernelLastStageS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage,
                                   T_stageOutType

                                   >(p_complexExpTable, p_fftInData, fftOutData_local);
        convertSuperStreamToArrayNScale<stage, transform_direction, 50000, t_L, t_R, T_stageOutType>(fftOutData_local,
                                                                                                     p_fftOutData);

    } // end ssr_fft_model function
};

template <int t_L,
          int t_R,
          int t_instanceID,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_fftIn,
          typename T_fftOut>
class FFTStageClassS2S<t_L,
                       t_R,
                       t_instanceID,
                       t_scalingMode,
                       transform_direction,
                       butterfly_rnd_mode,
                       SSR_FFT_NATURAL,
                       1,
                       T_complexExpTableType,
                       T_fftTwiddleType,
                       T_fftIn,
                       T_fftOut> {
    typedef typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType t_complexExpTableType;

    typedef typename TwiddleTraits<T_fftTwiddleType>::T_twiddleType T_twiddleType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageInType T_stageInType;

    static const int TWIDDLE_TALBE_LENGTH = TwiddleTableLENTraits<t_L, t_R>::TWIDDLE_TALBE_LENGTH;

    /// Typedef for next stage:
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_expTabType T2_expTabType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_twiddleType T2_twiddleType;
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageOutType T_stageOutType;

    typedef
        typename FFTScaledOutput<t_L, transform_direction, T_stageOutType>::T_scaledFFTOutputType T_scaledFFTOutputType;

   public:
    // SSR_FFT_VIVADO_BEGIN
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         T_twiddleType p_twiddleTable[TWIDDLE_TALBE_LENGTH],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         T_fftOut p_fftOutData[t_R][t_L / t_R]) {
#pragma HLS INLINE /// Stage Level inline pragma required for proper implementation of SSR FFT
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
    /****************************  Function call : Dataflow Pipeline Part 1 ****************************************/
    L_FFT_DATAFLOW_PIPELINE_FUNC1:
        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local depth = 8

        hls::stream<T_scaledFFTOutputType> fftOutData_local2[t_R];

//#pragma HLS data_pack variable = fftOutData_local2
#pragma HLS RESOURCE variable = fftOutData_local2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local2 depth = 8

        fftStageKernelLastStageS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage,
                                   T_stageOutType>(p_complexExpTable, p_fftInData, fftOutData_local);

        /*convertSuperStreamToArray<
                                  stage,
                                  50000,
                                  t_L,
                                  t_R,
                                  T_stageOutType
                                  >
                                 (
                                   fftOutData_local,
                                   fftOutData_local2
                                  );*/
        convertSuperStreamToArrayNScale<stage, transform_direction, 50000, t_L, t_R, T_stageOutType>(fftOutData_local,
                                                                                                     fftOutData_local2);
        // last stage so write to fft output to buffer
        digitReversedDataReOrder<t_L, t_R>(fftOutData_local2, p_fftOutData);

    } // end ssr_fft_model function
      // SSR_FFT_VIVADO_END
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         T_twiddleType p_twiddleTable[TWIDDLE_TALBE_LENGTH],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         hls::stream<T_fftOut> p_fftOutData[t_R]) {
#pragma HLS dataflow
        //#pragma HLS INLINE /// Stage Level inline pragma required for proper implementation of SSR FFT
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
    /****************************  Function call : Dataflow Pipeline Part 1 ****************************************/
    L_FFT_DATAFLOW_PIPELINE_FUNC1:
        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local depth = 8

        hls::stream<T_scaledFFTOutputType> fftOutData_local2[t_R];

//#pragma HLS data_pack variable = fftOutData_local2
#pragma HLS RESOURCE variable = fftOutData_local2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local2 depth = 8

        fftStageKernelLastStageS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage,
                                   T_stageOutType>(p_complexExpTable, p_fftInData, fftOutData_local);

        /*convertSuperStreamToArray<
                                  stage,
                                  50000,
                                  t_L,
                                  t_R,
                                  T_stageOutType
                                  >
                                 (
                                   fftOutData_local,
                                   fftOutData_local2
                                  );*/
        convertSuperStreamToArrayNScale<stage, transform_direction, 50000, t_L, t_R, T_stageOutType>(fftOutData_local,
                                                                                                     fftOutData_local2);
        // last stage so write to fft output to buffer
        digitReversedDataReOrder<t_L, t_R>(fftOutData_local2, p_fftOutData);

    } // end ssr_fft_model function
};

template <int t_L,
          int t_R,
          int t_instanceID,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          fft_output_order_enum tp_output_data_order,
          int stage,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_fftIn,
          typename T_fftOut>
class FFTStageClassS2SWithTable {
    typedef typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType t_complexExpTableType;
    typedef typename TwiddleTraits<T_fftTwiddleType>::T_twiddleType T_twiddleType;

    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_stageInType T_stageInType;

    static const int TWIDDLE_TALBE_LENGTH = TwiddleTableLENTraits<t_L, t_R>::TWIDDLE_TALBE_LENGTH;

    /// Typedef for next stage:
    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_expTabType T2_expTabType;

    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_twiddleType T2_twiddleType;
    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_stageOutType T2_stageOutType;
    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_stageOutType T_stageOutType;

   public:
    // SSR_FFT_VIVADO_BEGIN
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,

                         T_fftOut p_fftOutData[t_R][t_L / t_R]) {
/// Stage Level inline pragma required for
/// proper implementation of SSR FFT
#pragma HLS INLINE
        static TwiddleTableWrapper<t_L, t_R, t_instanceID, T_fftTwiddleType> twiddleObj;
        //#pragma HLS RESOURCE variable = twiddleObj.twiddleTable core = ROM_nP_LUTRAM
        //#pragma HLS RESOURCE variable=twiddleObj.twiddleTable
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
        const int this_stage_pf = ssrFFTPow<t_R, stage - 2>::val;
    /****************************  Function call : Dataflow Pipeline Part 1 ****************************************/
    L_FFT_DATAFLOW_PIPELINE_FUNC1:

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local depth = 8

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local2;

//#pragma HLS data_pack variable = fftOutData_local2
#pragma HLS RESOURCE variable = fftOutData_local2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local2 depth = 8

        fftStageKernelS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage, T_stageOutType>(
            p_complexExpTable, twiddleObj.twiddleTable, p_fftInData, fftOutData_local);

        static const int t_isLargeMemFlag =
            ((this_stage_pf * t_R > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

        DataCommutationsS2Streaming<t_instanceID, stage, 10000, t_R, t_L, t_R, this_stage_pf, t_isLargeMemFlag> dcomObj;

        dcomObj.template streamingDataCommutor<T_stageOutType>(fftOutData_local, fftOutData_local2);
        FFTStageClassS2SWithTable<t_L, t_R, t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                  tp_output_data_order, stage - 1, t_complexExpTableType, T_twiddleType, T_stageOutType,
                                  T_fftOut>::fftStage(p_complexExpTable, fftOutData_local2, p_fftOutData);
    } // end ssr_fft_model function
    // SSR_FFT_VIVADO_END
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,

                         hls::stream<T_fftOut> p_fftOutData[t_R]) {
/// Stage Level inline pragma required for
/// proper implementation of SSR FFT
//#pragma HLS INLINE
#pragma HLS dataflow disable_start_propagation
        static TwiddleTableWrapper<t_L, t_R, t_instanceID, T_fftTwiddleType> twiddleObj;
        //#pragma HLS RESOURCE variable = twiddleObj.twiddleTable core = ROM_nP_LUTRAM
        //#pragma HLS RESOURCE variable=twiddleObj.twiddleTable
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
        const int this_stage_pf = ssrFFTPow<t_R, stage - 2>::val;
    /****************************  Function call : Dataflow Pipeline Part 1 ****************************************/
    L_FFT_DATAFLOW_PIPELINE_FUNC1:

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local depth = 8

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local2;

//#pragma HLS data_pack variable = fftOutData_local2
#pragma HLS RESOURCE variable = fftOutData_local2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local2 depth = 8

        fftStageKernelS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage, T_stageOutType>(
            p_complexExpTable, twiddleObj.twiddleTable, p_fftInData, fftOutData_local);

        static const int t_isLargeMemFlag =
            ((this_stage_pf * t_R > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

        DataCommutationsS2Streaming<t_instanceID, stage, 10000, t_R, t_L, t_R, this_stage_pf, t_isLargeMemFlag> dcomObj;

        dcomObj.template streamingDataCommutor<T_stageOutType>(fftOutData_local, fftOutData_local2);
        FFTStageClassS2SWithTable<t_L, t_R, t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                  tp_output_data_order, stage - 1, t_complexExpTableType, T_twiddleType, T_stageOutType,
                                  T_fftOut>::fftStage(p_complexExpTable, fftOutData_local2, p_fftOutData);
    } // end ssr_fft_model function
};
template <int t_L,
          int t_R,
          int t_instanceID,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_fftIn,
          typename T_fftOut>
class FFTStageClassS2SWithTable<t_L,
                                t_R,
                                t_instanceID,
                                t_scalingMode,
                                transform_direction,
                                butterfly_rnd_mode,
                                SSR_FFT_DIGIT_REVERSED_TRANSPOSED,
                                1,
                                T_complexExpTableType,
                                T_fftTwiddleType,
                                T_fftIn,
                                T_fftOut> {
    typedef typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType t_complexExpTableType;

    typedef typename TwiddleTraits<T_fftTwiddleType>::T_twiddleType T_twiddleType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageInType T_stageInType;

    static const int TWIDDLE_TALBE_LENGTH = TwiddleTableLENTraits<t_L, t_R>::TWIDDLE_TALBE_LENGTH;

    /// Typedef for next stage:
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_expTabType T2_expTabType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_twiddleType T2_twiddleType;
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageOutType T_stageOutType;
    typedef
        typename FFTScaledOutput<t_L, transform_direction, T_stageOutType>::T_scaledFFTOutputType T_scaledFFTOutputType;

   public:
    // SSR_FFT_VIVADO_BEGIN
    static void fftStage(t_complexExpTableType p_complexExpTable[],

                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,

                         T_fftOut p_fftOutData[t_R][t_L / t_R]) {
/// Stage Level inline pragma required for
/// proper implementation of SSR FFT
#pragma HLS INLINE
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
    /****************************  Function call : Dataflow Pipeline Part 1 ****************************************/
    L_FFT_DATAFLOW_PIPELINE_FUNC1:

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local depth = 8

        fftStageKernelLastStageS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage,
                                   T_stageOutType

                                   >(p_complexExpTable, p_fftInData, fftOutData_local);
        convertSuperStreamToArrayNScale<stage, transform_direction, 50000, t_L, t_R, T_stageOutType>(fftOutData_local,
                                                                                                     p_fftOutData);

    } // end ssr_fft_model function
      // SSR_FFT_VIVADO_END
    static void fftStage(t_complexExpTableType p_complexExpTable[],

                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,

                         hls::stream<T_fftOut> p_fftOutData[t_R]) {
/// Stage Level inline pragma required for
/// proper implementation of SSR FFT
//#pragma HLS INLINE
#pragma HLS dataflow
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
    /****************************  Function call : Dataflow Pipeline Part 1 ****************************************/
    L_FFT_DATAFLOW_PIPELINE_FUNC1:

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local depth = 8

        fftStageKernelLastStageS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage,
                                   T_stageOutType

                                   >(p_complexExpTable, p_fftInData, fftOutData_local);
        convertSuperStreamToArrayNScale<stage, transform_direction, 50000, t_L, t_R, T_stageOutType>(fftOutData_local,
                                                                                                     p_fftOutData);

    } // end ssr_fft_model function
};

template <int t_L,
          int t_R,
          int t_instanceID,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_fftIn,
          typename T_fftOut>
class FFTStageClassS2SWithTable<t_L,
                                t_R,
                                t_instanceID,
                                t_scalingMode,
                                transform_direction,
                                butterfly_rnd_mode,
                                SSR_FFT_NATURAL,
                                1,
                                T_complexExpTableType,
                                T_fftTwiddleType,
                                T_fftIn,
                                T_fftOut> {
    typedef typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType t_complexExpTableType;

    typedef typename TwiddleTraits<T_fftTwiddleType>::T_twiddleType T_twiddleType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageInType T_stageInType;

    static const int TWIDDLE_TALBE_LENGTH = TwiddleTableLENTraits<t_L, t_R>::TWIDDLE_TALBE_LENGTH;

    /// Typedef for next stage:
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_expTabType T2_expTabType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_twiddleType T2_twiddleType;
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageOutType T_stageOutType;

    typedef
        typename FFTScaledOutput<t_L, transform_direction, T_stageOutType>::T_scaledFFTOutputType T_scaledFFTOutputType;

   public:
    // SSR_FFT_VIVADO_BEGIN
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         T_fftOut p_fftOutData[t_R][t_L / t_R]) {
#pragma HLS INLINE /// Stage Level inline pragma required for proper implementation of SSR FFT
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
    /****************************  Function call : Dataflow Pipeline Part 1 ****************************************/
    L_FFT_DATAFLOW_PIPELINE_FUNC1:
        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local depth = 8

        hls::stream<T_scaledFFTOutputType> fftOutData_local2[t_R];

//#pragma HLS data_pack variable = fftOutData_local2
#pragma HLS RESOURCE variable = fftOutData_local2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local2 depth = 8

        fftStageKernelLastStageS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage,
                                   T_stageOutType>(p_complexExpTable, p_fftInData, fftOutData_local);

        convertSuperStreamToArrayNScale<stage, transform_direction, 50000, t_L, t_R, T_stageOutType>(fftOutData_local,
                                                                                                     fftOutData_local2);
        // last stage so write to fft output to buffer
        digitReversedDataReOrder<t_L, t_R>(fftOutData_local2, p_fftOutData);

    } // end ssr_fft_model function
    // SSR_FFT_VIVADO_END
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         hls::stream<T_fftOut> p_fftOutData[t_R]) {
//#pragma HLS INLINE /// Stage Level inline pragma required for proper implementation of SSR FFT
#pragma HLS dataflow
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
    /****************************  Function call : Dataflow Pipeline Part 1 ****************************************/
    L_FFT_DATAFLOW_PIPELINE_FUNC1:
        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local depth = 8

        hls::stream<T_scaledFFTOutputType> fftOutData_local2[t_R];

//#pragma HLS data_pack variable = fftOutData_local2
#pragma HLS RESOURCE variable = fftOutData_local2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = fftOutData_local2 depth = 8

        fftStageKernelLastStageS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage,
                                   T_stageOutType>(p_complexExpTable, p_fftInData, fftOutData_local);

        convertSuperStreamToArrayNScale<stage, transform_direction, 50000, t_L, t_R, T_stageOutType>(fftOutData_local,
                                                                                                     fftOutData_local2);
        // last stage so write to fft output to buffer
        digitReversedDataReOrder<t_L, t_R>(fftOutData_local2, p_fftOutData);

    } // end ssr_fft_model function
};
// newS2S-end

/*
 * ======================================================================================================================
 * 2nd version of  FFT Stages class used for creating generalized SSR FFT for the cases where FFT Length is
 * not integer power of Radix, It is a fixed specialization of FFT stages classes which will support forking
 * and also deal with exception that are needed for supporting structural model for SSR FFT
 * ======================================================================================================================
 */
template <int tp_outputForkingFactor,
          int t_L,
          int t_R,
          int t_instanceID,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          fft_output_order_enum tp_output_data_order,
          int stage,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_fftIn,
          typename T_fftOut>
class FFTStageClassForkingOutputS2S {
    typedef typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType t_complexExpTableType;

    typedef typename TwiddleTraits<T_fftTwiddleType>::T_twiddleType T_twiddleType;

    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_stageInType T_stageInType;

    static const int TWIDDLE_TALBE_LENGTH = TwiddleTableLENTraits<t_L, t_R>::TWIDDLE_TALBE_LENGTH;

    /// Typedef for next stage:
    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_expTabType T2_expTabType;

    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_twiddleType T2_twiddleType;
    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_stageOutType T_stageOutType;

   public:
    // SSR_FFT_VIVADO_BEGIN
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         T_twiddleType p_twiddleTable[TWIDDLE_TALBE_LENGTH],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         T_fftOut p_fftOutData[t_R][t_L / t_R]) {
/// Stage Level inline pragma required
// for proper implementation of SSR FFT
#pragma HLS INLINE
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
        const int this_stage_pf = (ssrFFTPow<t_R, stage - 2>::val) / tp_outputForkingFactor;
    L_FFT_DATAFLOW_PIPELINE_FUNC1:
        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
        //#pragma HLS data_pack variable = fftOutData_local

        fftStageKernelFullForForkingS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage,
                                        T_stageOutType>(p_complexExpTable, p_twiddleTable, p_fftInData,
                                                        fftOutData_local);
    //**RW Info :: Read and Written in order and declared
    // as stream, p_fftInData declared as stream in top level

    /************  Function call : Dataflow Pipeline Part 2 ************/
    L_FFT_DATAFLOW_PIPELINE_FUNC2:

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local2;

#pragma HLS STREAM variable = fftOutData_local2 depth = 8
#pragma HLS RESOURCE variable = fftOutData_local2 core = FIFO_LUTRAM
        //#pragma HLS data_pack variable = fftOutData_local2

        static const int t_isLargeMemFlag =
            ((this_stage_pf * t_R > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

        DataCommutationsS2Streaming<t_instanceID, stage, 20000, t_R, t_L, t_R, this_stage_pf, t_isLargeMemFlag> dcomObj;
        dcomObj.template streamingDataCommutor<T_stageOutType>(fftOutData_local, fftOutData_local2);

        FFTStageClassForkingOutputS2S<tp_outputForkingFactor, t_L, t_R, t_instanceID, t_scalingMode,
                                      transform_direction, butterfly_rnd_mode, tp_output_data_order, stage - 1,
                                      T_complexExpTableType, T_fftTwiddleType, T_stageOutType,
                                      T_fftOut>::fftStage(p_complexExpTable, p_complexExpTableForkingStage,
                                                          p_twiddleTable, fftOutData_local2, p_fftOutData);
    } // end ssr_fft_model function
      // SSR_FFT_VIVADO_END
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         T_twiddleType p_twiddleTable[TWIDDLE_TALBE_LENGTH],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         hls::stream<T_fftOut> p_fftOutData[t_R]) {
#pragma HLS dataflow disable_start_propagation
        /// Stage Level inline pragma required
        // for proper implementation of SSR FFT
        //#pragma HLS INLINE
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
        const int this_stage_pf = (ssrFFTPow<t_R, stage - 2>::val) / tp_outputForkingFactor;
    L_FFT_DATAFLOW_PIPELINE_FUNC1:
        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
        //#pragma HLS data_pack variable = fftOutData_local

        fftStageKernelFullForForkingS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage,
                                        T_stageOutType>(p_complexExpTable, p_twiddleTable, p_fftInData,
                                                        fftOutData_local);
    //**RW Info :: Read and Written in order and declared
    // as stream, p_fftInData declared as stream in top level

    /************  Function call : Dataflow Pipeline Part 2 ************/
    L_FFT_DATAFLOW_PIPELINE_FUNC2:

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local2;

#pragma HLS STREAM variable = fftOutData_local2 depth = 8
#pragma HLS RESOURCE variable = fftOutData_local2 core = FIFO_LUTRAM
        //#pragma HLS data_pack variable = fftOutData_local2

        static const int t_isLargeMemFlag =
            ((this_stage_pf * t_R > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

        DataCommutationsS2Streaming<t_instanceID, stage, 20000, t_R, t_L, t_R, this_stage_pf, t_isLargeMemFlag> dcomObj;
        dcomObj.template streamingDataCommutor<T_stageOutType>(fftOutData_local, fftOutData_local2);

        FFTStageClassForkingOutputS2S<tp_outputForkingFactor, t_L, t_R, t_instanceID, t_scalingMode,
                                      transform_direction, butterfly_rnd_mode, tp_output_data_order, stage - 1,
                                      T_complexExpTableType, T_fftTwiddleType, T_stageOutType,
                                      T_fftOut>::fftStage(p_complexExpTable, p_complexExpTableForkingStage,
                                                          p_twiddleTable, fftOutData_local2, p_fftOutData);
    } // end ssr_fft_model function
};
/*
 * ======================================================================================================================
 * stage==2 Template Specialization of "fft_stage_class_forkingOutput" (This specialization will always create 2nd last
 * stage in the SSR FFT stages and this stage has special construction, the data commutations blocks is not present in
 * this stage) the 2nd version of  FFT Stages class used for creating generalized SSR FFT for the cases where FFT Length
 * is not integer power of Radix, It is a fixed specialization of FFT stages classes which will support forking and also
 * deal with exception that are needed for supporting structural model for SSR FFT
 * ======================================================================================================================
 */

template <int tp_outputForkingFactor,
          int t_L,
          int t_R,
          int t_instanceID,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          fft_output_order_enum tp_output_data_order,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_fftIn,
          typename T_fftOut>
class FFTStageClassForkingOutputS2S<tp_outputForkingFactor,
                                    t_L,
                                    t_R,
                                    t_instanceID,
                                    t_scalingMode,
                                    transform_direction,
                                    butterfly_rnd_mode,
                                    tp_output_data_order,
                                    2,
                                    T_complexExpTableType,
                                    T_fftTwiddleType,
                                    T_fftIn,
                                    T_fftOut>

{
    typedef typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType t_complexExpTableType;

    typedef typename TwiddleTraits<T_fftTwiddleType>::T_twiddleType T_twiddleType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 2, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageInType T_stageInType;

    static const int TWIDDLE_TALBE_LENGTH = TwiddleTableLENTraits<t_L, t_R>::TWIDDLE_TALBE_LENGTH;

    /// Typedef for next stage:
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 2, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_expTabType T2_expTabType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 2, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_twiddleType T2_twiddleType;
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 2, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageOutType T_stageOutType;

   public:
    // SSR_FFT_VIVADO_BEGIN
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         T_twiddleType p_twiddleTable[TWIDDLE_TALBE_LENGTH],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         T_fftOut p_fftOutData[t_R][t_L / t_R]) {
/// Stage Level inline pragma required
// for proper implementation of SSR FFT
#pragma HLS INLINE
        // +1 is added for rounding in this case, bcoz log L is not multple of log R

        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val + 1;

        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - 2;

        const int this_stage_pf = (ssrFFTPow<t_R, 2 - 2>::val / tp_outputForkingFactor)
                                      ? (ssrFFTPow<t_R, 2 - 2>::val / tp_outputForkingFactor)
                                      : 1;

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM

        /// Replaced for : fft_stage_kernel , because of stage
        // calculation exceptions
        fftStageKernelFullForForkingS2S<

            t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, 2, T_stageOutType>(
            p_complexExpTable, p_twiddleTable, p_fftInData, fftOutData_local);
        //**RW Ifno :: Read and Written in order and declared
        // as stream, p_fftInData declared as stream in top level
        // Array of Streams to be used for forking the output
        // into forkingFactor number of arrays
        hls::stream<SuperSampleContainer<t_R / tp_outputForkingFactor, T_stageOutType> >
            superStreamArray[tp_outputForkingFactor];

#pragma HLS STREAM variable = superStreamArray depth = 8
#pragma HLS RESOURCE variable = superStreamArray core = FIFO_LUTRAM
        //#pragma HLS ARRAY_PARTITION variable = superStreamArray complete dim = 1
        //#pragma HLS data_pack variable = superStreamArray

        hls::stream<SuperSampleContainer<t_R / tp_outputForkingFactor, T_stageOutType> >
            superStreamArray_out[tp_outputForkingFactor];

#pragma HLS STREAM variable = superStreamArray_out depth = 8
#pragma HLS RESOURCE variable = superStreamArray_out core = FIFO_LUTRAM
        //#pragma HLS ARRAY_PARTITION variable = superStreamArray_out complete dim = 1
        //#pragma HLS data_pack variable = superStreamArray_out

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > mergedSuperStream;

#pragma HLS STREAM variable = mergedSuperStream depth = 8
//#pragma HLS data_pack variable = mergedSuperStream
#pragma HLS RESOURCE variable = mergedSuperStream core = FIFO_LUTRAM

        forkSuperSampleStream<t_L, t_R, tp_outputForkingFactor, tp_outputForkingFactor, T_stageOutType>(
            fftOutData_local, superStreamArray);
        StreamingDataCommutorForkS2S<t_instanceID, 2, 30000, tp_outputForkingFactor, t_L, t_R, this_stage_pf,
                                     tp_outputForkingFactor>
            dcom_obj;
        dcom_obj.template forkedCompute<T_stageOutType>(superStreamArray, superStreamArray_out);
        mergeSuperSampleStreamNonInvertOut<t_L, t_R, tp_outputForkingFactor, tp_outputForkingFactor, T_stageOutType>(
            superStreamArray_out, mergedSuperStream);

        FFTStageClassForkingOutputS2S<tp_outputForkingFactor, t_L, t_R, t_instanceID, t_scalingMode,
                                      transform_direction, butterfly_rnd_mode, tp_output_data_order, 2 - 1,
                                      T_complexExpTableType, T_fftTwiddleType, T_stageOutType,
                                      T_fftOut>::fftStage(p_complexExpTable, p_complexExpTableForkingStage,
                                                          p_twiddleTable, mergedSuperStream, p_fftOutData);
    } // end ssr_fft_model function
      // SSR_FFT_VIVADO_END
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         T_twiddleType p_twiddleTable[TWIDDLE_TALBE_LENGTH],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         hls::stream<T_fftOut> p_fftOutData[t_R]) {
#pragma HLS dataflow
        /// Stage Level inline pragma required
        // for proper implementation of SSR FFT
        //#pragma HLS INLINE
        // +1 is added for rounding in this case, bcoz log L is not multple of log R

        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val + 1;

        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - 2;

        const int this_stage_pf = (ssrFFTPow<t_R, 2 - 2>::val / tp_outputForkingFactor)
                                      ? (ssrFFTPow<t_R, 2 - 2>::val / tp_outputForkingFactor)
                                      : 1;

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM

        /// Replaced for : fft_stage_kernel , because of stage
        // calculation exceptions
        fftStageKernelFullForForkingS2S<

            t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, 2, T_stageOutType>(
            p_complexExpTable, p_twiddleTable, p_fftInData, fftOutData_local);
        //**RW Ifno :: Read and Written in order and declared
        // as stream, p_fftInData declared as stream in top level
        // Array of Streams to be used for forking the output
        // into forkingFactor number of arrays
        hls::stream<SuperSampleContainer<t_R / tp_outputForkingFactor, T_stageOutType> >
            superStreamArray[tp_outputForkingFactor];

#pragma HLS STREAM variable = superStreamArray depth = 8
#pragma HLS RESOURCE variable = superStreamArray core = FIFO_LUTRAM
        //#pragma HLS ARRAY_PARTITION variable = superStreamArray complete dim = 1
        //#pragma HLS data_pack variable = superStreamArray

        hls::stream<SuperSampleContainer<t_R / tp_outputForkingFactor, T_stageOutType> >
            superStreamArray_out[tp_outputForkingFactor];

#pragma HLS STREAM variable = superStreamArray_out depth = 8
#pragma HLS RESOURCE variable = superStreamArray_out core = FIFO_LUTRAM
        //#pragma HLS ARRAY_PARTITION variable = superStreamArray_out complete dim = 1
        //#pragma HLS data_pack variable = superStreamArray_out

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > mergedSuperStream;

#pragma HLS STREAM variable = mergedSuperStream depth = 8
//#pragma HLS data_pack variable = mergedSuperStream
#pragma HLS RESOURCE variable = mergedSuperStream core = FIFO_LUTRAM

        forkSuperSampleStream<t_L, t_R, tp_outputForkingFactor, tp_outputForkingFactor, T_stageOutType>(
            fftOutData_local, superStreamArray);
        StreamingDataCommutorForkS2S<t_instanceID, 2, 30000, tp_outputForkingFactor, t_L, t_R, this_stage_pf,
                                     tp_outputForkingFactor>
            dcom_obj;
        dcom_obj.template forkedCompute<T_stageOutType>(superStreamArray, superStreamArray_out);
        mergeSuperSampleStreamNonInvertOut<t_L, t_R, tp_outputForkingFactor, tp_outputForkingFactor, T_stageOutType>(
            superStreamArray_out, mergedSuperStream);

        FFTStageClassForkingOutputS2S<tp_outputForkingFactor, t_L, t_R, t_instanceID, t_scalingMode,
                                      transform_direction, butterfly_rnd_mode, tp_output_data_order, 2 - 1,
                                      T_complexExpTableType, T_fftTwiddleType, T_stageOutType,
                                      T_fftOut>::fftStage(p_complexExpTable, p_complexExpTableForkingStage,
                                                          p_twiddleTable, mergedSuperStream, p_fftOutData);
    } // end ssr_fft_model function
};

/*
 * ======================================================================================================================
 * STAGE==1 , fft_output_data_order=SSR_FFT_DIGIT_REVERSED_TRANSPOSED, Template Specialization of
 * "fft_stage_class_forkingOutput" (This specialization will always create last stage in the SSR FFT stages and this
 * stage has special CONSTRUCTION, two blocks are not present the data commutations and ComplexMul
 * @@ Also this stage will fork kernel computation blocks.
 * the 2nd version of  FFT Stages class used for creating generalized SSR FFT for the cases where FFT Length is not
 * integer power of Radix, It is a fixed specialization of FFT stages classes which will support forking and also deal
 * with exception that are needed for supporting structural model for SSR FFT
 * ======================================================================================================================
 */

template <int tp_outputForkingFactor,
          int t_L,
          int t_R,
          int t_instanceID,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_fftIn,
          typename T_fftOut>
class FFTStageClassForkingOutputS2S<tp_outputForkingFactor,
                                    t_L,
                                    t_R,
                                    t_instanceID,
                                    t_scalingMode,
                                    transform_direction,
                                    butterfly_rnd_mode,
                                    SSR_FFT_DIGIT_REVERSED_TRANSPOSED,
                                    1,
                                    T_complexExpTableType,
                                    T_fftTwiddleType,
                                    T_fftIn,
                                    T_fftOut>

{
    typedef typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType t_complexExpTableType;

    typedef typename TwiddleTraits<T_fftTwiddleType>::T_twiddleType T_twiddleType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageInType T_stageInType;

    static const int TWIDDLE_TALBE_LENGTH = TwiddleTableLENTraits<t_L, t_R>::TWIDDLE_TALBE_LENGTH;

    /// Typedef for next stage:
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_expTabType T2_expTabType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_twiddleType T2_twiddleType;
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageOutType T_stageOutType;

   public:
    // SSR_FFT_VIVADO_BEGIN
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         T_twiddleType p_twiddleTable[TWIDDLE_TALBE_LENGTH],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         T_fftOut p_fftOutData[t_R][t_L / t_R]) {
#pragma HLS INLINE
        /// Stage Level inline pragma required
        // for proper implementation of SSR FFT
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM

        /*
         * ==================================================================================================================
         * Last kernel stage that will create fork of PARFFT or kernel computation block without data commutors and
         * complexMul
         * ==================================================================================================================
         */

        KernelProcessForkS2S<tp_outputForkingFactor, tp_outputForkingFactor, t_instanceID> KernelProcessFork_obj;

        KernelProcessFork_obj.template fftStageKernelLastStageFork<
            t_L, t_R, t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode, stage, T_stageOutType>(
            p_complexExpTableForkingStage, p_fftInData, fftOutData_local);

        convertSuperStreamToArrayNScale<stage, transform_direction, 59999, t_L, t_R>(fftOutData_local, p_fftOutData);

    } // end ssr_fft_model function
      // SSR_FFT_VIVADO_END
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         T_twiddleType p_twiddleTable[TWIDDLE_TALBE_LENGTH],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         hls::stream<T_fftOut> p_fftOutData[t_R]) {
#pragma HLS dataflow
        //#pragma HLS INLINE
        /// Stage Level inline pragma required
        // for proper implementation of SSR FFT
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM

        /*
         * ==================================================================================================================
         * Last kernel stage that will create fork of PARFFT or kernel computation block without data commutors and
         * complexMul
         * ==================================================================================================================
         */

        KernelProcessForkS2S<tp_outputForkingFactor, tp_outputForkingFactor, t_instanceID> KernelProcessFork_obj;

        KernelProcessFork_obj.template fftStageKernelLastStageFork<
            t_L, t_R, t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode, stage, T_stageOutType>(
            p_complexExpTableForkingStage, p_fftInData, fftOutData_local);

        convertSuperStreamToArrayNScale<stage, transform_direction, 59999, t_L, t_R>(fftOutData_local, p_fftOutData);

    } // end ssr_fft_model function
};

/*
 * ======================================================================================================================
 * STAGE==1 Template Specialization, fft_output_data_order=SSR_FFT_NATURAL
 *  of "fft_stage_class_forkingOutput" (This specialization will always create last stage
 * in the SSR FFT stages and this stage has special CONSTRUCTION, two blocks are not present
 * the data commutations and ComplexMul
 * @@ Also this stage will fork kernel computation blocks.
 * the 2nd version of  FFT Stages class used for creating generalized SSR FFT for the cases where FFT Length is not
 * integer power of Radix, It is a fixed specialization of FFT stages classes which will support forking and also deal
 * with exception that are needed for supporting structural model for SSR FFT
 * ======================================================================================================================
 */

template <int tp_outputForkingFactor,
          int t_L,
          int t_R,
          int t_instanceID,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_fftIn,
          typename T_fftOut>
class FFTStageClassForkingOutputS2S<tp_outputForkingFactor,
                                    t_L,
                                    t_R,
                                    t_instanceID,
                                    t_scalingMode,
                                    transform_direction,
                                    butterfly_rnd_mode,
                                    SSR_FFT_NATURAL,
                                    1,
                                    T_complexExpTableType,
                                    T_fftTwiddleType,
                                    T_fftIn,
                                    T_fftOut>

{
    typedef typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType t_complexExpTableType;

    typedef typename TwiddleTraits<T_fftTwiddleType>::T_twiddleType T_twiddleType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageInType T_stageInType;

    static const int TWIDDLE_TALBE_LENGTH = TwiddleTableLENTraits<t_L, t_R>::TWIDDLE_TALBE_LENGTH;

    /// Typedef for next stage:
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_expTabType T2_expTabType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_twiddleType T2_twiddleType;
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageOutType T_stageOutType;

    typedef
        typename FFTScaledOutput<t_L, transform_direction, T_stageOutType>::T_scaledFFTOutputType T_scaledFFTOutputType;

   public:
    // SSR_FFT_VIVADO_BEGIN
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         T_twiddleType p_twiddleTable[TWIDDLE_TALBE_LENGTH],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         T_fftOut p_fftOutData[t_R][t_L / t_R]) {
/// Stage Level inline pragma required
// for proper implementation of SSR FFT
#pragma HLS INLINE
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM

        hls::stream<SuperSampleContainer<t_R, T_scaledFFTOutputType> > fftOutData_scaled;
#pragma HLS STREAM variable = fftOutData_scaled depth = 8
//#pragma HLS data_pack variable = fftOutData_scaled
#pragma HLS RESOURCE variable = fftOutData_scaled core = FIFO_LUTRAM
        KernelProcessForkS2S<tp_outputForkingFactor, tp_outputForkingFactor, t_instanceID> KernelProcessFork_obj;

        KernelProcessFork_obj.template fftStageKernelLastStageFork<
            t_L, t_R, t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode, stage, T_stageOutType>(

            p_complexExpTableForkingStage, p_fftInData, fftOutData_local);

        superStreamNScale<1, transform_direction, t_instanceID, t_L, t_R>(fftOutData_local, fftOutData_scaled);
        OutputDataReOrder<(t_L) / (t_R * t_R)> OutputDataReOrder_obj;

        OutputDataReOrder_obj.template digitReversal2Phase<t_L, t_R>(fftOutData_scaled, p_fftOutData);

    } // end ssr_fft_model function
      // SSR_FFT_VIVADO_END
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         T_twiddleType p_twiddleTable[TWIDDLE_TALBE_LENGTH],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         hls::stream<T_fftOut> p_fftOutData[t_R]) {
#pragma HLS dataflow
        /// Stage Level inline pragma required
        // for proper implementation of SSR FFT
        //#pragma HLS INLINE
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM

        hls::stream<SuperSampleContainer<t_R, T_scaledFFTOutputType> > fftOutData_scaled;
#pragma HLS STREAM variable = fftOutData_scaled depth = 8
//#pragma HLS data_pack variable = fftOutData_scaled
#pragma HLS RESOURCE variable = fftOutData_scaled core = FIFO_LUTRAM
        KernelProcessForkS2S<tp_outputForkingFactor, tp_outputForkingFactor, t_instanceID> KernelProcessFork_obj;

        KernelProcessFork_obj.template fftStageKernelLastStageFork<
            t_L, t_R, t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode, stage, T_stageOutType>(

            p_complexExpTableForkingStage, p_fftInData, fftOutData_local);

        superStreamNScale<1, transform_direction, t_instanceID, t_L, t_R>(fftOutData_local, fftOutData_scaled);
        OutputDataReOrder<(t_L) / (t_R * t_R)> OutputDataReOrder_obj;

        OutputDataReOrder_obj.template digitReversal2Phase<t_L, t_R>(fftOutData_scaled, p_fftOutData);

    } // end ssr_fft_model function
};

template <int tp_outputForkingFactor,
          int t_L,
          int t_R,
          int t_instanceID,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          fft_output_order_enum tp_output_data_order,
          int stage,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_fftIn,
          typename T_fftOut>
class FFTForkingStage {
    typedef typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType t_complexExpTableType;

    typedef typename TwiddleTraits<T_fftTwiddleType>::T_twiddleType T_twiddleType;

    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_stageInType T_stageInType;

    static const int TWIDDLE_TALBE_LENGTH = TwiddleTableLENTraits<t_L, t_R>::TWIDDLE_TALBE_LENGTH;

    /// Typedef for next stage:
    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_expTabType T2_expTabType;

    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_twiddleType T2_twiddleType;
    typedef
        typename FFTTraits<t_scalingMode, t_L, t_R, stage, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
            T_stageOutType T_stageOutType;

   public:
    // SSR_FFT_VIVADO_BEGIN
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         T_fftOut p_fftOutData[t_R][t_L / t_R]) {
/// Stage Level inline pragma required
// for proper implementation of SSR FFT
#pragma HLS INLINE

        static TwiddleTableWrapper<t_L, t_R, t_instanceID, T_fftTwiddleType> twiddleObj;
        //#pragma HLS RESOURCE variable = twiddleObj.twiddleTable core = ROM_nP_LUTRAM

        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
        const int this_stage_pf = (ssrFFTPow<t_R, stage - 2>::val) / tp_outputForkingFactor;
    L_FFT_DATAFLOW_PIPELINE_FUNC1:
        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
        //#pragma HLS data_pack variable = fftOutData_local

        fftStageKernelFullForForkingS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage,
                                        T_stageOutType>(p_complexExpTable, twiddleObj.twiddleTable, p_fftInData,
                                                        fftOutData_local);
    //**RW Info :: Read and Written in order and declared
    // as stream, p_fftInData declared as stream in top level

    /************  Function call : Dataflow Pipeline Part 2 ************/
    L_FFT_DATAFLOW_PIPELINE_FUNC2:

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local2;

#pragma HLS STREAM variable = fftOutData_local2 depth = 8
#pragma HLS RESOURCE variable = fftOutData_local2 core = FIFO_LUTRAM
        //#pragma HLS data_pack variable = fftOutData_local2

        static const int t_isLargeMemFlag =
            ((this_stage_pf * t_R > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

        DataCommutationsS2Streaming<t_instanceID, stage, 20000, t_R, t_L, t_R, this_stage_pf, t_isLargeMemFlag> dcomObj;
        dcomObj.template streamingDataCommutor<T_stageOutType>(fftOutData_local, fftOutData_local2);

        FFTForkingStage<tp_outputForkingFactor, t_L, t_R, t_instanceID, t_scalingMode, transform_direction,
                        butterfly_rnd_mode, tp_output_data_order, stage - 1, T_complexExpTableType, T_fftTwiddleType,
                        T_stageOutType, T_fftOut>::fftStage(p_complexExpTable, p_complexExpTableForkingStage,
                                                            fftOutData_local2, p_fftOutData);
    } // end ssr_fft_model function
      // SSR_FFT_VIVADO_END
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         hls::stream<T_fftOut> p_fftOutData[t_R]) {
/// Stage Level inline pragma required
// for proper implementation of SSR FFT
//#pragma HLS INLINE
#pragma HLS dataflow disable_start_propagation

        static TwiddleTableWrapper<t_L, t_R, t_instanceID, T_fftTwiddleType> twiddleObj;
        //#pragma HLS RESOURCE variable = twiddleObj.twiddleTable core = ROM_nP_LUTRAM

        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;
        const int this_stage_pf = (ssrFFTPow<t_R, stage - 2>::val) / tp_outputForkingFactor;
    L_FFT_DATAFLOW_PIPELINE_FUNC1:
        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM
        //#pragma HLS data_pack variable = fftOutData_local

        fftStageKernelFullForForkingS2S<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, stage,
                                        T_stageOutType>(p_complexExpTable, twiddleObj.twiddleTable, p_fftInData,
                                                        fftOutData_local);
    //**RW Info :: Read and Written in order and declared
    // as stream, p_fftInData declared as stream in top level

    /************  Function call : Dataflow Pipeline Part 2 ************/
    L_FFT_DATAFLOW_PIPELINE_FUNC2:

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local2;

#pragma HLS STREAM variable = fftOutData_local2 depth = 8
#pragma HLS RESOURCE variable = fftOutData_local2 core = FIFO_LUTRAM
        //#pragma HLS data_pack variable = fftOutData_local2

        static const int t_isLargeMemFlag =
            ((this_stage_pf * t_R > SSR_FFT_URAM_SELECTION_THRESHHOLD) && SSR_FFT_USE_URAMS);

        DataCommutationsS2Streaming<t_instanceID, stage, 20000, t_R, t_L, t_R, this_stage_pf, t_isLargeMemFlag> dcomObj;
        dcomObj.template streamingDataCommutor<T_stageOutType>(fftOutData_local, fftOutData_local2);

        FFTForkingStage<tp_outputForkingFactor, t_L, t_R, t_instanceID, t_scalingMode, transform_direction,
                        butterfly_rnd_mode, tp_output_data_order, stage - 1, T_complexExpTableType, T_fftTwiddleType,
                        T_stageOutType, T_fftOut>::fftStage(p_complexExpTable, p_complexExpTableForkingStage,
                                                            fftOutData_local2, p_fftOutData);
    } // end ssr_fft_model function
};
/*
 * ======================================================================================================================
 * stage==2 Template Specialization of "fft_stage_class_forkingOutput" (This specialization will always create 2nd last
 * stage in the SSR FFT stages and this stage has special construction, the data commutations blocks is not present in
 * this stage) the 2nd version of  FFT Stages class used for creating generalized SSR FFT for the cases where FFT Length
 * is not integer power of Radix, It is a fixed specialization of FFT stages classes which will support forking and also
 * deal with exception that are needed for supporting structural model for SSR FFT
 * ======================================================================================================================
 */

template <int tp_outputForkingFactor,
          int t_L,
          int t_R,
          int t_instanceID,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          fft_output_order_enum tp_output_data_order,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_fftIn,
          typename T_fftOut>
class FFTForkingStage<tp_outputForkingFactor,
                      t_L,
                      t_R,
                      t_instanceID,
                      t_scalingMode,
                      transform_direction,
                      butterfly_rnd_mode,
                      tp_output_data_order,
                      2,
                      T_complexExpTableType,
                      T_fftTwiddleType,
                      T_fftIn,
                      T_fftOut>

{
    typedef typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType t_complexExpTableType;

    typedef typename TwiddleTraits<T_fftTwiddleType>::T_twiddleType T_twiddleType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 2, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageInType T_stageInType;

    static const int TWIDDLE_TALBE_LENGTH = TwiddleTableLENTraits<t_L, t_R>::TWIDDLE_TALBE_LENGTH;

    /// Typedef for next stage:
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 2, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_expTabType T2_expTabType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 2, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_twiddleType T2_twiddleType;
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 2, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageOutType T_stageOutType;

   public:
    // SSR_FFT_VIVADO_BEGIN
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         T_fftOut p_fftOutData[t_R][t_L / t_R]) {
/// Stage Level inline pragma required
// for proper implementation of SSR FFT
#pragma HLS INLINE
        static TwiddleTableWrapper<t_L, t_R, t_instanceID, T_fftTwiddleType> twiddleObj;
        //#pragma HLS RESOURCE variable = twiddleObj.twiddleTable core = ROM_nP_LUTRAM

        // +1 is added for rounding in this case, bcoz log L is not multple of log R

        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val + 1;

        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - 2;

        const int this_stage_pf = (ssrFFTPow<t_R, 2 - 2>::val / tp_outputForkingFactor)
                                      ? (ssrFFTPow<t_R, 2 - 2>::val / tp_outputForkingFactor)
                                      : 1;

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM

        /// Replaced for : fft_stage_kernel , because of stage
        // calculation exceptions
        fftStageKernelFullForForkingS2S<

            t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, 2, T_stageOutType>(
            p_complexExpTable, twiddleObj.twiddleTable, p_fftInData, fftOutData_local);
        //**RW Ifno :: Read and Written in order and declared
        // as stream, p_fftInData declared as stream in top level
        // Array of Streams to be used for forking the output
        // into forkingFactor number of arrays
        hls::stream<SuperSampleContainer<t_R / tp_outputForkingFactor, T_stageOutType> >
            superStreamArray[tp_outputForkingFactor];

#pragma HLS STREAM variable = superStreamArray depth = 8
#pragma HLS RESOURCE variable = superStreamArray core = FIFO_LUTRAM
        //#pragma HLS ARRAY_PARTITION variable = superStreamArray complete dim = 1
        //#pragma HLS data_pack variable = superStreamArray

        hls::stream<SuperSampleContainer<t_R / tp_outputForkingFactor, T_stageOutType> >
            superStreamArray_out[tp_outputForkingFactor];

#pragma HLS STREAM variable = superStreamArray_out depth = 8
#pragma HLS RESOURCE variable = superStreamArray_out core = FIFO_LUTRAM
        //#pragma HLS ARRAY_PARTITION variable = superStreamArray_out complete dim = 1
        //#pragma HLS data_pack variable = superStreamArray_out

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > mergedSuperStream;

#pragma HLS STREAM variable = mergedSuperStream depth = 8
//#pragma HLS data_pack variable = mergedSuperStream
#pragma HLS RESOURCE variable = mergedSuperStream core = FIFO_LUTRAM

        forkSuperSampleStream<t_L, t_R, tp_outputForkingFactor, tp_outputForkingFactor, T_stageOutType>(
            fftOutData_local, superStreamArray);
        StreamingDataCommutorForkS2S<t_instanceID, 2, 30000, tp_outputForkingFactor, t_L, t_R, this_stage_pf,
                                     tp_outputForkingFactor>
            dcom_obj;
        dcom_obj.template forkedCompute<T_stageOutType>(superStreamArray, superStreamArray_out);
        mergeSuperSampleStreamNonInvertOut<t_L, t_R, tp_outputForkingFactor, tp_outputForkingFactor, T_stageOutType>(
            superStreamArray_out, mergedSuperStream);

        FFTForkingStage<tp_outputForkingFactor, t_L, t_R, t_instanceID, t_scalingMode, transform_direction,
                        butterfly_rnd_mode, tp_output_data_order, 2 - 1, T_complexExpTableType, T_fftTwiddleType,
                        T_stageOutType, T_fftOut>::fftStage(p_complexExpTable, p_complexExpTableForkingStage,
                                                            mergedSuperStream, p_fftOutData);
    } // end ssr_fft_model function
      // SSR_FFT_VIVADO_END
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         hls::stream<T_fftOut> p_fftOutData[t_R]) {
/// Stage Level inline pragma required
// for proper implementation of SSR FFT
//#pragma HLS INLINE
#pragma HLS dataflow disable_start_propagation
        static TwiddleTableWrapper<t_L, t_R, t_instanceID, T_fftTwiddleType> twiddleObj;
        //#pragma HLS RESOURCE variable = twiddleObj.twiddleTable core = ROM_nP_LUTRAM

        // +1 is added for rounding in this case, bcoz log L is not multple of log R

        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val + 1;

        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - 2;

        const int this_stage_pf = (ssrFFTPow<t_R, 2 - 2>::val / tp_outputForkingFactor)
                                      ? (ssrFFTPow<t_R, 2 - 2>::val / tp_outputForkingFactor)
                                      : 1;

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM

        /// Replaced for : fft_stage_kernel , because of stage
        // calculation exceptions
        fftStageKernelFullForForkingS2S<

            t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode, 2, T_stageOutType>(
            p_complexExpTable, twiddleObj.twiddleTable, p_fftInData, fftOutData_local);
        //**RW Ifno :: Read and Written in order and declared
        // as stream, p_fftInData declared as stream in top level
        // Array of Streams to be used for forking the output
        // into forkingFactor number of arrays
        hls::stream<SuperSampleContainer<t_R / tp_outputForkingFactor, T_stageOutType> >
            superStreamArray[tp_outputForkingFactor];

#pragma HLS STREAM variable = superStreamArray depth = 8
#pragma HLS RESOURCE variable = superStreamArray core = FIFO_LUTRAM
        //#pragma HLS ARRAY_PARTITION variable = superStreamArray complete dim = 1
        //#pragma HLS data_pack variable = superStreamArray

        hls::stream<SuperSampleContainer<t_R / tp_outputForkingFactor, T_stageOutType> >
            superStreamArray_out[tp_outputForkingFactor];

#pragma HLS STREAM variable = superStreamArray_out depth = 8
#pragma HLS RESOURCE variable = superStreamArray_out core = FIFO_LUTRAM
        //#pragma HLS ARRAY_PARTITION variable = superStreamArray_out complete dim = 1
        //#pragma HLS data_pack variable = superStreamArray_out

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > mergedSuperStream;

#pragma HLS STREAM variable = mergedSuperStream depth = 8
//#pragma HLS data_pack variable = mergedSuperStream
#pragma HLS RESOURCE variable = mergedSuperStream core = FIFO_LUTRAM

        forkSuperSampleStream<t_L, t_R, tp_outputForkingFactor, tp_outputForkingFactor, T_stageOutType>(
            fftOutData_local, superStreamArray);
        StreamingDataCommutorForkS2S<t_instanceID, 2, 30000, tp_outputForkingFactor, t_L, t_R, this_stage_pf,
                                     tp_outputForkingFactor>
            dcom_obj;
        dcom_obj.template forkedCompute<T_stageOutType>(superStreamArray, superStreamArray_out);
        mergeSuperSampleStreamNonInvertOut<t_L, t_R, tp_outputForkingFactor, tp_outputForkingFactor, T_stageOutType>(
            superStreamArray_out, mergedSuperStream);

        FFTForkingStage<tp_outputForkingFactor, t_L, t_R, t_instanceID, t_scalingMode, transform_direction,
                        butterfly_rnd_mode, tp_output_data_order, 2 - 1, T_complexExpTableType, T_fftTwiddleType,
                        T_stageOutType, T_fftOut>::fftStage(p_complexExpTable, p_complexExpTableForkingStage,
                                                            mergedSuperStream, p_fftOutData);
    } // end ssr_fft_model function
};

/*
 * ======================================================================================================================
 * STAGE==1 , fft_output_data_order=SSR_FFT_DIGIT_REVERSED_TRANSPOSED, Template Specialization of
 * "fft_stage_class_forkingOutput" (This specialization will always create last stage in the SSR FFT stages and this
 * stage has special CONSTRUCTION, two blocks are not present the data commutations and ComplexMul
 * @@ Also this stage will fork kernel computation blocks.
 * the 2nd version of  FFT Stages class used for creating generalized SSR FFT for the cases where FFT Length is not
 * integer power of Radix, It is a fixed specialization of FFT stages classes which will support forking and also deal
 * with exception that are needed for supporting structural model for SSR FFT
 * ======================================================================================================================
 */

template <int tp_outputForkingFactor,
          int t_L,
          int t_R,
          int t_instanceID,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_fftIn,
          typename T_fftOut>
class FFTForkingStage<tp_outputForkingFactor,
                      t_L,
                      t_R,
                      t_instanceID,
                      t_scalingMode,
                      transform_direction,
                      butterfly_rnd_mode,
                      SSR_FFT_DIGIT_REVERSED_TRANSPOSED,
                      1,
                      T_complexExpTableType,
                      T_fftTwiddleType,
                      T_fftIn,
                      T_fftOut>

{
    typedef typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType t_complexExpTableType;

    typedef typename TwiddleTraits<T_fftTwiddleType>::T_twiddleType T_twiddleType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageInType T_stageInType;

    static const int TWIDDLE_TALBE_LENGTH = TwiddleTableLENTraits<t_L, t_R>::TWIDDLE_TALBE_LENGTH;

    /// Typedef for next stage:
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_expTabType T2_expTabType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_twiddleType T2_twiddleType;
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageOutType T_stageOutType;

   public:
    // SSR_FFT_VIVADO_BEGIN
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         T_fftOut p_fftOutData[t_R][t_L / t_R]) {
#pragma HLS INLINE
        /// Stage Level inline pragma required
        // for proper implementation of SSR FFT
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM

        /*
         * ==================================================================================================================
         * Last kernel stage that will create fork of PARFFT or kernel computation block without data commutors and
         * complexMul
         * ==================================================================================================================
         */

        KernelProcessForkS2S<tp_outputForkingFactor, tp_outputForkingFactor, t_instanceID> KernelProcessFork_obj;

        KernelProcessFork_obj.template fftStageKernelLastStageFork<
            t_L, t_R, t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode, stage, T_stageOutType>(
            p_complexExpTableForkingStage, p_fftInData, fftOutData_local);

        convertSuperStreamToArrayNScale<stage, transform_direction, 59999, t_L, t_R>(fftOutData_local, p_fftOutData);

    } // end ssr_fft_model function
      // SSR_FFT_VIVADO_END
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         hls::stream<T_fftOut> p_fftOutData[t_R]) {
#pragma HLS dataflow
        //#pragma HLS INLINE
        /// Stage Level inline pragma required
        // for proper implementation of SSR FFT
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM

        /*
         * ==================================================================================================================
         * Last kernel stage that will create fork of PARFFT or kernel computation block without data commutors and
         * complexMul
         * ==================================================================================================================
         */

        KernelProcessForkS2S<tp_outputForkingFactor, tp_outputForkingFactor, t_instanceID> KernelProcessFork_obj;

        KernelProcessFork_obj.template fftStageKernelLastStageFork<
            t_L, t_R, t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode, stage, T_stageOutType>(
            p_complexExpTableForkingStage, p_fftInData, fftOutData_local);

        convertSuperStreamToArrayNScale<stage, transform_direction, 59999, t_L, t_R>(fftOutData_local, p_fftOutData);

    } // end ssr_fft_model function
};

/*
 * ======================================================================================================================
 * STAGE==1 Template Specialization, fft_output_data_order=SSR_FFT_NATURAL
 *  of "fft_stage_class_forkingOutput" (This specialization will always create last stage
 * in the SSR FFT stages and this stage has special CONSTRUCTION, two blocks are not present
 * the data commutations and ComplexMul
 * @@ Also this stage will fork kernel computation blocks.
 * the 2nd version of  FFT Stages class used for creating generalized SSR FFT for the cases where FFT Length is not
 * integer power of Radix, It is a fixed specialization of FFT stages classes which will support forking and also deal
 * with exception that are needed for supporting structural model for SSR FFT
 * ======================================================================================================================
 */

template <int tp_outputForkingFactor,
          int t_L,
          int t_R,
          int t_instanceID,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          typename T_complexExpTableType,
          typename T_fftTwiddleType,
          typename T_fftIn,
          typename T_fftOut>
class FFTForkingStage<tp_outputForkingFactor,
                      t_L,
                      t_R,
                      t_instanceID,
                      t_scalingMode,
                      transform_direction,
                      butterfly_rnd_mode,
                      SSR_FFT_NATURAL,
                      1,
                      T_complexExpTableType,
                      T_fftTwiddleType,
                      T_fftIn,
                      T_fftOut>

{
    typedef typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType t_complexExpTableType;

    typedef typename TwiddleTraits<T_fftTwiddleType>::T_twiddleType T_twiddleType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageInType T_stageInType;

    static const int TWIDDLE_TALBE_LENGTH = TwiddleTableLENTraits<t_L, t_R>::TWIDDLE_TALBE_LENGTH;

    /// Typedef for next stage:
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_expTabType T2_expTabType;

    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_twiddleType T2_twiddleType;
    typedef typename FFTTraits<t_scalingMode, t_L, t_R, 1, T_fftTwiddleType, T_complexExpTableType, T_fftIn, T_fftOut>::
        T_stageOutType T_stageOutType;

    typedef
        typename FFTScaledOutput<t_L, transform_direction, T_stageOutType>::T_scaledFFTOutputType T_scaledFFTOutputType;

   public:
    // SSR_FFT_VIVADO_BEGIN
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         T_fftOut p_fftOutData[t_R][t_L / t_R]) {
/// Stage Level inline pragma required
// for proper implementation of SSR FFT
#pragma HLS INLINE
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM

        hls::stream<SuperSampleContainer<t_R, T_scaledFFTOutputType> > fftOutData_scaled;
#pragma HLS STREAM variable = fftOutData_scaled depth = 8
//#pragma HLS data_pack variable = fftOutData_scaled
#pragma HLS RESOURCE variable = fftOutData_scaled core = FIFO_LUTRAM
        KernelProcessForkS2S<tp_outputForkingFactor, tp_outputForkingFactor, t_instanceID> KernelProcessFork_obj;

        KernelProcessFork_obj.template fftStageKernelLastStageFork<
            t_L, t_R, t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode, stage, T_stageOutType>(

            p_complexExpTableForkingStage, p_fftInData, fftOutData_local);

        superStreamNScale<1, transform_direction, t_instanceID, t_L, t_R>(fftOutData_local, fftOutData_scaled);
        OutputDataReOrder<(t_L) / (t_R * t_R)> OutputDataReOrder_obj;

        OutputDataReOrder_obj.template digitReversal2Phase<t_L, t_R>(fftOutData_scaled, p_fftOutData);

    } // end ssr_fft_model function
      // SSR_FFT_VIVADO_END
    static void fftStage(t_complexExpTableType p_complexExpTable[],
                         t_complexExpTableType p_complexExpTableForkingStage[],
                         hls::stream<SuperSampleContainer<t_R, T_stageInType> >& p_fftInData,
                         hls::stream<T_fftOut> p_fftOutData[t_R]) {
/// Stage Level inline pragma required
// for proper implementation of SSR FFT
//#pragma HLS INLINE
#pragma HLS dataflow
        const int stage = 1;
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        const int tp_log2R = ssrFFTLog2<t_R>::val;
        const unsigned int s = NO_OF_FFT_STAGES - stage;

        hls::stream<SuperSampleContainer<t_R, T_stageOutType> > fftOutData_local;

#pragma HLS STREAM variable = fftOutData_local depth = 8
//#pragma HLS data_pack variable = fftOutData_local
#pragma HLS RESOURCE variable = fftOutData_local core = FIFO_LUTRAM

        hls::stream<SuperSampleContainer<t_R, T_scaledFFTOutputType> > fftOutData_scaled;
#pragma HLS STREAM variable = fftOutData_scaled depth = 8
//#pragma HLS data_pack variable = fftOutData_scaled
#pragma HLS RESOURCE variable = fftOutData_scaled core = FIFO_LUTRAM
        KernelProcessForkS2S<tp_outputForkingFactor, tp_outputForkingFactor, t_instanceID> KernelProcessFork_obj;

        KernelProcessFork_obj.template fftStageKernelLastStageFork<
            t_L, t_R, t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode, stage, T_stageOutType>(

            p_complexExpTableForkingStage, p_fftInData, fftOutData_local);

        superStreamNScale<1, transform_direction, t_instanceID, t_L, t_R>(fftOutData_local, fftOutData_scaled);
        OutputDataReOrder<(t_L) / (t_R * t_R)> OutputDataReOrder_obj;

        OutputDataReOrder_obj.template digitReversal2Phase<t_L, t_R>(fftOutData_scaled, p_fftOutData);

    } // end ssr_fft_model function
};

template <int t_L, int t_R, typename T_in, typename T_out>
void castArray(T_in p_inData[t_R][t_L / t_R], T_out p_outData[t_R][t_L / t_R]) {
#pragma HLS INLINE off
    for (int t = 0; t < t_L / t_R; t++) {
#pragma HLS PIPELINE II = 1 rewind
        for (int r = 0; r < t_R; r++) {
            p_outData[r][t] = p_inData[r][t];
        }
    }
}

template <int t_L,
          int t_R,
          scaling_mode_enum t_scalingMode,
          transform_direction_enum transform_direction,
          butterfly_rnd_mode_enum butterfly_rnd_mode,
          int stage,
          typename T_complexMulOutType,
          typename T_complexExpTableType,
          typename T_in,
          typename T_out>
void fftStageKernelLastStageOld(T_complexExpTableType complexExpTable[],
                                T_in p_fftReOrderedInput[t_R][t_L / t_R],
                                T_out p_fftOutDataLocal[t_R][t_L / t_R]) {
#pragma HLS INLINE recursive
    const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
    const unsigned int s = NO_OF_FFT_STAGES - stage;
    static const bool isFirstStage = (NO_OF_FFT_STAGES == stage);
    const int no_of_ffts_in_stage = ssrFFTPow<t_R, s>::val; //((int)pow(t_R, (s)));  // t_L / ((int)pow(t_R, (s + 1)));
    const int current_fft_length = t_L / ssrFFTPow<t_R, s>::val; // t_L / (int)pow(t_R, s); ///(int)pow(t_R, (s + 1));
    const int no_bflys_per_fft = current_fft_length / t_R;       //(int)pow(t_R, (s));
L_FFTs_LOOP: // fft_length/(radix^(stage+1)) L=64,R=4, S=0 the FFTs=16
    for (int f = 0; f < no_of_ffts_in_stage; f++) {
#pragma HLS PIPELINE II = 1 rewind // This rewind created apparent deadlock that are not detected.
    L_BFLYs_LOOP:                  // This loop calculates butterflies in a sub FFT that is part of a stage
        for (int k = 0; k < no_bflys_per_fft; k++) // Here bf is actually the k in FFT : frequency index
        {
#pragma HLS LOOP_FLATTEN
            T_in X_of_ns[t_R];
            //#pragma HLS data_pack variable = X_of_ns
            T_out bflyOutData[t_R];
        //#pragma HLS data_pack variable = bflyOutData
        L_READ_R_IN_SAMPLES:
            for (int n = 0; n < t_R; n++) {
#pragma HLS UNROLL
                // replcaed//X_of_ns[n] = p_fftReOrderedInput[n][(no_bflys_per_fft*f + k)];
                X_of_ns[n] = p_fftReOrderedInput[n][(f << ssrFFTLog2<no_bflys_per_fft>::val) + k];
            }
            Butterfly<t_R> Butterfly_obj;
            Butterfly_obj
                .template calcButterFly<t_L, isFirstStage, t_scalingMode, transform_direction, butterfly_rnd_mode>(
                    X_of_ns, bflyOutData, complexExpTable);
        //*******************************************************************************/
        // For last stage there is no need to multiply with twiddles since they are unity.
        // twiddleFactorMul<t_L, t_R>
        //(bflyOutData, complexExpMulOut, twiddleTable, k*(t_L / current_fft_length));
        //*******************************************************************************/
        L_WRITE_R_BUTTERFLY_OUT_SAMPLES:
            for (int r = 0; r < t_R; r++) {
#pragma HLS UNROLL
                // replaced//p_fftOutDataLocal[r][no_bflys_per_fft*f + k]=bflyOutData[r];
                p_fftOutDataLocal[r][(f << ssrFFTLog2<no_bflys_per_fft>::val) + k] = bflyOutData[r];
            }
        } // butterflies loop
    }     // sub ffts loop
}

template <int t_isForkedFFT, int t_isTiny, int t_instanceID>
struct FFTWrapper // Declaration only
{
    // SSR_FFT_VIVADO_BEGIN
    template <int t_L,
              int t_R,
              int iid,
              scaling_mode_enum t_scalingMode,
              transform_direction_enum transform_direction,
              butterfly_rnd_mode_enum butterfly_rnd_mode,
              fft_output_order_enum tp_output_data_order,
              typename T_complexExpTableType,
              typename T_fftTwiddleType,
              typename T_in,
              typename T_out>
    static void innerFFT(T_in p_fftInData[t_R][t_L / t_R], T_out p_fftOutData[t_R][t_L / t_R]);
    // SSR_FFT_VIVADO_END
    template <int t_L,
              int t_R,
              int iid,
              scaling_mode_enum t_scalingMode,
              transform_direction_enum transform_direction,
              butterfly_rnd_mode_enum butterfly_rnd_mode,
              fft_output_order_enum tp_output_data_order,
              typename T_complexExpTableType,
              typename T_fftTwiddleType,
              typename T_in,
              typename T_out>
    static void innerFFT(hls::stream<T_in> p_fftInData[t_R], hls::stream<T_out> p_fftOutData[t_R]);
};

/*
 * ===========================================================================================================================================
 * // The Specialization when the FFT is not forked, tiny does not matter so this specialization is used in all the
 * cases
 * ===========================================================================================================================================
 */
template <int t_instanceID>
struct FFTWrapper<0, 0, t_instanceID> // The Specialization when the FFT is not forked, tiny does not matter so this
                                      // specialization is used in all the cases
{
    // SSR_FFT_VIVADO_BEGIN
    template <int t_L,
              int t_R,
              int iid,
              scaling_mode_enum t_scalingMode,
              transform_direction_enum transform_direction,
              butterfly_rnd_mode_enum butterfly_rnd_mode,
              fft_output_order_enum tp_output_data_order,
              typename T_complexExpTableType,
              typename T_fftTwiddleType,
              typename T_in,
              typename T_out>
    static void innerFFT(T_in p_fftInData[t_R][t_L / t_R], T_out p_fftOutData[t_R][t_L / t_R]) {
#pragma HLS INLINE
#ifndef SSR_FFT_SEPERATE_REAL_IMAG_PARTS

//#pragma HLS data_pack variable = p_fftInData
//#pragma HLS data_pack variable = p_fftOutData

#endif

#ifdef SSR_FFT_PARTITION_IO_ARRAYS // SSR_FFT_PARTITION_INTERFACE_ARRAYS

#pragma HLS ARRAY_PARTITION variable = p_fftInData complete dim = 1
#pragma HLS ARRAY_PARTITION variable = p_fftOutData complete dim = 1

#else

#pragma HLS ARRAY_RESHAPE variable = p_fftInData complete dim = 1
#pragma HLS ARRAY_RESHAPE variable = p_fftOutData complete dim = 1

#endif

#pragma HLS STREAM variable = p_fftInData depth = 8
#pragma HLS STREAM variable = p_fftOutData depth = 8
#pragma HLS RESOURCE variable = p_fftInData core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = p_fftOutData core = FIFO_LUTRAM

#ifndef __SYNTHESIS__
        assert((t_R) == (ssrFFTPow<2, ssrFFTLog2<t_R>::val>::val)); // radix should be power of 2 always
        assert((t_L) == (ssrFFTPow<2, ssrFFTLog2<t_L>::val>::val)); // Length of FFT should be power of 2 always
#endif
        T_complexExpTableType complexExpTable[ComplexExpTableLENTraits<0, t_L, t_R>::EXTENDED_EXP_TALBE_LENGTH];
#pragma HLS ARRAY_PARTITION variable = complexExpTable complete dim = 1
        ComplexExpTable<t_R, transform_direction, butterfly_rnd_mode,
                        typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType>::
            initComplexExpTable(complexExpTable);
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        typedef typename FFTInputTraits<T_in>::T_castedType casted_type;
        hls::stream<SuperSampleContainer<t_R, casted_type> > p_fftInData_reOrdered;
//#pragma HLS data_pack variable = p_fftInData_reOrdered
#pragma HLS STREAM variable = p_fftInData_reOrdered depth = 8
#pragma HLS RESOURCE variable = p_fftInData_reOrdered core = FIFO_LUTRAM
        hls::stream<SuperSampleContainer<t_R, casted_type> > casted_output;
//#pragma HLS data_pack variable = casted_output
#pragma HLS STREAM variable = casted_output depth = 8
#pragma HLS RESOURCE variable = casted_output core = FIFO_LUTRAM

        castArrayS2Streaming<t_L, t_R, T_in, casted_type>(p_fftInData, casted_output);
        // InputTransposeChainStreamingS2S<NO_OF_FFT_STAGES-1,1,iid> swapObj;
        // swapObj.template swap<t_L,t_R,iid,1>(casted_output, p_fftInData_reOrdered);
        // template <int t_instanceID, int stage,int sub_stage,int tp_fork_number,int t_L, int t_R, int t_PF, int
        // tp_outputForkingFactor>
        InputTransposeChainStreamingS2S<t_instanceID, NO_OF_FFT_STAGES - 1, 40000, t_R, t_L, t_R, 1, 1> swapObj;

        swapObj.template swap<casted_type>(casted_output, p_fftInData_reOrdered);
        FFTStageClassS2SWithTable<
            t_L, t_R, iid, t_scalingMode, transform_direction, butterfly_rnd_mode, tp_output_data_order,
            NO_OF_FFT_STAGES, T_complexExpTableType, T_fftTwiddleType,
            typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType, T_complexExpTableType,
                               casted_type, T_out>::T_stageInType,
            typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                     casted_type>::T_FFTOutType>::fftStage(complexExpTable, p_fftInData_reOrdered,
                                                                           p_fftOutData);
    }
    // SSR_FFT_VIVADO_END
    template <int t_L,
              int t_R,
              int iid,
              scaling_mode_enum t_scalingMode,
              transform_direction_enum transform_direction,
              butterfly_rnd_mode_enum butterfly_rnd_mode,
              fft_output_order_enum tp_output_data_order,
              typename T_complexExpTableType,
              typename T_fftTwiddleType,
              typename T_in,
              typename T_out>
    static void innerFFT(hls::stream<T_in> p_fftInData[t_R], hls::stream<T_out> p_fftOutData[t_R]) {
#pragma HLS dataflow disable_start_propagation
//#pragma HLS INLINE

#ifndef __SYNTHESIS__
        assert((t_R) == (ssrFFTPow<2, ssrFFTLog2<t_R>::val>::val)); // radix should be power of 2 always
        assert((t_L) == (ssrFFTPow<2, ssrFFTLog2<t_L>::val>::val)); // Length of FFT should be power of 2 always
#endif
        T_complexExpTableType complexExpTable[ComplexExpTableLENTraits<0, t_L, t_R>::EXTENDED_EXP_TALBE_LENGTH];
#pragma HLS ARRAY_PARTITION variable = complexExpTable complete dim = 1
        ComplexExpTable<t_R, transform_direction, butterfly_rnd_mode,
                        typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType>::
            initComplexExpTable(complexExpTable);
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        typedef typename FFTInputTraits<T_in>::T_castedType casted_type;
        hls::stream<SuperSampleContainer<t_R, casted_type> > p_fftInData_reOrdered;
//#pragma HLS data_pack variable = p_fftInData_reOrdered
#pragma HLS STREAM variable = p_fftInData_reOrdered depth = 8
#pragma HLS RESOURCE variable = p_fftInData_reOrdered core = FIFO_LUTRAM
        hls::stream<SuperSampleContainer<t_R, casted_type> > casted_output;
//#pragma HLS data_pack variable = casted_output
#pragma HLS STREAM variable = casted_output depth = 8
#pragma HLS RESOURCE variable = casted_output core = FIFO_LUTRAM

        castArrayS2Streaming<t_L, t_R, T_in, casted_type>(p_fftInData, casted_output);
        // InputTransposeChainStreamingS2S<NO_OF_FFT_STAGES-1,1,iid> swapObj;
        // swapObj.template swap<t_L,t_R,iid,1>(casted_output, p_fftInData_reOrdered);
        // template <int t_instanceID, int stage,int sub_stage,int tp_fork_number,int t_L, int t_R, int t_PF, int
        // tp_outputForkingFactor>
        InputTransposeChainStreamingS2S<t_instanceID, NO_OF_FFT_STAGES - 1, 40000, t_R, t_L, t_R, 1, 1> swapObj;

        swapObj.template swap<casted_type>(casted_output, p_fftInData_reOrdered);
        FFTStageClassS2SWithTable<
            t_L, t_R, iid, t_scalingMode, transform_direction, butterfly_rnd_mode, tp_output_data_order,
            NO_OF_FFT_STAGES, T_complexExpTableType, T_fftTwiddleType,
            typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType, T_complexExpTableType,
                               casted_type, T_out>::T_stageInType,
            typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                     casted_type>::T_FFTOutType>::fftStage(complexExpTable, p_fftInData_reOrdered,
                                                                           p_fftOutData);
    }
};

template <int t_instanceID>
struct FFTWrapper<0, 1, t_instanceID> // The Specialization when the FFT is not forked, tiny does not matter so this
                                      // specialization is used in all the cases
{
    // SSR_FFT_VIVADO_BEGIN
    template <int t_L,
              int t_R,
              int iid,
              scaling_mode_enum t_scalingMode,
              transform_direction_enum transform_direction,
              butterfly_rnd_mode_enum butterfly_rnd_mode,
              fft_output_order_enum tp_output_data_order,
              typename T_complexExpTableType,
              typename T_fftTwiddleType,
              typename T_in,
              typename T_out>
    static void innerFFT(T_in p_fftInData[t_R][t_L / t_R], T_out p_fftOutData[t_R][t_L / t_R]) {
#pragma HLS INLINE
#ifndef SSR_FFT_SEPERATE_REAL_IMAG_PARTS

//#pragma HLS data_pack variable = p_fftInData
//#pragma HLS data_pack variable = p_fftOutData

#endif

#ifdef SSR_FFT_PARTITION_IO_ARRAYS // SSR_FFT_PARTITION_INTERFACE_ARRAYS

#pragma HLS ARRAY_PARTITION variable = p_fftInData complete dim = 1
#pragma HLS ARRAY_PARTITION variable = p_fftOutData complete dim = 1

#else

#pragma HLS ARRAY_RESHAPE variable = p_fftInData complete dim = 1
#pragma HLS ARRAY_RESHAPE variable = p_fftOutData complete dim = 1

#endif

#pragma HLS STREAM variable = p_fftInData depth = 8
#pragma HLS STREAM variable = p_fftOutData depth = 8
#pragma HLS RESOURCE variable = p_fftInData core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = p_fftOutData core = FIFO_LUTRAM

#ifndef __SYNTHESIS__
        assert((t_R) == (ssrFFTPow<2, ssrFFTLog2<t_R>::val>::val)); // radix should be power of 2 always
        assert((t_L) == (ssrFFTPow<2, ssrFFTLog2<t_L>::val>::val)); // Length of FFt should be power of 2 always
#endif

        T_complexExpTableType complexExpTable[ComplexExpTableLENTraits<0, t_L, t_R>::EXTENDED_EXP_TALBE_LENGTH];

#pragma HLS ARRAY_PARTITION variable = complexExpTable complete dim = 1
        ComplexExpTable<t_R, transform_direction, butterfly_rnd_mode,
                        typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType>::
            initComplexExpTable(complexExpTable);
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        typedef typename FFTInputTraits<T_in>::T_castedType casted_type;
        hls::stream<SuperSampleContainer<t_R, casted_type> > p_fftInData_reOrdered;
//#pragma HLS data_pack variable = p_fftInData_reOrdered
#pragma HLS STREAM variable = p_fftInData_reOrdered depth = 8
#pragma HLS RESOURCE variable = p_fftInData_reOrdered core = FIFO_LUTRAM
        hls::stream<SuperSampleContainer<t_R, casted_type> > casted_output;
//#pragma HLS data_pack variable = casted_output
#pragma HLS STREAM variable = casted_output depth = 8
#pragma HLS RESOURCE variable = casted_output core = FIFO_LUTRAM
        castArrayS2Streaming<t_L, t_R, T_in, casted_type>(p_fftInData, casted_output);
        InputTransposeChainStreamingS2S<t_instanceID, NO_OF_FFT_STAGES - 1, 50000, t_R, t_L, t_R, 1, 1> swapObj;
        swapObj.template swap<casted_type>(casted_output, p_fftInData_reOrdered);

        FFTStageClassS2S<t_L, t_R, iid, t_scalingMode, transform_direction, butterfly_rnd_mode, tp_output_data_order,
                         NO_OF_FFT_STAGES,
                         typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType,
                                            T_complexExpTableType, casted_type, T_out>::T_expTabType,
                         typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType,
                                            T_complexExpTableType, casted_type, T_out>::T_twiddleType,
                         typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType,
                                            T_complexExpTableType, casted_type, T_out>::T_stageInType,
                         typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                                  casted_type>::T_FFTOutType>::fftStage(complexExpTable,
                                                                                        p_fftInData_reOrdered,
                                                                                        p_fftOutData);
    }
    // SSR_FFT_VIVADO_END
    template <int t_L,
              int t_R,
              int iid,
              scaling_mode_enum t_scalingMode,
              transform_direction_enum transform_direction,
              butterfly_rnd_mode_enum butterfly_rnd_mode,
              fft_output_order_enum tp_output_data_order,
              typename T_complexExpTableType,
              typename T_fftTwiddleType,
              typename T_in,
              typename T_out>
    static void innerFFT(hls::stream<T_in> p_fftInData[t_R], hls::stream<T_out> p_fftOutData[t_R]) {
//#pragma HLS INLINE
#pragma HLS dataflow

#ifndef __SYNTHESIS__
        assert((t_R) == (ssrFFTPow<2, ssrFFTLog2<t_R>::val>::val)); // radix should be power of 2 always
        assert((t_L) == (ssrFFTPow<2, ssrFFTLog2<t_L>::val>::val)); // Length of FFt should be power of 2 always
#endif

        T_complexExpTableType complexExpTable[ComplexExpTableLENTraits<0, t_L, t_R>::EXTENDED_EXP_TALBE_LENGTH];
#pragma HLS ARRAY_PARTITION variable = complexExpTable complete dim = 1
        ComplexExpTable<t_R, transform_direction, butterfly_rnd_mode,
                        typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType>::
            initComplexExpTable(complexExpTable);
        const int NO_OF_FFT_STAGES = ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val;
        typedef typename FFTInputTraits<T_in>::T_castedType casted_type;
        hls::stream<SuperSampleContainer<t_R, casted_type> > p_fftInData_reOrdered;
//#pragma HLS data_pack variable = p_fftInData_reOrdered
#pragma HLS STREAM variable = p_fftInData_reOrdered depth = 8
#pragma HLS RESOURCE variable = p_fftInData_reOrdered core = FIFO_LUTRAM
        hls::stream<SuperSampleContainer<t_R, casted_type> > casted_output;
//#pragma HLS data_pack variable = casted_output
#pragma HLS STREAM variable = casted_output depth = 8
#pragma HLS RESOURCE variable = casted_output core = FIFO_LUTRAM
        castArrayS2Streaming<t_L, t_R, T_in, casted_type>(p_fftInData, casted_output);
        InputTransposeChainStreamingS2S<t_instanceID, NO_OF_FFT_STAGES - 1, 50000, t_R, t_L, t_R, 1, 1> swapObj;
        swapObj.template swap<casted_type>(casted_output, p_fftInData_reOrdered);

        FFTStageClassS2S<t_L, t_R, iid, t_scalingMode, transform_direction, butterfly_rnd_mode, tp_output_data_order,
                         NO_OF_FFT_STAGES,
                         typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType,
                                            T_complexExpTableType, casted_type, T_out>::T_expTabType,
                         typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType,
                                            T_complexExpTableType, casted_type, T_out>::T_twiddleType,
                         typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType,
                                            T_complexExpTableType, casted_type, T_out>::T_stageInType,
                         typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                                  casted_type>::T_FFTOutType>::fftStage(complexExpTable,
                                                                                        p_fftInData_reOrdered,
                                                                                        p_fftOutData);
    }
};
// The Specialization when the FFT is Forked and Also Tiny needs special Handling for for input arrays,
// the reshaping cannot be used for input arrays instead partitioning is required
// t_isForkedFFT > 0 means it is generalized forked SSR FFT ,this implementation specifically deals with the case when
// t_L not power of t_R
template <int t_instanceID>
struct FFTWrapper<1, 0, t_instanceID> {
    // SSR_FFT_VIVADO_BEGIN
    template <int t_L,
              int t_R,
              int iid,
              scaling_mode_enum t_scalingMode,
              transform_direction_enum transform_direction,
              butterfly_rnd_mode_enum butterfly_rnd_mode,
              fft_output_order_enum tp_output_data_order,
              typename T_complexExpTableType,
              typename T_fftTwiddleType,
              typename T_in,
              typename T_out>
    static void innerFFT(T_in p_fftInData[t_R][t_L / t_R], T_out p_fftOutData[t_R][t_L / t_R]) {
#pragma HLS INLINE

#ifndef SSR_FFT_SEPERATE_REAL_IMAG_PARTS

//#pragma HLS data_pack variable = p_fftInData
//#pragma HLS data_pack variable = p_fftOutData

#endif

#pragma HLS STREAM variable = p_fftInData depth = 8
#pragma HLS STREAM variable = p_fftOutData depth = 8
#pragma HLS RESOURCE variable = p_fftInData core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = p_fftOutData core = FIFO_LUTRAM

#ifdef SSR_FFT_PARTITION_IO_ARRAYS // SSR_FFT_PARTITION_INTERFACE_ARRAYS
#pragma HLS ARRAY_PARTITION variable = p_fftInData complete dim = 1
#pragma HLS ARRAY_PARTITION variable = p_fftOutData complete dim = 1
#else
#pragma HLS ARRAY_RESHAPE variable = p_fftInData complete dim = 1
#pragma HLS ARRAY_RESHAPE variable = p_fftOutData complete dim = 1
#endif
        const int tp_outputForkingFactor =
            ssrFFTPow<t_R, ((ssrFFTLog2<t_L>::val) / (ssrFFTLog2<t_R>::val) + 1)>::val / (t_L);
        T_complexExpTableType
            complexExpTable[ComplexExpTableLENTraits<tp_outputForkingFactor, t_L, t_R>::EXTENDED_EXP_TALBE_LENGTH];
#pragma HLS ARRAY_PARTITION variable = complexExpTable complete dim = 1
        T_complexExpTableType complexExpTable_forkingStage[ComplexExpTableLENTraits<tp_outputForkingFactor, t_L,
                                                                                    t_R>::EXTENDED_EXP_TALBE_LENGTH];
#pragma HLS ARRAY_PARTITION variable = complexExpTable_forkingStage complete dim = 1
        ComplexExpTable<ComplexExpTableLENTraits<tp_outputForkingFactor, t_L, t_R>::EXP_TALBE_LENGTH,
                        transform_direction, butterfly_rnd_mode,
                        typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType>::
            initComplexExpTable(complexExpTable);
        ComplexExpTableLastStage<
            t_R, transform_direction, butterfly_rnd_mode,
            ComplexExpTableLENTraits<tp_outputForkingFactor, t_L, t_R>::EXP_TALBE_LENGTH_LAST_STAGE,
            typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType>::
            initComplexExpTable(complexExpTable_forkingStage);
        const int NO_OF_FFT_STAGES = (ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val) + 1;
        typedef typename FFTInputTraits<T_in>::T_castedType casted_type;
        hls::stream<SuperSampleContainer<t_R, casted_type> > p_fftInData_reOrdered;
//#pragma HLS data_pack variable = p_fftInData_reOrdered
#pragma HLS STREAM variable = p_fftInData_reOrdered depth = 8
#pragma HLS RESOURCE variable = p_fftInData_reOrdered core = FIFO_LUTRAM
        hls::stream<SuperSampleContainer<t_R, casted_type> > casted_output;
//#pragma HLS data_pack variable = casted_output
#pragma HLS STREAM variable = casted_output depth = 8
#pragma HLS RESOURCE variable = casted_output core = FIFO_LUTRAM
        castArrayS2Streaming<t_L, t_R, T_in, casted_type>(p_fftInData, casted_output);
        InputTransposeChainStreamingS2S<t_instanceID, NO_OF_FFT_STAGES - 1, 60000, t_R, t_L, t_R, 1,
                                        tp_outputForkingFactor>
            swapObj;
        swapObj.template swap<casted_type>(casted_output, p_fftInData_reOrdered);

        FFTForkingStage<

            tp_outputForkingFactor, t_L, t_R, iid, t_scalingMode, transform_direction, butterfly_rnd_mode,
            tp_output_data_order, NO_OF_FFT_STAGES,
            typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType, T_complexExpTableType,
                               casted_type, T_out>::T_expTabType,
            typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType, T_complexExpTableType,
                               casted_type, T_out>::T_twiddleType,
            typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType, T_complexExpTableType,
                               casted_type, T_out>::T_stageInType,
            typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                     casted_type>::T_FFTOutType>::fftStage(complexExpTable,
                                                                           complexExpTable_forkingStage,
                                                                           p_fftInData_reOrdered, p_fftOutData);
    }
    // SSR_FFT_VIVADO_END
    template <int t_L,
              int t_R,
              int iid,
              scaling_mode_enum t_scalingMode,
              transform_direction_enum transform_direction,
              butterfly_rnd_mode_enum butterfly_rnd_mode,
              fft_output_order_enum tp_output_data_order,
              typename T_complexExpTableType,
              typename T_fftTwiddleType,
              typename T_in,
              typename T_out>
    static void innerFFT(hls::stream<T_in> p_fftInData[t_R], hls::stream<T_out> p_fftOutData[t_R]) {
//#pragma HLS INLINE
#pragma HLS dataflow disable_start_propagation

        const int tp_outputForkingFactor =
            ssrFFTPow<t_R, ((ssrFFTLog2<t_L>::val) / (ssrFFTLog2<t_R>::val) + 1)>::val / (t_L);
        T_complexExpTableType
            complexExpTable[ComplexExpTableLENTraits<tp_outputForkingFactor, t_L, t_R>::EXTENDED_EXP_TALBE_LENGTH];
#pragma HLS ARRAY_PARTITION variable = complexExpTable complete dim = 1
        T_complexExpTableType complexExpTable_forkingStage[ComplexExpTableLENTraits<tp_outputForkingFactor, t_L,
                                                                                    t_R>::EXTENDED_EXP_TALBE_LENGTH];
#pragma HLS ARRAY_PARTITION variable = complexExpTable_forkingStage complete dim = 1
        ComplexExpTable<ComplexExpTableLENTraits<tp_outputForkingFactor, t_L, t_R>::EXP_TALBE_LENGTH,
                        transform_direction, butterfly_rnd_mode,
                        typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType>::
            initComplexExpTable(complexExpTable);
        ComplexExpTableLastStage<
            t_R, transform_direction, butterfly_rnd_mode,
            ComplexExpTableLENTraits<tp_outputForkingFactor, t_L, t_R>::EXP_TALBE_LENGTH_LAST_STAGE,
            typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType>::
            initComplexExpTable(complexExpTable_forkingStage);
        const int NO_OF_FFT_STAGES = (ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val) + 1;
        typedef typename FFTInputTraits<T_in>::T_castedType casted_type;
        hls::stream<SuperSampleContainer<t_R, casted_type> > p_fftInData_reOrdered;
#pragma HLS STREAM variable = p_fftInData_reOrdered depth = 8
#pragma HLS RESOURCE variable = p_fftInData_reOrdered core = FIFO_LUTRAM
        hls::stream<SuperSampleContainer<t_R, casted_type> > casted_output;
#pragma HLS STREAM variable = casted_output depth = 8
#pragma HLS RESOURCE variable = casted_output core = FIFO_LUTRAM
        castArrayS2Streaming<t_L, t_R, T_in, casted_type>(p_fftInData, casted_output);
        InputTransposeChainStreamingS2S<t_instanceID, NO_OF_FFT_STAGES - 1, 60000, t_R, t_L, t_R, 1,
                                        tp_outputForkingFactor>
            swapObj;
        swapObj.template swap<casted_type>(casted_output, p_fftInData_reOrdered);

        FFTForkingStage<

            tp_outputForkingFactor, t_L, t_R, iid, t_scalingMode, transform_direction, butterfly_rnd_mode,
            tp_output_data_order, NO_OF_FFT_STAGES,
            typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType, T_complexExpTableType,
                               casted_type, T_out>::T_expTabType,
            typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType, T_complexExpTableType,
                               casted_type, T_out>::T_twiddleType,
            typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType, T_complexExpTableType,
                               casted_type, T_out>::T_stageInType,
            typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                     casted_type>::T_FFTOutType>::fftStage(complexExpTable,
                                                                           complexExpTable_forkingStage,
                                                                           p_fftInData_reOrdered, p_fftOutData);
    }
};

template <int t_instanceID>
struct FFTWrapper<1, 1, t_instanceID> {
    // SSR_FFT_VIVADO_BEGIN
    template <int t_L,
              int t_R,
              int iid,
              scaling_mode_enum t_scalingMode,
              transform_direction_enum transform_direction,
              butterfly_rnd_mode_enum butterfly_rnd_mode,
              fft_output_order_enum tp_output_data_order,
              typename T_complexExpTableType,
              typename T_fftTwiddleType,
              typename T_in,
              typename T_out>
    static void innerFFT(T_in p_fftInData[t_R][t_L / t_R], T_out p_fftOutData[t_R][t_L / t_R]) {
#pragma HLS INLINE
#ifndef SSR_FFT_SEPERATE_REAL_IMAG_PARTS
//////////////////////////////////////////////
//#pragma HLS data_pack variable = p_fftInData
//#pragma HLS data_pack variable = p_fftOutData
/////////////////////////////////////////////
#endif

#ifdef SSR_FFT_PARTITION_IO_ARRAYS // SSR_FFT_PARTITION_INTERFACE_ARRAYS
////////////////////////////////////////////////////////////////
#pragma HLS ARRAY_PARTITION variable = p_fftInData complete dim = 1
#pragma HLS ARRAY_PARTITION variable = p_fftOutData complete dim = 1
////////////////////////////////////////////////////////////////
#else
////////////////////////////////////////////////////////////////
#pragma HLS ARRAY_RESHAPE variable = p_fftInData complete dim = 1
#pragma HLS ARRAY_RESHAPE variable = p_fftOutData complete dim = 1
////////////////////////////////////////////////////////////////
#endif
#pragma HLS STREAM variable = p_fftInData depth = 8
#pragma HLS STREAM variable = p_fftOutData depth = 8
#pragma HLS RESOURCE variable = p_fftInData core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = p_fftOutData core = FIFO_LUTRAM

        const int tp_outputForkingFactor =
            ssrFFTPow<t_R, ((ssrFFTLog2<t_L>::val) / (ssrFFTLog2<t_R>::val) + 1)>::val / (t_L);
        T_complexExpTableType
            complexExpTable[ComplexExpTableLENTraits<tp_outputForkingFactor, t_L, t_R>::EXTENDED_EXP_TALBE_LENGTH];
#pragma HLS ARRAY_PARTITION variable = complexExpTable complete dim = 1
        T_complexExpTableType complexExpTable_forkingStage[ComplexExpTableLENTraits<tp_outputForkingFactor, t_L,
                                                                                    t_R>::EXTENDED_EXP_TALBE_LENGTH];

//#pragma HLS data_pack variable = complexExpTable_forkingStage
#pragma HLS ARRAY_PARTITION variable = complexExpTable_forkingStage complete dim = 1

        ComplexExpTable<ComplexExpTableLENTraits<tp_outputForkingFactor, t_L, t_R>::EXP_TALBE_LENGTH,
                        transform_direction, butterfly_rnd_mode,
                        typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType>::
            initComplexExpTable(complexExpTable);
        ComplexExpTableLastStage<
            t_R, transform_direction, butterfly_rnd_mode,
            ComplexExpTableLENTraits<tp_outputForkingFactor, t_L, t_R>::EXP_TALBE_LENGTH_LAST_STAGE,
            typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType>::
            initComplexExpTable(complexExpTable_forkingStage);
        const int NO_OF_FFT_STAGES = (ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val) + 1;

        typedef typename FFTInputTraits<T_in>::T_castedType casted_type;

        hls::stream<SuperSampleContainer<t_R, casted_type> > p_fftInData_reOrdered;
//#pragma HLS data_pack variable = p_fftInData_reOrdered
#pragma HLS STREAM variable = p_fftInData_reOrdered depth = 8
#pragma HLS RESOURCE variable = p_fftInData_reOrdered core = FIFO_LUTRAM

        hls::stream<SuperSampleContainer<t_R, casted_type> > casted_output;
//#pragma HLS data_pack variable = casted_output
#pragma HLS STREAM variable = casted_output depth = 8
#pragma HLS RESOURCE variable = casted_output core = FIFO_LUTRAM

        /* The casted output cannot be reshaped in this case because it is the TINY case where the
         * casted output will be directly consumed by multiple data commuter processes and when it
         * re-shaped it will produced data-flow error of "multiple functions have read accesses to the streamed
         * variable"
         */

        castArrayS2Streaming<t_L, t_R, T_in, casted_type>(p_fftInData, casted_output);

        InputTransposeChainStreamingS2S<t_instanceID, NO_OF_FFT_STAGES - 1, 70000, t_R, t_L, t_R, 1,
                                        tp_outputForkingFactor>
            swapObj;
        swapObj.template swap<casted_type>(casted_output, p_fftInData_reOrdered);

        FFTForkingStage<tp_outputForkingFactor, t_L, t_R, iid, t_scalingMode, transform_direction, butterfly_rnd_mode,
                        tp_output_data_order, NO_OF_FFT_STAGES,
                        typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType,
                                           T_complexExpTableType, casted_type, T_out>::T_expTabType,
                        typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType,
                                           T_complexExpTableType, casted_type, T_out>::T_twiddleType,
                        typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType,
                                           T_complexExpTableType, casted_type, T_out>::T_stageInType,
                        typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                                 casted_type>::T_FFTOutType>::fftStage(complexExpTable,
                                                                                       complexExpTable_forkingStage,
                                                                                       // twiddleTable,
                                                                                       p_fftInData_reOrdered,
                                                                                       p_fftOutData);
    }
    // SSR_FFT_VIVADO_END
    template <int t_L,
              int t_R,
              int iid,
              scaling_mode_enum t_scalingMode,
              transform_direction_enum transform_direction,
              butterfly_rnd_mode_enum butterfly_rnd_mode,
              fft_output_order_enum tp_output_data_order,
              typename T_complexExpTableType,
              typename T_fftTwiddleType,
              typename T_in,
              typename T_out>
    static void innerFFT(hls::stream<T_in> p_fftInData[t_R], hls::stream<T_out> p_fftOutData[t_R]) {
//#pragma HLS INLINE
#pragma HLS dataflow disable_start_propagation

        const int tp_outputForkingFactor =
            ssrFFTPow<t_R, ((ssrFFTLog2<t_L>::val) / (ssrFFTLog2<t_R>::val) + 1)>::val / (t_L);
        T_complexExpTableType
            complexExpTable[ComplexExpTableLENTraits<tp_outputForkingFactor, t_L, t_R>::EXTENDED_EXP_TALBE_LENGTH];
#pragma HLS ARRAY_PARTITION variable = complexExpTable complete dim = 1
        T_complexExpTableType complexExpTable_forkingStage[ComplexExpTableLENTraits<tp_outputForkingFactor, t_L,
                                                                                    t_R>::EXTENDED_EXP_TALBE_LENGTH];
//#pragma HLS data_pack variable = complexExpTable_forkingStage
#pragma HLS ARRAY_PARTITION variable = complexExpTable_forkingStage complete dim = 1

        ComplexExpTable<ComplexExpTableLENTraits<tp_outputForkingFactor, t_L, t_R>::EXP_TALBE_LENGTH,
                        transform_direction, butterfly_rnd_mode,
                        typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType>::
            initComplexExpTable(complexExpTable);
        ComplexExpTableLastStage<
            t_R, transform_direction, butterfly_rnd_mode,
            ComplexExpTableLENTraits<tp_outputForkingFactor, t_L, t_R>::EXP_TALBE_LENGTH_LAST_STAGE,
            typename ComplexExpTableTraits<T_complexExpTableType>::t_complexExpTableType>::
            initComplexExpTable(complexExpTable_forkingStage);
        const int NO_OF_FFT_STAGES = (ssrFFTLog2<t_L>::val / ssrFFTLog2<t_R>::val) + 1;

        typedef typename FFTInputTraits<T_in>::T_castedType casted_type;

        hls::stream<SuperSampleContainer<t_R, casted_type> > p_fftInData_reOrdered;
//#pragma HLS data_pack variable = p_fftInData_reOrdered
#pragma HLS STREAM variable = p_fftInData_reOrdered depth = 8
#pragma HLS RESOURCE variable = p_fftInData_reOrdered core = FIFO_LUTRAM

        hls::stream<SuperSampleContainer<t_R, casted_type> > casted_output;
//#pragma HLS data_pack variable = casted_output
#pragma HLS STREAM variable = casted_output depth = 8
#pragma HLS RESOURCE variable = casted_output core = FIFO_LUTRAM

        /* The casted output cannot be reshaped in this case because it is the TINY case where the
         * casted output will be directly consumed by multiple data commuter processes and when it
         * re-shaped it will produced data-flow error of "multiple functions have read accesses to the streamed
         * variable"
         */

        castArrayS2Streaming<t_L, t_R, T_in, casted_type>(p_fftInData, casted_output);

        InputTransposeChainStreamingS2S<t_instanceID, NO_OF_FFT_STAGES - 1, 70000, t_R, t_L, t_R, 1,
                                        tp_outputForkingFactor>
            swapObj;
        swapObj.template swap<casted_type>(casted_output, p_fftInData_reOrdered);

        FFTForkingStage<tp_outputForkingFactor, t_L, t_R, iid, t_scalingMode, transform_direction, butterfly_rnd_mode,
                        tp_output_data_order, NO_OF_FFT_STAGES,
                        typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType,
                                           T_complexExpTableType, casted_type, T_out>::T_expTabType,
                        typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType,
                                           T_complexExpTableType, casted_type, T_out>::T_twiddleType,
                        typename FFTTraits<t_scalingMode, t_L, t_R, NO_OF_FFT_STAGES, T_fftTwiddleType,
                                           T_complexExpTableType, casted_type, T_out>::T_stageInType,
                        typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                                 casted_type>::T_FFTOutType>::fftStage(complexExpTable,
                                                                                       complexExpTable_forkingStage,
                                                                                       // twiddleTable,
                                                                                       p_fftInData_reOrdered,
                                                                                       p_fftOutData);
    }
};

#ifndef __SYNTHESIS__
template <int L, int R, int tw_WL, int tw_IL>
void checkFFTparams() {
    // powerOf2CheckonL<1> L_checker;
    // L_checker.check();
    // powerOf2CheckonRadix<1> R_checker;
    // R_checker.check();
    // powerOf2CheckonRadix<R != ssrFFTPow<2,(ssrFFTLog2<R>::val)>::val>::check();
    // powerOf2CheckonL<L != ssrFFTPow<2,(ssrFFTLog2<L>::val)>::val>::check();
    // powerOf2CheckonRadix<R != ssrFFTPow<2,(ssrFFTLog2<R>::val)>::val>::check();
    if (L < 16) {
        std::cerr << "\n\n\n====================================================================\n";
        std::cerr << "The Minimum FFT Length allowed is  : 16\n";
        std::cerr << "The Provided FFT Length:" << L << " is less than 16\n";
        std::cerr << "====================================================================\n\n\n";

        exit(1);
    }
    if (L > (1024 * 16)) {
        std::cerr << "\n\n\n====================================================================\n";
        std::cerr << "The Maximum FFT Length allowed is  : 16384 : 16K \n";
        std::cerr << "The Provided FFT Length:" << L << " is greater than 16384 \n";
        std::cerr << "====================================================================\n\n\n";

        exit(1);
    }
    if (L != ssrFFTPow<2, (ssrFFTLog2<L>::val)>::val) {
        std::cerr << "\n\n\n====================================================================\n";
        std::cerr << "The FFT Length should be an integer power of 2\n";
        std::cerr << "The Provided FFT Length:" << L << " is not a power of 2\n";
        std::cerr << "====================================================================\n\n\n";

        exit(1);
    }
    if (R != ssrFFTPow<2, (ssrFFTLog2<R>::val)>::val) {
        std::cerr << "\n\n\n====================================================================\n";
        std::cerr << "The FFT SSR should be an integer power of 2\n";
        std::cerr << "The Provided FFT Length:" << R << " is not a power of 2\n";
        std::cerr << "====================================================================\n\n\n";

        exit(1);
    }
    if (tw_IL < 1) {
        std::cerr << "\n\n\n====================================================================\n";
        std::cerr << "The twiddle table should have at least 2 bits for integer part storage\n";
        std::cerr << "The provided twiddle integer part width=" << tw_IL << "\n";
        std::cerr << "====================================================================\n\n\n";
        exit(1);
    }

    if (R == 2 && L < 16) {
        std::cerr << "\n\n\n====================================================================\n";
        std::cerr << "For Radix/SSR=2 the minimum allowed SIZE or length of FFT is 16\n";
        std::cerr << "====================================================================\n\n\n";
        exit(1);
    }

    if (R == 8 && L < 16) {
        std::cerr << "\n\n\n====================================================================\n";
        std::cerr << "For Radix/SSR=8 the minimum allowed SIZE or length of FFT is 16\n";
        std::cerr << "====================================================================\n\n\n";
        exit(1);
    }

    if (R == 16 && L < 32) {
        std::cerr << "\n\n\n====================================================================\n";
        std::cerr << "For Radix/SSR=16 the minimum allowed SIZE or length of FFT is 32\n";
        std::cerr << "====================================================================\n\n\n";
        exit(1);
    }

    if (R > 128) {
        std::cerr << "\n\n\n====================================================================\n";
        std::cerr << "Currently SSR FFT Supports Radix/R/SSR = 2,4,8,16,32.\n The values of Radix/R/SSR larger than 32 "
                     " are not supported. \n";
        std::cerr << "====================================================================\n\n\n";
        exit(1);
    }
}
//// Simulation only utility functions added to re order data in the case when the user want to have
//// fft output in digit reversed transposed form. The utlity function can help to re order the fft out
//// for simulation and verification purposes, these function wont synthesize. If user need to synthesize
//// them he should use the fft param struct to have natural order for fft output.
template <int is_R_x_R_form>
struct outputTransposerSimulationModelOnly {
    template <int t_R, int t_L, typename T_in, typename T_out>
    void reOrderOutput(T_in p_inData[t_R][t_L / t_R], T_out p_outData[t_R][t_L / t_R]);
};
template <>
struct outputTransposerSimulationModelOnly<1> {
    template <int t_R, int t_L, typename T_in, typename T_out>
    void reOrderOutput(T_in p_inData[t_R][t_L / t_R], T_out p_outData[t_R][t_L / t_R]) {
        digitReversedDataReOrder<t_L, t_R>(p_inData, p_outData);
    }
};
template <>
struct outputTransposerSimulationModelOnly<0> {
    template <int t_R, int t_L, typename T_in, typename T_out>
    void reOrderOutput(T_in p_inData[t_R][t_L / t_R], T_out p_outData[t_R][t_L / t_R]) {
        hls::stream<SuperSampleContainer<t_R, T_in> > temp_stream;
        OutputDataReOrder<(t_L) / (t_R * t_R)> OutputDataReOrder_obj1;
        convertArrayToSuperStream<1010, 2020, t_L, t_R>(p_inData, temp_stream);
        OutputDataReOrder_obj1.template digitReversal2Phase<t_L, t_R>(temp_stream, p_outData);
    }
};

template <int t_R, int t_L, typename T_in, typename T_out>
void fftOutputReorderSimulationModelOnly(T_in p_inData[t_R][t_L / t_R], T_out p_outData[t_R][t_L / t_R]) {
    static const int is_R_x_R_form = (ssrFFTLog2<t_L>::val) % (ssrFFTLog2<t_R>::val) == 0;
    outputTransposerSimulationModelOnly<is_R_x_R_form> outputTransposer_simOnly_obj;
    outputTransposer_simOnly_obj.template reOrderOutput<t_R, t_L>(p_inData, p_outData);
}

#endif

template <typename ssr_fft_param_struct, typename T_in>
void fftKernelInputAdapter(
    T_in p_inData[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R],
    T_in p_outDataStream[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R]) {
#pragma HLS INLINE off
//#pragma HLS data_pack variable = p_inData
#pragma HLS ARRAY_RESHAPE variable = p_inData complete dim = 1
    const static int T = ssr_fft_param_struct::N / ssr_fft_param_struct::R;
    const static int R = ssr_fft_param_struct::R;
    for (int t = 0; t < T; t++) {
#pragma HLS PIPELINE II = 1
        for (int r = 0; r < R; ++r) {
#pragma HLS UNROLL
            p_outDataStream[r][t] = p_inData[r][t];
        }
    }
}

template <typename ssr_fft_param_struct, typename T_in>
void fftKernelOutputAdapter(
    typename FFTOutputTraits<ssr_fft_param_struct::N,
                             ssr_fft_param_struct::R,
                             ssr_fft_param_struct::scaling_mode,
                             ssr_fft_param_struct::transform_direction,
                             ssr_fft_param_struct::butterfly_rnd_mode,
                             typename FFTInputTraits<T_in>::T_castedType>::T_FFTOutType
        p_inDataStream[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R],
    typename FFTOutputTraits<ssr_fft_param_struct::N,
                             ssr_fft_param_struct::R,
                             ssr_fft_param_struct::scaling_mode,
                             ssr_fft_param_struct::transform_direction,
                             ssr_fft_param_struct::butterfly_rnd_mode,
                             typename FFTInputTraits<T_in>::T_castedType>::T_FFTOutType
        p_outData[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R]) {
#pragma HLS INLINE off
//#pragma HLS data_pack variable = p_outData
#pragma HLS ARRAY_RESHAPE variable = p_outData complete dim = 1

    const static int T = ssr_fft_param_struct::N / ssr_fft_param_struct::R;
    const static int R = ssr_fft_param_struct::R;
    for (int t = 0; t < T; t++) {
#pragma HLS PIPELINE II = 1
        for (int r = 0; r < R; ++r) {
#pragma HLS UNROLL
            p_outData[r][t] = p_inDataStream[r][t];
        }
    }
}

#if 0
// SSR_FFT_VIVADO_BEGIN
template <typename ssr_fft_param_struct, typename T_in>
void fft(T_in p_fftInData[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R],
         typename FFTIOTypes<ssr_fft_param_struct, T_in>::T_outType
             p_fftOutData[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R]) {
#pragma HLS INLINE
#pragma HLS DATAFLOW disable_start_propagation

    static const int t_L = ssr_fft_param_struct::N;
    static const int t_R = ssr_fft_param_struct::R;
    static const scaling_mode_enum t_scalingMode = ssr_fft_param_struct::scaling_mode;
    static const fft_output_order_enum tp_output_data_order = ssr_fft_param_struct::output_data_order;
    static const int tw_WL = ssr_fft_param_struct::twiddle_table_word_length;
    static const int tw_IL = ssr_fft_param_struct::twiddle_table_intger_part_length;
    static const int default_t_instanceID = ssr_fft_param_struct::default_t_instanceID;
    static const transform_direction_enum transform_direction = ssr_fft_param_struct::transform_direction;
    static const butterfly_rnd_mode_enum butterfly_rnd_mode = ssr_fft_param_struct::butterfly_rnd_mode;
    typedef typename FFTInputTraits<T_in>::T_castedType casted_type;

#ifndef __SYNTHESIS__
    checkFFTparams<t_L, t_R, tw_WL, tw_IL>();
// std::cout<<"SRR FFT INSTANCE_ID = "<<default_t_instanceID<<std::endl;
#endif
    typedef typename InputBasedTwiddleTraits<ssr_fft_param_struct, casted_type>::T_twiddleType T_fftTwiddleType;
    typedef typename InputBasedTwiddleTraits<ssr_fft_param_struct, casted_type>::T_expTabType T_complexExpTableType;

#ifndef __SYNTHESIS__
    assert((t_R) == (ssrFFTPow<2, ssrFFTLog2<t_R>::val>::val)); // radix should be power of 2 always
    assert((t_L) == (ssrFFTPow<2, ssrFFTLog2<t_L>::val>::val)); // Length of FFt should be power of 2 always
#endif
    FFTWrapper<(((ssrFFTLog2<t_L>::val) % (ssrFFTLog2<t_R>::val)) > 0), (t_L) < ((t_R * t_R)), default_t_instanceID>
        ssr_fft_wrapper_obj;
    // The 1st template arguments select : if the FFT is forked , if it is then a different architecture is required
    // 2nd template argument select if the L < (t_R^2) , which requires removal of interface bundle pragma
    ssr_fft_wrapper_obj
        .template innerFFT<t_L, t_R, default_t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode,
                           tp_output_data_order, T_complexExpTableType, T_fftTwiddleType, T_in,
                           typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                                    casted_type>::T_FFTOutType>(p_fftInData, p_fftOutData);
}
// SSR_FFT_VIVADO_END
#endif
template <int t_L, int t_R, typename T_in>
void array2Stream(T_in arrayIn[t_R][t_L / t_R], hls::stream<T_in> strmOut[t_R]) {
    for (int i = 0; i < t_L / t_R; i++) {
        for (int j = 0; j < t_R; j++) {
#pragma HLS pipeline II = 1
            strmOut[j].write(arrayIn[j][i]);
        }
    }
}
template <int t_L, int t_R, typename T_out>
void stream2Array(hls::stream<T_out> strmIn[t_R], T_out arrayOut[t_R][t_L / t_R]) {
    for (int i = 0; i < t_L / t_R; i++) {
        for (int j = 0; j < t_R; j++) {
#pragma HLS pipeline II = 1
            arrayOut[j][i] = strmIn[j].read();
        }
    }
}
template <typename ssr_fft_param_struct, typename T_in>
void fft(T_in p_fftInData[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R],
         typename FFTIOTypes<ssr_fft_param_struct, T_in>::T_outType
             p_fftOutData[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R]) {
    enum { FIFO_SIZE = ssr_fft_param_struct::N / ssr_fft_param_struct::R };
    //#pragma HLS INLINE
    //#pragma HLS DATAFLOW  disable_start_propagation

    static const int t_L = ssr_fft_param_struct::N;
    static const int t_R = ssr_fft_param_struct::R;
    static const scaling_mode_enum t_scalingMode = ssr_fft_param_struct::scaling_mode;
    static const fft_output_order_enum tp_output_data_order = ssr_fft_param_struct::output_data_order;
    static const int tw_WL = ssr_fft_param_struct::twiddle_table_word_length;
    static const int tw_IL = ssr_fft_param_struct::twiddle_table_intger_part_length;
    static const int default_t_instanceID = ssr_fft_param_struct::default_t_instanceID;
    static const transform_direction_enum transform_direction = ssr_fft_param_struct::transform_direction;
    static const butterfly_rnd_mode_enum butterfly_rnd_mode = ssr_fft_param_struct::butterfly_rnd_mode;
    typedef typename FFTInputTraits<T_in>::T_castedType casted_type;

#ifndef __SYNTHESIS__
    checkFFTparams<t_L, t_R, tw_WL, tw_IL>();
// std::cout<<"SRR FFT INSTANCE_ID = "<<default_t_instanceID<<std::endl;
#endif
    typedef typename InputBasedTwiddleTraits<ssr_fft_param_struct, casted_type>::T_twiddleType T_fftTwiddleType;
    typedef typename InputBasedTwiddleTraits<ssr_fft_param_struct, casted_type>::T_expTabType T_complexExpTableType;

#ifndef __SYNTHESIS__
    assert((t_R) == (ssrFFTPow<2, ssrFFTLog2<t_R>::val>::val)); // radix should be power of 2 always
    assert((t_L) == (ssrFFTPow<2, ssrFFTLog2<t_L>::val>::val)); // Length of FFt should be power of 2 always
#endif
    hls::stream<T_in> fftInStrm[t_R];
#pragma HLS stream variable = fftInStrm depth = FIFO_SIZE
    hls::stream<typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                         casted_type>::T_FFTOutType>
        fftOutStrm[t_R];
#pragma HLS stream variable = fftOutStrm depth = FIFO_SIZE
    array2Stream<t_L, t_R, T_in>(p_fftInData, fftInStrm);
    FFTWrapper<(((ssrFFTLog2<t_L>::val) % (ssrFFTLog2<t_R>::val)) > 0), (t_L) < ((t_R * t_R)), default_t_instanceID>
        ssr_fft_wrapper_obj;
    // The 1st template arguments select : if the FFT is forked , if it is then a different architecture is required
    // 2nd template argument select if the L < (t_R^2) , which requires removal of interface bundle pragma
    ssr_fft_wrapper_obj
        .template innerFFT<t_L, t_R, default_t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode,
                           tp_output_data_order, T_complexExpTableType, T_fftTwiddleType, T_in,
                           typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                                    casted_type>::T_FFTOutType>(fftInStrm, fftOutStrm);
    stream2Array<t_L, t_R, typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                                    casted_type>::T_FFTOutType>(fftOutStrm, p_fftOutData);
}

template <typename ssr_fft_param_struct, int t_instanceID, typename T_in>
void fft(T_in p_fftInData[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R],
         typename FFTIOTypes<ssr_fft_param_struct, T_in>::T_outType
             p_fftOutData[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R]) {
    enum { FIFO_SIZE = ssr_fft_param_struct::N / ssr_fft_param_struct::R };
    //#pragma HLS INLINE
    //#pragma HLS DATAFLOW disable_start_propagation

    static const int t_L = ssr_fft_param_struct::N;
    static const int t_R = ssr_fft_param_struct::R;
    static const scaling_mode_enum t_scalingMode = ssr_fft_param_struct::scaling_mode;
    static const fft_output_order_enum tp_output_data_order = ssr_fft_param_struct::output_data_order;
    static const int tw_WL = ssr_fft_param_struct::twiddle_table_word_length;
    static const int tw_IL = ssr_fft_param_struct::twiddle_table_intger_part_length;
    static const transform_direction_enum transform_direction = ssr_fft_param_struct::transform_direction;
    static const butterfly_rnd_mode_enum butterfly_rnd_mode = ssr_fft_param_struct::butterfly_rnd_mode;
    typedef typename FFTInputTraits<T_in>::T_castedType casted_type;

#ifndef __SYNTHESIS__
    checkFFTparams<t_L, t_R, tw_WL, tw_IL>();
// std::cout<<"SRR FFT INSTNACE_ID = "<<t_instanceID<<std::endl;
#endif
    typedef typename InputBasedTwiddleTraits<ssr_fft_param_struct, casted_type>::T_twiddleType T_fftTwiddleType;
    typedef typename InputBasedTwiddleTraits<ssr_fft_param_struct, casted_type>::T_expTabType T_complexExpTableType;

#ifndef __SYNTHESIS__
    assert((t_R) == (ssrFFTPow<2, ssrFFTLog2<t_R>::val>::val)); // radix should be power of 2 always
    assert((t_L) == (ssrFFTPow<2, ssrFFTLog2<t_L>::val>::val)); // Length of FFt should be power of 2 always
#endif
    hls::stream<T_in> fftInStrm[t_R];
#pragma HLS stream variable = fftInStrm depth = FIFO_SIZE
    hls::stream<typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                         casted_type>::T_FFTOutType>
        fftOutStrm[t_R];
#pragma HLS stream variable = fftOutStrm depth = FIFO_SIZE
    array2Stream<t_L, t_R, T_in>(p_fftInData, fftInStrm);
    FFTWrapper<(((ssrFFTLog2<t_L>::val) % (ssrFFTLog2<t_R>::val)) > 0), (t_L) < ((t_R * t_R)), t_instanceID>
        ssr_fft_wrapper_obj;
    // The 1st template arguments select : if the FFT is forked , if it is then a different architecture is required
    // 2nd template argument select if the L < (t_R^2) , which requires removal of interface bundle pragma
    ssr_fft_wrapper_obj
        .template innerFFT<t_L, t_R, t_instanceID, t_scalingMode, transform_direction, butterfly_rnd_mode,
                           tp_output_data_order, T_complexExpTableType, T_fftTwiddleType, T_in,
                           typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                                    casted_type>::T_FFTOutType>(fftInStrm, fftOutStrm);
    stream2Array<t_L, t_R, typename FFTOutputTraits<t_L, t_R, t_scalingMode, transform_direction, butterfly_rnd_mode,
                                                    casted_type>::T_FFTOutType>(fftOutStrm, p_fftOutData);
}

template <typename ssr_fft_param_struct, typename T_in>
void fftKernel(T_in p_fftInData[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R],
               typename FFTOutputTraits<ssr_fft_param_struct::N,
                                        ssr_fft_param_struct::R,
                                        ssr_fft_param_struct::scaling_mode,
                                        ssr_fft_param_struct::transform_direction,
                                        ssr_fft_param_struct::butterfly_rnd_mode,
                                        typename FFTInputTraits<T_in>::T_castedType>::T_FFTOutType
                   p_fftOutData[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R]) {
#pragma HLS INLINE
    //#pragma HLS DATAFLOW disable_start_propagation
    // const static int N = ssr_fft_param_struct::N;
    // const static int R = ssr_fft_param_struct::R;

    // T_in inData_stream[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R];
    // typename FFTOutputTraits<N, R, ssr_fft_param_struct::scaling_mode, ssr_fft_param_struct::transform_direction,
    //                         ssr_fft_param_struct::butterfly_rnd_mode,
    //                         typename FFTInputTraits<T_in>::T_castedType>::T_FFTOutType
    //    outData_stream[ssr_fft_param_struct::R][ssr_fft_param_struct::N / ssr_fft_param_struct::R];
    // fftKernelInputAdapter<ssr_fft_param_struct, T_in>(p_fftInData, inData_stream);
    // fft<ssr_fft_param_struct, T_in>(inData_stream, outData_stream);
    // fftKernelOutputAdapter<ssr_fft_param_struct, T_in>(outData_stream, p_fftOutData);
    fft<ssr_fft_param_struct, T_in>(p_fftInData, p_fftOutData);
}

} // namespace fft
} // namespace dsp
} // namespace xf

#endif // !HLS_SSR_FFT_H
