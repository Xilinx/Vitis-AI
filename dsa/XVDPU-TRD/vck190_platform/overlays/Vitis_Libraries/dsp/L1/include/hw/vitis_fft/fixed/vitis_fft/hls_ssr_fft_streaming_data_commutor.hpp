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

// File Name : hls_ssr_fft_streaming_data_commutor.hpp
#ifndef __HLS_SSR_FFT_STREAMING_DATA_COMMUTOR_H__
#define __HLS_SSR_FFT_STREAMING_DATA_COMMUTOR_H__

#include <complex>
#include <ap_shift_reg.h>
#include <hls_stream.h>
#include <ap_int.h>
//#include <ap_utils.h>

#include "vitis_fft/hls_ssr_fft_mux_chain.hpp"
#include "vitis_fft/hls_ssr_fft_triangle_delay.hpp"
#include "vitis_fft/hls_ssr_fft_super_sample.hpp"
#include "vitis_fft/hls_ssr_fft_utilities.hpp"
#include "vitis_fft/hls_ssr_fft_fork_merge_utils.hpp"
#include "vitis_fft/fft_complex.hpp"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace dsp {
namespace fft {

// Streaming data commutor
// template <int t_stage, int id, int iid, typename T_T, unsigned int t_L,unsigned int t_R, unsigned int t_PF>
template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF, typename T_in>
void streamingDataCommutorInner(hls::stream<SuperSampleContainer<t_R, T_in> >& p_sampleIn,
                                hls::stream<SuperSampleContainer<t_R, T_in> >& p_sampleOut) {
#pragma HLS INLINE off
    typedef tagged_sample<SuperSampleContainer<t_R, T_in> > tagged_super_sample;
    typedef tagged_sample<T_in> taggeg_single_sample;
    const int control_count_width = ((ssrFFTLog2<t_R>::val) > 0) ? (ssrFFTLog2<t_R>::val) : 1;
    // If t_PF log2 is smaller then 1 then assign bit width as 1
    const int pf_count_width = (ssrFFTLog2<t_PF>::val > 0) ? (ssrFFTLog2<t_PF>::val) : 1;
    static ap_uint<control_count_width> control_count = 0;
    static ap_uint<pf_count_width> pf_count = 0;
    static ap_uint<control_count_width> control_bits;
    SuperSampleContainer<t_R, T_in> temp_super_sample;
#pragma HLS ARRAY_PARTITION variable = temp_super_sample.superSample complete dim = 1

    tagged_sample<T_in> temp_tagged_input_triangle_delay_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_input_triangle_delay_input complete dim = 1

    taggeg_single_sample temp_tagged_mux_chain_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_mux_chain_input complete dim = 1

    taggeg_single_sample temp_tagged_output_triangle_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_output_triangle_input complete dim = 1

    taggeg_single_sample commuted_output[t_R];
#pragma HLS ARRAY_PARTITION variable = commuted_output complete dim = 1

    // TriangleDelay<t_stage,id,iid,t_R> input_triangle_delay_block;
    // TriangleDelay<t_stage,id,iid,t_R> output_triangle_delay_block;
    /*template < 1-t_instanceID,
     2- t_stage,
     3- t_subStage,
     4-t_forkNumber,
     5-t_PF,
     6-delay_on_low_index_first,
     7-t_regTriangleHeight
     >*/
    TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, false, t_R> input_triangle_delay_block;
    TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, true, t_R> output_triangle_delay_block;

    // template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber,int t_PF, unsigned int
    // t_regTriangleHeight>
    MuxChain<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, t_R> muxChain_block;
    tagged_sample<SuperSampleContainer<t_R, T_in> > temp_tagged_sample;
    for (int t = 0; t < t_L / t_R + (t_R - 1) * t_PF; t++) {
#pragma HLS PIPELINE II = 1 rewind
        /// Read Sample from the input fifo /////////////////////////////////////////////////////////////////////////
        // If the input fifo has data read it, tag it and push it into the commutor network
        if (p_sampleIn.read_nb(temp_super_sample)) {
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                temp_tagged_input_triangle_delay_input[c].sample = temp_super_sample.superSample[c];
                temp_tagged_input_triangle_delay_input[c].valid = true;
            }
            control_bits = (control_count);
            bool pf_tick;
            if (pf_count == t_PF - 1) {
                pf_count = 0;
                pf_tick = true;
            } else {
                pf_count++;
                pf_tick = false;
            }
            if (pf_tick) {
                if (control_count == t_R - 1)
                    control_count = 0;
                else
                    control_count++;
            }
        } else // if the input fifo has no data push zeros input the commutor with invalid data tag :: here the
               // assumption is that there are no bules inside a frame, they can appear between frames
        {
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                temp_tagged_input_triangle_delay_input[c].sample = 0;
                temp_tagged_input_triangle_delay_input[c].valid = false;
            }
            control_bits = 0; //(control_count>>2);
            control_count = 0;
            pf_count = 0;
        }
        /// Read Sample from the input fifo
        input_triangle_delay_block.template process<taggeg_single_sample>(temp_tagged_input_triangle_delay_input,
                                                                          temp_tagged_mux_chain_input);
        muxChain_block.template genChain<t_R, ap_uint<control_count_width>, taggeg_single_sample>(
            control_bits, temp_tagged_mux_chain_input, temp_tagged_output_triangle_input);
        output_triangle_delay_block.template process<taggeg_single_sample>(temp_tagged_output_triangle_input,
                                                                           commuted_output);
        SuperSampleContainer<t_R, T_in> temp_output;
#pragma HLS ARRAY_PARTITION variable = temp_output.superSample complete dim = 1

        for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
            temp_output.superSample[c] = commuted_output[c].sample;
        }
        bool valid_flag = true;
        for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
            valid_flag = valid_flag && commuted_output[c].valid;
        }
        if (valid_flag == true) {
            p_sampleOut.write(temp_output);
        }
    }
}

// template < int t_stage, int id , int t_instanceID, int t_PF, int t_isLargeMem>
template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF, int t_isLargeMem>
struct streamingDataCommutations {
    template <typename T_in>
    void streamingDataCommutor(T_in p_in[t_R][t_L / t_R], T_in p_out[t_R][t_L / t_R]);
};
// template <int t_stage,int id,int t_instanceID,int t_PF, int t_isLargeMem>
template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF, int t_isLargeMem>
template <typename T_in>
void streamingDataCommutations<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L, t_R, t_PF, t_isLargeMem>::
    streamingDataCommutor(T_in p_in[t_R][t_L / t_R], T_in p_out[t_R][t_L / t_R]) {
#pragma HLS INLINE
#pragma HLS STREAM variable = p_in
#pragma HLS STREAM variable = p_out
#pragma HLS RESOURCE variable = p_in core = FIFO_LUTRAM
#pragma HLS RESOURCE variable = p_out core = FIFO_LUTRAM

#pragma HLS ARRAY_PARTITION variable = p_in complete dim = 1
#pragma HLS ARRAY_PARTITION variable = p_out complete dim = 1

    hls::stream<SuperSampleContainer<t_R, T_in> > superSample_in;
#pragma HLS STREAM variable = superSample_in depth = 8
#pragma HLS RESOURCE variable = superSample_in core = FIFO_LUTRAM

    hls::stream<SuperSampleContainer<t_R, T_in> > superSample_out;
#pragma HLS STREAM variable = superSample_out depth = 8
#pragma HLS RESOURCE variable = superSample_out core = FIFO_LUTRAM

    convertArrayToSuperStream<t_stage, t_instanceID, t_L, t_R, T_in>(p_in, superSample_in);
    streamingDataCommutorInner<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L, t_R, t_PF, T_in>(superSample_in,
                                                                                                      superSample_out);
    convertSuperStreamToArray<t_stage, t_instanceID, t_L, t_R, T_in>(superSample_out, p_out);
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF, int t_isLargeMem>
struct DataCommutationsS2Streaming {
    template <typename T_in>
    void streamingDataCommutor(hls::stream<SuperSampleContainer<t_R, T_in> >& p_sampleIn,
                               hls::stream<SuperSampleContainer<t_R, T_in> >& p_sampleOut);
};

template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF, int t_isLargeMem>
template <typename T_in>
void DataCommutationsS2Streaming<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L, t_R, t_PF, t_isLargeMem>::
    streamingDataCommutor(hls::stream<SuperSampleContainer<t_R, T_in> >& p_sampleIn,
                          hls::stream<SuperSampleContainer<t_R, T_in> >& p_sampleOut)

{
#pragma HLS INLINE off
    enum { delay_factor = 2 };
    typedef tagged_sample<SuperSampleContainer<t_R, T_in> > tagged_super_sample;
    typedef tagged_sample<T_in> taggeg_single_sample;
    const int control_count_width = ((ssrFFTLog2<t_R>::val) > 0) ? (ssrFFTLog2<t_R>::val) : 1;
    const int sample_count_width = ((ssrFFTLog2<t_L / t_R>::val) > 0) ? (ssrFFTLog2<t_L / t_R>::val) : 1;
    // If t_PF log2 is smaller then 1 then assign bit width as 1
    const int pf_count_width = (ssrFFTLog2<t_PF>::val > 0) ? (ssrFFTLog2<t_PF>::val) : 1;

    static ap_uint<control_count_width> control_count = 0;
    static ap_uint<pf_count_width> pf_count = 0;
    static ap_uint<sample_count_width> sample_in_read_count = 0;
    static bool delay_line_stall = false;
    static ap_uint<control_count_width> control_bits;

    SuperSampleContainer<t_R, T_in> temp_super_sample;
#pragma HLS ARRAY_PARTITION variable = temp_super_sample.superSample complete dim = 1
    tagged_sample<T_in> temp_tagged_input_triangle_delay_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_input_triangle_delay_input complete dim = 1
    taggeg_single_sample temp_tagged_mux_chain_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_mux_chain_input complete dim = 1
    taggeg_single_sample temp_tagged_output_triangle_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_output_triangle_input complete dim = 1
    taggeg_single_sample commuted_output[t_R];
#pragma HLS ARRAY_PARTITION variable = commuted_output complete dim = 1

    TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, false, t_R> input_triangle_delay_block;
    TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, true, t_R> output_triangle_delay_block;

    MuxChain<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, t_R> muxChain_block;

    tagged_sample<SuperSampleContainer<t_R, T_in> > temp_tagged_sample;
    for (int t = 0; t < t_L / t_R + delay_factor * (t_R - 1) * t_PF; t++) {
#pragma HLS PIPELINE II = 1 rewind
        /// Read Sample from the input fifo
        // If the input fifo has data read it, tag it and push it into the commutor network
        // bool fifo_has_next_sample = p_sampleIn.read_nb(temp_super_sample);
        bool fifo_has_next_sample = !p_sampleIn.empty();
        if (fifo_has_next_sample) {
            p_sampleIn.read(temp_super_sample);
        }
        if (fifo_has_next_sample) {
            control_bits = (control_count);
            bool pf_tick;
            if (pf_count == t_PF - 1) {
                pf_count = 0;
                pf_tick = true;
            } else {
                pf_count++;
                pf_tick = false;
            }
            if (pf_tick) {
                if (control_count == t_R - 1)
                    control_count = 0;
                else
                    control_count++;
            }
            if (sample_in_read_count == (t_L / t_R) - 1) {
                sample_in_read_count = 0;
                delay_line_stall = false;
            } else {
                sample_in_read_count++;
                delay_line_stall = true;
            }
        }
        if (fifo_has_next_sample) {
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                temp_tagged_input_triangle_delay_input[c].sample = temp_super_sample.superSample[c];
                temp_tagged_input_triangle_delay_input[c].valid = true;
            }
        } else {
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                temp_tagged_input_triangle_delay_input[c].sample = 0;
                temp_tagged_input_triangle_delay_input[c].valid = false;
            }
        }

        // Push next sample into the delay line network if the fifo read produced valid sample
        // else depending on the condition if the stall is needed stall the delay line network
        // or push a zero sample.
        if (fifo_has_next_sample || !delay_line_stall) {
            input_triangle_delay_block.template process<taggeg_single_sample>(temp_tagged_input_triangle_delay_input,
                                                                              temp_tagged_mux_chain_input);
            muxChain_block.template genChain<t_R, ap_uint<control_count_width>, taggeg_single_sample>(
                control_bits, temp_tagged_mux_chain_input, temp_tagged_output_triangle_input);
            output_triangle_delay_block.template process<taggeg_single_sample>(temp_tagged_output_triangle_input,
                                                                               commuted_output);
            SuperSampleContainer<t_R, T_in> temp_output;
#pragma HLS ARRAY_PARTITION variable = temp_output.superSample complete dim = 1
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                temp_output.superSample[c] = commuted_output[c].sample;
            }
            bool valid_flag = true;
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                valid_flag = valid_flag && commuted_output[c].valid;
            }
            if (valid_flag == true) {
                p_sampleOut.write(temp_output);
#ifdef __DEBUG_STREAMING_DATA_COMMUTOR_
                std::cout << "============================================\n";
                for (int r = 0; r < t_R; ++r) {
                    std::cout << temp_output.superSample[r] << std::endl;
                }
                std::cout << "============================================\n";

#endif
            }
        } // stall if end
    }     // for loop end
    // ap_wait_n(10);
    //#ifndef __SYNTHESIS__
    //    std::cout << "p_sampleOut.size() = " << p_sampleOut.size() << std::endl;
    //#endif
} // function end
#if 0
template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF, int t_isLargeMem>
template <typename T_in>
void DataCommutationsS2Streaming<t_instanceID, t_stage, t_subStage, t_forkNumber, t_L, t_R, t_PF, t_isLargeMem>::
    streamingDataCommutor(hls::stream<SuperSampleContainer<t_R, T_in> >& p_sampleIn,
                          hls::stream<SuperSampleContainer<t_R, T_in> >& p_sampleOut)

{
#pragma HLS INLINE off
    typedef tagged_sample<SuperSampleContainer<t_R, T_in> > tagged_super_sample;
    typedef tagged_sample<T_in> taggeg_single_sample;
    const int control_count_width = ((ssrFFTLog2<t_R>::val) > 0) ? (ssrFFTLog2<t_R>::val) : 1;
    const int sample_count_width = ((ssrFFTLog2<t_L / t_R>::val) > 0) ? (ssrFFTLog2<t_L / t_R>::val) : 1;
    // If t_PF log2 is smaller then 1 then assign bit width as 1
    const int pf_count_width = (ssrFFTLog2<t_PF>::val > 0) ? (ssrFFTLog2<t_PF>::val) : 1;

    static ap_uint<control_count_width> control_count = 0;
    static ap_uint<pf_count_width> pf_count = 0;
    static ap_uint<sample_count_width> sample_in_read_count = 0;
    static bool delay_line_stall = false;
    static ap_uint<control_count_width> control_bits;

    SuperSampleContainer<t_R, T_in> temp_super_sample;
#pragma HLS ARRAY_PARTITION variable = temp_super_sample.superSample complete dim = 1
    tagged_sample<T_in> temp_tagged_input_triangle_delay_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_input_triangle_delay_input complete dim = 1
    taggeg_single_sample temp_tagged_mux_chain_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_mux_chain_input complete dim = 1
    taggeg_single_sample temp_tagged_output_triangle_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_output_triangle_input complete dim = 1
    taggeg_single_sample commuted_output[t_R];
#pragma HLS ARRAY_PARTITION variable = commuted_output complete dim = 1

    TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, false, t_R> input_triangle_delay_block;
    TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, true, t_R> output_triangle_delay_block;

    MuxChain<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, t_R> muxChain_block;

    tagged_sample<SuperSampleContainer<t_R, T_in> > temp_tagged_sample;

    int cnt = 0;
    while (1) {
#pragma HLS PIPELINE II = 1 rewind
        /// Read Sample from the input fifo
        // If the input fifo has data read it, tag it and push it into the commutor network
        bool fifo_has_next_sample = p_sampleIn.read_nb(temp_super_sample);

        if (fifo_has_next_sample) {
            control_bits = (control_count);
            bool pf_tick;
            if (pf_count == t_PF - 1) {
                pf_count = 0;
                pf_tick = true;
            } else {
                pf_count++;
                pf_tick = false;
            }
            if (pf_tick) {
                if (control_count == t_R - 1)
                    control_count = 0;
                else
                    control_count++;
            }
            if (sample_in_read_count == (t_L / t_R) - 1) {
                sample_in_read_count = 0;
                delay_line_stall = false;
            } else {
                sample_in_read_count++;
                delay_line_stall = true;
            }
        }
        if (fifo_has_next_sample) {
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                temp_tagged_input_triangle_delay_input[c].sample = temp_super_sample.superSample[c];
                temp_tagged_input_triangle_delay_input[c].valid = true;
            }
        } else {
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                temp_tagged_input_triangle_delay_input[c].sample = 0;
                temp_tagged_input_triangle_delay_input[c].valid = false;
            }
        }

        // Push next sample into the delay line network if the fifo read produced valid sample
        // else depending on the condition if the stall is needed stall the delay line network
        // or push a zero sample.
        if (fifo_has_next_sample || !delay_line_stall) {
            input_triangle_delay_block.template process<taggeg_single_sample>(temp_tagged_input_triangle_delay_input,
                                                                              temp_tagged_mux_chain_input);
            muxChain_block.template genChain<t_R, ap_uint<control_count_width>, taggeg_single_sample>(
                control_bits, temp_tagged_mux_chain_input, temp_tagged_output_triangle_input);
            output_triangle_delay_block.template process<taggeg_single_sample>(temp_tagged_output_triangle_input,
                                                                               commuted_output);
            SuperSampleContainer<t_R, T_in> temp_output;
#pragma HLS ARRAY_PARTITION variable = temp_output.superSample complete dim = 1
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                temp_output.superSample[c] = commuted_output[c].sample;
            }
            bool valid_flag = true;
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                valid_flag = valid_flag && commuted_output[c].valid;
            }
            if (valid_flag == true) {
                p_sampleOut.write(temp_output);
                cnt++;
                if (cnt == t_L / t_R) {
                    break;
                }
#ifdef __DEBUG_STREAMING_DATA_COMMUTOR_
                std::cout << "============================================\n";
                for (int r = 0; r < t_R; ++r) {
                    std::cout << temp_output.superSample[r] << std::endl;
                }
                std::cout << "============================================\n";

#endif
            }
        } // stall if end
    }     // for loop end
    // ap_wait_n(10);
    //#ifndef __SYNTHESIS__
    //    std::cout << "p_sampleOut.size() = " << p_sampleOut.size() << std::endl;
    //#endif
} // function end
#endif

// template <int t_stage, int id, int iid, typename T_T, unsigned int t_L,unsigned int t_R, unsigned int t_PF>
template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF, typename T_in>

void streamingDataCommutorInnerS2StreamNoStall(hls::stream<SuperSampleContainer<t_R, T_in> >& p_sampleIn,
                                               hls::stream<SuperSampleContainer<t_R, T_in> >& p_sampleOut) {
#pragma HLS INLINE off
    typedef tagged_sample<SuperSampleContainer<t_R, T_in> > tagged_super_sample;
    typedef tagged_sample<T_in> taggeg_single_sample;
    const int control_count_width = ((ssrFFTLog2<t_R>::val) > 0) ? (ssrFFTLog2<t_R>::val) : 1;
    // If t_PF log2 is smaller then 1 then assign bit width as 1
    const int pf_count_width = (ssrFFTLog2<t_PF>::val > 0) ? (ssrFFTLog2<t_PF>::val) : 1;

    static ap_uint<control_count_width> control_count = 0;
    // static unsigned int control_count = 0;

    static ap_uint<pf_count_width> pf_count = 0;
    // static unsigned int pf_count=0;

    // static unsigned int control_bits;
    static ap_uint<control_count_width> control_bits;
    SuperSampleContainer<t_R, T_in> temp_super_sample;
#pragma HLS ARRAY_PARTITION variable = temp_super_sample.superSample complete dim = 1

    tagged_sample<T_in> temp_tagged_input_triangle_delay_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_input_triangle_delay_input complete dim = 1

    taggeg_single_sample temp_tagged_mux_chain_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_mux_chain_input complete dim = 1

    taggeg_single_sample temp_tagged_output_triangle_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_output_triangle_input complete dim = 1

    taggeg_single_sample commuted_output[t_R];
#pragma HLS ARRAY_PARTITION variable = commuted_output complete dim = 1

    TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, false, t_R> input_triangle_delay_block;
    TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, true, t_R> output_triangle_delay_block;

    MuxChain<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, t_R> muxChain_block;

    tagged_sample<SuperSampleContainer<t_R, T_in> > temp_tagged_sample;
    for (int t = 0; t < t_L / t_R + (t_R - 1) * t_PF; t++) {
#pragma HLS PIPELINE II = 1 rewind
        /// Read Sample from the input fifo
        // If the input fifo has data read it, tag it and push it into the commutor network
        if (p_sampleIn.read_nb(temp_super_sample)) {
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                temp_tagged_input_triangle_delay_input[c].sample = temp_super_sample.superSample[c];
                temp_tagged_input_triangle_delay_input[c].valid = true;
            }

            control_bits = (control_count);
            bool pf_tick;
            if (pf_count == t_PF - 1) {
                pf_count = 0;
                pf_tick = true;
            } else {
                pf_count++;
                pf_tick = false;
            }
            if (pf_tick) {
                if (control_count == t_R - 1)
                    control_count = 0;
                else
                    control_count++;
            }

        }
        // if the input fifo has no data push zeros input the commutor with invalid data tag here
        // the assumption is that there are no bules inside a frame, they can appear between frames
        else {
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                temp_tagged_input_triangle_delay_input[c].sample = 0;
                temp_tagged_input_triangle_delay_input[c].valid = false;
            }
            control_bits = 0; //(control_count>>2);
            control_count = 0;
            pf_count = 0;
        }
        /// Read Sample from the input fifo
        input_triangle_delay_block.template process<taggeg_single_sample>(temp_tagged_input_triangle_delay_input,
                                                                          temp_tagged_mux_chain_input);
        muxChain_block.template genChain<t_R, ap_uint<control_count_width>, taggeg_single_sample>(
            control_bits, temp_tagged_mux_chain_input, temp_tagged_output_triangle_input);
        output_triangle_delay_block.template process<taggeg_single_sample>(temp_tagged_output_triangle_input,
                                                                           commuted_output);
        SuperSampleContainer<t_R, T_in> temp_output;
#pragma HLS ARRAY_PARTITION variable = temp_output.superSample complete dim = 1
        for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
            temp_output.superSample[c] = commuted_output[c].sample;
        }
        bool valid_flag = true;
        for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
            valid_flag = valid_flag && commuted_output[c].valid;
        }
        if (valid_flag == true) {
            p_sampleOut.write(temp_output);
        }
    }
}

template <int t_instanceID, int t_stage, int t_subStage, int t_forkNumber, int t_L, int t_R, int t_PF, typename T_in>
void streamingDataCommutorInnerS2streaming(hls::stream<SuperSampleContainer<t_R, T_in> >& p_sampleIn,
                                           hls::stream<SuperSampleContainer<t_R, T_in> >& p_sampleOut) {
#pragma HLS INLINE off
    typedef tagged_sample<SuperSampleContainer<t_R, T_in> > tagged_super_sample;
    typedef tagged_sample<T_in> taggeg_single_sample;
    const int control_count_width = ((ssrFFTLog2<t_R>::val) > 0) ? (ssrFFTLog2<t_R>::val) : 1;
    const int sample_count_width = ((ssrFFTLog2<t_L / t_R>::val) > 0) ? (ssrFFTLog2<t_L / t_R>::val) : 1;
    // If t_PF log2 is smaller then 1 then assign bit width as 1
    const int pf_count_width = (ssrFFTLog2<t_PF>::val > 0) ? (ssrFFTLog2<t_PF>::val) : 1;

    static ap_uint<control_count_width> control_count = 0;
    static ap_uint<pf_count_width> pf_count = 0;
    static ap_uint<sample_count_width> sample_in_read_count = 0;
    static bool delay_line_stall = false;
    static ap_uint<control_count_width> control_bits;

    SuperSampleContainer<t_R, T_in> temp_super_sample;
#pragma HLS ARRAY_PARTITION variable = temp_super_sample.superSample complete dim = 1
    tagged_sample<T_in> temp_tagged_input_triangle_delay_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_input_triangle_delay_input complete dim = 1
    taggeg_single_sample temp_tagged_mux_chain_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_mux_chain_input complete dim = 1
    taggeg_single_sample temp_tagged_output_triangle_input[t_R];
#pragma HLS ARRAY_PARTITION variable = temp_tagged_output_triangle_input complete dim = 1
    taggeg_single_sample commuted_output[t_R];
#pragma HLS ARRAY_PARTITION variable = commuted_output complete dim = 1

    TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, false, t_R> input_triangle_delay_block;
    TriangleDelay<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, true, t_R> output_triangle_delay_block;

    MuxChain<t_instanceID, t_stage, t_subStage, t_forkNumber, t_PF, t_R> muxChain_block;

    tagged_sample<SuperSampleContainer<t_R, T_in> > temp_tagged_sample;
    for (int t = 0; t < t_L / t_R + (t_R - 1) * t_PF; t++) {
#pragma HLS PIPELINE II = 1 rewind
        /// Read Sample from the input fifo
        // If the input fifo has data read it, tag it and push it into the commutor network
        bool fifo_has_next_sample = p_sampleIn.read_nb(temp_super_sample);
        if (fifo_has_next_sample) {
            control_bits = (control_count);
            bool pf_tick;
            if (pf_count == t_PF - 1) {
                pf_count = 0;
                pf_tick = true;
            } else {
                pf_count++;
                pf_tick = false;
            }
            if (pf_tick) {
                if (control_count == t_R - 1)
                    control_count = 0;
                else
                    control_count++;
            }
            if (sample_in_read_count == (t_L / t_R) - 1) {
                sample_in_read_count = 0;
                delay_line_stall = false;
            } else {
                sample_in_read_count++;
                delay_line_stall = true;
            }
        }
        if (fifo_has_next_sample) {
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                temp_tagged_input_triangle_delay_input[c].sample = temp_super_sample.superSample[c];
                temp_tagged_input_triangle_delay_input[c].valid = true;
            }
        } else {
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                temp_tagged_input_triangle_delay_input[c].sample = 0;
                temp_tagged_input_triangle_delay_input[c].valid = false;
            }
        }

        // Push next sample into the delay line network if the fifo read produced valid sample
        // else depending on the condition if the stall is needed stall the delay line network
        // or push a zero sample.
        if (fifo_has_next_sample || !delay_line_stall) {
            input_triangle_delay_block.template process<taggeg_single_sample>(temp_tagged_input_triangle_delay_input,
                                                                              temp_tagged_mux_chain_input);
            muxChain_block.template genChain<t_R, ap_uint<control_count_width>, taggeg_single_sample>(
                control_bits, temp_tagged_mux_chain_input, temp_tagged_output_triangle_input);
            output_triangle_delay_block.template process<taggeg_single_sample>(temp_tagged_output_triangle_input,
                                                                               commuted_output);
            SuperSampleContainer<t_R, T_in> temp_output;
#pragma HLS ARRAY_PARTITION variable = temp_output.superSample complete dim = 1
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                temp_output.superSample[c] = commuted_output[c].sample;
            }
            bool valid_flag = true;
            for (int c = 0; c < t_R; c++) {
#pragma HLS UNROLL
                valid_flag = valid_flag && commuted_output[c].valid;
            }
            if (valid_flag == true) {
                p_sampleOut.write(temp_output);
            }
        } // stall if end
    }     // for loop end
} // function end

} // end namespace fft
} // end namespace dsp
} // end namespace xf

#endif //__HLS_SSR_FFT_STREAMING_DATA_COMMUTOR_H__
