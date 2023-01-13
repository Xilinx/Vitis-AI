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

#ifndef _XF_RESIZE_DOWN_AREA_
#define _XF_RESIZE_DOWN_AREA_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "core/xf_math.h"
#include "common/xf_utility.hpp"

#define AREADOWN_PARTIAL_RESULT_BITS 16

static uint32_t xFUdivResizeDownArea(unsigned short in_n, unsigned short in_d) {
    uint32_t out_div = uint32_t(in_n) * POW16 / in_d;
    return out_div;
}

template <int NUM_INPB, int LOG2_PB, int NUM_PB, int NPC>
static void flag_index_generator(ap_uint<32> Xscale,
                                 ap_uint<32> X_1PixelWeight,
                                 ap_uint<32> Y_1PixelWeight,
                                 ap_uint<16> row_index,
                                 int in_col_index,
                                 ap_uint<32> Xindex_output[NUM_PB],
                                 ap_uint<32>* Xindex_output_next,
                                 ap_uint<16> output_buffer_index[NUM_PB + 1],
                                 bool inflag_TA[NUM_PB][NUM_INPB],
                                 ap_uint<16>* skip_count,
                                 ap_uint<17> Wx[NUM_PB][NUM_INPB],
                                 bool* inflag_for_Nplus1_Procblock,
                                 ap_uint<17>* Wx_for_Nplus1_Procblock,
                                 bool* DDR_wr_en,
                                 bool* out_buffer_wr_en,
                                 bool Yaxis_overlap_en,
                                 ap_uint<32> Yindex_output,
                                 ap_uint<32> Yindex_output_prev,
                                 ap_uint<16> ouput_index_write_counter,
                                 unsigned short in_height,
                                 unsigned short in_width,
                                 unsigned short inImg_ncpr,
                                 ap_uint<16>* output_buffer_index_next_out) {
// clang-format off
    #pragma HLS inline
    // clang-format on

    ap_int<16> skip_count_tmp;
    ap_int<16> skip_count_tmp_opt;
    skip_count_tmp_opt = Xindex_output[0].range(31, 16); // - (Xindex_output[0].range(15,0)<X_1PixelWeight);

    ap_uint<16> output_buffer_index_start;
    if (Xscale == 0x10000)
        output_buffer_index_start = in_col_index * NPC;
    else {
        ap_uint<16> index_fract_value = Xindex_output[0].range(15, 0);
        ap_uint<16> weight_value = X_1PixelWeight.range(15, 0);
        ap_uint<16> sub_value = weight_value - index_fract_value;
        if (index_fract_value < weight_value && sub_value > 0x41)
            output_buffer_index_start = Xindex_output[0].range(31, 16) - 1;
        else
            output_buffer_index_start = Xindex_output[0].range(31, 16);
    }

    for (int pb_in = 0; pb_in < NUM_INPB + 1; pb_in++) {
        output_buffer_index[pb_in] = output_buffer_index_start + pb_in;
    }

    ap_uint<16> output_buffer_index_next;
    if (Xscale == 0x10000)
        output_buffer_index_next = (in_col_index + 1) * NPC;
    else {
        ap_uint<16> index_fract_value = Xindex_output_next[0].range(15, 0);
        ap_uint<16> weight_value = X_1PixelWeight.range(15, 0);
        ap_uint<16> sub_value = weight_value - index_fract_value;
        if (index_fract_value < weight_value && sub_value > 0x41)
            output_buffer_index_next = Xindex_output_next[0].range(31, 16) - 1;
        else
            output_buffer_index_next = Xindex_output_next[0].range(31, 16);
    }
    *output_buffer_index_next_out = output_buffer_index_next;

    ap_uint<16> int_bits_Xindex_out_previous;
    if (in_col_index == 0)
        int_bits_Xindex_out_previous = 0;
    else
        int_bits_Xindex_out_previous = (Xindex_output[0] - X_1PixelWeight - 0x41) >> 16;

    ap_uint<16> fract_bits_Xindex_out_previous;
    if (in_col_index == 0)
        fract_bits_Xindex_out_previous = 0;
    else
        fract_bits_Xindex_out_previous = (ap_uint<16>)(Xindex_output[0] - X_1PixelWeight);

    for (int ta_idx = 0; ta_idx < NUM_PB; ta_idx++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        for (int pb_in = 0; pb_in < NUM_INPB; pb_in++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            ap_uint<16> int_bits_Xindex_out = (Xindex_output[pb_in] - 0x41) >> 16;
            ap_uint<16> fract_bits_Xindex_out = Xindex_output[pb_in].range(15, 0);
            ap_uint<16> int_bits_Xindex_out_min1;   // = (Xindex_output[pb_in-1]-0x41)>>16;
            ap_uint<16> fract_bits_Xindex_out_min1; // = Xindex_output[pb_in-1].range(15,0);

            if (pb_in == 0) {
                int_bits_Xindex_out_min1 = 0;
                fract_bits_Xindex_out_min1 = 0;
            } else {
                int_bits_Xindex_out_min1 = (Xindex_output[pb_in - 1] - 0x41) >> 16;
                fract_bits_Xindex_out_min1 = Xindex_output[pb_in - 1].range(15, 0);
            }

            ap_uint<16> index_value = output_buffer_index[ta_idx];
            bool t1 = index_value == int_bits_Xindex_out;
            bool t2 = ((int_bits_Xindex_out - int_bits_Xindex_out_previous) == 1) && (pb_in == 0) &&
                      (index_value == int_bits_Xindex_out - 1) && fract_bits_Xindex_out < X_1PixelWeight &&
                      ((X_1PixelWeight - fract_bits_Xindex_out) > 0x41);
            bool t3 = ((int_bits_Xindex_out - int_bits_Xindex_out_min1) == 1) && (pb_in > 0) &&
                      (index_value == int_bits_Xindex_out - 1) && fract_bits_Xindex_out < X_1PixelWeight &&
                      ((X_1PixelWeight - fract_bits_Xindex_out) > 0x41);

            if (((t1 || t2 || t3) && (Xscale != 0x10000)) || ((Xscale == 0x10000) && (pb_in == ta_idx)))
                inflag_TA[ta_idx][pb_in] = 1;
            else
                inflag_TA[ta_idx][pb_in] = 0;
        }
    }

    ap_uint<32> input_index__for_Nplus1_Procblock = output_buffer_index_next * Xscale;
    ap_uint<16> intBits_input_index__for_Nplus1_Procblock = input_index__for_Nplus1_Procblock.range(31, 16);
    ap_uint<16> col_idx_x_NPC = (in_col_index + 1) * NPC;

    ap_uint<32> Xindex_for_Nplus1_Procblock = Xindex_output[NUM_PB - 1]; //+X_1PixelWeight;

    ap_uint<32> overlap_next_pixel = (((ap_uint<32>)output_buffer_index_next) << 16) - Xindex_output[NUM_PB - 1];

    if (NPC != 1) {
        // x scale is less than 1.5, then N+1 output pixel(inclusing partial output) can be generated using N input
        // pixel if( (output_buffer_index_start+4) == output_buffer_index_next && Xscale!=65536 && Xscale<98304 &&
        // overlap_next_pixel>0x41){
        if ((output_buffer_index_start + NPC) == output_buffer_index_next && Xscale != 65536 && Xscale < 98304 &&
            overlap_next_pixel > 0x41) {
            *inflag_for_Nplus1_Procblock = 1;
            *Wx_for_Nplus1_Procblock = Xindex_for_Nplus1_Procblock.range(15, 0);
        } else {
            *inflag_for_Nplus1_Procblock = 0;
            *Wx_for_Nplus1_Procblock = 0;
        }
    } else {
        if ((output_buffer_index_start + NPC) == output_buffer_index_next && overlap_next_pixel > 0x41) {
            *inflag_for_Nplus1_Procblock = 1;
            *Wx_for_Nplus1_Procblock = Xindex_for_Nplus1_Procblock.range(15, 0);
        } else {
            *inflag_for_Nplus1_Procblock = 0;
            *Wx_for_Nplus1_Procblock = 0;
        }
    }

    ap_uint<32> Yindex_output_tmp = Yindex_output; // - 0x41;
    ap_uint<32> overlap_with_next_row = 0x10000 - Yindex_output.range(15, 0);
    ap_uint<32> overlap_with_prev_row = 0x10000 - Yindex_output_prev.range(15, 0);
    ap_uint<32> Yindex_output_prev_tmp = Yindex_output_prev; // - 0x41;

    bool t1 = (ouput_index_write_counter <= output_buffer_index_next);
    bool t2 = Yaxis_overlap_en == 1;
    bool t3 = (Yindex_output_tmp.range(31, 16) != Yindex_output_prev_tmp.range(31, 16));
    bool if_test = t1 && (t2 || t3);
    int current_Yidx_int = Yindex_output_tmp.range(31, 16);
    int next_Yidx_int = Yindex_output_prev_tmp.range(31, 16);

    bool scale1_en = X_1PixelWeight[16] == 1;
    bool write_en_pixel_in_same_row = (ouput_index_write_counter <= output_buffer_index_next);
    bool overlap_en_next_row = (overlap_with_next_row > 0x41);
    bool overlap_en_prev_row = (overlap_with_prev_row > 0x41);
    bool output_row_en =
        (Yaxis_overlap_en == 1 || (Yindex_output_tmp.range(31, 16) != Yindex_output_prev_tmp.range(31, 16)));

    //## Yindex precision error when current index value is close to integeger number.. like 8.999928 is close
    // to 9.00000
    //## precision error upto 10^-2 is accepted.
    ap_uint<32> Yindex_output_precision_error = 0x10000 - Yindex_output.range(15, 0);
    ap_uint<32> Yindex_output_prev_precision_error = 0x10000 - Yindex_output_prev.range(15, 0);

    bool DDR_wr_en_tmp;
    if ((((Yaxis_overlap_en == 1 || (Yindex_output_tmp.range(31, 16) != Yindex_output_prev_tmp.range(31, 16)))) ||
         (Yindex_output_precision_error <= 65)) ||
        row_index == (in_height - 1)) {
        if ((ouput_index_write_counter <= output_buffer_index_next || (in_col_index == (inImg_ncpr)-1)) &&
            (Yindex_output_prev_precision_error > 65))
            DDR_wr_en_tmp = 1;
        else
            DDR_wr_en_tmp = 0;
    } else {
        DDR_wr_en_tmp = 0;
    }

    if (X_1PixelWeight[16] == 1 && Y_1PixelWeight[16] == 1)
        *DDR_wr_en = 1;
    else
        *DDR_wr_en = DDR_wr_en_tmp;

    if ((X_1PixelWeight[16] == 1) || (ouput_index_write_counter <= output_buffer_index_next) ||
        (in_col_index == (inImg_ncpr)-1)) {
        *out_buffer_wr_en = 1;
    } else {
        *out_buffer_wr_en = 0;
    }

    for (int ta_idx = 0; ta_idx < NUM_PB; ta_idx++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        for (int pb_in = 0; pb_in < NUM_INPB; pb_in++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on

            bool rangeA_0_to_scale = Xindex_output[pb_in].range(15, 0) <= X_1PixelWeight.range(15, 0); // Q0.16
            ap_uint<16> sub_result = X_1PixelWeight.range(15, 0) - Xindex_output[pb_in].range(15, 0);

            if (rangeA_0_to_scale == true && Xscale != 0x10000) {
                //				if(int_bits_wo_th_for_Wx[pb_in].range(LOG2_PB-1,0) == ta_idx)
                if (output_buffer_index[ta_idx] == Xindex_output[pb_in].range(31, 16))
                    Wx[ta_idx][pb_in] = Xindex_output[pb_in].range(15, 0);
                else
                    Wx[ta_idx][pb_in] = sub_result;
            } else
                Wx[ta_idx][pb_in] = X_1PixelWeight.range(16, 0);
        }
    }
}

template <int SIZE>
void treeAdder(ap_uint<32> in1[SIZE], ap_uint<32>* output) {
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=in1 complete dim=1
    #pragma HLS inline
    // clang-format on

    ap_uint<32> add1_out[SIZE / 2];
    ap_uint<32> add2_out[SIZE / 4];
    ap_uint<32> add3_out[SIZE / 8];
    ap_uint<32> add4_out[SIZE / 16];

    if ((SIZE / 2) != 0) {
        for (ap_uint<10> idx = 0; idx < (SIZE / 2); idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            add1_out[idx] = in1[2 * idx] + in1[2 * idx + 1];
        }
    }

    if ((SIZE / 4) != 0) {
        for (ap_uint<10> idx = 0; idx < (SIZE / 4); idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            add2_out[idx] = add1_out[2 * idx] + add1_out[2 * idx + 1];
        }
    }

    if ((SIZE / 8) != 0) {
        for (ap_uint<10> idx = 0; idx < (SIZE / 8); idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            add3_out[idx] = add2_out[2 * idx] + add2_out[2 * idx + 1];
        }
    }

    if ((SIZE / 16) != 0) {
        for (ap_uint<10> idx = 0; idx < (SIZE / 16); idx++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            add4_out[idx] = add3_out[2 * idx] + add3_out[2 * idx + 1];
        }
    }

    ap_uint<32> add_out;
    if ((SIZE / 2) == 1)
        add_out = add1_out[0];
    else if ((SIZE / 4) == 1)
        add_out = add2_out[0];
    else if ((SIZE / 8) == 1)
        add_out = add3_out[0];
    else if ((SIZE / 16) == 1)
        add_out = add4_out[0];
    else
        add_out = in1[0];

    *output = (add_out);
}

template <int NUM_INPB, int NUM_PB, int NPC>
static void processBlock(bool inflag_TA[NUM_INPB],
                         ap_uint<8> input_1plane[NUM_INPB],
                         ap_uint<17> Wx[NUM_INPB],
                         ap_uint<17> Wy,
                         ap_uint<AREADOWN_PARTIAL_RESULT_BITS>* procBlock_out) {
    ap_uint<32> mul_out[NPC];
    for (int pixelproc = 0; pixelproc < NPC; pixelproc++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        ap_uint<8> in_data;
        if (inflag_TA[pixelproc] == 1)
            in_data = input_1plane[pixelproc];
        else
            in_data = 0;
        //##x_mul:Q8.16 = Q1.16 x Q8.0
        ap_uint<24> x_mul = Wx[pixelproc] * in_data;
        ap_uint<24> x_mul_round = x_mul + (1 << (8 - 1));
        //##mul_out:Q8.24 = Q1.16 x Q8.8
        mul_out[pixelproc] = Wy * (x_mul_round >> 8);
    }
    //##ta_out Q8.24
    ap_uint<32> ta_out;
    treeAdder<NPC>(mul_out, &ta_out);
    ap_uint<32> ta_out_round = ta_out + (1 << ((32 - AREADOWN_PARTIAL_RESULT_BITS) - 1));
    //##procBlock_out:Q8.8
    *procBlock_out = ta_out_round >> (32 - AREADOWN_PARTIAL_RESULT_BITS);
}

/*
 * Core Processing Block
 *
 *  PixelValue = Wx0*Wy0*data0[0] + Wx1*Wy0*data0[1] + Wx2*Wy0*data0[2] + Wx3*Wy0*data0[3] + Wx4*Wy0*data0[4] +
 *  			 Wx0*Wy1*data1[0] + Wx1*Wy1*data1[1] + Wx2*Wy1*data1[2] + Wx3*Wy1*data1[3] + Wx4*Wy1*data1[4] +
 *  			 Wx0*Wy2*data2[0] + Wx1*Wy2*data2[1] + Wx2*Wy2*data2[2] + Wx3*Wy2*data2[3] + Wx4*Wy2*data2[4] +
 *  			 Wx0*Wy3*data3[0] + Wx1*Wy3*data3[1] + Wx2*Wy3*data3[2] + Wx3*Wy3*data3[3] + Wx4*Wy3*data3[4] +
 *  			 Wx0*Wy4*data4[0] + Wx1*Wy4*data4[1] + Wx2*Wy4*data4[2] + Wx3*Wy4*data4[3] +; Wx4*Wy4*data4[4] +
 */
/**
 * Stream implementation of resizing the image using area interpolation technique.
 */
template <int NUM_INPB, int NUM_PB, int DEPTH, int WORDWIDTH, int PLANES, int NPC>
static void CoreProcessDownArea(ap_uint<17> Wx[NUM_PB][NUM_INPB],
                                ap_uint<17> Wy,
                                bool inflag_TA[NUM_PB][NUM_INPB],
                                XF_TNAME(DEPTH, NPC) read_word,
                                ap_uint<AREADOWN_PARTIAL_RESULT_BITS> output_PB[PLANES][NUM_PB + 1],
                                ap_uint<17> Wx_for_Nplus1_Procblock) {
    ap_uint<8> read_word_extract[PLANES][NUM_PB];
    for (int pixel = 0, bit1 = 0; pixel < NUM_PB; pixel++, bit1 += (PLANES * 8)) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        for (int channel = 0, bit2 = 0; channel < PLANES; channel++, bit2 += 8) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            if (pixel < NPC)
                read_word_extract[channel][pixel] = read_word.range(bit1 + (bit2 + 7), bit1 + bit2);
            else
                read_word_extract[channel][pixel] = 0;
            //		fprintf(stderr,"\n.range( %d,%d )",bit1+(bit2+7),bit1+bit2);
        }
    }

    for (int procblock_index = 0; procblock_index < NUM_PB + 1; procblock_index++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        for (int plane_index = 0, bit = 0; plane_index < PLANES; plane_index++, bit += 8) {
// clang-format off
            #pragma HLS unroll
            // clang-format on

            ap_uint<8> input_1plane[NUM_INPB];
            for (int in_index = 0; in_index < NUM_INPB; in_index++) {
                input_1plane[in_index] = read_word_extract[plane_index][in_index];
            }

            if (procblock_index != NUM_PB) {
                ap_uint<AREADOWN_PARTIAL_RESULT_BITS> procBlock_out; // Q8.8
                processBlock<NUM_INPB, NUM_PB, NPC>(inflag_TA[procblock_index], input_1plane, Wx[procblock_index], Wy,
                                                    &procBlock_out);
                output_PB[plane_index][procblock_index] = procBlock_out;
            } else {
                // if(NPC!=1)
                {
                    //##x_mul:Q8.16 = Q1.16 x Q8.0
                    ap_uint<24> x_mul = Wx_for_Nplus1_Procblock * input_1plane[NUM_INPB - 1];
                    //##mul_out:Q8.24 = Q1.16 x Q8.8
                    ap_uint<32> mul_out = Wy * (x_mul >> 8);
                    output_PB[plane_index][procblock_index] = mul_out >> (32 - AREADOWN_PARTIAL_RESULT_BITS);
                    //				fprintf(stderr,"\n last PB: in x Wx x Wy = %d x %f x %f = %f",
                    //(int)input_1plane[NUM_INPB-1],(float)Wx_for_Nplus1_Procblock/(float)(1<<16),
                    //(float)Wy/(float)(1<<16), (float)mul_out/(float)(1<<24));
                }
            }
        }
    }
}

template <int PLANES, int NUM_PB, int LOG2_PB, int DST_COLS, int DEPTH_OUTBUFFER, int NPC>
static void update_output_buffer(bool DDR_write_en,
                                 bool out_buffer_wr_en,
                                 ap_uint<32> write_index,
                                 ap_uint<16> write_index_col,
                                 unsigned short out_width,
                                 ap_uint<AREADOWN_PARTIAL_RESULT_BITS> accum_reg[PLANES][2 * NPC],
                                 ap_uint<AREADOWN_PARTIAL_RESULT_BITS> accum_reg_overlap[PLANES][2 * NPC],
                                 ap_uint<AREADOWN_PARTIAL_RESULT_BITS> ouput_buffer[PLANES][NUM_PB][DEPTH_OUTBUFFER],
                                 ap_uint<16> output_buffer_Colindex[NUM_PB + 1],
                                 ap_uint<AREADOWN_PARTIAL_RESULT_BITS> PB_out[PLANES][NUM_PB + 1],
                                 int in_col_index,
                                 ap_uint<AREADOWN_PARTIAL_RESULT_BITS> PB_out_overlap[PLANES][NUM_PB + 1],
                                 bool Yaxis_overlap_en,
                                 ap_uint<8> DDR_write_data[PLANES][NPC]) {
// clang-format off
    #pragma HLS inline
    // clang-format on
    bool output_col_index_bit0 = write_index_col[0];

    ap_uint<AREADOWN_PARTIAL_RESULT_BITS> DDR_write0_temp[PLANES][NPC];
    ap_uint<AREADOWN_PARTIAL_RESULT_BITS> DDR_write1_temp[PLANES][NPC];

    ap_uint<AREADOWN_PARTIAL_RESULT_BITS> DDR_write0_temp_overlap[PLANES][NPC];
    ap_uint<AREADOWN_PARTIAL_RESULT_BITS> DDR_write1_temp_overlap[PLANES][NPC];

    for (int plane_id = 0; plane_id < PLANES; plane_id++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on

        for (ap_uint<8> accum_idx = 0, index_pixel = 0; accum_idx < 2 * NPC; accum_idx++, index_pixel++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on

            ap_uint<AREADOWN_PARTIAL_RESULT_BITS> data_mux_out = 0;         //
            ap_uint<AREADOWN_PARTIAL_RESULT_BITS> data_mux_out_overlap = 0; //
            ap_uint<NUM_PB + 1> data_mux_out_status = 0;
            for (ap_uint<16> out_idx = 0; out_idx < (NUM_PB + 1); out_idx++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                ap_uint<LOG2_PB + 1> out_index_val = output_buffer_Colindex[out_idx].range(LOG2_PB, 0);
                if (out_index_val == accum_idx) {
                    data_mux_out = PB_out[plane_id][out_idx];
                    data_mux_out_overlap = PB_out_overlap[plane_id][out_idx];
                    data_mux_out_status[out_idx] = 1;
                } else
                    data_mux_out_status[out_idx] = 0;
            }

            ap_uint<AREADOWN_PARTIAL_RESULT_BITS> data_previous;
            ap_uint<AREADOWN_PARTIAL_RESULT_BITS> data_previous_overlap;
            if (in_col_index == 0) {
                data_previous = 0;
                data_previous_overlap = 0;
            } else // if(DDR_write_en==0)
            {
                data_previous = accum_reg[plane_id][index_pixel];
                data_previous_overlap = accum_reg_overlap[plane_id][index_pixel];
            }

            ap_uint<AREADOWN_PARTIAL_RESULT_BITS> update;         // = data_mux_out + data_previous;
            ap_uint<AREADOWN_PARTIAL_RESULT_BITS> update_overlap; // = data_mux_out_overlap + data_previous_overlap;

            if (data_mux_out_status != 0) {
                update = data_mux_out + data_previous;
                update_overlap = data_mux_out_overlap + data_previous_overlap;
            } else {
                update = data_previous;
                update_overlap = data_previous_overlap;
            }

            if (((output_col_index_bit0 == 0 && accum_idx < NPC) || (output_col_index_bit0 == 1 && accum_idx >= NPC)) &&
                (DDR_write_en == 1 || out_buffer_wr_en == 1)) {
                accum_reg[plane_id][accum_idx] = 0;
                accum_reg_overlap[plane_id][accum_idx] = 0;
            } else {
                accum_reg[plane_id][accum_idx] = update;
                accum_reg_overlap[plane_id][accum_idx] = update_overlap;
            }

            if (accum_idx < NPC) {
                DDR_write0_temp[plane_id][accum_idx] = update;
                DDR_write0_temp_overlap[plane_id][accum_idx] = update_overlap;
            } else {
                DDR_write1_temp[plane_id][accum_idx - NPC] = update;
                DDR_write1_temp_overlap[plane_id][accum_idx - NPC] = update_overlap;
            }
        }
    }

    for (int plane_id = 0; plane_id < PLANES; plane_id++) {
// clang-format off
        #pragma HLS unroll
        // clang-format on
        for (ap_uint<8> index_pixel = 0; index_pixel < NPC; index_pixel++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            ap_uint<AREADOWN_PARTIAL_RESULT_BITS> temp_sum;
            ap_uint<AREADOWN_PARTIAL_RESULT_BITS> buffer_updated_data;
            ap_uint<AREADOWN_PARTIAL_RESULT_BITS> read_buffer_data =
                ouput_buffer[plane_id][index_pixel][write_index_col];
            if (output_col_index_bit0 == 0)
                temp_sum = (read_buffer_data + DDR_write0_temp[plane_id][index_pixel]);
            else
                temp_sum = (read_buffer_data + DDR_write1_temp[plane_id][index_pixel]);

            if (DDR_write_en == 1) {
                ap_uint<16> sum_rounding = temp_sum + (1 << (AREADOWN_PARTIAL_RESULT_BITS - 9));
                DDR_write_data[plane_id][index_pixel] = sum_rounding >> (AREADOWN_PARTIAL_RESULT_BITS - 8);
                ap_uint<AREADOWN_PARTIAL_RESULT_BITS> buffer_data_temp;
                if (Yaxis_overlap_en == 1)
                    if (output_col_index_bit0 == 0)
                        buffer_data_temp = DDR_write0_temp_overlap[plane_id][index_pixel];
                    else
                        buffer_data_temp = DDR_write1_temp_overlap[plane_id][index_pixel];
                else
                    buffer_data_temp = 0;

                ouput_buffer[plane_id][index_pixel][write_index_col] = buffer_data_temp;

            } else if (out_buffer_wr_en == 1) {
                ouput_buffer[plane_id][index_pixel][write_index_col] = temp_sum;
            }
        }
    }
}

template <int SRC_ROWS,
          int SRC_COLS,
          int PLANES,
          int DEPTH,
          int NPC,
          int WORDWIDTH,
          int DST_ROWS,
          int DST_COLS,
          int SRC_TC,
          int DST_TC>
void xFResizeAreaDownScale(xf::cv::Mat<DEPTH, SRC_ROWS, SRC_COLS, NPC>& stream_in,
                           xf::cv::Mat<DEPTH, DST_ROWS, DST_COLS, NPC>& resize_out) {
    unsigned short height = stream_in.rows;
    unsigned short width = stream_in.cols;
    unsigned short out_height = resize_out.rows;
    unsigned short out_width = resize_out.cols;

    unsigned short imgInput_ncpr = (width + (NPC - 1)) >> XF_BITSHIFT(NPC);
    unsigned short imgOutput_ncpr = (out_width + (NPC - 1)) >> XF_BITSHIFT(NPC);
    unsigned short imgOutput_width_align_npc = imgOutput_ncpr << XF_BITSHIFT(NPC);
    unsigned short in_col_loop_bound;
    if (imgOutput_width_align_npc != out_width)
        in_col_loop_bound = imgInput_ncpr + 1;
    else
        in_col_loop_bound = imgInput_ncpr;

    enum { NUM_PB = NPC, NUM_INPB = NPC, LOG2_PB = XF_BITSHIFT(NPC) };

    ap_uint<32> Xscale, Yscale; // Q16.16 format
    Xscale = xFUdivResizeDownArea((width), (out_width));
    Yscale = xFUdivResizeDownArea(height, out_height);
    ap_uint<32> X_1PixelWeight, Y_1PixelWeight; // Q16.16 format
    X_1PixelWeight = xFUdivResizeDownArea(out_width, width);
    Y_1PixelWeight = xFUdivResizeDownArea(out_height, height);

    //## X-direction output index(Q16.16), which is used for each Process block output.
    ap_uint<32> Xindex_output[NUM_PB];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=Xindex_output complete dim=0
    // clang-format on
    ap_uint<32> Xindex_output_initial[NUM_PB];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=Xindex_output_initial complete dim=0
    // clang-format on

    //## input flag for each input of last process block.
    //## TRUE - input data is mappped to multiplier
    //## FALSE- multiplier input is zero
    bool inflag_TA_prev[NUM_INPB];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=inflag_TA_prev complete dim=0
    // clang-format on
    for (ap_uint<8> idx = 1; idx <= NUM_PB; idx++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on
        Xindex_output_initial[idx - 1] = X_1PixelWeight * idx;
        //		inflag_TA_prev[idx-1] = true;
    }

    ap_uint<32> Xindex_output_initial_next;
    Xindex_output_initial_next = X_1PixelWeight * (1 + NUM_PB);

    ap_uint<32> Xindex_output_next;

    //## Y-direction output index(Q16.16)
    ap_uint<32> Yindex_output = Y_1PixelWeight;
    ap_uint<32> Yindex_output_prev = 0;

    //## skip_count Q16.0, it is used for mapping input data to process block
    ap_uint<16> skip_count = 0;

    //##DDR index
    uint32_t read_index = 0;
    ap_uint<32> write_index = 0;
    ap_uint<16> write_col_index = 0;

    //## overlap flag in Y-direction
    bool Yaxis_overlap_en = 0;
    bool Yaxis_overlap_nextrow_en = 0;
    bool Yaxis_overlap_prevrow_en = 0;

    enum { DEPTH_OUTBUFFER = (DST_COLS + NPC - 1) / NPC };

    //## output buffer
    ap_uint<AREADOWN_PARTIAL_RESULT_BITS> ouput_buffer[PLANES][NUM_PB][DEPTH_OUTBUFFER];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=ouput_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=ouput_buffer complete dim=2
    // clang-format on

    ap_uint<AREADOWN_PARTIAL_RESULT_BITS> accum_reg[PLANES][NPC * 2];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=accum_reg complete dim=0
    // clang-format on
    ap_uint<AREADOWN_PARTIAL_RESULT_BITS> accum_reg_overlap[PLANES][NPC * 2];
// clang-format off
    #pragma HLS ARRAY_PARTITION variable=accum_reg_overlap complete dim=0
    // clang-format on
    for (int dim2 = 0; dim2 < NPC * 2; dim2++) {
// clang-format off
        #pragma HLS unroll
        #pragma HLS unroll
        // clang-format on
        for (int dim1 = 0; dim1 < PLANES; dim1++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            accum_reg[dim1][dim2] = 0;
            accum_reg_overlap[dim1][dim2] = 0;
        }
    }

    for (ap_uint<16> dim3 = 0; dim3 < DEPTH_OUTBUFFER; dim3++) {
// clang-format off
        #pragma HLS pipeline
        // clang-format on
        for (int dim2 = 0; dim2 < NUM_PB; dim2++) {
// clang-format off
            #pragma HLS unroll
            // clang-format on
            for (int dim1 = 0; dim1 < PLANES; dim1++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                ouput_buffer[dim1][dim2][dim3] = 0;
            }
        }
    }

    int out_col_index = 0;
    ap_uint<16> output_row_index_for_pingpong = 0; // Q16.0
    bool prev_output_row_index_for_pingpong_bit0 = 0;
    ap_uint<16> ouput_index_write_counter = NPC;

    XF_TNAME(DEPTH, NPC) read_word;

    int display_write_per_row = 0;
    int display_write_rowID = 0;

LOOP_ROW:
    for (ap_uint<16> row_index = 0; row_index < height; row_index++) {
// clang-format off
        #pragma HLS LOOP_TRIPCOUNT min=1 max=SRC_ROWS
    // clang-format on

    LOOP_COL:
        for (int col_index = 0, col_index_next = 1; col_index < in_col_loop_bound; col_index++, col_index_next++) {
// clang-format off
            #pragma HLS LOOP_TRIPCOUNT min=1 max=SRC_TC
            #pragma HLS pipeline
            #pragma HLS DEPENDENCE variable=ouput_buffer inter false
            // clang-format on

            for (int idx = 0; idx < NUM_PB; idx++) {
// clang-format off
                #pragma HLS unroll
                // clang-format on
                if (col_index == 0) {
                    Xindex_output[idx] = Xindex_output_initial[idx];

                    inflag_TA_prev[idx] = true;
                } else {
                    if (NPC == 1)
                        Xindex_output[idx] += X_1PixelWeight;
                    else
                        Xindex_output[idx] += (X_1PixelWeight * NUM_PB);
                }
            }
            if (col_index == 0) {
                Xindex_output_next = Xindex_output_initial_next;
            } else {
                if (NPC == 1)
                    Xindex_output_next += X_1PixelWeight;
                else
                    Xindex_output_next += (X_1PixelWeight * NUM_PB);
            }
            ///////////////////////////////////////////////////////////

            ap_uint<16> output_buffer_index[NUM_PB + 1];
// clang-format off
            #pragma HLS ARRAY_PARTITION variable=output_buffer_index complete dim=0
            // clang-format on
            //## input flag for each input of process block.
            bool inflag_TA[NUM_PB][NUM_INPB];
// clang-format off
            #pragma HLS ARRAY_PARTITION variable=inflag_TA complete dim=0
            // clang-format on
            bool inflag_for_Nplus1_Procblock;
            ap_uint<17> Wx_for_Nplus1_Procblock;
            ap_uint<17> Wx[NUM_PB][NUM_INPB]; // Q1.16
// clang-format off
            #pragma HLS ARRAY_PARTITION variable=Wx complete dim=0
            // clang-format on

            bool DDR_wr_en;
            bool out_buffer_wr_en;

            ap_uint<16> output_buffer_index_next_out;
            flag_index_generator<NUM_PB, LOG2_PB, NUM_PB, NPC>(
                Xscale, X_1PixelWeight, Y_1PixelWeight, row_index, col_index, Xindex_output, &Xindex_output_next,
                output_buffer_index, inflag_TA, &skip_count, Wx, &inflag_for_Nplus1_Procblock, &Wx_for_Nplus1_Procblock,
                &DDR_wr_en, &out_buffer_wr_en, Yaxis_overlap_en, Yindex_output, Yindex_output_prev,
                ouput_index_write_counter, height, width, in_col_loop_bound, &output_buffer_index_next_out);

            if (col_index == (in_col_loop_bound)-1)
                ouput_index_write_counter = NPC;
            else if (ouput_index_write_counter <= output_buffer_index_next_out)
                ouput_index_write_counter += NPC;

            if (col_index < imgInput_ncpr) read_word = stream_in.read(read_index++);

            // TODO: Wy weight generation
            ap_uint<17> Wy0, Wy1; // Q1.16
            if (Yaxis_overlap_en == 1) {
                Wy0 = Y_1PixelWeight.range(15, 0) - Yindex_output.range(15, 0);
                Wy1 = Yindex_output.range(15, 0);
            } else {
                Wy0 = Y_1PixelWeight;
                Wy1 = 0;
            }
            //## output data of each process block
            ap_uint<AREADOWN_PARTIAL_RESULT_BITS> PB_out[PLANES][NUM_PB + 1]; // Q8.8
// clang-format off
            #pragma HLS ARRAY_PARTITION variable=PB_out complete dim=0
            // clang-format on
            ap_uint<AREADOWN_PARTIAL_RESULT_BITS> PB_out_overlap[PLANES][NUM_PB + 1]; // Q8.8
// clang-format off
            #pragma HLS ARRAY_PARTITION variable=PB_out_overlap complete dim=0
            // clang-format on

            //## CoreProcess has "NUM_PB" process blocks. Each process block has "NUM_INPB" 3-input multiplier and Tree
            // adder to accumulate multiplier output.
            CoreProcessDownArea<NUM_INPB, NUM_PB, DEPTH, WORDWIDTH, PLANES, NPC>(Wx, Wy0, inflag_TA, read_word, PB_out,
                                                                                 Wx_for_Nplus1_Procblock);

            //## Extra CoreProcess to process next output in case of overlap.
            CoreProcessDownArea<NUM_INPB, NUM_PB, DEPTH, WORDWIDTH, PLANES, NPC>(
                Wx, Wy1, inflag_TA, read_word, PB_out_overlap, Wx_for_Nplus1_Procblock);

            ap_uint<8> DDR_write_data[PLANES][NPC];
// clang-format off
            #pragma HLS ARRAY_PARTITION variable=DDR_write_data complete dim=0
            // clang-format on
            update_output_buffer<PLANES, NUM_PB, LOG2_PB, DST_COLS, DEPTH_OUTBUFFER, NPC>(
                DDR_wr_en, out_buffer_wr_en, write_index, write_col_index, out_width, accum_reg, accum_reg_overlap,
                ouput_buffer, output_buffer_index, PB_out, col_index, PB_out_overlap, Yaxis_overlap_en, DDR_write_data);

            if (DDR_wr_en == 1) {
                display_write_per_row++;
                XF_TNAME(DEPTH, NPC) out_pix;
                ap_uint<PLANES * 8> plane_tmp;
                for (int pixel = 0, bit1 = 0; pixel < NPC; pixel++, bit1 += (PLANES * 8)) {
// clang-format off
                    #pragma HLS unroll
                    // clang-format on
                    for (int channel = 0, bit2 = 0; channel < PLANES; channel++, bit2 += 8) {
// clang-format off
                        #pragma HLS unroll
                        // clang-format on
                        plane_tmp.range(bit2 + 7, bit2) = DDR_write_data[channel][pixel];
                    }
                    out_pix.range(bit1 + (PLANES * 8) - 1, bit1) = plane_tmp;
                }
                if (out_col_index < imgOutput_ncpr) resize_out.write(write_index++, out_pix);

                if (col_index == ((in_col_loop_bound)-1))
                    out_col_index = 0;
                else
                    out_col_index++;
            }

            if (col_index == ((in_col_loop_bound)-1))
                write_col_index = 0;
            else if (out_buffer_wr_en)
                write_col_index++;

            // last iteration of col loop
            if (col_index == ((in_col_loop_bound)-1)) {
                Yindex_output += Y_1PixelWeight;
                Yindex_output_prev += Y_1PixelWeight;
            }

            int t1 = Yindex_output.range(15, 0);
            int t2 = Y_1PixelWeight;
            int t3 = Yindex_output;

            ap_uint<32> Yindex_threshold = Yindex_output - 0x41;

            if (col_index == ((in_col_loop_bound)-1)) Yaxis_overlap_prevrow_en = Yaxis_overlap_en;

            if (Yindex_output.range(15, 0) < Y_1PixelWeight && (Y_1PixelWeight - Yindex_output.range(15, 0)) > 0x41 &&
                Y_1PixelWeight[16] == 0)
                Yaxis_overlap_en = 1;
            else
                Yaxis_overlap_en = 0;

            if (col_index == ((in_col_loop_bound)-1))
                prev_output_row_index_for_pingpong_bit0 = output_row_index_for_pingpong[0];

            if (Yaxis_overlap_en == 0) {
                if ((Yindex_output.range(15, 0) < Y_1PixelWeight) && (Yindex_output.range(15, 0) > 0x41) &&
                    (col_index == ((in_col_loop_bound)-1)))
                    output_row_index_for_pingpong = Yindex_threshold.range(31, 16) - 1;
                else
                    output_row_index_for_pingpong = Yindex_threshold.range(31, 16);
            }
        } // col loop
    }     // row loop
}

#endif //_XF_RESIZE_DOWN_AREA_
