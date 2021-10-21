/**
* Copyright (C) 2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include "ap_int.h"
#include "hls_stream.h"
#include "xf_data_analytics/text/editDistance.hpp"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

template <int M>
void readToVec(const int addr, const int nrow, ap_uint<128>* vec_ptr, hls::stream<ap_uint<128> >& vec_strm) {
READ_CSV_LOOP:
    for (int i = 0; i < nrow; i++) {
#pragma HLS pipeline
        ap_uint<128> t = vec_ptr[addr + i];
        vec_strm.write(t);
    }
}

template <int M>
void splitToString(const int nrow,
                   hls::stream<ap_uint<128> >& vec_strm,

                   hls::stream<ap_uint<8> >& len_strm,
                   hls::stream<ap_uint<8 * M> >& comp_str_strm) {
    const int elem_per_line = 128 / (M + 1);

    ap_uint<8 * M> str;
    for (int i = 0; i < nrow; i++) {
#pragma HLS pipeline
        ap_uint<128> t = vec_strm.read();
        ap_uint<8> index = i % elem_per_line;
        ap_uint<8> len;
        if (index == 0) {
            len = t(127, 120);
            str(279, 160) = t(119, 0);
        } else if (index < elem_per_line - 1) {
            str(159, 32) = t;
        } else {
            str(31, 0) = t(127, 96);
            // write out
            comp_str_strm.write(str);
            len_strm.write(len);
        }
    }
}

template <int M>
void readCSV(const int addr,
             const int nrow,
             ap_uint<128>* csv_ptr,

             hls::stream<ap_uint<8> >& len_strm,
             hls::stream<ap_uint<8 * M> >& string_strm) {
#pragma HLS dataflow
    hls::stream<ap_uint<128> > vec_strm;
#pragma HLS STREAM variable = vec_strm depth = 64

    readToVec<M>(addr, nrow, csv_ptr, vec_strm);

    splitToString<M>(nrow, vec_strm, len_strm, string_strm);
}

template <int N, int M, int BIT>
void max_filter(const int nrow,
                hls::stream<ap_uint<8> >& len_of_ptn,
                hls::stream<ap_uint<8 * N> >& ptn_strm,
                hls::stream<ap_uint<8> >& len_of_input,
                hls::stream<ap_uint<8 * M> >& input_strm,

                hls::stream<ap_uint<BIT> >& len1_strm,
                hls::stream<ap_uint<8 * N> >& str1_strm,
                hls::stream<bool>& o_e_strm,
                hls::stream<ap_uint<BIT> >& len2_strm,
                hls::stream<ap_uint<8> >& o_med_strm,
                hls::stream<ap_uint<8 * M> >& o_str_strm) {
    const int elem_per_line = 128 / (M + 1);
    const int nread = nrow / elem_per_line;
    ap_uint<8> len1 = len_of_ptn.read();
    ap_uint<8 * N> str1 = ptn_strm.read();
    len1_strm.write(len1(BIT - 1, 0));
    str1_strm.write(str1);

    int cnt = 0;

    for (int i = 0; i < nread; i++) {
#pragma HLS pipeline
        ap_uint<8> len2 = len_of_input.read();
        ap_uint<8 * M> str2 = input_strm.read();

        ap_uint<8> minLen = (len1 < len2) ? len1 : len2;
        ap_uint<8> med;
        if (minLen >= 0 && minLen < 10)
            med = 0;
        else if (minLen >= 10 && minLen < 20)
            med = 1;
        else if (minLen >= 20 && minLen < 30)
            med = 2;
        else
            med = 3;

        ap_uint<8> difLen = (len1 < len2) ? (len2 - len1) : (len1 - len2);
        if (difLen <= med) { // Filter out
            len2_strm.write(len2(BIT - 1, 0));
            o_med_strm.write(med);
            o_str_strm.write(str2);
            o_e_strm.write(false);
#ifndef __SYNTHESIS__
            cnt++;
#endif
        }
    }
#ifndef __SYNTHESIS__
// std::cout << "After filter, string number is: " << cnt << std::endl;
#endif

    o_e_strm.write(true);
}

template <int N, int M, int BIT, int ED_NUM>
void dispatch(hls::stream<ap_uint<BIT> >& i_len1_strm,
              hls::stream<ap_uint<8 * N> >& i_str1_strm,
              hls::stream<bool>& i_e_strm,
              hls::stream<ap_uint<BIT> >& i_len2_strm,
              hls::stream<ap_uint<8> >& i_med_strm,
              hls::stream<ap_uint<8 * M> >& i_str2_strm,

              hls::stream<ap_uint<BIT> > o_len1_strm[ED_NUM],
              hls::stream<ap_uint<64> > o_str1_strm[ED_NUM],
              hls::stream<ap_uint<BIT> > o_len2_strm[ED_NUM],
              hls::stream<ap_uint<BIT> > o_med_strm[ED_NUM],
              hls::stream<ap_uint<64> > o_str2_strm[ED_NUM],
              hls::stream<bool> o_e_strm[ED_NUM]) {
    const int fold_num = (8 * M + 63) / 64;
    ap_uint<8> index = 0;

    ap_uint<BIT> len1 = i_len1_strm.read();
    ap_uint<8 * N> str1 = i_str1_strm.read();
    for (int i = 0; i < ED_NUM; i++) {
#pragma HLS unroll
        o_len1_strm[i].write(len1);
    }
    for (int j = 0; j < fold_num; j++) {
#pragma HLS pipeline
        ap_uint<64> t0 = ((j + 1) * 64 > 8 * M) ? str1(8 * N - 1, 64 * j) : str1(64 * j + 63, 64 * j);
        for (int i = 0; i < ED_NUM; i++) {
#pragma HLS unroll
            o_str1_strm[i].write(t0);
        }
    }

    bool last = i_e_strm.read();
    while (!last) {
        ap_uint<BIT> len2 = i_len2_strm.read();
        ap_uint<8> med = i_med_strm.read();
        ap_uint<8 * M> t = i_str2_strm.read();
        last = i_e_strm.read();

        o_len2_strm[index].write(len2);
        o_med_strm[index].write(med);
        o_e_strm[index].write(false);
        for (int j = 0; j < fold_num; j++) {
#pragma HLS pipeline
            ap_uint<64> t0 = ((j + 1) * 64 > 8 * M) ? t(8 * M - 1, 64 * j) : t(64 * j + 63, 64 * j);
            o_str2_strm[index].write(t0);
        }

        if (index == ED_NUM - 1)
            index = 0;
        else
            index++;
    }

    // padding to align
    if (index > 0) {
        for (int i = index; i < ED_NUM; i++) {
            o_len2_strm[i].write(0);
            o_med_strm[i].write(0);
            o_e_strm[i].write(false);
            for (int j = 0; j < fold_num; j++) {
#pragma HLS pipeline
                o_str2_strm[i].write(0);
            }
        }
    }

    for (int i = 0; i < ED_NUM; i++) {
#pragma HLS unroll
        o_e_strm[i].write(true);
    }
}

/**
 * @brief read until all channels have been empty
 */
template <int ED_NUM>
void collect(hls::stream<bool> i_match_strm[ED_NUM],
             hls::stream<bool> i_e_strm[ED_NUM],

             hls::stream<bool>& o_match_strm) {
    bool is_match = false;
    bool last[ED_NUM];
    bool cond = false;

    for (int i = 0; i < ED_NUM; i++) {
#pragma HLS unroll
        last[i] = i_e_strm[i].read();
        cond |= last[i];
    }

    while (!cond) {
#pragma HLS pipeline
        for (int i = 0; i < ED_NUM; i++) {
#pragma HLS unroll
            is_match |= i_match_strm[i].read();
            last[i] = i_e_strm[i].read();
            cond |= last[i];
        }
    }

    o_match_strm.write(is_match);
}

template <int N, int M, int BIT, int ED_NUM>
void process_unit(const int addr,
                  const int nrow,
                  hls::stream<ap_uint<8 * N> >& str1,
                  hls::stream<ap_uint<8> >& len1,
                  ap_uint<128>* csv_ptr,
                  hls::stream<bool>& o_match_strm) {
#pragma HLS dataflow

    hls::stream<ap_uint<8> > len2_strm;
#pragma HLS STREAM variable = len2_strm depth = 4
    hls::stream<ap_uint<8 * M> > str2_strm;
#pragma HLS STREAM variable = str2_strm depth = 4
#pragma HLS resource variable = str2_strm core = FIFO_LUTRAM

    hls::stream<ap_uint<BIT> > filter_len1_strm;
#pragma HLS STREAM variable = filter_len1_strm depth = 2
    hls::stream<ap_uint<BIT> > filter_len2_strm;
#pragma HLS STREAM variable = filter_len2_strm depth = 2
    hls::stream<bool> filter_e_strm;
#pragma HLS STREAM variable = filter_e_strm depth = 4
    hls::stream<ap_uint<8> > filter_med_strm;
#pragma HLS STREAM variable = filter_med_strm depth = 4
    hls::stream<ap_uint<8 * N> > filter_str1_strm;
#pragma HLS STREAM variable = filter_str1_strm depth = 2
#pragma HLS resource variable = filter_str1_strm core = FIFO_LUTRAM
    hls::stream<ap_uint<8 * M> > filter_str2_strm;
#pragma HLS STREAM variable = filter_str2_strm depth = 4
#pragma HLS resource variable = filter_str2_strm core = FIFO_LUTRAM

    hls::stream<ap_uint<BIT> > dispatch_len1_strm[ED_NUM];
#pragma HLS STREAM variable = dispatch_len1_strm depth = 2
#pragma HLS resource variable = dispatch_len1_strm core = FIFO_LUTRAM
    hls::stream<ap_uint<BIT> > dispatch_len2_strm[ED_NUM];
#pragma HLS STREAM variable = dispatch_len2_strm depth = 4
#pragma HLS resource variable = dispatch_len2_strm core = FIFO_LUTRAM
    hls::stream<bool> dispatch_e_strm[ED_NUM];
#pragma HLS STREAM variable = dispatch_e_strm depth = 4
#pragma HLS resource variable = dispatch_e_strm core = FIFO_LUTRAM
    hls::stream<ap_uint<6> > dispatch_med_strm[ED_NUM];
#pragma HLS STREAM variable = dispatch_med_strm depth = 4
#pragma HLS resource variable = dispatch_med_strm core = FIFO_LUTRAM
    hls::stream<ap_uint<64> > dispatch_str1_strm[ED_NUM];
#pragma HLS STREAM variable = dispatch_str1_strm depth = 8
#pragma HLS resource variable = dispatch_str1_strm core = FIFO_LUTRAM
    hls::stream<ap_uint<64> > dispatch_str2_strm[ED_NUM];
#pragma HLS STREAM variable = dispatch_str2_strm depth = 8
#pragma HLS resource variable = dispatch_str2_strm core = FIFO_LUTRAM

    hls::stream<bool> ed_e_strm[ED_NUM];
#pragma HLS STREAM variable = ed_e_strm depth = 4
    hls::stream<bool> ed_match_strm[ED_NUM];
#pragma HLS STREAM variable = ed_match_strm depth = 4

    readCSV<M>(addr, nrow, csv_ptr, len2_strm, str2_strm);

    max_filter<N, M, BIT>(nrow, len1, str1, len2_strm, str2_strm, filter_len1_strm, filter_str1_strm, filter_e_strm,
                          filter_len2_strm, filter_med_strm, filter_str2_strm);

    dispatch<N, M, BIT, ED_NUM>(filter_len1_strm, filter_str1_strm, filter_e_strm, filter_len2_strm, filter_med_strm,
                                filter_str2_strm, dispatch_len1_strm, dispatch_str1_strm, dispatch_len2_strm,
                                dispatch_med_strm, dispatch_str2_strm, dispatch_e_strm);

    for (int i = 0; i < ED_NUM; i++) {
#pragma HLS unroll
        xf::data_analytics::text::editDistance<N, M, BIT>(
            dispatch_len1_strm[i], dispatch_str1_strm[i], dispatch_len2_strm[i], dispatch_str2_strm[i],
            dispatch_med_strm[i], dispatch_e_strm[i], ed_e_strm[i], ed_match_strm[i]);
    }

    collect<ED_NUM>(ed_match_strm, ed_e_strm, o_match_strm);
}

template <int N, int PU_NUM>
void readAndDispatch(ap_uint<32>* buf_in,
                     hls::stream<ap_uint<8 * N> > str[PU_NUM],
                     hls::stream<ap_uint<8> > len[PU_NUM]) {
    const int kdepth = (32 + 3) / 4;

    ap_uint<8 * N> str_tmp;
    ap_uint<8> len_tmp = buf_in[0](31, 24);
    str_tmp(8 * N - 1, 8 * N - 24) = buf_in[0](23, 0);

    for (int i = 0; i < kdepth; i++) {
#pragma HLS pipeline
        ap_uint<32> t = buf_in[i + 1];
        str_tmp(32 * (kdepth - i - 1) + 31, 32 * (kdepth - i - 1)) = t;
    }

    for (int i = 0; i < PU_NUM; i++) {
#pragma HLS unroll
        str[i].write(str_tmp);
        len[i].write(len_tmp);
    }
}

template <int PU_NUM>
void mergeAndWrite(hls::stream<bool> i_match_strm[PU_NUM], ap_uint<32>* buf_out) {
    bool match = false;
    for (int i = 0; i < PU_NUM; i++) {
#pragma HLS unroll
        match |= i_match_strm[i].read();
    }

    ap_uint<32> t = match ? 1 : 0;
    buf_out[0] = t;
}

extern "C" void fuzzy_kernel(int base_addr,
                             int entry_num,
                             ap_uint<32>* buf_in,

                             ap_uint<128>* csv_part0,
                             ap_uint<128>* csv_part1,
                             ap_uint<128>* csv_part2,
                             ap_uint<128>* csv_part3,
                             ap_uint<128>* csv_part4,
                             ap_uint<128>* csv_part5,
                             ap_uint<128>* csv_part6,
                             ap_uint<128>* csv_part7,

                             ap_uint<32>* buf_out) {
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_read_outstanding = 2 max_read_burst_length = 8 bundle = \
    gmem0_0 port = buf_in
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_read_outstanding = 2 max_read_burst_length = 32 bundle = \
    gmem1_0 port = csv_part0
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_read_outstanding = 2 max_read_burst_length = 32 bundle = \
    gmem1_1 port = csv_part1
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_read_outstanding = 2 max_read_burst_length = 32 bundle = \
    gmem1_2 port = csv_part2
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_read_outstanding = 2 max_read_burst_length = 32 bundle = \
    gmem1_3 port = csv_part3
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_read_outstanding = 2 max_read_burst_length = 32 bundle = \
    gmem1_4 port = csv_part4
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_read_outstanding = 2 max_read_burst_length = 32 bundle = \
    gmem1_5 port = csv_part5
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_read_outstanding = 2 max_read_burst_length = 32 bundle = \
    gmem1_6 port = csv_part6
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_read_outstanding = 2 max_read_burst_length = 32 bundle = \
    gmem1_7 port = csv_part7
#pragma HLS INTERFACE m_axi offset = slave latency = 64 num_write_outstanding = 2 max_write_burst_length = 2 bundle = \
    gmem2_0 port = buf_out

#pragma HLS INTERFACE s_axilite port = base_addr bundle = control
#pragma HLS INTERFACE s_axilite port = entry_num bundle = control
#pragma HLS INTERFACE s_axilite port = buf_in bundle = control
#pragma HLS INTERFACE s_axilite port = csv_part0 bundle = control
#pragma HLS INTERFACE s_axilite port = csv_part1 bundle = control
#pragma HLS INTERFACE s_axilite port = csv_part2 bundle = control
#pragma HLS INTERFACE s_axilite port = csv_part3 bundle = control
#pragma HLS INTERFACE s_axilite port = csv_part4 bundle = control
#pragma HLS INTERFACE s_axilite port = csv_part5 bundle = control
#pragma HLS INTERFACE s_axilite port = csv_part6 bundle = control
#pragma HLS INTERFACE s_axilite port = csv_part7 bundle = control
#pragma HLS INTERFACE s_axilite port = buf_out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    const int N = 35;
    const int M = 35;
    const int BIT = 6;
    const int ED_NUM = 10;
    const int PU_NUM = 8;

#pragma HLS dataflow

    hls::stream<ap_uint<8 * N> > str1[PU_NUM];
#pragma HLS STREAM variable = str1 depth = 2
#pragma HLS resource variable = str1 core = FIFO_LUTRAM
    hls::stream<ap_uint<8> > len1[PU_NUM];
#pragma HLS STREAM variable = len1 depth = 2

    hls::stream<bool> pu_match[PU_NUM];
#pragma HLS STREAM variable = pu_match depth = 2

    readAndDispatch<N, PU_NUM>(buf_in, str1, len1);

    process_unit<N, M, BIT, ED_NUM>(base_addr, entry_num, str1[0], len1[0], csv_part0, pu_match[0]);
    process_unit<N, M, BIT, ED_NUM>(base_addr, entry_num, str1[1], len1[1], csv_part1, pu_match[1]);
    process_unit<N, M, BIT, ED_NUM>(base_addr, entry_num, str1[2], len1[2], csv_part2, pu_match[2]);
    process_unit<N, M, BIT, ED_NUM>(base_addr, entry_num, str1[3], len1[3], csv_part3, pu_match[3]);
    process_unit<N, M, BIT, ED_NUM>(base_addr, entry_num, str1[4], len1[4], csv_part4, pu_match[4]);
    process_unit<N, M, BIT, ED_NUM>(base_addr, entry_num, str1[5], len1[5], csv_part5, pu_match[5]);
    process_unit<N, M, BIT, ED_NUM>(base_addr, entry_num, str1[6], len1[6], csv_part6, pu_match[6]);
    process_unit<N, M, BIT, ED_NUM>(base_addr, entry_num, str1[7], len1[7], csv_part7, pu_match[7]);

    mergeAndWrite<PU_NUM>(pu_match, buf_out);
}
