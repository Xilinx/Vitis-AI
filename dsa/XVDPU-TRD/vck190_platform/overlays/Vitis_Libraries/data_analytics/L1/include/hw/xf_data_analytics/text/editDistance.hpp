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

/**
 * @file editDistance.hpp
 * @brief Implementation for Levenshtein distance to quantify how dissimilar two strings.
 *
 * This editDistance output one stream to indicate each edit distance of each input string against one query string
 * based on diagonal approach.
 */

#ifndef XF_TEXT_EDIT_DISTANCE_H
#define XF_TEXT_EDIT_DISTANCE_H

#include "ap_int.h"
#include "hls_stream.h"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace data_analytics {
namespace text {
namespace internal {

/**
 * @brief compute one step against diagonal line
 *
 * @param pattern_string shifted input string
 * @param input_string input query string
 * @param northwest cache bits at northwest direction
 * @param north cache bits at north direction
 * @param west cache bits at west direction
 *
 */
template <int N, int BITS>
void compute_ed(char pattern_string[N], // must be shifted
                char input_string[N],

                ap_uint<BITS> northwest[N],
                ap_uint<BITS> north[N],
                ap_uint<BITS> west[N]) {
#pragma HLS inline
    for (int i = N - 1; i >= 0; i--) { // Remove the dependence
#pragma HLS unroll
        const char ichar = input_string[i];
        const char pchar = pattern_string[i];

        const ap_uint<BITS> nw_value =
            (pchar != ichar && (northwest[i] <= (ap_uint<BITS>)-2)) ? (ap_uint<BITS>)(northwest[i] + 1) : northwest[i];
        const ap_uint<BITS> n_value = (north[i] > (ap_uint<BITS>)-2) ? north[i] : (ap_uint<BITS>)(north[i] + 1);
        const ap_uint<BITS> w_value = (west[i] > (ap_uint<BITS>)-2) ? west[i] : (ap_uint<BITS>)(west[i] + 1);

        if (nw_value <= n_value && nw_value <= w_value) { // north west
            northwest[i] = north[i];
            west[i] = nw_value;
            north[i] = nw_value;
        } else if (n_value <= w_value) { // north
            northwest[i] = north[i];
            west[i] = n_value;
            north[i] = n_value;
        } else { // west
            northwest[i] = north[i];
            west[i] = w_value;
            north[i] = w_value;
        }
    }
}

/**
 * @brief left shift one string toward the LSB char, with feeding the MSB char with given char
 *
 * @param in_char input char to fill the MSB location
 * @param str input string
 *
 */
template <int N>
void char_shift(char in_char, char str[N]) {
#pragma HLS inline
    for (int i = 1; i < N; i++) {
#pragma HLS unroll
        str[i - 1] = str[i];
    }
    str[N - 1] = in_char;
}

/**
 * @brief circular left shifter, each call will move one char for two input string independently
 *
 * @param northwest input string 1
 * @param west input string 2
 *
 */
template <int N, int BITS>
void left_shift(ap_uint<BITS> northwest[N], ap_uint<BITS> west[N]) {
#pragma HLS inline
    ap_uint<8> nw0 = northwest[0];
    ap_uint<8> w0 = west[0];
    for (int i = 1; i < N; i++) {
#pragma HLS unroll
        northwest[i - 1] = northwest[i];
        west[i - 1] = west[i];
    }
    northwest[N - 1] = nw0;
    west[N - 1] = w0;
}

} // namespace internal

/**
 * @brief Levenshtein distance implementation
 *
 * @tparam N maximum length of query string.
 * @tparam M maximum length of input stream string, N must be less than M.
 * @tparam BITS data width of internal edit distance in bits.
 *
 * @param len1_strm length of the query string in bytes.
 * @param query_strm the query string folded into mutiple 8B elements.
 * @param len2_strm length of each input string in bytes.
 * @param input_strm input strings to compute edit distance against the given query string, which is folded into
 * multiple 8B elements.
 * @param max_ed_strm the maximum threshold of edit distance.
 * @param i_e_strm end flag of input_strm and max_ed_strm.
 * @param o_e_strm end flag of output matched stream.
 * @param o_match_strm only the calculated ED less than threshold in max_ed_strm will be TRUE.
 *
 */
template <int N, int M, int BITS>
void editDistance(hls::stream<ap_uint<BITS> >& len1_strm,
                  hls::stream<ap_uint<64> >& query_strm,
                  hls::stream<ap_uint<BITS> >& len2_strm,
                  hls::stream<ap_uint<64> >& input_strm,
                  hls::stream<ap_uint<BITS> >& max_ed_strm,
                  hls::stream<bool>& i_e_strm,

                  hls::stream<bool>& o_e_strm,
                  hls::stream<bool>& o_match_strm) {
    ap_uint<BITS> north[N];
#pragma HLS ARRAY_PARTITION variable = north complete dim = 1
    ap_uint<BITS> west[N];
#pragma HLS ARRAY_PARTITION variable = west complete dim = 1
    ap_uint<BITS> northwest[N];
#pragma HLS ARRAY_PARTITION variable = northwest complete dim = 1

    char str1[N], str3[N];
#pragma HLS ARRAY_PARTITION variable = str1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = str3 complete dim = 1
    char str2[M];
#pragma HLS ARRAY_PARTITION variable = str2 complete dim = 1

    const int fold_num = (8 * M + 63) / 64;

    // read the query string
    const ap_uint<BITS> len1 = len1_strm.read();
    for (int i = 0; i < fold_num; i++) {
        ap_uint<64> str1_part = query_strm.read();
        for (int j = 0; j < 8; j++) {
#pragma HLS unroll
            if (i * 8 + j < N) str1[i * 8 + j] = str1_part(j * 8 + 7, j * 8);
        }
    }

    bool last = i_e_strm.read();
    while (!last) {
        // initialize the cache lines
        northwest[N - 1] = 0;
        north[N - 1] = 1;
        west[N - 1] = 1;
        for (int i = 0; i < N - 1; i++) {
#pragma HLS unroll
            northwest[i] = (ap_uint<BITS>)-1;
            north[i] = (i == 0) ? (ap_uint<BITS>)1 : (ap_uint<BITS>)-1;
            west[i] = (i == N - 2) ? (ap_uint<BITS>)1 : (ap_uint<BITS>)-1;
        }
        for (int i = 0; i < N; i++) {
#pragma HLS unroll
            str3[i] = 255;
        }

        // read one input string to compute the distance against the query string
        const ap_uint<BITS> len2 = len2_strm.read();
        ap_uint<BITS> med = max_ed_strm.read();
        for (int i = 0; i < fold_num; i++) {
            ap_uint<64> str2_part = input_strm.read();
            for (int j = 0; j < 8; j++) {
#pragma HLS unroll
                if (i * 8 + j < M) str2[i * 8 + j] = str2_part(j * 8 + 7, j * 8);
            }
        }

    ED_CORE_LOOP:
        for (unsigned short i = 0; i < len1 + len2 - 1; i++) {
#pragma HLS pipeline
            internal::char_shift<N>(str2[M - 1 - i], str3);
            internal::compute_ed<N, BITS>(str3, str1, northwest, north, west);
            internal::left_shift<N, BITS>(northwest, west);
        }

        ap_uint<BITS> ed = north[N - len1]; // edit distance is at the last element
        o_match_strm.write(ed <= med);
        o_e_strm.write(false);

        last = i_e_strm.read();
    }

    o_e_strm.write(true);
}

} // namespace text
} // namespace data_analytics
} // namespace xf

#endif // XF_TEXT_EDIT_DISTANCE_H
