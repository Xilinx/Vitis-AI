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
 * @file scan_cmp_str_col.hpp
 * @brief This file is part of Vitis Database Library, contains implentments on
 * scan and filter string column.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_SCAN_CMP_STR_COL_H
#define XF_DATABASE_SCAN_CMP_STR_COL_H

#ifndef __cplusplus
#error "Database Library only works with C++."
#endif

#include "ap_int.h"
#include "hls_stream.h"

#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace database {
namespace details {

/**
 * @brief string compare with condition, support mask input or inverse output
 *
 *
 *
 */
void str_equal_cond(hls::stream<ap_uint<512> >& str_stream,
                    hls::stream<bool>& e_str_i,
                    hls::stream<ap_uint<512> >& cnst_stream,
                    hls::stream<ap_uint<64> >& mask_stream,
                    hls::stream<bool>& inv_ctl_stream,
                    hls::stream<bool>& out_stream,
                    hls::stream<bool>& e_str_o) {
    // read a constant string for multiple column of input string stream
    const ap_uint<512> str_a = cnst_stream.read();
#ifndef __SYNTHESIS__
    std::cout << "STR_EQUAL_COND-->Input Constant String:" << std::endl;
    for (int i = 62; i >= 0; i--) {
        unsigned char c = str_a(8 * i + 7, 8 * i);
        if ((c >= 32) && (c <= 126)) std::cout << c;
    }
    std::cout << std::endl;
#endif
    const ap_uint<64> mask = mask_stream.read();
    const bool is_inv = inv_ctl_stream.read();
    bool is_end = e_str_i.read();
// start read and compare process
input_stream_loop:
    while (!is_end) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
#pragma HLS PIPELINE II = 1
        bool tmp_result = true;
        is_end = e_str_i.read();
        ap_uint<512> str_b = str_stream.read();
        // compare string length firstly
        if ((str_a.range(511, 504) == str_b.range(511, 504)) || mask[63]) {
        // then compare char sequence
        char_compare_loop:
            for (int i = 503; i > 0; i = i - 8) {
#pragma HLS unroll
                if (!mask[(i - 7) / 8] && (str_a.range(i, i - 7) != str_b.range(i, i - 7))) tmp_result &= false;
            }
        } else {
            tmp_result = false;
        }

        // write out result
        bool final_result = is_inv ? !tmp_result : tmp_result;
        out_stream << final_result;
        e_str_o << false;
    }

    // End of transfer
    e_str_o << true;
}

/**
 * @brief      compare a string stream with a constant string in padding format
 *
 * @param      str_stream   input string stream for comparasion, 512 bits in
 *                          heading-length and padding-zero format.
 * @param      e_str_i      end flag stream for input data.
 * @param      cnst_stream  input constant string stream, 512 bits in
 *                          heading-length and padding-zero format.
 * @param      out_stream   output whether each string is equal to the constant
 *                          string, true indicates they are equal.
 * @param      e_str_o      end flag stream for output data.
 */
void str_equal(hls::stream<ap_uint<512> >& str_stream,
               hls::stream<bool>& e_str_i,
               hls::stream<ap_uint<512> >& cnst_stream,
               hls::stream<bool>& out_stream,
               hls::stream<bool>& e_str_o) {
    // read a constant string for multiple column of input string stream
    ap_uint<512> str_a = cnst_stream.read();
#ifndef __SYNTHESIS__
    std::cout << "STR_EQUAL-->Input Constant String:" << std::endl;
    for (int i = 62; i >= 0; i--) {
        unsigned char c = str_a(8 * i + 7, 8 * i);
        if ((c >= 32) && (c <= 126)) std::cout << c;
    }
    std::cout << std::endl;
#endif
    bool is_end = e_str_i.read();
// start read and compare process
input_stream_loop:
    while (!is_end) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
#pragma HLS PIPELINE II = 1
        bool tmp_result = true;
        is_end = e_str_i.read();
        ap_uint<512> str_b = str_stream.read();
        // compare string length firstly
        if (str_a.range(511, 504) == str_b.range(511, 504)) {
        // then compare char sequence
        char_compare_loop:
            for (int i = 503; i > 0; i = i - 8) {
#pragma HLS unroll
                if (str_a.range(i, i - 7) != str_b.range(i, i - 7)) tmp_result &= false;
            }
        } else {
            tmp_result = false;
        }

        // write out result
        out_stream << tmp_result;
        e_str_o << false;
    }

    // End of transfer
    e_str_o << true;
}

/**
 * @brief Read multiple columns from global memory and transform into stream
 *
 * @param ddr_input  input string array stored in global memory.
 * @param size       the number of reading global memory
 * @param stream_o   output string stream in semi-pact format.
 * @param e_str_o    output end flag
 */
void read_ddr(ap_uint<512>* ddr_input,
              hls::stream<int>& size,
              hls::stream<ap_uint<512> >& stream_o,
              hls::stream<bool>& e_str_o) {
ddr_read:
    int t = size.read();
    for (int i = 0; i < t; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
        stream_o << ddr_input[i];
        e_str_o << false;
    }
    e_str_o << true;
}

/**
 * @brief      transform string stream from semi-pact into padding-zero format
 *
 * @param      pact_str_stream  input string stream in seim-pact format.
 * @param      e_str_i          end flag stream for input data.
 * @param      num_str          the number of actual strings.
 * @param      out_str_stream   output string stream in padding-zero format.
 * @param      e_str_o          end flag stream for output data.
 */
void padding_stream_out(hls::stream<ap_uint<512> >& pact_str_stream,
                        hls::stream<bool>& e_str_i,
                        hls::stream<int>& num_str,
                        hls::stream<ap_uint<512> >& out_str_stream,
                        hls::stream<bool>& e_str_o) {
    ap_uint<512> upper_part;
    ap_uint<512> up_copy;

    ap_uint<3> len[8];
#pragma HLS ARRAY_PARTITION variable = len complete dim = 1
    ap_uint<3> loc = 0;
    ap_uint<512> padding_str;

    int counter = 0;
    bool is_cross = true;

    int row_num = num_str.read();
    bool is_end = e_str_i.read();
    if (!is_end) pact_str_stream >> upper_part; // start up
    ap_uint<512> lower_part = upper_part;

main_loop:
    while (counter < row_num) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1

        if (is_cross) {
            upper_part = lower_part;
            e_str_i >> is_end;
            if (!is_end) pact_str_stream >> lower_part;
        }

        for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
            len[i] = upper_part(509 - 64 * i, 507 - 64 * i); // read length of all string
        }

        switch (loc) { // left shift 1~7
            case 0:
                up_copy = upper_part;
                break;
            case 1:
                up_copy.range(511, 64) = upper_part.range(448, 0);
                up_copy.range(63, 0) = lower_part.range(511, 448);
                break;
            case 2:
                up_copy.range(511, 128) = upper_part.range(384, 0);
                up_copy.range(127, 0) = lower_part.range(511, 384);
                break;
            case 3:
                up_copy.range(511, 192) = upper_part.range(320, 0);
                up_copy.range(191, 0) = lower_part.range(511, 320);
                break;
            case 4:
                up_copy.range(511, 256) = upper_part.range(256, 0);
                up_copy.range(255, 0) = lower_part.range(511, 256);
                break;
            case 5:
                up_copy.range(511, 320) = upper_part.range(192, 0);
                up_copy.range(319, 0) = lower_part.range(511, 192);
                break;
            case 6:
                up_copy.range(511, 384) = upper_part.range(128, 0);
                up_copy.range(383, 0) = lower_part.range(511, 128);
                break;
            case 7:
                up_copy.range(511, 448) = upper_part.range(64, 0);
                up_copy.range(447, 0) = lower_part.range(511, 64);
                break;
        }

        padding_str.range(511, 448) = up_copy.range(511, 448); // pick up string
        if (len[loc] >= 1)
            padding_str.range(447, 384) = up_copy.range(447, 384);
        else
            padding_str(447, 384) = 0;
        if (len[loc] >= 2)
            padding_str.range(383, 320) = up_copy.range(383, 320);
        else
            padding_str(383, 320) = 0;
        if (len[loc] >= 3)
            padding_str.range(319, 256) = up_copy.range(319, 256);
        else
            padding_str(319, 256) = 0;
        if (len[loc] >= 4)
            padding_str.range(255, 192) = up_copy.range(255, 192);
        else
            padding_str(255, 192) = 0;
        if (len[loc] >= 5)
            padding_str.range(191, 128) = up_copy.range(191, 128);
        else
            padding_str(191, 128) = 0;
        if (len[loc] >= 6)
            padding_str.range(127, 64) = up_copy.range(127, 64);
        else
            padding_str(127, 64) = 0;
        if (len[loc] == 7)
            padding_str.range(63, 0) = up_copy.range(63, 0);
        else
            padding_str(63, 0) = 0;

        out_str_stream << padding_str; // write out padding string
        e_str_o << false;
        counter++;

        is_cross = (loc + len[loc] >= 7);
        loc += (len[loc] + 1);

#ifndef __SYNTHESIS__
// std::cout << "len: " << std::dec << len[loc] << ", ";
// std::cout << "loc: " << std::dec << loc << ", ";
// std::cout << "cnt: " << std::dec << counter << ", ";
// std::cout << "is_cross: " << std::dec << is_cross << std::endl;
#endif

    } // end read stream loop

    // empty e_str_i stream
    if (!is_end) e_str_i >> is_end;
    // if (!e_str_i.empty()) e_str_i >> is_end; // Maybe hang in cosim

    //  end of transfer
    e_str_o << true;
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief      sacn multiple columns of string in global memory, and compare each
 *             of them with constant string
 *
 * @param      ddr_ptr      input string array stored in global memory.
 * @param  size         the number of times reading global memory
 * @param  num_str      the number of actual strings
 * @param      cnst_stream  input constant string stream, 512 bits in
 *                          heading-length and padding-zero format,
 *                          read only once as configuration.
 * @param      out_stream   output whether each string is equal to the constant
 *                          string, true indicates they are equal.
 * @param      e_str_o      end flag stream for output stream.
 */
void scanCmpStrCol(ap_uint<512>* ddr_ptr,
                   hls::stream<int>& size,
                   hls::stream<int>& num_str,
                   hls::stream<ap_uint<512> >& cnst_stream,
                   hls::stream<bool>& out_stream,
                   hls::stream<bool>& e_str_o) {
//#pragma HLS INTERFACE s_axilite port=ddr_ptr
//#pragma HLS INTERFACE m_axi depth=8 port=ddr_ptr

#pragma HLS DATAFLOW
    hls::stream<ap_uint<512> > stream_t1, stream_t2;
#pragma HLS STREAM variable = stream_t1 depth = 8
#pragma HLS STREAM variable = stream_t2 depth = 8

    hls::stream<bool> stream_f1, stream_f2;
#pragma HLS STREAM variable = stream_f1 depth = 8
#pragma HLS STREAM variable = stream_f2 depth = 8

    details::read_ddr(ddr_ptr, size, stream_t1, stream_f1);
    details::padding_stream_out(stream_t1, stream_f1, num_str, stream_t2, stream_f2);
    details::str_equal(stream_t2, stream_f2, cnst_stream, out_stream, e_str_o);
}

} // namespace database
} // namespace xf

#endif // ifndef XF_DATABASE_SCAN_CMP_STR_COL_H
