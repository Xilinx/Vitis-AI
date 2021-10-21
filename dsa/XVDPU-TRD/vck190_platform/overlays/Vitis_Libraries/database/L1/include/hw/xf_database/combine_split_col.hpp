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
 * @file combine_split_col.hpp
 * @brief combine/split unit template function implementation.
 *
 * This file is part of Vitis Database Library.
 */

#ifndef XF_DATABASE_COMBINE_SPLIT_COL_H
#define XF_DATABASE_COMBINE_SPLIT_COL_H

#ifndef __cplusplus
#error "Vitis Database Library only works with C++."
#endif

#include "hls_stream.h"

// for wide output
#include <ap_int.h>

// for uint64_t
#include "xf_database/types.hpp"

namespace xf {
namespace database {

/**
 * @brief Combines two columns into one.
 *
 * Columns are passed through streams of certain width in hardware. Normally, each column uses one stream, but for some
 * primitives, the processing semantic abstract the columns into a couple of groups, and trait each group as a whole.
 * To make calling such primitives easier, the combine column primitive fuses data of same row but different columns
 * into one wide column.
 *
 * The counter part of this primitive is ``splitCol``.
 *
 * @tparam _WCol1 the width of 1st input stream.
 * @tparam _WCol2 the width of 2nd input stream.
 * @tparam _WColOut the width of output stream.
 *
 * @param din1_strm 1st input data stream.
 * @param din2_strm 2nd input data stream.
 * @param in_e_strm end flag stream for input data.
 * @param dout_strm output data stream.
 * @param out_e_strm end flag stream for output data.
 */
template <int _WCol1, int _WCol2, int _WColOut>
void combineCol(hls::stream<ap_uint<_WCol1> >& din1_strm,
                hls::stream<ap_uint<_WCol2> >& din2_strm,
                hls::stream<bool>& in_e_strm,
                hls::stream<ap_uint<_WColOut> >& dout_strm,
                hls::stream<bool>& out_e_strm) {
    bool e = in_e_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<_WCol1> keyin1 = din1_strm.read();
        ap_uint<_WCol2> keyin2 = din2_strm.read();
        e = in_e_strm.read();
        ap_uint<_WColOut> dout_value = (keyin1, keyin2);
        dout_strm.write(dout_value);
        out_e_strm.write(0);
    }
    out_e_strm.write(1);
}
/**
 * @brief Combines three columns into one.
 *
 * Columns are passed through streams of certain width in hardware. Normally, each column uses one stream, but for some
 * primitives, the processing semantic abstract the columns into a couple of groups, and trait each group as a whole.
 * To make calling such primitives easier, the combine column primitive fuses data of same row but different columns
 * into one wide column.
 *
 * The counter part of this primitive is ``splitCol``.
 *
 * @tparam _WCol1 the width of 1st input stream.
 * @tparam _WCol2 the width of 2nd input stream.
 * @tparam _WCol3 the width of 3rd input stream.
 * @tparam _WColOut the width of output stream.
 *
 * @param din1_strm 1st input data stream.
 * @param din2_strm 2nd input data stream.
 * @param din3_strm 3rd input data stream.
 * @param in_e_strm end flag stream for input data.
 * @param dout_strm output data stream.
 * @param out_e_strm end flag stream for output data.
 */
template <int _WCol1, int _WCol2, int _WCol3, int _WColOut>
void combineCol(hls::stream<ap_uint<_WCol1> >& din1_strm,
                hls::stream<ap_uint<_WCol2> >& din2_strm,
                hls::stream<ap_uint<_WCol3> >& din3_strm,
                hls::stream<bool>& in_e_strm,
                hls::stream<ap_uint<_WColOut> >& dout_strm,
                hls::stream<bool>& out_e_strm) {
    bool e = in_e_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<_WCol1> keyin1 = din1_strm.read();
        ap_uint<_WCol2> keyin2 = din2_strm.read();
        ap_uint<_WCol3> keyin3 = din3_strm.read();
        e = in_e_strm.read();
        ap_uint<_WColOut> dout_value = (keyin1, keyin2, keyin3);
        dout_strm.write(dout_value);
        out_e_strm.write(0);
    }
    out_e_strm.write(1);
}
/**
 * @brief Combines four columns into one.
 *
 * Columns are passed through streams of certain width in hardware. Normally, each column uses one stream, but for some
 * primitives, the processing semantic abstract the columns into a couple of groups, and trait each group as a whole.
 * To make calling such primitives easier, the combine column primitive fuses data of same row but different columns
 * into one wide column.
 *
 * The counter part of this primitive is ``splitCol``.
 *
 * @tparam _WCol1 the width of 1st input stream.
 * @tparam _WCol2 the width of 2nd input stream.
 * @tparam _WCol3 the width of 3rd input stream.
 * @tparam _WCol4 the width of 4th input stream.
 * @tparam _WColOut the width of output stream.
 *
 * @param din1_strm 1st input data stream.
 * @param din2_strm 2nd input data stream.
 * @param din3_strm 3rd input data stream.
 * @param din4_strm 4th input data stream.
 * @param in_e_strm end flag stream for input data.
 * @param dout_strm output data stream.
 * @param out_e_strm end flag stream for output data.
 */
template <int _WCol1, int _WCol2, int _WCol3, int _WCol4, int _WColOut>
void combineCol(hls::stream<ap_uint<_WCol1> >& din1_strm,
                hls::stream<ap_uint<_WCol2> >& din2_strm,
                hls::stream<ap_uint<_WCol3> >& din3_strm,
                hls::stream<ap_uint<_WCol4> >& din4_strm,
                hls::stream<bool>& in_e_strm,
                hls::stream<ap_uint<_WColOut> >& dout_strm,
                hls::stream<bool>& out_e_strm) {
    bool e = in_e_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<_WCol1> keyin1 = din1_strm.read();
        ap_uint<_WCol2> keyin2 = din2_strm.read();
        ap_uint<_WCol3> keyin3 = din3_strm.read();
        ap_uint<_WCol4> keyin4 = din4_strm.read();
        e = in_e_strm.read();
        ap_uint<_WColOut> dout_value = (keyin1, keyin2, keyin3, keyin4);
        dout_strm.write(dout_value);
        out_e_strm.write(0);
    }
    out_e_strm.write(1);
}
/**
 * @brief Combines five columns into one.
 *
 * Columns are passed through streams of certain width in hardware. Normally, each column uses one stream, but for some
 * primitives, the processing semantic abstract the columns into a couple of groups, and trait each group as a whole.
 * To make calling such primitives easier, the combine column primitive fuses data of same row but different columns
 * into one wide column.
 *
 * The counter part of this primitive is ``splitCol``.
 *
 * @tparam _WCol1 the width of 1st input stream.
 * @tparam _WCol2 the width of 2nd input stream.
 * @tparam _WCol3 the width of 3rd input stream.
 * @tparam _WCol4 the width of 4th input stream.
 * @tparam _WCol5 the width of 5th input stream.
 * @tparam _WColOut the width of output stream.
 *
 * @param din1_strm 1st input data stream.
 * @param din2_strm 2nd input data stream.
 * @param din3_strm 3rd input data stream.
 * @param din4_strm 4th input data stream.
 * @param din5_strm 5th input data stream.
 * @param in_e_strm end flag stream for input data.
 * @param dout_strm output data stream.
 * @param out_e_strm end flag stream for output data.
 */
template <int _WCol1, int _WCol2, int _WCol3, int _WCol4, int _WCol5, int _WColOut>
void combineCol(hls::stream<ap_uint<_WCol1> >& din1_strm,
                hls::stream<ap_uint<_WCol2> >& din2_strm,
                hls::stream<ap_uint<_WCol3> >& din3_strm,
                hls::stream<ap_uint<_WCol4> >& din4_strm,
                hls::stream<ap_uint<_WCol5> >& din5_strm,
                hls::stream<bool>& in_e_strm,
                hls::stream<ap_uint<_WColOut> >& dout_strm,
                hls::stream<bool>& out_e_strm) {
    bool e = in_e_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<_WCol1> keyin1 = din1_strm.read();
        ap_uint<_WCol2> keyin2 = din2_strm.read();
        ap_uint<_WCol3> keyin3 = din3_strm.read();
        ap_uint<_WCol4> keyin4 = din4_strm.read();
        ap_uint<_WCol5> keyin5 = din5_strm.read();
        e = in_e_strm.read();
        ap_uint<_WColOut> dout_value = (keyin1, keyin2, keyin3, keyin4, keyin5);
        dout_strm.write(dout_value);
        out_e_strm.write(0);
    }
    out_e_strm.write(1);
}

/**
 * @brief Split previously combined columns into two.
 *
 * Columns are passed through streams of certain width in hardware. Normally, each column uses one stream, but for some
 * primitives, the processing semantic abstract the columns into a couple of groups, and trait each group as a whole.
 * To make calling such primitives easier, the split column primitive breaks the wide output stream into independent
 * column-specific streams.
 *
 * The counter part of this primitive is ``combineCol``.
 *
 * @tparam _WColIn the width of input stream.
 * @tparam _WCol1 the width of 1st output stream.
 * @tparam _WCol2 the width of 2nd output stream.
 *
 * @param din_strm input data stream.
 * @param in_e_strm end flag stream for input data.
 * @param dout1_strm 1st output data stream.
 * @param dout2_strm 2nd output data stream.
 * @param out_e_strm end flag stream for output data.
 */
template <int _WColIn, int _WCol1, int _WCol2>
void splitCol(hls::stream<ap_uint<_WColIn> >& din_strm,
              hls::stream<bool>& in_e_strm,
              hls::stream<ap_uint<_WCol1> >& dout1_strm,
              hls::stream<ap_uint<_WCol2> >& dout2_strm,
              hls::stream<bool>& out_e_strm) {
    bool e = in_e_strm.read();
    ap_uint<_WCol1> keyout1;
    ap_uint<_WCol2> keyout2;
    ap_uint<_WColIn> keyin;
    uint64_t width_t1 = keyout1.length();
    uint64_t width_t2 = keyout2.length();
    uint64_t width_in = keyin.length();
    while (!e) {
#pragma HLS pipeline II = 1
        keyin = din_strm.read();
        e = in_e_strm.read();
        keyout1 = keyin.range(width_t1 - 1, 0);
        keyout2 = keyin.range(width_in - 1, width_t1);
        dout1_strm.write(keyout1);
        dout2_strm.write(keyout2);
        out_e_strm.write(0);
    }
    out_e_strm.write(1);
}
/**
 * @brief Split previously combined columns into three.
 *
 * Columns are passed through streams of certain width in hardware. Normally, each column uses one stream, but for some
 * primitives, the processing semantic abstract the columns into a couple of groups, and trait each group as a whole.
 * To make calling such primitives easier, the split column primitive breaks the wide output stream into independent
 * column-specific streams.
 *
 * The counter part of this primitive is ``combineCol``.
 *
 * @tparam _WColIn the width of input stream.
 * @tparam _WCol1 the width of 1st output stream.
 * @tparam _WCol2 the width of 2nd output stream.
 * @tparam _WCol3 the width of 3rd output stream.
 *
 * @param din_strm input data stream
 * @param in_e_strm end flag stream for input data
 * @param dout1_strm 1st output data stream
 * @param dout2_strm 2nd output data stream
 * @param dout3_strm 3rd output data stream
 * @param out_e_strm end flag stream for output data
 */
template <int _WColIn, int _WCol1, int _WCol2, int _WCol3>
void splitCol(hls::stream<ap_uint<_WColIn> >& din_strm,
              hls::stream<bool>& in_e_strm,
              hls::stream<ap_uint<_WCol1> >& dout1_strm,
              hls::stream<ap_uint<_WCol2> >& dout2_strm,
              hls::stream<ap_uint<_WCol3> >& dout3_strm,
              hls::stream<bool>& out_e_strm) {
    bool e = in_e_strm.read();
    ap_uint<_WCol1> keyout1;
    ap_uint<_WCol2> keyout2;
    ap_uint<_WCol3> keyout3;
    ap_uint<_WColIn> keyin;
    uint64_t width_t1 = keyout1.length();
    uint64_t width_t2 = keyout2.length();
    uint64_t width_t3 = keyout3.length();
    uint64_t width_in = keyin.length();
    while (!e) {
#pragma HLS pipeline II = 1
        keyin = din_strm.read();
        e = in_e_strm.read();
        keyout1 = keyin.range(width_t1 - 1, 0);
        keyout2 = keyin.range(width_t1 + width_t2 - 1, width_t1);
        keyout3 = keyin.range(width_in - 1, width_t1 + width_t2);
        dout1_strm.write(keyout1);
        dout2_strm.write(keyout2);
        dout3_strm.write(keyout3);
        out_e_strm.write(0);
    }
    out_e_strm.write(1);
}
/**
 * @brief Split previously combined columns into four.
 *
 * Columns are passed through streams of certain width in hardware. Normally, each column uses one stream, but for some
 * primitives, the processing semantic abstract the columns into a couple of groups, and trait each group as a whole.
 * To make calling such primitives easier, the split column primitive breaks the wide output stream into independent
 * column-specific streams.
 *
 * The counter part of this primitive is ``combineCol``.
 *
 * @tparam _WColIn the width of input stream.
 * @tparam _WCol1 the width of 1st output stream.
 * @tparam _WCol2 the width of 2nd output stream.
 * @tparam _WCol3 the width of 3rd output stream.
 * @tparam _WCol4 the width of 4th output stream.
 *
 * @param din_strm input data stream
 * @param in_e_strm end flag stream for input data
 * @param dout1_strm 1st output data stream
 * @param dout2_strm 2nd output data stream
 * @param dout3_strm 3rd output data stream
 * @param dout4_strm 4th output data stream
 * @param out_e_strm end flag stream for output data
 */
template <int _WColIn, int _WCol1, int _WCol2, int _WCol3, int _WCol4>
void splitCol(hls::stream<ap_uint<_WColIn> >& din_strm,
              hls::stream<bool>& in_e_strm,
              hls::stream<ap_uint<_WCol1> >& dout1_strm,
              hls::stream<ap_uint<_WCol2> >& dout2_strm,
              hls::stream<ap_uint<_WCol3> >& dout3_strm,
              hls::stream<ap_uint<_WCol4> >& dout4_strm,
              hls::stream<bool>& out_e_strm) {
    bool e = in_e_strm.read();
    ap_uint<_WCol1> keyout1;
    ap_uint<_WCol2> keyout2;
    ap_uint<_WCol3> keyout3;
    ap_uint<_WCol4> keyout4;
    ap_uint<_WColIn> keyin;
    uint64_t width_t1 = keyout1.length();
    uint64_t width_t2 = keyout2.length();
    uint64_t width_t3 = keyout3.length();
    uint64_t width_t4 = keyout4.length();
    uint64_t width_in = keyin.length();
    while (!e) {
#pragma HLS pipeline II = 1
        keyin = din_strm.read();
        e = in_e_strm.read();
        keyout1 = keyin.range(width_t1 - 1, 0);
        keyout2 = keyin.range(width_t1 + width_t2 - 1, width_t1);
        keyout3 = keyin.range(width_t1 + width_t2 + width_t3 - 1, width_t1 + width_t2);
        keyout4 = keyin.range(width_in - 1, width_t1 + width_t2 + width_t3);
        dout1_strm.write(keyout1);
        dout2_strm.write(keyout2);
        dout3_strm.write(keyout3);
        dout4_strm.write(keyout4);
        out_e_strm.write(0);
    }
    out_e_strm.write(1);
}
/**
 * @brief Split previously combined columns into five.
 *
 * Columns are passed through streams of certain width in hardware. Normally, each column uses one stream, but for some
 * primitives, the processing semantic abstract the columns into a couple of groups, and trait each group as a whole.
 * To make calling such primitives easier, the split column primitive breaks the wide output stream into independent
 * column-specific streams.
 *
 * The counter part of this primitive is ``combineCol``.
 *
 * @tparam _WColIn the width of input stream.
 * @tparam _WCol1 the width of 1st output stream.
 * @tparam _WCol2 the width of 2nd output stream.
 * @tparam _WCol3 the width of 3rd output stream.
 * @tparam _WCol4 the width of 4th output stream.
 * @tparam _WCol5 the width of 5th output stream.
 *
 * @param din_strm input data stream
 * @param in_e_strm end flag stream for input data
 * @param dout1_strm 1st output data stream
 * @param dout2_strm 2nd output data stream
 * @param dout3_strm 3rd output data stream
 * @param dout4_strm 4th output data stream
 * @param dout5_strm 5th output data stream
 * @param out_e_strm end flag stream for output data
 */
template <int _WColIn, int _WCol1, int _WCol2, int _WCol3, int _WCol4, int _WCol5>
void splitCol(hls::stream<ap_uint<_WColIn> >& din_strm,
              hls::stream<bool>& in_e_strm,
              hls::stream<ap_uint<_WCol1> >& dout1_strm,
              hls::stream<ap_uint<_WCol2> >& dout2_strm,
              hls::stream<ap_uint<_WCol3> >& dout3_strm,
              hls::stream<ap_uint<_WCol4> >& dout4_strm,
              hls::stream<ap_uint<_WCol5> >& dout5_strm,
              hls::stream<bool>& out_e_strm) {
    bool e = in_e_strm.read();
    ap_uint<_WCol1> keyout1;
    ap_uint<_WCol2> keyout2;
    ap_uint<_WCol3> keyout3;
    ap_uint<_WCol4> keyout4;
    ap_uint<_WCol5> keyout5;
    ap_uint<_WColIn> keyin;
    uint64_t width_t1 = keyout1.length();
    uint64_t width_t2 = keyout2.length();
    uint64_t width_t3 = keyout3.length();
    uint64_t width_t4 = keyout4.length();
    uint64_t width_t5 = keyout5.length();
    uint64_t width_in = keyin.length();
    while (!e) {
#pragma HLS pipeline II = 1
        keyin = din_strm.read();
        e = in_e_strm.read();
        keyout1 = keyin.range(width_t1 - 1, 0);
        keyout2 = keyin.range(width_t1 + width_t2 - 1, width_t1);
        keyout3 = keyin.range(width_t1 + width_t2 + width_t3 - 1, width_t1 + width_t2);
        keyout4 = keyin.range(width_in - width_t5 - 1, width_t1 + width_t2 + width_t3);
        keyout5 = keyin.range(width_in - 1, width_in - width_t5);
        dout1_strm.write(keyout1);
        dout2_strm.write(keyout2);
        dout3_strm.write(keyout3);
        dout4_strm.write(keyout4);
        dout5_strm.write(keyout5);
        out_e_strm.write(0);
    }
    out_e_strm.write(1);
}

} // namespace database
} // namespace xf
#endif
