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
 * @file scan_col_2.hpp
 * @brief This file is part of Vitis Database Library, contains SCAN by column
 * functions.
 *
 * Unlike the functions in scan_col.h, these functions obtains the number of
 * rows to read from head of buffer instead of stand-alone arguments. This
 * enables the kernel to be scheduled without knowing the length of intermediate
 * results.
 */
#ifndef XF_DATABASE_SCAN_COL_2_H
#define XF_DATABASE_SCAN_COL_2_H

#if defined(AP_INT_MAX_W) && (AP_INT_MAX_W < 4096)
#error "database::scan requires define AP_INT_MAX_W to 4096 or larger"
#else
#define AP_INT_MAX_W 4096 // Must be defined before next line
#include <ap_int.h>
#endif

#include "xf_database/types.hpp"
#include "xf_database/utils.hpp"

#include <hls_stream.h>

namespace xf {
namespace database {
namespace details {

template <int burst_len, int vec_len, int size>
void read_1st_col_vec(ap_uint<8 * size * vec_len>* vec_ptr,
                      hls::stream<ap_uint<8 * size * vec_len> >& vec_strm,
                      hls::stream<int>& nrow_strm) {
    // the first vec only contains row number
    ap_uint<8 * size* vec_len> t = vec_ptr[0];
    int nrow = t.to_int();
    nrow_strm.write(nrow);
    // now read the vec in multiple bursts
    int nread = (nrow + vec_len - 1) / vec_len;
#ifndef __SYNTHESIS__
    printf("nrow = %d, burst_len = %d, nread = %d\n", nrow, burst_len, nread);
#endif
BURST_READS:
    for (int i = 0; i < nread; i += burst_len) {
        const int len = ((i + burst_len) > nread) ? (nread - i) : burst_len;
    READ_VECS_P:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            vec_strm.write(vec_ptr[1 + i + j]);
        }
    }
}

template <int burst_len, int vec_len, int size>
void read_col_vec(ap_uint<8 * size * vec_len>* vec_ptr,
                  hls::stream<ap_uint<8 * size * vec_len> >& vec_strm,
                  hls::stream<int>& nrow_i_strm,
                  hls::stream<int>& nrow_o_strm) {
    // following first vector's nrow
    int nrow = nrow_i_strm.read();
    nrow_o_strm.write(nrow);
    // decide whether to really load data from the buffer.
    bool empty_buf = vec_ptr[0].to_int() == 0;
    // now read the vec in multiple bursts
    int nread = (nrow + vec_len - 1) / vec_len;
#ifndef __SYNTHESIS__
    printf("nrow = %d, burst_len = %d, nread = %d\n", nrow, burst_len, nread);
#endif
    if (empty_buf) {
        for (int i = 0; i < nread; ++i) {
#pragma HLS pipeline II = 1
            vec_strm.write(0);
        }
    } else {
    BURST_READS_2:
        for (int i = 0; i < nread; i += burst_len) {
            const int len = ((i + burst_len) > nread) ? (nread - i) : burst_len;
        READ_VECS_2_P:
            for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
                ap_uint<8 * size* vec_len> t = vec_ptr[1 + i + j];
                vec_strm.write(t);
            }
        }
    }
}

} // namespace details
} // namespace database
} // namespace xf

// ---------------------- scan_col 2 cols ---------------------------------
namespace xf {
namespace database {
namespace details {

template <int vec_len, int ch_nm, int size0, int size1>
static void split_col_vec(                                  //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    hls::stream<int>& nrow_strm,                            //
    hls::stream<ap_uint<8 * size0> > c0_strm[ch_nm],        //
    hls::stream<ap_uint<8 * size1> > c1_strm[ch_nm],        //
    hls::stream<bool> e_strm[ch_nm]) {
    enum { split_col_vec_ii = vec_len / ch_nm };
    int nrow = nrow_strm.read();
SPLIT_COL_VEC:
    for (int i = 0; i < nrow; i += vec_len) {
#pragma HLS pipeline II = split_col_vec_ii
        ap_uint<8 * size0* vec_len> c0vec = c0vec_strm.read();
        ap_uint<8 * size1* vec_len> c1vec = c1vec_strm.read();
        int n = (i + vec_len) > nrow ? (nrow - i) : vec_len;
        XF_DATABASE_ASSERT((vec_len % ch_nm == 0) && (vec_len >= ch_nm));
        for (int j = 0; j < vec_len / ch_nm; ++j) {
            for (int k = 0; k < ch_nm; ++k) {
#pragma HLS unroll
                ap_uint<8 * size0> c0 = c0vec.range(8 * size0 * (j * ch_nm + k + 1) - 1, 8 * size0 * (j * ch_nm + k));
                ap_uint<8 * size1> c1 = c1vec.range(8 * size1 * (j * ch_nm + k + 1) - 1, 8 * size1 * (j * ch_nm + k));
                if ((j * ch_nm + k) < n) {
                    c0_strm[k].write(c0);
                    c1_strm[k].write(c1);
                    e_strm[k].write(false);
                }
            }
        }
    }
    for (int k = 0; k < ch_nm; ++k) {
#pragma HLS unroll
        e_strm[k].write(true);
    }
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief scan 2 columns from DDR/HBM buffers.
 *
 * The LSB of first vector of first column specifies the number of rows to be
 * scanned. For a following buffer, if the first vector is zero, same number of
 * zeros will be emitted, otherwise, same number of rows will be read from the
 * buffer.
 *
 * @tparam burst_len burst read length, must be supported by MC.
 * @tparam vec_len scan this number of items as a vector from AXI port.
 * @tparam ch_nm number of concurrent output channels per column.
 * @tparam size0 size of column 0, in byte.
 * @tparam size1 size of column 1, in byte.
 *
 * @param c0vec_ptr buffer pointer to column 0.
 * @param c1vec_ptr buffer pointer to column 1.
 * @param c0_strm array of column 0 stream.
 * @param c1_strm array of column 1 stream.
 * @param e_row_strm array of output end flag stream.
 */
template <int burst_len, int vec_len, int ch_nm, int size0, int size1>
static void scanCol(                                 //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,         //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,         //
    hls::stream<ap_uint<8 * size0> > c0_strm[ch_nm], //
    hls::stream<ap_uint<8 * size1> > c1_strm[ch_nm], //
    hls::stream<bool> e_row_strm[ch_nm]) {
#pragma HLS DATAFLOW
    enum { fifo_depth = burst_len * 2 };

    hls::stream<ap_uint<8 * size0 * vec_len> > c0vec_strm("c0vec_strm");
#pragma HLS STREAM variable = c0vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size1 * vec_len> > c1vec_strm("c1vec_strm");
#pragma HLS STREAM variable = c1vec_strm depth = fifo_depth

    static hls::stream<int> nrow_strm0("nrow_strm0");
#pragma HLS STREAM variable = nrow_strm0 depth = 2

    static hls::stream<int> nrow_strm1("nrow_strm1");
#pragma HLS STREAM variable = nrow_strm1 depth = 2

    details::read_1st_col_vec<burst_len, vec_len, size0>(c0vec_ptr, c0vec_strm, nrow_strm0);

    details::read_col_vec<burst_len, vec_len, size1>(c1vec_ptr, c1vec_strm, nrow_strm0, nrow_strm1);

    details::split_col_vec<vec_len, ch_nm, size0, size1>(c0vec_strm, c1vec_strm, nrow_strm1, c0_strm, c1_strm,
                                                         e_row_strm);
}

} // namespace database
} // namespace xf

// ---------------------- scan_col 3 cols ---------------------------------
namespace xf {
namespace database {
namespace details {

template <int vec_len, int ch_nm, int size0, int size1, int size2>
static void split_col_vec(                                  //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    hls::stream<ap_uint<8 * size2 * vec_len> >& c2vec_strm, //
    hls::stream<int>& nrow_strm,                            //
    hls::stream<ap_uint<8 * size0> > c0_strm[ch_nm],        //
    hls::stream<ap_uint<8 * size1> > c1_strm[ch_nm],        //
    hls::stream<ap_uint<8 * size2> > c2_strm[ch_nm],        //
    hls::stream<bool> e_strm[ch_nm]) {
    enum { split_col_vec_ii = vec_len / ch_nm };
    int nrow = nrow_strm.read();
SPLIT_COL_VEC_II_NOT_1:
    for (int i = 0; i < nrow; i += vec_len) {
#pragma HLS pipeline II = split_col_vec_ii
        ap_uint<8 * size0* vec_len> c0vec = c0vec_strm.read();
        ap_uint<8 * size1* vec_len> c1vec = c1vec_strm.read();
        ap_uint<8 * size2* vec_len> c2vec = c2vec_strm.read();
        int n = (i + vec_len) > nrow ? (nrow - i) : vec_len;
        XF_DATABASE_ASSERT((vec_len >= ch_nm) && (vec_len % ch_nm == 0));
        for (int j = 0; j < vec_len / ch_nm; ++j) {
            for (int k = 0; k < ch_nm; ++k) {
#pragma HLS unroll
                ap_uint<8 * size0> c0 = c0vec.range(8 * size0 * (j * ch_nm + k + 1) - 1, 8 * size0 * (j * ch_nm + k));
                ap_uint<8 * size1> c1 = c1vec.range(8 * size1 * (j * ch_nm + k + 1) - 1, 8 * size1 * (j * ch_nm + k));
                ap_uint<8 * size2> c2 = c2vec.range(8 * size2 * (j * ch_nm + k + 1) - 1, 8 * size2 * (j * ch_nm + k));
                if ((j * ch_nm + k) < n) {
                    c0_strm[k].write(c0);
                    c1_strm[k].write(c1);
                    c2_strm[k].write(c2);
                    e_strm[k].write(false);
                }
            }
        }
    }
    for (int k = 0; k < ch_nm; ++k) {
#pragma HLS UNROLL
        e_strm[k].write(true);
    }
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief scan 3 columns from DDR/HBM buffers.
 *
 * The LSB of first vector of first column specifies the number of rows to be
 * scanned. For a following buffer, if the first vector is zero, same number of
 * zeros will be emitted, otherwise, same number of rows will be read from the
 * buffer.
 *
 * @tparam burst_len burst read length, must be supported by MC.
 * @tparam vec_len scan this number of items as a vector from AXI port.
 * @tparam ch_nm number of concurrent output channels per column.
 * @tparam size0 size of column 0, in byte.
 * @tparam size1 size of column 1, in byte.
 * @tparam size2 size of column 2, in byte.
 *
 * @param c0vec_ptr buffer pointer to column 0.
 * @param c1vec_ptr buffer pointer to column 1.
 * @param c2vec_ptr buffer pointer to column 2.
 * @param c0_strm array of column 0 stream.
 * @param c1_strm array of column 1 stream.
 * @param c2_strm array of column 2 stream.
 * @param e_row_strm array of output end flag stream.
 */
template <int burst_len,
          int vec_len,
          int ch_nm,
          int size0,
          int size1,
          int size2>
static void scanCol(                                 //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,         //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,         //
    ap_uint<8 * size2 * vec_len>* c2vec_ptr,         //
    hls::stream<ap_uint<8 * size0> > c0_strm[ch_nm], //
    hls::stream<ap_uint<8 * size1> > c1_strm[ch_nm], //
    hls::stream<ap_uint<8 * size2> > c2_strm[ch_nm], //
    hls::stream<bool> e_row_strm[ch_nm]) {
#pragma HLS DATAFLOW
    enum { fifo_depth = burst_len * 2 };

    hls::stream<ap_uint<8 * size0 * vec_len> > c0vec_strm("c0vec_strm");
#pragma HLS STREAM variable = c0vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size1 * vec_len> > c1vec_strm("c1vec_strm");
#pragma HLS STREAM variable = c1vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size2 * vec_len> > c2vec_strm("c2vec_strm");
#pragma HLS STREAM variable = c2vec_strm depth = fifo_depth

    hls::stream<int> nrow_strm0("nrow_strm0");
#pragma HLS STREAM variable = nrow_strm0 depth = 2

    hls::stream<int> nrow_strm1("nrow_strm1");
#pragma HLS STREAM variable = nrow_strm1 depth = 2

    hls::stream<int> nrow_strm2("nrow_strm2");
#pragma HLS STREAM variable = nrow_strm2 depth = 2

    details::read_1st_col_vec<burst_len, vec_len, size0>(c0vec_ptr, c0vec_strm, nrow_strm0);

    details::read_col_vec<burst_len, vec_len, size1>(c1vec_ptr, c1vec_strm, nrow_strm0, nrow_strm1);

    details::read_col_vec<burst_len, vec_len, size2>(c2vec_ptr, c2vec_strm, nrow_strm1, nrow_strm2);

    details::split_col_vec<vec_len, ch_nm, size0, size1, size2>( //
        c0vec_strm, c1vec_strm, c2vec_strm, nrow_strm2,          //
        c0_strm, c1_strm, c2_strm, e_row_strm);
}

} // namespace database
} // namespace xf

// ---------------------- scan_col 4 cols ---------------------------------
namespace xf {
namespace database {
namespace details {

template <int vec_len, int ch_nm, int size0, int size1, int size2, int size3>
static void split_col_vec(                                  //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    hls::stream<ap_uint<8 * size2 * vec_len> >& c2vec_strm, //
    hls::stream<ap_uint<8 * size3 * vec_len> >& c3vec_strm, //
    hls::stream<int>& nrow_strm,                            //
    hls::stream<ap_uint<8 * size0> > c0_strm[ch_nm],        //
    hls::stream<ap_uint<8 * size1> > c1_strm[ch_nm],        //
    hls::stream<ap_uint<8 * size2> > c2_strm[ch_nm],        //
    hls::stream<ap_uint<8 * size3> > c3_strm[ch_nm],        //
    hls::stream<bool> e_strm[ch_nm]) {
    enum { split_col_vec_ii = vec_len / ch_nm };
    int nrow = nrow_strm.read();
SPLIT_COL_VEC_II_NOT_1:
    for (int i = 0; i < nrow; i += vec_len) {
#pragma HLS pipeline II = split_col_vec_ii
        ap_uint<8 * size0* vec_len> c0vec = c0vec_strm.read();
        ap_uint<8 * size1* vec_len> c1vec = c1vec_strm.read();
        ap_uint<8 * size2* vec_len> c2vec = c2vec_strm.read();
        ap_uint<8 * size3* vec_len> c3vec = c3vec_strm.read();
        int n = (i + vec_len) > nrow ? (nrow - i) : vec_len;
        XF_DATABASE_ASSERT((vec_len >= ch_nm) && (vec_len % ch_nm == 0));
        for (int j = 0; j < vec_len / ch_nm; ++j) {
            for (int k = 0; k < ch_nm; ++k) {
#pragma HLS unroll
                ap_uint<8 * size0> c0 = c0vec.range(8 * size0 * (j * ch_nm + k + 1) - 1, 8 * size0 * (j * ch_nm + k));
                ap_uint<8 * size1> c1 = c1vec.range(8 * size1 * (j * ch_nm + k + 1) - 1, 8 * size1 * (j * ch_nm + k));
                ap_uint<8 * size2> c2 = c2vec.range(8 * size2 * (j * ch_nm + k + 1) - 1, 8 * size2 * (j * ch_nm + k));
                ap_uint<8 * size3> c3 = c3vec.range(8 * size3 * (j * ch_nm + k + 1) - 1, 8 * size3 * (j * ch_nm + k));
                if ((j * ch_nm + k) < n) {
                    c0_strm[k].write(c0);
                    c1_strm[k].write(c1);
                    c2_strm[k].write(c2);
                    c3_strm[k].write(c3);
                    e_strm[k].write(false);
                }
            }
        }
    }
    for (int k = 0; k < ch_nm; ++k) {
#pragma HLS UNROLL
        e_strm[k].write(true);
    }
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief scan 4 columns from DDR/HBM buffers.
 *
 * The LSB of first vector of first column specifies the number of rows to be
 * scanned. For a following buffer, if the first vector is zero, same number of
 * zeros will be emitted, otherwise, same number of rows will be read from the
 * buffer.
 *
 * @tparam burst_len burst read length, must be supported by MC.
 * @tparam vec_len scan this number of items as a vector from AXI port.
 * @tparam ch_nm number of concurrent output channels per column.
 * @tparam size0 size of column 0, in byte.
 * @tparam size1 size of column 1, in byte.
 * @tparam size2 size of column 2, in byte.
 * @tparam size3 size of column 3, in byte.
 *
 * @param c0vec_ptr buffer pointer to column 0.
 * @param c1vec_ptr buffer pointer to column 1.
 * @param c2vec_ptr buffer pointer to column 2.
 * @param c3vec_ptr buffer pointer to column 3.
 * @param c0_strm array of column 0 stream.
 * @param c1_strm array of column 1 stream.
 * @param c2_strm array of column 2 stream.
 * @param c3_strm array of column 3 stream.
 * @param e_row_strm array of output end flag stream.
 */
template <int burst_len, int vec_len, int ch_nm, int size0, int size1, int size2, int size3>
static void scanCol(                                 //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,         //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,         //
    ap_uint<8 * size2 * vec_len>* c2vec_ptr,         //
    ap_uint<8 * size3 * vec_len>* c3vec_ptr,         //
    hls::stream<ap_uint<8 * size0> > c0_strm[ch_nm], //
    hls::stream<ap_uint<8 * size1> > c1_strm[ch_nm], //
    hls::stream<ap_uint<8 * size2> > c2_strm[ch_nm], //
    hls::stream<ap_uint<8 * size3> > c3_strm[ch_nm], //
    hls::stream<bool> e_row_strm[ch_nm]) {
#pragma HLS DATAFLOW
    enum { fifo_depth = burst_len * 2 };

    hls::stream<ap_uint<8 * size0 * vec_len> > c0vec_strm("c0vec_strm");
#pragma HLS STREAM variable = c0vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size1 * vec_len> > c1vec_strm("c1vec_strm");
#pragma HLS STREAM variable = c1vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size2 * vec_len> > c2vec_strm("c2vec_strm");
#pragma HLS STREAM variable = c2vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size3 * vec_len> > c3vec_strm("c3vec_strm");
#pragma HLS STREAM variable = c3vec_strm depth = fifo_depth

    hls::stream<int> nrow_strm0("nrow_strm0");
#pragma HLS STREAM variable = nrow_strm0 depth = 2

    hls::stream<int> nrow_strm1("nrow_strm1");
#pragma HLS STREAM variable = nrow_strm1 depth = 2

    hls::stream<int> nrow_strm2("nrow_strm2");
#pragma HLS STREAM variable = nrow_strm2 depth = 2

    hls::stream<int> nrow_strm3("nrow_strm3");
#pragma HLS STREAM variable = nrow_strm3 depth = 2

    details::read_1st_col_vec<burst_len, vec_len, size0>(c0vec_ptr, c0vec_strm, nrow_strm0);

    details::read_col_vec<burst_len, vec_len, size1>(c1vec_ptr, c1vec_strm, nrow_strm0, nrow_strm1);

    details::read_col_vec<burst_len, vec_len, size2>(c2vec_ptr, c2vec_strm, nrow_strm1, nrow_strm2);

    details::read_col_vec<burst_len, vec_len, size3>(c3vec_ptr, c3vec_strm, nrow_strm2, nrow_strm3);

    details::split_col_vec<vec_len, ch_nm, size0, size1, size2,
                           size3>(                                  //
        c0vec_strm, c1vec_strm, c2vec_strm, c3vec_strm, nrow_strm3, //
        c0_strm, c1_strm, c2_strm, c3_strm, e_row_strm);
}

} // namespace database
} // namespace xf
  // ---------------------- scan_col 5 cols ---------------------------------

namespace xf {
namespace database {
namespace details {

template <int vec_len,
          int ch_nm,
          int size0,
          int size1,
          int size2,
          int size3,
          int size4>
static void split_col_vec(                                  //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    hls::stream<ap_uint<8 * size2 * vec_len> >& c2vec_strm, //
    hls::stream<ap_uint<8 * size3 * vec_len> >& c3vec_strm, //
    hls::stream<ap_uint<8 * size4 * vec_len> >& c4vec_strm, //
    hls::stream<int>& nrow_strm,                            //
    hls::stream<ap_uint<8 * size0> > c0_strm[ch_nm],        //
    hls::stream<ap_uint<8 * size1> > c1_strm[ch_nm],        //
    hls::stream<ap_uint<8 * size2> > c2_strm[ch_nm],        //
    hls::stream<ap_uint<8 * size3> > c3_strm[ch_nm],        //
    hls::stream<ap_uint<8 * size4> > c4_strm[ch_nm],        //
    hls::stream<bool> e_strm[ch_nm]) {
    enum { split_col_vec_ii = vec_len / ch_nm };
    int nrow = nrow_strm.read();
SPLIT_COL_VEC_II_NOT_1:
    for (int i = 0; i < nrow; i += vec_len) {
#pragma HLS pipeline II = split_col_vec_ii
        ap_uint<8 * size0* vec_len> c0vec = c0vec_strm.read();
        ap_uint<8 * size1* vec_len> c1vec = c1vec_strm.read();
        ap_uint<8 * size2* vec_len> c2vec = c2vec_strm.read();
        ap_uint<8 * size3* vec_len> c3vec = c3vec_strm.read();
        ap_uint<8 * size4* vec_len> c4vec = c4vec_strm.read();
        int n = (i + vec_len) > nrow ? (nrow - i) : vec_len;
        XF_DATABASE_ASSERT((vec_len >= ch_nm) && (vec_len % ch_nm == 0));
        for (int j = 0; j < vec_len / ch_nm; ++j) {
            for (int k = 0; k < ch_nm; ++k) {
#pragma HLS unroll
                ap_uint<8 * size0> c0 = c0vec.range(8 * size0 * (j * ch_nm + k + 1) - 1, 8 * size0 * (j * ch_nm + k));
                ap_uint<8 * size1> c1 = c1vec.range(8 * size1 * (j * ch_nm + k + 1) - 1, 8 * size1 * (j * ch_nm + k));
                ap_uint<8 * size2> c2 = c2vec.range(8 * size2 * (j * ch_nm + k + 1) - 1, 8 * size2 * (j * ch_nm + k));
                ap_uint<8 * size3> c3 = c3vec.range(8 * size3 * (j * ch_nm + k + 1) - 1, 8 * size3 * (j * ch_nm + k));
                ap_uint<8 * size4> c4 = c4vec.range(8 * size4 * (j * ch_nm + k + 1) - 1, 8 * size4 * (j * ch_nm + k));
                if ((j * ch_nm + k) < n) {
                    c0_strm[k].write(c0);
                    c1_strm[k].write(c1);
                    c2_strm[k].write(c2);
                    c3_strm[k].write(c3);
                    c4_strm[k].write(c4);
                    e_strm[k].write(false);
                }
            }
        }
    }
    for (int k = 0; k < ch_nm; ++k) {
#pragma HLS UNROLL
        e_strm[k].write(true);
    }
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief scan 5 columns from DDR/HBM buffers.
 *
 * The LSB of first vector of first column specifies the number of rows to be
 * scanned. For a following buffer, if the first vector is zero, same number of
 * zeros will be emitted, otherwise, same number of rows will be read from the
 * buffer.
 *
 * @tparam burst_len burst read length, must be supported by MC.
 * @tparam vec_len scan this number of items as a vector from AXI port.
 * @tparam ch_nm number of concurrent output channels per column.
 * @tparam size0 size of column 0, in byte.
 * @tparam size1 size of column 1, in byte.
 * @tparam size2 size of column 2, in byte.
 * @tparam size3 size of column 3, in byte.
 * @tparam size4 size of column 4, in byte.
 *
 * @param c0vec_ptr buffer pointer to column 0.
 * @param c1vec_ptr buffer pointer to column 1.
 * @param c2vec_ptr buffer pointer to column 2.
 * @param c3vec_ptr buffer pointer to column 3.
 * @param c4vec_ptr buffer pointer to column 4.
 * @param c0_strm array of column 0 stream.
 * @param c1_strm array of column 1 stream.
 * @param c2_strm array of column 2 stream.
 * @param c3_strm array of column 3 stream.
 * @param c4_strm array of column 4 stream.
 * @param e_row_strm array of output end flag stream.
 */
template <int burst_len, int vec_len, int ch_nm, int size0, int size1, int size2, int size3, int size4>
static void scanCol(                                 //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,         //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,         //
    ap_uint<8 * size2 * vec_len>* c2vec_ptr,         //
    ap_uint<8 * size3 * vec_len>* c3vec_ptr,         //
    ap_uint<8 * size4 * vec_len>* c4vec_ptr,         //
    hls::stream<ap_uint<8 * size0> > c0_strm[ch_nm], //
    hls::stream<ap_uint<8 * size1> > c1_strm[ch_nm], //
    hls::stream<ap_uint<8 * size2> > c2_strm[ch_nm], //
    hls::stream<ap_uint<8 * size3> > c3_strm[ch_nm], //
    hls::stream<ap_uint<8 * size4> > c4_strm[ch_nm], //
    hls::stream<bool> e_row_strm[ch_nm]) {
#pragma HLS DATAFLOW
    enum { fifo_depth = burst_len * 2 };

    hls::stream<ap_uint<8 * size0 * vec_len> > c0vec_strm("c0vec_strm");
#pragma HLS STREAM variable = c0vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size1 * vec_len> > c1vec_strm("c1vec_strm");
#pragma HLS STREAM variable = c1vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size2 * vec_len> > c2vec_strm("c2vec_strm");
#pragma HLS STREAM variable = c2vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size3 * vec_len> > c3vec_strm("c3vec_strm");
#pragma HLS STREAM variable = c3vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size4 * vec_len> > c4vec_strm("c4vec_strm");
#pragma HLS STREAM variable = c4vec_strm depth = fifo_depth

    hls::stream<int> nrow_strm0("nrow_strm0");
#pragma HLS STREAM variable = nrow_strm0 depth = 2

    hls::stream<int> nrow_strm1("nrow_strm1");
#pragma HLS STREAM variable = nrow_strm1 depth = 2

    hls::stream<int> nrow_strm2("nrow_strm2");
#pragma HLS STREAM variable = nrow_strm2 depth = 2

    hls::stream<int> nrow_strm3("nrow_strm3");
#pragma HLS STREAM variable = nrow_strm3 depth = 2

    hls::stream<int> nrow_strm4("nrow_strm4");
#pragma HLS STREAM variable = nrow_strm4 depth = 2

    details::read_1st_col_vec<burst_len, vec_len, size0>(c0vec_ptr, c0vec_strm, nrow_strm0);

    details::read_col_vec<burst_len, vec_len, size1>(c1vec_ptr, c1vec_strm, nrow_strm0, nrow_strm1);

    details::read_col_vec<burst_len, vec_len, size2>(c2vec_ptr, c2vec_strm, nrow_strm1, nrow_strm2);

    details::read_col_vec<burst_len, vec_len, size3>(c3vec_ptr, c3vec_strm, nrow_strm2, nrow_strm3);

    details::read_col_vec<burst_len, vec_len, size4>(c4vec_ptr, c4vec_strm, nrow_strm3, nrow_strm4);

    details::split_col_vec<vec_len, ch_nm, size0, size1, size2, size3,
                           size4>(                                              //
        c0vec_strm, c1vec_strm, c2vec_strm, c3vec_strm, c4vec_strm, nrow_strm4, //
        c0_strm, c1_strm, c2_strm, c3_strm, c4_strm, e_row_strm);
}

} // namespace database
} // namespace xf
  // -----------------------------------------------------------------------

#endif // !defined(XF_DATABASE_SCAN_COL_2_H)
