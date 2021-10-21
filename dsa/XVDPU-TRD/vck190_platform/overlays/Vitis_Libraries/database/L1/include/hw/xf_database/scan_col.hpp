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
 * @file scan_col.hpp
 * @brief This file is part of Vitis Database Library, contains SCAN by column
 * functions.
 */
#ifndef XF_DATABASE_SCAN_COL_H
#define XF_DATABASE_SCAN_COL_H

#if defined(AP_INT_MAX_W) && (AP_INT_MAX_W < 4096)
#error "database::scan requires define AP_INT_MAX_W to 4096 or larger"
#else
#define AP_INT_MAX_W 4096 // Must be defined before next line
#include <ap_int.h>
#endif

#include <hls_stream.h>

#include "xf_database/types.hpp"
#include "xf_database/utils.hpp"

// ---------------------- scan_col 1 cols ---------------------------------

namespace xf {
namespace database {
namespace details {
template <int burst_len, int vec_len, int size0>
void read_to_col_vec(                        //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr, //
    const int nrow,                          //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm) {
    //
    int nread = (nrow + vec_len - 1) / vec_len;

READ_TO_COL_VEC:
    for (int i = 0; i < nread; i += burst_len) {
#pragma HLS dataflow
        const int len = ((i + burst_len) > nread) ? (nread - i) : burst_len;
    READ_VEC0:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c0vec_strm.write(c0vec_ptr[i + j]);
        }
        // printf("%d burst len %d\n", i / burst_len, len);
    }
}

template <int vec_len, int size0>
void split_col_vec(                                         //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    const int nrow,                                         //
    hls::stream<ap_uint<8 * size0> >& c0_strm,              //
    hls::stream<bool>& e_strm) {
//
SPLIT_COL_VEC:
    for (int i = 0; i < nrow; i += vec_len) {
#pragma HLS pipeline II = vec_len
        ap_uint<8 * size0* vec_len> c0vec = c0vec_strm.read();
        int n = (i + vec_len) > nrow ? (nrow - i) : vec_len;
        for (int j = 0; j < vec_len; ++j) {
            ap_uint<8 * size0> c0 = c0vec.range(8 * size0 * (j + 1) - 1, 8 * size0 * j);
            if (j < n) {
                c0_strm.write(c0);
                e_strm.write(false);
            }
        }
    }
    e_strm.write(true);
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Scan 1 column from DDR/HBM buffers.
 *
 * @tparam burst_len burst read length, must be supported by MC.
 * @tparam vec_len number of items to be scanned as a vector from AXI port.
 * @tparam size0 size of column 0, in byte.
 *
 * @param c0vec_ptr buffer pointer to column 0.
 * @param nrow number of row to scan.
 * @param c0_strm column 0 stream.
 * @param e_row_strm output end flag stream.
 */
template <int burst_len, int vec_len, int size0>
void scanCol(                                  //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,   //
    const int nrow,                            //
    hls::stream<ap_uint<8 * size0> >& c0_strm, //
    hls::stream<bool>& e_row_strm) {
//
#pragma HLS dataflow
    const int fifo_depth = burst_len * 2;

    hls::stream<ap_uint<8 * size0 * vec_len> > c0vec_strm("c0vec_strm");
#pragma HLS stream variable = c0vec_strm depth = fifo_depth

    details::read_to_col_vec<burst_len, vec_len, size0>( //
        c0vec_ptr, nrow,                                 //
        c0vec_strm);

    details::split_col_vec<vec_len, size0>( //
        c0vec_strm, nrow,                   //
        c0_strm, e_row_strm);
}

} // namespace database
} // namespace xf

// ---------------------- scan_col 2 cols ---------------------------------

namespace xf {
namespace database {
namespace details {
template <int burst_len, int vec_len, int size0, int size1>
void read_to_col_vec(                                       //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,                //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,                //
    const int nrow,                                         //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm) {
    //
    int nread = (nrow + vec_len - 1) / vec_len;

READ_TO_COL_VEC:
    for (int i = 0; i < nread; i += burst_len) {
#pragma HLS dataflow
        const int len = ((i + burst_len) > nread) ? (nread - i) : burst_len;
    READ_VEC0:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c0vec_strm.write(c0vec_ptr[i + j]);
        }
    READ_VEC1:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c1vec_strm.write(c1vec_ptr[i + j]);
        }
        // printf("%d burst len %d\n", i / burst_len, len);
    }
}

template <int vec_len, int size0, int size1>
void split_col_vec(                                         //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    const int nrow,                                         //
    hls::stream<ap_uint<8 * size0> >& c0_strm,              //
    hls::stream<ap_uint<8 * size1> >& c1_strm,              //
    hls::stream<bool>& e_strm) {
//
SPLIT_COL_VEC:
    for (int i = 0; i < nrow; i += vec_len) {
#pragma HLS pipeline II = vec_len
        ap_uint<8 * size0* vec_len> c0vec = c0vec_strm.read();
        ap_uint<8 * size1* vec_len> c1vec = c1vec_strm.read();
        int n = (i + vec_len) > nrow ? (nrow - i) : vec_len;
        for (int j = 0; j < vec_len; ++j) {
            ap_uint<8 * size0> c0 = c0vec.range(8 * size0 * (j + 1) - 1, 8 * size0 * j);
            ap_uint<8 * size1> c1 = c1vec.range(8 * size1 * (j + 1) - 1, 8 * size1 * j);
            if (j < n) {
                c0_strm.write(c0);
                c1_strm.write(c1);
                e_strm.write(false);
            }
        }
    }
    e_strm.write(true);
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Scan 2 columns from DDR/HBM buffers.
 *
 * @tparam burst_len burst read length, must be supported by MC.
 * @tparam vec_len number of items to be scanned as a vector from AXI port.
 * @tparam size0 size of column 0, in byte.
 * @tparam size1 size of column 1, in byte.
 *
 * @param c0vec_ptr buffer pointer to column 0.
 * @param c1vec_ptr buffer pointer to column 1.
 * @param nrow number of row to scan.
 * @param c0_strm column 0 stream.
 * @param c1_strm column 1 stream.
 * @param e_row_strm output end flag stream.
 */
template <int burst_len, int vec_len, int size0, int size1>
void scanCol(                                  //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,   //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,   //
    const int nrow,                            //
    hls::stream<ap_uint<8 * size0> >& c0_strm, //
    hls::stream<ap_uint<8 * size1> >& c1_strm, //
    hls::stream<bool>& e_row_strm) {
//
#pragma HLS dataflow
    const int fifo_depth = burst_len * 2;

    hls::stream<ap_uint<8 * size0 * vec_len> > c0vec_strm("c0vec_strm");
#pragma HLS stream variable = c0vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size1 * vec_len> > c1vec_strm("c1vec_strm");
#pragma HLS stream variable = c1vec_strm depth = fifo_depth

    details::read_to_col_vec<burst_len, vec_len, size0, size1>( //
        c0vec_ptr, c1vec_ptr, nrow,                             //
        c0vec_strm, c1vec_strm);

    details::split_col_vec<vec_len, size0, size1>( //
        c0vec_strm, c1vec_strm, nrow,              //
        c0_strm, c1_strm, e_row_strm);
}

} // namespace database
} // namespace xf

// ---------------------- scan_col 3 cols ---------------------------------
namespace xf {
namespace database {
namespace details {
template <int burst_len, int vec_len, int size0, int size1, int size2>
void read_to_col_vec(                                       //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,                //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,                //
    ap_uint<8 * size2 * vec_len>* c2vec_ptr,                //
    const int nrow,                                         //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    hls::stream<ap_uint<8 * size2 * vec_len> >& c2vec_strm) {
    //
    int nread = (nrow + vec_len - 1) / vec_len;

READ_TO_COL_VEC:
    for (int i = 0; i < nread; i += burst_len) {
#pragma HLS dataflow
        const int len = ((i + burst_len) > nread) ? (nread - i) : burst_len;
    READ_VEC0:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c0vec_strm.write(c0vec_ptr[i + j]);
        }
    READ_VEC1:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c1vec_strm.write(c1vec_ptr[i + j]);
        }
    READ_VEC2:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c2vec_strm.write(c2vec_ptr[i + j]);
        }
        // printf("%d burst len %d\n", i / burst_len, len);
    }
}

template <int vec_len, int size0, int size1, int size2>
void split_col_vec(                                         //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    hls::stream<ap_uint<8 * size2 * vec_len> >& c2vec_strm, //
    const int nrow,                                         //
    hls::stream<ap_uint<8 * size0> >& c0_strm,              //
    hls::stream<ap_uint<8 * size1> >& c1_strm,              //
    hls::stream<ap_uint<8 * size2> >& c2_strm,              //
    hls::stream<bool>& e_strm) {
//
SPLIT_COL_VEC:
    for (int i = 0; i < nrow; i += vec_len) {
#pragma HLS pipeline II = vec_len
        ap_uint<8 * size0* vec_len> c0vec = c0vec_strm.read();
        ap_uint<8 * size1* vec_len> c1vec = c1vec_strm.read();
        ap_uint<8 * size2* vec_len> c2vec = c2vec_strm.read();
        int n = (i + vec_len) > nrow ? (nrow - i) : vec_len;
        for (int j = 0; j < vec_len; ++j) {
            ap_uint<8 * size0> c0 = c0vec.range(8 * size0 * (j + 1) - 1, 8 * size0 * j);
            ap_uint<8 * size1> c1 = c1vec.range(8 * size1 * (j + 1) - 1, 8 * size1 * j);
            ap_uint<8 * size2> c2 = c2vec.range(8 * size1 * (j + 1) - 1, 8 * size2 * j);
            if (j < n) {
                c0_strm.write(c0);
                c1_strm.write(c1);
                c2_strm.write(c2);
                e_strm.write(false);
            }
        }
    }
    e_strm.write(true);
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Scan 3 columns from DDR/HBM buffers.
 *
 * @tparam burst_len burst read length, must be supported by MC.
 * @tparam vec_len number of items to be scanned as a vector from AXI port.
 * @tparam size0 size of column 0, in byte.
 * @tparam size1 size of column 1, in byte.
 * @tparam size2 size of column 2, in byte.
 *
 * @param c0vec_ptr buffer pointer to column 0.
 * @param c1vec_ptr buffer pointer to column 1.
 * @param c2vec_ptr buffer pointer to column 2.
 * @param nrow number of row to scan.
 * @param c0_strm column 0 stream.
 * @param c1_strm column 1 stream.
 * @param c2_strm column 2 stream.
 * @param e_row_strm output end flag stream.
 */
template <int burst_len, int vec_len, int size0, int size1, int size2>
void scanCol(                                  //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,   //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,   //
    ap_uint<8 * size2 * vec_len>* c2vec_ptr,   //
    const int nrow,                            //
    hls::stream<ap_uint<8 * size0> >& c0_strm, //
    hls::stream<ap_uint<8 * size1> >& c1_strm, //
    hls::stream<ap_uint<8 * size2> >& c2_strm, //
    hls::stream<bool>& e_row_strm) {
//
#pragma HLS dataflow
    const int fifo_depth = burst_len * 2;

    hls::stream<ap_uint<8 * size0 * vec_len> > c0vec_strm("c0vec_strm");
#pragma HLS stream variable = c0vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size1 * vec_len> > c1vec_strm("c1vec_strm");
#pragma HLS stream variable = c1vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size2 * vec_len> > c2vec_strm("c2vec_strm");
#pragma HLS stream variable = c2vec_strm depth = fifo_depth

    details::read_to_col_vec<burst_len, vec_len, size0, size1, size2>(c0vec_ptr, c1vec_ptr, c2vec_ptr, nrow, //
                                                                      c0vec_strm, c1vec_strm, c2vec_strm);

    details::split_col_vec<vec_len, size0, size1, size2>( //
        c0vec_strm, c1vec_strm, c2vec_strm, nrow,         //
        c0_strm, c1_strm, c2_strm, e_row_strm);
}

} // namespace database
} // namespace xf

// ---------------------- scan_col 4 cols ---------------------------------
namespace xf {
namespace database {
namespace details {
template <int burst_len,
          int vec_len,
          int size0,
          int size1,
          int size2,
          int size3>
void read_to_col_vec(                                       //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,                //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,                //
    ap_uint<8 * size2 * vec_len>* c2vec_ptr,                //
    ap_uint<8 * size3 * vec_len>* c3vec_ptr,                //
    const int nrow,                                         //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    hls::stream<ap_uint<8 * size2 * vec_len> >& c2vec_strm, //
    hls::stream<ap_uint<8 * size3 * vec_len> >& c3vec_strm) {
    //
    int nread = (nrow + vec_len - 1) / vec_len;

READ_TO_COL_VEC:
    for (int i = 0; i < nread; i += burst_len) {
#pragma HLS dataflow
        const int len = ((i + burst_len) > nread) ? (nread - i) : burst_len;
    READ_VEC0:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c0vec_strm.write(c0vec_ptr[i + j]);
        }
    READ_VEC1:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c1vec_strm.write(c1vec_ptr[i + j]);
        }
    READ_VEC2:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c2vec_strm.write(c2vec_ptr[i + j]);
        }
    READ_VEC3:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c3vec_strm.write(c3vec_ptr[i + j]);
        }
        // printf("%d burst len %d\n", i / burst_len, len);
    }
}

template <int vec_len, int size0, int size1, int size2, int size3>
void split_col_vec(                                         //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    hls::stream<ap_uint<8 * size2 * vec_len> >& c2vec_strm, //
    hls::stream<ap_uint<8 * size3 * vec_len> >& c3vec_strm, //
    const int nrow,                                         //
    hls::stream<ap_uint<8 * size0> >& c0_strm,              //
    hls::stream<ap_uint<8 * size1> >& c1_strm,              //
    hls::stream<ap_uint<8 * size2> >& c2_strm,              //
    hls::stream<ap_uint<8 * size3> >& c3_strm,              //
    hls::stream<bool>& e_strm) {
//
SPLIT_COL_VEC:
    for (int i = 0; i < nrow; i += vec_len) {
#pragma HLS pipeline II = vec_len
        ap_uint<8 * size0* vec_len> c0vec = c0vec_strm.read();
        ap_uint<8 * size1* vec_len> c1vec = c1vec_strm.read();
        ap_uint<8 * size2* vec_len> c2vec = c2vec_strm.read();
        ap_uint<8 * size3* vec_len> c3vec = c3vec_strm.read();
        int n = (i + vec_len) > nrow ? (nrow - i) : vec_len;
        for (int j = 0; j < vec_len; ++j) {
            ap_uint<8 * size0> c0 = c0vec.range(8 * size0 * (j + 1) - 1, 8 * size0 * j);
            ap_uint<8 * size1> c1 = c1vec.range(8 * size1 * (j + 1) - 1, 8 * size1 * j);
            ap_uint<8 * size2> c2 = c2vec.range(8 * size2 * (j + 1) - 1, 8 * size2 * j);
            ap_uint<8 * size3> c3 = c3vec.range(8 * size3 * (j + 1) - 1, 8 * size3 * j);
            if (j < n) {
                c0_strm.write(c0);
                c1_strm.write(c1);
                c2_strm.write(c2);
                c3_strm.write(c3);
                e_strm.write(false);
            }
        }
    }
    e_strm.write(true);
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Scan 4 columns from DDR/HBM buffers.
 *
 * @tparam burst_len burst read length, must be supported by MC.
 * @tparam vec_len number of items to be scanned as a vector from AXI port.
 * @tparam size0 size of column 0, in byte.
 * @tparam size1 size of column 1, in byte.
 * @tparam size2 size of column 2, in byte.
 * @tparam size3 size of column 3, in byte.
 *
 * @param c0vec_ptr buffer pointer to column 0.
 * @param c1vec_ptr buffer pointer to column 1.
 * @param c2vec_ptr buffer pointer to column 2.
 * @param c3vec_ptr buffer pointer to column 3.
 * @param nrow number of row to scan.
 * @param c0_strm column 0 stream.
 * @param c1_strm column 1 stream.
 * @param c2_strm column 2 stream.
 * @param c3_strm column 3 stream.
 * @param e_row_strm output end flag stream.
 */
template <int burst_len,
          int vec_len,
          int size0,
          int size1,
          int size2,
          int size3>
void scanCol(                                  //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,   //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,   //
    ap_uint<8 * size2 * vec_len>* c2vec_ptr,   //
    ap_uint<8 * size3 * vec_len>* c3vec_ptr,   //
    const int nrow,                            //
    hls::stream<ap_uint<8 * size0> >& c0_strm, //
    hls::stream<ap_uint<8 * size1> >& c1_strm, //
    hls::stream<ap_uint<8 * size2> >& c2_strm, //
    hls::stream<ap_uint<8 * size3> >& c3_strm, //
    hls::stream<bool>& e_row_strm) {
//
#pragma HLS dataflow
    const int fifo_depth = burst_len * 2;

    hls::stream<ap_uint<8 * size0 * vec_len> > c0vec_strm("c0vec_strm");
#pragma HLS stream variable = c0vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size1 * vec_len> > c1vec_strm("c1vec_strm");
#pragma HLS stream variable = c1vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size2 * vec_len> > c2vec_strm("c2vec_strm");
#pragma HLS stream variable = c2vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size3 * vec_len> > c3vec_strm("c3vec_strm");
#pragma HLS stream variable = c3vec_strm depth = fifo_depth

    details::read_to_col_vec<burst_len, vec_len, size0, size1, size2, size3>(
        c0vec_ptr, c1vec_ptr, c2vec_ptr, c3vec_ptr, nrow, //
        c0vec_strm, c1vec_strm, c2vec_strm, c3vec_strm);

    details::split_col_vec<vec_len, size0, size1, size2, size3>(c0vec_strm, c1vec_strm, c2vec_strm, c3vec_strm, nrow, //
                                                                c0_strm, c1_strm, c2_strm, c3_strm, e_row_strm);
}

} // namespace database
} // namespace xf

// ---------------------- scan_col 5 cols ---------------------------------
namespace xf {
namespace database {
namespace details {
template <int burst_len, int vec_len, int size0, int size1, int size2, int size3, int size4>
void read_to_col_vec(                                       //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,                //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,                //
    ap_uint<8 * size2 * vec_len>* c2vec_ptr,                //
    ap_uint<8 * size3 * vec_len>* c3vec_ptr,                //
    ap_uint<8 * size4 * vec_len>* c4vec_ptr,                //
    const int nrow,                                         //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    hls::stream<ap_uint<8 * size2 * vec_len> >& c2vec_strm, //
    hls::stream<ap_uint<8 * size3 * vec_len> >& c3vec_strm, //
    hls::stream<ap_uint<8 * size4 * vec_len> >& c4vec_strm) {
    //
    int nread = (nrow + vec_len - 1) / vec_len;

READ_TO_COL_VEC:
    for (int i = 0; i < nread; i += burst_len) {
#pragma HLS dataflow
        const int len = ((i + burst_len) > nread) ? (nread - i) : burst_len;
    READ_VEC0:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c0vec_strm.write(c0vec_ptr[i + j]);
        }
    READ_VEC1:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c1vec_strm.write(c1vec_ptr[i + j]);
        }
    READ_VEC2:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c2vec_strm.write(c2vec_ptr[i + j]);
        }
    READ_VEC3:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c3vec_strm.write(c3vec_ptr[i + j]);
        }
    READ_VEC4:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c4vec_strm.write(c4vec_ptr[i + j]);
        }
        // printf("%d burst len %d\n", i / burst_len, len);
    }
}

template <int vec_len, int size0, int size1, int size2, int size3, int size4>
void split_col_vec(                                         //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    hls::stream<ap_uint<8 * size2 * vec_len> >& c2vec_strm, //
    hls::stream<ap_uint<8 * size3 * vec_len> >& c3vec_strm, //
    hls::stream<ap_uint<8 * size4 * vec_len> >& c4vec_strm, //
    const int nrow,                                         //
    hls::stream<ap_uint<8 * size0> >& c0_strm,              //
    hls::stream<ap_uint<8 * size1> >& c1_strm,              //
    hls::stream<ap_uint<8 * size2> >& c2_strm,              //
    hls::stream<ap_uint<8 * size3> >& c3_strm,              //
    hls::stream<ap_uint<8 * size4> >& c4_strm,              //
    hls::stream<bool>& e_strm) {
//
SPLIT_COL_VEC:
    for (int i = 0; i < nrow; i += vec_len) {
#pragma HLS pipeline II = vec_len
        ap_uint<8 * size0* vec_len> c0vec = c0vec_strm.read();
        ap_uint<8 * size1* vec_len> c1vec = c1vec_strm.read();
        ap_uint<8 * size2* vec_len> c2vec = c2vec_strm.read();
        ap_uint<8 * size3* vec_len> c3vec = c3vec_strm.read();
        ap_uint<8 * size4* vec_len> c4vec = c4vec_strm.read();
        int n = (i + vec_len) > nrow ? (nrow - i) : vec_len;
        for (int j = 0; j < vec_len; ++j) {
            ap_uint<8 * size0> c0 = c0vec.range(8 * size0 * (j + 1) - 1, 8 * size0 * j);
            ap_uint<8 * size1> c1 = c1vec.range(8 * size1 * (j + 1) - 1, 8 * size1 * j);
            ap_uint<8 * size2> c2 = c2vec.range(8 * size2 * (j + 1) - 1, 8 * size2 * j);
            ap_uint<8 * size3> c3 = c3vec.range(8 * size3 * (j + 1) - 1, 8 * size3 * j);
            ap_uint<8 * size4> c4 = c4vec.range(8 * size4 * (j + 1) - 1, 8 * size4 * j);
            if (j < n) {
                c0_strm.write(c0);
                c1_strm.write(c1);
                c2_strm.write(c2);
                c3_strm.write(c3);
                c4_strm.write(c4);
                e_strm.write(false);
            }
        }
    }
    e_strm.write(true);
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Scan 5 columns from DDR/HBM buffers.
 *
 * @tparam burst_len burst read length, must be supported by MC.
 * @tparam vec_len number of items to be scanned as a vector from AXI port.
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
 * @param nrow number of row to scan.
 * @param c0_strm column 0 stream.
 * @param c1_strm column 1 stream.
 * @param c2_strm column 2 stream.
 * @param c3_strm column 3 stream.
 * @param c4_strm column 4 stream.
 * @param e_row_strm output end flag stream.
 */
template <int burst_len, int vec_len, int size0, int size1, int size2, int size3, int size4>
void scanCol(                                  //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,   //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,   //
    ap_uint<8 * size2 * vec_len>* c2vec_ptr,   //
    ap_uint<8 * size3 * vec_len>* c3vec_ptr,   //
    ap_uint<8 * size4 * vec_len>* c4vec_ptr,   //
    const int nrow,                            //
    hls::stream<ap_uint<8 * size0> >& c0_strm, //
    hls::stream<ap_uint<8 * size1> >& c1_strm, //
    hls::stream<ap_uint<8 * size2> >& c2_strm, //
    hls::stream<ap_uint<8 * size3> >& c3_strm, //
    hls::stream<ap_uint<8 * size4> >& c4_strm, //
    hls::stream<bool>& e_row_strm) {
//
#pragma HLS dataflow
    const int fifo_depth = burst_len * 2;

    hls::stream<ap_uint<8 * size0 * vec_len> > c0vec_strm("c0vec_strm");
#pragma HLS stream variable = c0vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size1 * vec_len> > c1vec_strm("c1vec_strm");
#pragma HLS stream variable = c1vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size2 * vec_len> > c2vec_strm("c2vec_strm");
#pragma HLS stream variable = c2vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size3 * vec_len> > c3vec_strm("c3vec_strm");
#pragma HLS stream variable = c3vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size4 * vec_len> > c4vec_strm("c4vec_strm");
#pragma HLS stream variable = c4vec_strm depth = fifo_depth

    details::read_to_col_vec<burst_len, vec_len, size0, size1, size2, size3, size4>(
        c0vec_ptr, c1vec_ptr, c2vec_ptr, c3vec_ptr, c4vec_ptr, nrow, //
        c0vec_strm, c1vec_strm, c2vec_strm, c3vec_strm, c4vec_strm);

    details::split_col_vec<vec_len, size0, size1, size2, size3, size4>(
        c0vec_strm, c1vec_strm, c2vec_strm, c3vec_strm, c4vec_strm, nrow, //
        c0_strm, c1_strm, c2_strm, c3_strm, c4_strm, e_row_strm);
}

} // namespace database
} // namespace xf

// ---------------------- scan_col 6 cols ---------------------------------

namespace xf {
namespace database {
namespace details {
template <int burst_len, int vec_len, int size0, int size1, int size2, int size3, int size4, int size5>
void read_to_col_vec(                                       //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,                //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,                //
    ap_uint<8 * size2 * vec_len>* c2vec_ptr,                //
    ap_uint<8 * size3 * vec_len>* c3vec_ptr,                //
    ap_uint<8 * size4 * vec_len>* c4vec_ptr,                //
    ap_uint<8 * size5 * vec_len>* c5vec_ptr,                //
    const int nrow,                                         //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    hls::stream<ap_uint<8 * size2 * vec_len> >& c2vec_strm, //
    hls::stream<ap_uint<8 * size3 * vec_len> >& c3vec_strm, //
    hls::stream<ap_uint<8 * size4 * vec_len> >& c4vec_strm, //
    hls::stream<ap_uint<8 * size5 * vec_len> >& c5vec_strm) {
    //
    int nread = (nrow + vec_len - 1) / vec_len;

READ_TO_COL_VEC:
    for (int i = 0; i < nread; i += burst_len) {
#pragma HLS dataflow
        const int len = ((i + burst_len) > nread) ? (nread - i) : burst_len;
    READ_VEC0:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c0vec_strm.write(c0vec_ptr[i + j]);
        }
    READ_VEC1:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c1vec_strm.write(c1vec_ptr[i + j]);
        }
    READ_VEC2:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c2vec_strm.write(c2vec_ptr[i + j]);
        }
    READ_VEC3:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c3vec_strm.write(c3vec_ptr[i + j]);
        }
    READ_VEC4:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c4vec_strm.write(c4vec_ptr[i + j]);
        }
    READ_VEC5:
        for (int j = 0; j < len; ++j) {
#pragma HLS pipeline II = 1
            c5vec_strm.write(c5vec_ptr[i + j]);
        }
        // printf("%d burst len %d\n", i / burst_len, len);
    }
}

template <int vec_len,
          int size0,
          int size1,
          int size2,
          int size3,
          int size4,
          int size5>
void split_col_vec(                                         //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    hls::stream<ap_uint<8 * size2 * vec_len> >& c2vec_strm, //
    hls::stream<ap_uint<8 * size3 * vec_len> >& c3vec_strm, //
    hls::stream<ap_uint<8 * size4 * vec_len> >& c4vec_strm, //
    hls::stream<ap_uint<8 * size5 * vec_len> >& c5vec_strm, //
    const int nrow,                                         //
    hls::stream<ap_uint<8 * size0> >& c0_strm,              //
    hls::stream<ap_uint<8 * size1> >& c1_strm,              //
    hls::stream<ap_uint<8 * size2> >& c2_strm,              //
    hls::stream<ap_uint<8 * size3> >& c3_strm,              //
    hls::stream<ap_uint<8 * size4> >& c4_strm,              //
    hls::stream<ap_uint<8 * size5> >& c5_strm,              //
    hls::stream<bool>& e_strm) {
//
SPLIT_COL_VEC:
    for (int i = 0; i < nrow; i += vec_len) {
#pragma HLS pipeline II = vec_len
        ap_uint<8 * size0* vec_len> c0vec = c0vec_strm.read();
        ap_uint<8 * size1* vec_len> c1vec = c1vec_strm.read();
        ap_uint<8 * size2* vec_len> c2vec = c2vec_strm.read();
        ap_uint<8 * size3* vec_len> c3vec = c3vec_strm.read();
        ap_uint<8 * size4* vec_len> c4vec = c4vec_strm.read();
        ap_uint<8 * size5* vec_len> c5vec = c5vec_strm.read();
        int n = (i + vec_len) > nrow ? (nrow - i) : vec_len;
        for (int j = 0; j < vec_len; ++j) {
            ap_uint<8 * size0> c0 = c0vec.range(8 * size0 * (j + 1) - 1, 8 * size0 * j);
            ap_uint<8 * size1> c1 = c1vec.range(8 * size1 * (j + 1) - 1, 8 * size1 * j);
            ap_uint<8 * size2> c2 = c2vec.range(8 * size2 * (j + 1) - 1, 8 * size2 * j);
            ap_uint<8 * size3> c3 = c3vec.range(8 * size3 * (j + 1) - 1, 8 * size3 * j);
            ap_uint<8 * size4> c4 = c4vec.range(8 * size4 * (j + 1) - 1, 8 * size4 * j);
            ap_uint<8 * size5> c5 = c5vec.range(8 * size5 * (j + 1) - 1, 8 * size5 * j);
            if (j < n) {
                c0_strm.write(c0);
                c1_strm.write(c1);
                c2_strm.write(c2);
                c3_strm.write(c3);
                c4_strm.write(c4);
                c5_strm.write(c5);
                e_strm.write(false);
            }
        }
    }
    e_strm.write(true);
}

} // namespace details
} // namespace database
} // namespace xf

namespace xf {
namespace database {
/**
 * @brief Scan 6 columns from DDR/HBM buffers.
 *
 * @tparam burst_len burst read length, must be supported by MC.
 * @tparam vec_len number of items to be scanned as a vector from AXI port.
 * @tparam size0 size of column 0, in byte.
 * @tparam size1 size of column 1, in byte.
 * @tparam size2 size of column 2, in byte.
 * @tparam size3 size of column 3, in byte.
 * @tparam size4 size of column 4, in byte.
 * @tparam size5 size of column 5, in byte.
 *
 * @param c0vec_ptr buffer pointer to column 0.
 * @param c1vec_ptr buffer pointer to column 1.
 * @param c2vec_ptr buffer pointer to column 2.
 * @param c3vec_ptr buffer pointer to column 3.
 * @param c4vec_ptr buffer pointer to column 4.
 * @param c5vec_ptr buffer pointer to column 5.
 * @param nrow number of row to scan.
 * @param c0_strm column 0 stream.
 * @param c1_strm column 1 stream.
 * @param c2_strm column 2 stream.
 * @param c3_strm column 3 stream.
 * @param c4_strm column 4 stream.
 * @param c5_strm column 5 stream.
 * @param e_row_strm output end flag stream.
 */
template <int burst_len, int vec_len, int size0, int size1, int size2, int size3, int size4, int size5>
void scanCol(                                  //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,   //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,   //
    ap_uint<8 * size2 * vec_len>* c2vec_ptr,   //
    ap_uint<8 * size3 * vec_len>* c3vec_ptr,   //
    ap_uint<8 * size4 * vec_len>* c4vec_ptr,   //
    ap_uint<8 * size5 * vec_len>* c5vec_ptr,   //
    const int nrow,                            //
    hls::stream<ap_uint<8 * size0> >& c0_strm, //
    hls::stream<ap_uint<8 * size1> >& c1_strm, //
    hls::stream<ap_uint<8 * size2> >& c2_strm, //
    hls::stream<ap_uint<8 * size3> >& c3_strm, //
    hls::stream<ap_uint<8 * size4> >& c4_strm, //
    hls::stream<ap_uint<8 * size5> >& c5_strm, //
    hls::stream<bool>& e_row_strm) {
//
#pragma HLS dataflow
    const int fifo_depth = burst_len * 2;

    hls::stream<ap_uint<8 * size0 * vec_len> > c0vec_strm("c0vec_strm");
#pragma HLS stream variable = c0vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size1 * vec_len> > c1vec_strm("c1vec_strm");
#pragma HLS stream variable = c1vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size2 * vec_len> > c2vec_strm("c2vec_strm");
#pragma HLS stream variable = c2vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size3 * vec_len> > c3vec_strm("c3vec_strm");
#pragma HLS stream variable = c3vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size4 * vec_len> > c4vec_strm("c4vec_strm");
#pragma HLS stream variable = c4vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size5 * vec_len> > c5vec_strm("c5vec_strm");
#pragma HLS stream variable = c5vec_strm depth = fifo_depth

    details::read_to_col_vec<burst_len, vec_len, size0, size1, size2, size3, size4, size5>(
        c0vec_ptr, c1vec_ptr, c2vec_ptr, c3vec_ptr, c4vec_ptr, c5vec_ptr, nrow, //
        c0vec_strm, c1vec_strm, c2vec_strm, c3vec_strm, c4vec_strm, c5vec_strm);

    details::split_col_vec<vec_len, size0, size1, size2, size3, size4, size5>(
        c0vec_strm, c1vec_strm, c2vec_strm, c3vec_strm, c4vec_strm, c5vec_strm,
        nrow, //
        c0_strm, c1_strm, c2_strm, c3_strm, c4_strm, c5_strm, e_row_strm);
}

} // namespace database
} // namespace xf

// ----------------------------------------------------------------------- //
//                                                                         //
//                      Multi-channel scan_col                             //
//                                                                         //
// ----------------------------------------------------------------------- //

namespace xf {
namespace database {
namespace details {
template <int vec_len, int ch_num, int size0>
void split_col_vec(                                         //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    const int nrow,                                         //
    hls::stream<ap_uint<8 * size0> > c0_strm[ch_num],       //
    hls::stream<bool> e_strm[ch_num]) {
    //
    enum { per_ch = vec_len / ch_num };
SPLIT_COL_VEC:
    for (int i = 0; i < nrow; i += vec_len) {
#pragma HLS pipeline II = per_ch
        ap_uint<8 * size0* vec_len> c0vec = c0vec_strm.read();
        int n = (i + vec_len) > nrow ? (nrow - i) : vec_len;
        XF_DATABASE_ASSERT((vec_len % ch_num == 0) && (vec_len >= ch_num));
        for (int j = 0; j < per_ch; ++j) {
            for (int k = 0; k < ch_num; ++k) {
#pragma HLS unroll
                ap_uint<8 * size0> c0 = c0vec.range(8 * size0 * (j * ch_num + k + 1) - 1, 8 * size0 * (j * ch_num + k));
                if ((j * ch_num + k) < n) {
                    c0_strm[k].write(c0);
                    e_strm[k].write(false);
                }
            }
        }
    }
    for (int k = 0; k < ch_num; ++k) {
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
 * @brief Scan one column from DDR/HBM buffers, emit multiple rows
 * concurrently.
 *
 * @tparam burst_len burst read length, must be supported by MC.
 * @tparam vec_len number of items to be scanned as a vector from AXI port.
 * @tparam ch_num number of concurrent output channels per column.
 * @tparam size0 size of column 0, in byte.
 *
 * @param c0vec_ptr buffer pointer to column 0.
 * @param nrow number of row to scan.
 * @param c0_strm array of column 0 stream.
 * @param e_row_strm array of output end flag stream.
 */
template <int burst_len, int vec_len, int ch_num, int size0>
void scanCol(                                         //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,          //
    const int nrow,                                   //
    hls::stream<ap_uint<8 * size0> > c0_strm[ch_num], //
    hls::stream<bool> e_row_strm[ch_num]) {
//
#pragma HLS dataflow
    const int fifo_depth = burst_len * 2;

    hls::stream<ap_uint<8 * size0 * vec_len> > c0vec_strm("c0vec_strm");
#pragma HLS stream variable = c0vec_strm depth = fifo_depth

    details::read_to_col_vec<burst_len, vec_len, size0>( //
        c0vec_ptr, nrow,                                 //
        c0vec_strm);

    details::split_col_vec<vec_len, ch_num, size0>( //
        c0vec_strm, nrow,                           //
        c0_strm, e_row_strm);
}

} // namespace database
} // namespace xf

// -----------------------------------------------------------------------
namespace xf {
namespace database {
namespace details {
template <int vec_len, int ch_num, int size0, int size1>
void split_col_vec(                                         //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    const int nrow,                                         //
    hls::stream<ap_uint<8 * size0> > c0_strm[ch_num],       //
    hls::stream<ap_uint<8 * size1> > c1_strm[ch_num],       //
    hls::stream<bool> e_strm[ch_num]) {
    //
    enum { per_ch = vec_len / ch_num };
SPLIT_COL_VEC:
    for (int i = 0; i < nrow; i += vec_len) {
#pragma HLS pipeline II = per_ch
        ap_uint<8 * size0* vec_len> c0vec = c0vec_strm.read();
        ap_uint<8 * size1* vec_len> c1vec = c1vec_strm.read();
        int n = (i + vec_len) > nrow ? (nrow - i) : vec_len;
        XF_DATABASE_ASSERT((vec_len % ch_num == 0) && (vec_len >= ch_num));
        for (int j = 0; j < per_ch; ++j) {
            for (int k = 0; k < ch_num; ++k) {
#pragma HLS unroll
                ap_uint<8 * size0> c0 = c0vec.range(8 * size0 * (j * ch_num + k + 1) - 1, 8 * size0 * (j * ch_num + k));
                ap_uint<8 * size1> c1 = c1vec.range(8 * size1 * (j * ch_num + k + 1) - 1, 8 * size1 * (j * ch_num + k));
                if ((j * ch_num + k) < n) {
                    c0_strm[k].write(c0);
                    c1_strm[k].write(c1);
                    e_strm[k].write(false);
                }
            }
        }
    }
    for (int k = 0; k < ch_num; ++k) {
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
 * @brief Scan two columns from DDR/HBM buffers, emit multiple rows
 * concurrently.
 *
 * @tparam burst_len burst read length, must be supported by MC.
 * @tparam vec_len number of items to be scanned as a vector from AXI port.
 * @tparam ch_num number of concurrent output channels per column.
 * @tparam size0 size of column 0, in byte.
 * @tparam size1 size of column 1, in byte.
 *
 * @param c0vec_ptr buffer pointer to column 0.
 * @param c1vec_ptr buffer pointer to column 1.
 * @param nrow number of row to scan.
 * @param c0_strm array of column 0 stream.
 * @param c1_strm array of column 1 stream.
 * @param e_row_strm array of output end flag stream.
 */
template <int burst_len, int vec_len, int ch_num, int size0, int size1>
void scanCol(                                         //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,          //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,          //
    const int nrow,                                   //
    hls::stream<ap_uint<8 * size0> > c0_strm[ch_num], //
    hls::stream<ap_uint<8 * size1> > c1_strm[ch_num], //
    hls::stream<bool> e_row_strm[ch_num]) {
//
#pragma HLS dataflow
    const int fifo_depth = burst_len * 2;

    hls::stream<ap_uint<8 * size0 * vec_len> > c0vec_strm("c0vec_strm");
#pragma HLS stream variable = c0vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size1 * vec_len> > c1vec_strm("c1vec_strm");
#pragma HLS stream variable = c1vec_strm depth = fifo_depth

    details::read_to_col_vec<burst_len, vec_len, size0, size1>(c0vec_ptr, c1vec_ptr, nrow, //
                                                               c0vec_strm, c1vec_strm);

    details::split_col_vec<vec_len, ch_num, size0, size1>( //
        c0vec_strm, c1vec_strm, nrow,                      //
        c0_strm, c1_strm, e_row_strm);
}

} // namespace database
} // namespace xf

// -----------------------------------------------------------------------
namespace xf {
namespace database {
namespace details {
template <int vec_len, int ch_num, int size0, int size1, int size2>
void split_col_vec(                                         //
    hls::stream<ap_uint<8 * size0 * vec_len> >& c0vec_strm, //
    hls::stream<ap_uint<8 * size1 * vec_len> >& c1vec_strm, //
    hls::stream<ap_uint<8 * size2 * vec_len> >& c2vec_strm, //
    const int nrow,                                         //
    hls::stream<ap_uint<8 * size0> > c0_strm[ch_num],       //
    hls::stream<ap_uint<8 * size1> > c1_strm[ch_num],       //
    hls::stream<ap_uint<8 * size2> > c2_strm[ch_num],       //
    hls::stream<bool> e_strm[ch_num]) {
    //
    enum { per_ch = vec_len / ch_num };
SPLIT_COL_VEC_II_NOT_1:
    for (int i = 0; i < nrow; i += vec_len) {
#pragma HLS pipeline II = per_ch
        ap_uint<8 * size0* vec_len> c0vec = c0vec_strm.read();
        ap_uint<8 * size1* vec_len> c1vec = c1vec_strm.read();
        ap_uint<8 * size2* vec_len> c2vec = c2vec_strm.read();
        int n = (i + vec_len) > nrow ? (nrow - i) : vec_len;
        XF_DATABASE_ASSERT((vec_len >= ch_num) && (vec_len % ch_num == 0));
        for (int j = 0; j < per_ch; ++j) {
            for (int k = 0; k < ch_num; ++k) {
#pragma HLS unroll
                ap_uint<8 * size0> c0 = c0vec.range(8 * size0 * (j * ch_num + k + 1) - 1, 8 * size0 * (j * ch_num + k));
                ap_uint<8 * size1> c1 = c1vec.range(8 * size1 * (j * ch_num + k + 1) - 1, 8 * size1 * (j * ch_num + k));
                ap_uint<8 * size2> c2 = c2vec.range(8 * size1 * (j * ch_num + k + 1) - 1, 8 * size2 * (j * ch_num + k));
                if ((j * ch_num + k) < n) {
                    c0_strm[k].write(c0);
                    c1_strm[k].write(c1);
                    c2_strm[k].write(c2);
                    e_strm[k].write(false);
                }
            }
        }
    }
    for (int k = 0; k < ch_num; ++k) {
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
 * @brief Scan three columns from DDR/HBM buffers, emit multiple rows
 * concurrently.
 *
 * @tparam burst_len burst read length, must be supported by MC.
 * @tparam vec_len number of items to be scanned as a vector from AXI port.
 * @tparam ch_num number of concurrent output channels per column.
 * @tparam size0 size of column 0, in byte.
 * @tparam size1 size of column 1, in byte.
 * @tparam size2 size of column 2, in byte.
 *
 * @param c0vec_ptr buffer pointer to column 0.
 * @param c1vec_ptr buffer pointer to column 1.
 * @param c2vec_ptr buffer pointer to column 2.
 * @param nrow number of row to scan.
 * @param c0_strm array of column 0 stream.
 * @param c1_strm array of column 1 stream.
 * @param c2_strm array of column 2 stream.
 * @param e_row_strm array of output end flag stream.
 */
template <int burst_len,
          int vec_len,
          int ch_num,
          int size0,
          int size1,
          int size2>
void scanCol(                                         //
    ap_uint<8 * size0 * vec_len>* c0vec_ptr,          //
    ap_uint<8 * size1 * vec_len>* c1vec_ptr,          //
    ap_uint<8 * size2 * vec_len>* c2vec_ptr,          //
    const int nrow,                                   //
    hls::stream<ap_uint<8 * size0> > c0_strm[ch_num], //
    hls::stream<ap_uint<8 * size1> > c1_strm[ch_num], //
    hls::stream<ap_uint<8 * size2> > c2_strm[ch_num], //
    hls::stream<bool> e_row_strm[ch_num]) {
//
#pragma HLS dataflow
    const int fifo_depth = burst_len * 2;

    hls::stream<ap_uint<8 * size0 * vec_len> > c0vec_strm("c0vec_strm");
#pragma HLS stream variable = c0vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size1 * vec_len> > c1vec_strm("c1vec_strm");
#pragma HLS stream variable = c1vec_strm depth = fifo_depth

    hls::stream<ap_uint<8 * size2 * vec_len> > c2vec_strm("c2vec_strm");
#pragma HLS stream variable = c2vec_strm depth = fifo_depth

    details::read_to_col_vec<burst_len, vec_len, size0, size1, size2>( //
        c0vec_ptr, c1vec_ptr, c2vec_ptr, nrow,                         //
        c0vec_strm, c1vec_strm, c2vec_strm);

    details::split_col_vec<vec_len, ch_num, size0, size1, size2>( //
        c0vec_strm, c1vec_strm, c2vec_strm, nrow,                 //
        c0_strm, c1_strm, c2_strm, e_row_strm);
}
} // namespace database
} // namespace xf

// -----------------------------------------------------------------------

#endif // XF_DATABASE_SCAN_COL_H
