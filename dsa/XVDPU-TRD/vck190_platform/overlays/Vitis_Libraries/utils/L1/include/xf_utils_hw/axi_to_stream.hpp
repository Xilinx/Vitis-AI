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
 * @file axi_to_stream.hpp
 * @brief This file provides loading data from AXI master to stream APIs.
 *
 * This file is part of Vitis Utility Library.
 */

#ifndef XF_UTILS_HW_AXI_TO_STREAM_H
#define XF_UTILS_HW_AXI_TO_STREAM_H

#include "xf_utils_hw/common.hpp"

namespace xf {
namespace common {
namespace utils_hw {

// ---------------------- APIs ---------------------------------

/**
 * @brief Loading data elements from AXI master to stream.
 *
 * This module requires the data elements to align to its size in buffer.
 * In another word, the start offset is specified by element count from
 * the beginning of a vector of AXI port width.
 *
 * This primitive assumes the width of AXI port is positive integer
 * multiple of data element's alignment width.
 *
 * The alignment width is assumed to be multiple of 8-bit char.
 * The AXI master port width is power of two, and no less than 8.
 *
 * @tparam _BurstLen burst length of AXI buffer, default is 32.
 * @tparam _WAxi width of AXI port, must be power of 2 and between 8 to 512.
 * @tparam _TStrm stream's type, e.g. ap_uint<aligned_width> for a aligned_width
 * stream.
 *
 * @param rbuf input AXI port.
 * @param num number of data elements to load from AXI port.
 * @param ostrm output stream.
 * @param e_ostrm end flag for output stream.
 */
/* TODO
 * \rst
 * ::
 *
 *    +--------------------------------------------------------+
 *    | DDR  -> AXI_BUS  -> FIFO     -> stream(aligned to 16b) |
 *    | XXaa    XXaabbcc    XXaabbcc    aa                     |
 *    | bbcc    ddXX0000    ddXX0000    bb                     |
 *    | ddXX                            cc                     |
 *    |                                 dd                     |
 *    +--------------------------------------------------------+
 *
 * \endrst
 *
 * Add param offset_num offset from the beginning of the buffer, by num of element.
 */

template <int _BurstLen = 32, int _WAxi, typename _TStrm>
void axiToStream(ap_uint<_WAxi>* rbuf, const int num, hls::stream<_TStrm>& ostrm, hls::stream<bool>& e_ostrm);

/**
 * @brief Loading char data from AXI master to stream.
 *
 * This primitive relaxes the alignment requirement, and actually load data by
 * 8-bit char. The 8-bit chars are packed as output stream wide word.
 * The last word may contain invalid data in high-bits if enough char has
 * already been packed.
 *
 * The alignment width is assumed to be multiple of 8-bit char.
 * The AXI master port width is power of two, and no less than 8.
 *
 * \rst
 * ::
 *
 *    +-----------------------------------------------------------------+
 *    | DDR  -> AXI_BUS                           -> FIFO     -> stream |
 *    | XXX1    XXX1234567812345_6781234567812345    XXX12345    1234   |
 *    | ...     ...                                  67812345    5678   |
 *    |                                              ...         ...    |
 *    | 32XX    6780000321XXXXXX_XXXXXXXXXXXXXXXX    21XXXXXX    21XX   |
 *    +-----------------------------------------------------------------+
 *
 * \endrst
 *
 * @tparam _BurstLen burst length of AXI buffer, default is 32.
 * @tparam _WAxi width of AXI port, must be power of 2 and between 8 to 512.
 * @tparam _TStrm stream's type.
 *
 * @param rbuf input AXI port.
 * @param ostrm output stream.
 * @param e_ostrm end flag for output stream.
 * @param len number of char to load from AXI port.
 * @param offset offset from the beginning of the buffer, in number of char.
 */
template <int _BurstLen = 32, int _WAxi, typename _TStrm>
void axiToCharStream(
    ap_uint<_WAxi>* rbuf, hls::stream<_TStrm>& ostrm, hls::stream<bool>& e_ostrm, const int len, const int offset = 0);

/**
 * @brief Loading data elements from AXI master to stream.
 *
 * This module requires the data elements to align to its size in buffer.
 * In another word, the start offset is specified by element count from
 * the beginning of a vector of AXI port width.
 *
 * This primitive assumes the width of AXI port is positive integer
 * multiple of data element's alignment width.
 *
 * The alignment width is assumed to be multiple of 8-bit char.
 * The AXI master port width is power of two, and no less than 8.
 *
 * @tparam _BurstLen burst length of AXI buffer, default is 32.
 * @tparam _WAxi width of AXI port, must be power of 2 and between 8 to 512.
 * @tparam _TStrm stream's type, e.g. ap_uint<aligned_width> for a aligned_width
 * stream.
 *
 * @param rbuf input AXI port.
 * @param num number of data elements to load from AXI port.
 * @param ostrm output stream.
 * @param e_ostrm end flag for output stream.
 */
/* TODO
 * \rst
 * ::
 *
 *    +--------------------------------------------------------+
 *    | DDR  -> AXI_BUS  -> FIFO     -> stream(aligned to 16b) |
 *    | XXaa    XXaabbcc    XXaabbcc    aa                     |
 *    | bbcc    ddXX0000    ddXX0000    bb                     |
 *    | ddXX                            cc                     |
 *    |                                 dd                     |
 *    +--------------------------------------------------------+
 *
 * \endrst
 *
 * Add param offset_num offset from the beginning of the buffer, by num of element.
 */

template <int _BurstLen = 32, int _WAxi, typename _TStrm>
void axiToStream(ap_uint<_WAxi>* rbuf, const int num, hls::stream<_TStrm>& ostrm);

// ------------------- Implementation --------------------------

namespace details {

template <int _WAxi, int _BurstLen>
void read_to_vec(ap_uint<_WAxi>* vec_ptr, const int nrow, const int scal_vec, hls::stream<ap_uint<_WAxi> >& vec_strm) {
    const int nread = (nrow + scal_vec - 1) / scal_vec;
READ_TO_VEC:
    for (int i = 0; i < nread; ++i) {
#pragma HLS loop_tripcount min = 1 max = 1
#pragma HLS PIPELINE II = 1
        vec_strm.write(vec_ptr[i]);
    } // This pipeline must be no judgment, otherwise the tool will not be able
      // to derive the correct burst_len
}

template <int _WAxi, typename _TStrm, int scal_vec>
void split_vec(hls::stream<ap_uint<_WAxi> >& vec_strm,
               const int nrow,
               const int offset_AL,
               hls::stream<_TStrm>& r_strm,
               hls::stream<bool>& e_strm) {
    const int WStrm = 8 * sizeof(_TStrm);
    ap_uint<_WAxi> fst_vec = vec_strm.read();
    int fst_n = (scal_vec - offset_AL) > nrow ? (nrow + offset_AL) : scal_vec;

SPLIT_FEW_VEC:
    for (int j = 0; j < scal_vec; ++j) {
#pragma HLS loop_tripcount min = 1 max = 1
#pragma HLS PIPELINE II = 1
        ap_uint<WStrm> fst_r0 = fst_vec.range(WStrm * (j + 1) - 1, WStrm * j);
        if (j < fst_n && j >= offset_AL) {
            r_strm.write((_TStrm)fst_r0);
            e_strm.write(false);
        }
    }

SPLIT_VEC:
    for (int i = scal_vec - offset_AL; i < nrow; i += scal_vec) {
#pragma HLS loop_tripcount min = 1 max = 1
#pragma HLS PIPELINE II = scal_vec
        ap_uint<_WAxi> vec = vec_strm.read();
        int n = (i + scal_vec) > nrow ? (nrow - i) : scal_vec;

        for (int j = 0; j < scal_vec; ++j) {
#pragma HLS PIPELINE II = 1
            ap_uint<WStrm> r0 = vec.range(WStrm * (j + 1) - 1, WStrm * j);
            if (j < n) {
                r_strm.write((_TStrm)r0);
                e_strm.write(false);
            }
        }
    }
    e_strm.write(true);
}

template <int _WAxi, typename _TStrm, int scal_vec>
void split_vec(hls::stream<ap_uint<_WAxi> >& vec_strm,
               const int nrow,
               const int offset_AL,
               hls::stream<_TStrm>& r_strm) {
    const int WStrm = 8 * sizeof(_TStrm);
    ap_uint<_WAxi> fst_vec = vec_strm.read();
    int fst_n = (scal_vec - offset_AL) > nrow ? (nrow + offset_AL) : scal_vec;

SPLIT_FEW_VEC:
    for (int j = 0; j < scal_vec; ++j) {
#pragma HLS loop_tripcount min = 1 max = 1
#pragma HLS PIPELINE II = 1
        ap_uint<WStrm> fst_r0 = fst_vec.range(_WAxi - 1, WStrm * j);
        if (j < fst_n && j >= offset_AL) {
            r_strm.write((_TStrm)fst_r0);
        }
    }

SPLIT_VEC:
    for (int i = scal_vec - offset_AL; i < nrow; i += scal_vec) {
#pragma HLS loop_tripcount min = 1 max = 1
#pragma HLS PIPELINE II = scal_vec
        ap_uint<_WAxi> vec = vec_strm.read();
        int n = (i + scal_vec) > nrow ? (nrow - i) : scal_vec;

        for (int j = 0; j < scal_vec; ++j) {
            ap_uint<WStrm> r0 = vec.range(_WAxi - 1, WStrm * j);
            if (j < n) {
                r_strm.write((_TStrm)r0);
            }
        }
    }
}

template <int _WAxi, int _BurstLen>
void read_to_vec(ap_uint<_WAxi>* vec_ptr,
                 const int len,
                 const int scal_char,
                 const int offset,
                 hls::stream<ap_uint<_WAxi> >& vec_strm) {
    const int nread = (len + offset + scal_char - 1) / scal_char;

READ_TO_VEC:
    for (int i = 0; i < nread; ++i) {
#pragma HLS loop_tripcount min = 1 max = 1
#pragma HLS PIPELINE II = 1
        vec_strm.write(vec_ptr[i]);
    } // This pipeline must be no judgment, otherwise the tool will not be able
      // to derive the correct burst_len
}

template <int _WAxi, typename _TStrm, int scal_vec>
void split_vec_to_aligned(hls::stream<ap_uint<_WAxi> >& vec_strm,
                          const int len,
                          const int scal_char,
                          const int offset,
                          hls::stream<_TStrm>& r_strm,
                          hls::stream<bool>& e_strm) {
    const int nread = (len + offset + scal_char - 1) / scal_char;
    // n read times except the first read, n_read+1 = total read times
    int cnt_r = nread - 1;
    const int nwrite = (len + sizeof(_TStrm) - 1) / sizeof(_TStrm);
    const int WStrm = 8 * sizeof(_TStrm);
    // first read is specific
    ap_uint<_WAxi> vec_reg = vec_strm.read();
    ap_uint<_WAxi> vec_aligned = 0;

    if (offset) {
    LOOP_SPLIT_VEC_TO_ALIGNED:
        for (int i = 0; i < nwrite; i += scal_vec) {
#pragma HLS loop_tripcount min = 1 max = 1
#pragma HLS PIPELINE II = scal_vec
            vec_aligned(((scal_char - offset) << 3) - 1, 0) = vec_reg((scal_char << 3) - 1, offset << 3);
            if ((scal_char - offset) < len && (cnt_r != 0)) { // always need read
                                                              // again
                ap_uint<_WAxi> vec = vec_strm.read();
                vec_aligned((scal_char << 3) - 1, (scal_char - offset) << 3) = vec(offset << 3, 0);
                vec_reg((scal_char << 3) - 1, offset << 3) = vec((scal_char << 3) - 1, offset << 3);
                cnt_r--;
            } // else few cases no read again
            int n = (i + scal_vec) > nwrite ? (nwrite - i) : scal_vec;
            for (int j = 0; j < scal_vec; ++j) {
#pragma HLS PIPELINE II = 1
                ap_uint<WStrm> r0 = vec_aligned.range(WStrm * (j + 1) - 1, WStrm * j);
                if (j < n) {
                    r_strm.write((_TStrm)r0);
                    e_strm.write(false);
                } // end if
            }
        } // end loop
    }

    if (!offset) {
    // no read
    SPLIT_VEC:
        int fst_n = scal_vec > nwrite ? nwrite : scal_vec;
        for (int j = 0; j < scal_vec; ++j) {
#pragma HLS PIPELINE II = 1
            ap_uint<WStrm> r0 = vec_reg.range(WStrm * (j + 1) - 1, WStrm * j);
            if (j < fst_n) {
                r_strm.write((_TStrm)r0);
                e_strm.write(false);
            }
        }

        for (int i = scal_vec; i < nwrite; i += scal_vec) {
#pragma HLS loop_tripcount min = 1 max = 1
#pragma HLS PIPELINE II = scal_vec
            ap_uint<_WAxi> vec = vec_strm.read();
            int n = (i + scal_vec) > nwrite ? (nwrite - i) : scal_vec;

            for (int j = 0; j < scal_vec; ++j) {
#pragma HLS PIPELINE II = 1
                ap_uint<WStrm> r0 = vec.range(WStrm * (j + 1) - 1, WStrm * j);
                if (j < n) {
                    r_strm.write((_TStrm)r0);
                    e_strm.write(false);
                }
            }
        }
    }
    e_strm.write(true);
}

template <int _WAxi, typename _TStrm, int scal_vec>
void split_vec_to_aligned(hls::stream<ap_uint<_WAxi> >& vec_strm,
                          const int len,
                          const int scal_char,
                          const int offset,
                          hls::stream<_TStrm>& r_strm) {
    const int nread = (len + offset + scal_char - 1) / scal_char;
    // n read times except the first read, n_read+1 = total read times
    int cnt_r = nread - 1;
    const int nwrite = (len + sizeof(_TStrm) - 1) / sizeof(_TStrm);
    const int WStrm = 8 * sizeof(_TStrm);
    // first read is specific
    ap_uint<_WAxi> vec_reg = vec_strm.read();
    ap_uint<_WAxi> vec_aligned = 0;

    if (offset) {
    LOOP_SPLIT_VEC_TO_ALIGNED:
        for (int i = 0; i < nwrite; i += scal_vec) {
#pragma HLS loop_tripcount min = 1 max = 1
#pragma HLS PIPELINE II = scal_vec
            vec_aligned(((scal_char - offset) << 3) - 1, 0) = vec_reg((scal_char << 3) - 1, offset << 3);
            if ((scal_char - offset) < len && (cnt_r != 0)) { // always need read
                                                              // again
                ap_uint<_WAxi> vec = vec_strm.read();
                vec_aligned((scal_char << 3) - 1, (scal_char - offset) << 3) = vec(offset << 3, 0);
                vec_reg((scal_char << 3) - 1, offset << 3) = vec((scal_char << 3) - 1, offset << 3);
                cnt_r--;
            } // else few cases no read again
            int n = (i + scal_vec) > nwrite ? (nwrite - i) : scal_vec;
            for (int j = 0; j < scal_vec; ++j) {
#pragma HLS PIPELINE II = 1
                ap_uint<WStrm> r0 = vec_aligned.range(WStrm * (j + 1) - 1, WStrm * j);
                if (j < n) {
                    r_strm.write((_TStrm)r0);
                } // end if
            }
        } // end loop
    }

    if (!offset) {
    // no read
    SPLIT_VEC:
        int fst_n = scal_vec > nwrite ? nwrite : scal_vec;
        for (int j = 0; j < scal_vec; ++j) {
#pragma HLS PIPELINE II = 1
            ap_uint<WStrm> r0 = vec_reg.range(WStrm * (j + 1) - 1, WStrm * j);
            if (j < fst_n) {
                r_strm.write((_TStrm)r0);
            }
        }

        for (int i = scal_vec; i < nwrite; i += scal_vec) {
#pragma HLS loop_tripcount min = 1 max = 1
#pragma HLS PIPELINE II = scal_vec
            ap_uint<_WAxi> vec = vec_strm.read();
            int n = (i + scal_vec) > nwrite ? (nwrite - i) : scal_vec;

            for (int j = 0; j < scal_vec; ++j) {
#pragma HLS PIPELINE II = 1
                ap_uint<WStrm> r0 = vec.range(WStrm * (j + 1) - 1, WStrm * j);
                if (j < n) {
                    r_strm.write((_TStrm)r0);
                }
            }
        }
    }
}

} // namespace details

template <int _BurstLen, int _WAxi, typename _TStrm>
void axiToStream(ap_uint<_WAxi>* rbuf, const int num, hls::stream<_TStrm>& ostrm, hls::stream<bool>& e_ostrm) {
    XF_UTILS_HW_STATIC_ASSERT(_WAxi % sizeof(_TStrm) == 0, "AXI port width is not multiple of stream element width.");
    XF_UTILS_HW_STATIC_ASSERT((_WAxi == 8) || (_WAxi == 16) || (_WAxi == 32) || (_WAxi == 64) || (_WAxi == 128) ||
                                  (_WAxi == 256) || (_WAxi == 512) || (_WAxi == 1024),
                              "AXI port width must be power of 2 and between 8 to 1024.");

#pragma HLS DATAFLOW
    static const int fifo_depth = _BurstLen * 2;
    static const int size0 = sizeof(_TStrm);
    static const int scal_vec = _WAxi / (8 * size0);
    static const int scal_char = _WAxi / 8;

    hls::stream<ap_uint<_WAxi> > vec_strm;
#pragma HLS RESOURCE variable = vec_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = vec_strm depth = fifo_depth

    details::read_to_vec<_WAxi, _BurstLen>(rbuf, num, scal_vec, vec_strm);

    details::split_vec<_WAxi, _TStrm, scal_vec>(vec_strm, num, 0, ostrm, e_ostrm);
}

template <int _BurstLen, int _WAxi, typename _TStrm>
void axiToCharStream(ap_uint<_WAxi>* rbuf,
                     hls::stream<_TStrm>& ostrm,
                     hls::stream<bool>& e_ostrm,
                     const int len,
                     const int offset /* = 0 in decl */) {
    XF_UTILS_HW_STATIC_ASSERT(_WAxi % sizeof(_TStrm) == 0, "AXI port width is not multiple of stream element width.");
    XF_UTILS_HW_STATIC_ASSERT((_WAxi == 8) || (_WAxi == 16) || (_WAxi == 32) || (_WAxi == 64) || (_WAxi == 128) ||
                                  (_WAxi == 256) || (_WAxi == 512) || (_WAxi == 1024),
                              "AXI port width must be power of 2 and between 8 to 1024.");
#pragma HLS DATAFLOW
    static const int fifo_depth = _BurstLen * 2;
    static const int size0 = sizeof(_TStrm);
    static const int scal_vec = _WAxi / (8 * size0);
    static const int scal_char = _WAxi / 8;

    hls::stream<ap_uint<_WAxi> > vec_strm;
#pragma HLS RESOURCE variable = vec_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = vec_strm depth = fifo_depth

    details::read_to_vec<_WAxi, _BurstLen>(rbuf, len, scal_char, offset, vec_strm);

    details::split_vec_to_aligned<_WAxi, _TStrm, scal_vec>(vec_strm, len, scal_char, offset, ostrm, e_ostrm);
}

template <int _BurstLen, int _WAxi, typename _TStrm>
void axiToStream(ap_uint<_WAxi>* rbuf, const int num, hls::stream<_TStrm>& ostrm) {
#pragma HLS DATAFLOW
    static const int fifo_depth = _BurstLen * 2;
    static const int size0 = sizeof(_TStrm);
    // static const int size0 = _WAxi / 8;
    static const int scal_vec = _WAxi / (8 * size0);
    static const int scal_char = _WAxi / 8;

    hls::stream<ap_uint<_WAxi> > vec_strm;
#pragma HLS RESOURCE variable = vec_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = vec_strm depth = fifo_depth

    details::read_to_vec<_WAxi, _BurstLen>(rbuf, num, scal_vec, vec_strm);

    details::split_vec<_WAxi, _TStrm, scal_vec>(vec_strm, num, 0, ostrm);
}

} // namespace utils_hw
} // namespace common
} // namespace xf

#endif // XF_UTILS_HW_AXI_TO_STRM_H
