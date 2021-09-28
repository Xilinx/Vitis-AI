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
 * @file L2_utils.hpp
 * @brief This file contains the utilities for L2.
 */

#ifndef _XF_GRAPH_L2_UTILS_HPP_
#define _XF_GRAPH_L2_UTILS_HPP_

#include "hls_math.h"
#include <hls_stream.h>
#include <ap_int.h>

#ifndef __SYNTHESIS__
#include <iostream>
#endif

namespace xf {
namespace graph {
namespace internal {

template <typename uint512>
void burstRead2Strm(int len, uint512* inArr, hls::stream<uint512>& outStrm) {
    for (int i = 0; i < len; i++) {
#pragma HLS loop_tripcount min = 1000 max = 1000
#pragma HLS pipeline ii = 1
        outStrm.write(inArr[i]);
    }
}

template <typename DT>
void writeDDRByAddr(int len, hls::stream<DT>& addrStrm, hls::stream<DT>& dataStrm, DT* outArr) {
    for (int i = 0; i < len; i++) {
#pragma HLS loop_tripcount min = 1000 max = 1000
#pragma HLS pipeline ii = 1
        outArr[addrStrm.read()] = dataStrm.read();
    }
}

template <typename uint512>
void burstWrite2Strm(int len, hls::stream<uint512>& inStrm, uint512* outArr) {
    for (int i = 0; i < len; i++) {
#pragma HLS loop_tripcount min = 1000 max = 1000
#pragma HLS pipeline ii = 1
        outArr[i] = inStrm.read();
    }
}
template <typename DT, typename uint512, int K, int W>
void splitStrm(int len, int beginAddr, hls::stream<uint512>& inStrm, hls::stream<DT>& outStrm) {
    uint512 tmp;
    for (int i = beginAddr; i < len; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount min = 10000 max = 10000
        if (i % K == 0 || i == beginAddr) {
            tmp = inStrm.read();
        }
        DT val = tmp.range(i % K * W + W - 1, i % K * W);
        outStrm.write(val);
        // std::cout << "outArr[" << i << "]=" << val << std::endl;
    }
}

template <typename DT, typename uint512, int K, int W>
void burstReadSplit2Strm(int len, int beginAddr, uint512* inArr, hls::stream<DT>& outStrm) {
#pragma HLS dataflow
    hls::stream<uint512> arrStrm("arrStrm");
#pragma HLS stream variable = arrStrm depth = 16
#pragma HLS resource variable = arrStrm core = FIFO_LUTRAM
    burstRead2Strm<uint512>((len + K - 1) / K, inArr, arrStrm);
    splitStrm<DT, uint512, K, W>(len, beginAddr, arrStrm, outStrm);
}

template <typename DT, typename uint512, int K, int W>
void addrReadData(int len, uint512* inArr, hls::stream<DT>& addrStrm, hls::stream<DT>& dataStrm) {
    int indexPre = -1;
    uint512 arr;
    for (int i = 0; i < len; i++) {
#pragma HLS loop_tripcount max = 20000 min = 20000
#pragma HLS pipeline ii = 1
        DT addr = addrStrm.read();
        int index = addr / K;
        int offset = addr % K;
        if (indexPre != index) {
            arr = inArr[index];
            index = indexPre;
        }
        DT data = arr.range(offset * W + W - 1, offset * W);
        dataStrm.write(data);
    }
}

template <typename DT, typename DT2, int K, int W>
void combineStrm(int len, hls::stream<DT>& inStrm, hls::stream<DT2>& outStrm) {
    DT2 tmp;
    int offset = 0;
    for (int i = 0; i < len; i++) {
#pragma HLS loop_tripcount max = 1000 min = 1000
#pragma HLS pipeline ii = 1
        DT indata = inStrm.read();
        offset = i % K;
        tmp.range(offset * W + W - 1, offset * W) = indata;
        if (offset + 1 == K) outStrm.write(tmp);
    }
    if (offset + 1 != K) outStrm.write(tmp);
}

template <int _WAxi, int _BurstLen>
void read_to_vec(ap_uint<_WAxi>* vec_ptr,
                 const int len,
                 const int scal_char,
                 const int offset,
                 hls::stream<ap_uint<_WAxi> >& vec_strm) {
    const int nread = (len + offset + scal_char - 1) / scal_char;

READ_TO_VEC:
    for (int i = 0; i < nread; i += _BurstLen) {
#pragma HLS loop_tripcount min = 1 max = 1
        //#pragma HLS PIPELINE II = _BurstLen
        int len1 = ((i + _BurstLen) > nread) ? (nread - i) : _BurstLen;

    READ_VEC0:
        for (int j = 0; j < len1; ++j) {
#pragma HLS loop_tripcount min = len max = len
#pragma HLS PIPELINE II = 1
            vec_strm.write(vec_ptr[i + j]);
        } // This pipeline must be no judgment, otherwise the tool will not be able
        // to derive the correct burst_len
    }
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
            vec_aligned((scal_char - offset << 3) - 1, 0) = vec_reg((scal_char << 3) - 1, offset << 3);
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
template <int _BurstLen = 32, int _WAxi, typename _TStrm>
void axiToCharStream(ap_uint<_WAxi>* rbuf, hls::stream<_TStrm>& ostrm, const int len, const int offset = 0) {
#pragma HLS DATAFLOW
    static const int fifo_depth = _BurstLen * 2;
    static const int size0 = sizeof(_TStrm);
    static const int scal_vec = _WAxi / (8 * size0);
    static const int scal_char = _WAxi / 8;

    hls::stream<ap_uint<_WAxi> > vec_strm;
#pragma HLS RESOURCE variable = vec_strm core = FIFO_LUTRAM
#pragma HLS STREAM variable = vec_strm depth = fifo_depth

    read_to_vec<_WAxi, _BurstLen>(rbuf, len, scal_char, offset, vec_strm);

    split_vec_to_aligned<_WAxi, _TStrm, scal_vec>(vec_strm, len, scal_char, offset, ostrm);
}

} // internal
} // graph
} // xf
#endif
