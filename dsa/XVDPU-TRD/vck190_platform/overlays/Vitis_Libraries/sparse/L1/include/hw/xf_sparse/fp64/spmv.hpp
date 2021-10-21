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
 * @file spmv.hpp
 * @brief SPARSE Level 1 template function implementation for double precision.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_SPMV_HPP
#define XF_SPARSE_SPMV_HPP

#ifndef __cplusplus
#error "SPARSE Library only works with C++."
#endif

#include <cstdint>
#include "ap_int.h"
#include "hls_math.h"
#include "hls_stream.h"
#include "xf_blas.hpp"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

using namespace xf::blas;

namespace xf {
namespace sparse {

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxCols,
          unsigned int t_ParEntries,
          unsigned int t_DataBits = 64,
          unsigned int t_IndexBits = 16>
void selVecX(const unsigned int p_cols,
             const unsigned int p_nnzs,
             hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_xStr,
             hls::stream<ap_uint<t_IndexBits> >& p_idxStr,
             hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_x4NnzStr) {
#pragma HLS INLINE
    constexpr unsigned int t_ColBlocks = t_MaxCols / t_ParEntries;
#ifndef __SYNTHESIS__
    assert(t_MaxCols % t_ParEntries == 0);
    assert(p_cols % t_ParEntries == 0);
    assert(p_nnzs % t_ParEntries == 0);
#endif
    t_DataType l_xStore[t_ColBlocks][t_ParEntries];
#pragma HLS ARRAY_PARTITION variable = l_xStore complete dim = 2
    unsigned int l_colBlocks = p_cols / t_ParEntries;
    unsigned int l_nnzBlocks = p_nnzs / t_ParEntries;

    for (unsigned int i = 0; i < l_colBlocks; ++i) {
#pragma HLS PIPELINE
        WideType<t_DataType, t_ParEntries> l_val = p_xStr.read();
#pragma HLS ARRAY_PARTITION variable = l_val complete
        for (unsigned int j = 0; j < t_ParEntries; ++j) {
            l_xStore[i][j] = l_val[j];
        }
    }
    for (unsigned int i = 0; i < l_nnzBlocks; ++i) {
#pragma HLS PIPELINE
        t_IndexType l_idx = p_idxStr.read();
        WideType<t_DataType, t_ParEntries> l_val;
#pragma HLS ARRAY_PARTITION variable = l_val complete
        for (unsigned int j = 0; j < t_ParEntries; ++j) {
            l_val[j] = l_xStore[l_idx][j];
        }
        p_x4NnzStr.write(l_val);
    }
}

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxCols,
          unsigned int t_ParEntries,
          unsigned int t_DataBits = 64,
          unsigned int t_IndexBits = 16>
void selVecXstr(hls::stream<uint32_t>& p_inParamStr,
                hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_xStr,
                hls::stream<ap_uint<t_IndexBits> >& p_idxStr,
                hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_x4NnzStr) {
    unsigned int l_cols = p_inParamStr.read();
    unsigned int l_nnzs = p_inParamStr.read();
    selVecX<t_DataType, t_IndexType, t_MaxCols, t_ParEntries, t_DataBits, t_IndexBits>(l_cols, l_nnzs, p_xStr, p_idxStr,
                                                                                       p_x4NnzStr);
}

template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_DataBits>
void multX(const unsigned int p_nnzs,
           hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_nnzStr,
           hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_xStr,
           hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_outDatStr) {
#pragma HLS INLINE

    unsigned int l_nnzBlocks = p_nnzs / t_ParEntries;

LOOP_MULX:
    for (unsigned int i = 0; i < l_nnzBlocks; ++i) {
#pragma HLS PIPELINE
        WideType<t_DataType, t_ParEntries> l_nnzVal = p_nnzStr.read();
        WideType<t_DataType, t_ParEntries> l_xVal = p_xStr.read();
        WideType<t_DataType, t_ParEntries> l_outVal;
        for (unsigned int k = 0; k < t_ParEntries; ++k) {
            l_outVal[k] = l_nnzVal[k] * l_xVal[k];
        }
        p_outDatStr.write(l_outVal);
    }
}

template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_DataBits>
void multXstr(hls::stream<uint32_t>& p_inParamStr,
              hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_nnzStr,
              hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_xStr,
              hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_outDatStr) {
    unsigned int l_nnzs = p_inParamStr.read();
    multX<t_DataType, t_ParEntries, t_DataBits>(l_nnzs, p_nnzStr, p_xStr, p_outDatStr);
}

template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_DataBits>
void multAddX(const unsigned int p_nnzBks,
              hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_nnzStr,
              hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_xStr,
              hls::stream<ap_uint<t_DataBits> >& p_outDatStr) {
LOOP_MULX:
    for (unsigned int i = 0; i < p_nnzBks; ++i) {
#pragma HLS PIPELINE
        WideType<t_DataType, t_ParEntries> l_nnzVal = p_nnzStr.read();
        WideType<t_DataType, t_ParEntries> l_xVal = p_xStr.read();
        WideType<t_DataType, t_ParEntries> l_outVal;
        for (unsigned int k = 0; k < t_ParEntries; ++k) {
            l_outVal[k] = l_nnzVal[k] * l_xVal[k];
        }
        t_DataType l_sum = 0;
        for (unsigned int k = 0; k < t_ParEntries; ++k) {
            l_sum += l_outVal[k];
        }
        ap_uint<t_DataBits> l_sumBits = *reinterpret_cast<ap_uint<t_DataBits>*>(&l_sum);
        p_outDatStr.write(l_sumBits);
    }
}

template <typename t_DataType, unsigned int t_DataBits = 64, unsigned int t_Latency = 8>
void regAcc(const unsigned int p_parBlocks,
            hls::stream<ap_uint<t_DataBits> >& p_entryStr,
            hls::stream<ap_uint<t_DataBits> >& p_accValStr) {
    unsigned int l_macBlocks = p_parBlocks / t_Latency;

regAccLoop:
    for (unsigned int i = 0; i < l_macBlocks; ++i) {
#pragma HLS PIPELINE II = t_Latency
        t_DataType l_accReg[t_Latency];
#pragma HLS ARRAY_PARTITION variable = l_accReg complete dim = 1
        for (unsigned int j = 0; j < t_Latency; ++j) {
            ap_uint<t_DataBits> l_val = p_entryStr.read();
            l_accReg[j] = *reinterpret_cast<t_DataType*>(&l_val);
        }
        t_DataType l_sum = 0;
        for (unsigned int j = 0; j < t_Latency; ++j) {
#pragma HLS UNROLL
            l_sum += l_accReg[j];
        }
        ap_uint<t_DataBits> l_sumBits = *reinterpret_cast<ap_uint<t_DataBits>*>(&l_sum);
        p_accValStr.write(l_sumBits);
    }
}

template <typename t_DataType,
          unsigned int t_ParEntries,
          unsigned int t_Latency,
          unsigned int t_DataBits,
          unsigned int t_IndexBits>
void multAddXstr(hls::stream<uint32_t>& p_inParamStr,
                 hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_nnzStr,
                 hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_xStr,
                 hls::stream<ap_uint<t_DataBits> >& p_outDatStr) {
    hls::stream<ap_uint<t_DataBits> > l_datStr;
#pragma HLS STREAM variable = l_datStr depth = 16
    unsigned int l_nnzBks = p_inParamStr.read();
#pragma HLS DATAFLOW
    multAddX<t_DataType, t_ParEntries, t_DataBits>(l_nnzBks, p_nnzStr, p_xStr, l_datStr);
    regAcc<t_DataType, t_DataBits, t_Latency>(l_nnzBks, l_datStr, p_outDatStr);
}

template <unsigned int t_ParEntries, unsigned int t_DataBits>
void fwdNnzStr(hls::stream<uint32_t>& p_inParamStr,
               hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_inNnzStr,
               hls::stream<uint32_t>& p_outParamStr,
               hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_outNnzStr) {
    unsigned int l_nnzBks = p_inParamStr.read();
    p_outParamStr.write(l_nnzBks);
    xf::blas::duplicateStream<1, ap_uint<t_DataBits * t_ParEntries> >(l_nnzBks, p_inNnzStr, &p_outNnzStr);
}

template <unsigned int t_ParEntries, unsigned int t_Latency, unsigned int t_DataBits, unsigned int t_IndexBits>
void splitValColIdx(hls::stream<uint32_t>& p_inParamStr,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_nnzStr,
                    hls::stream<uint32_t>& p_outParamStr1,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_wideColIdxStr,
                    hls::stream<uint32_t>& p_outParamStr2,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_valRowIdxStr) {
    constexpr unsigned int t_ParIdx = t_DataBits * t_ParEntries / t_IndexBits;
    constexpr unsigned int t_RowBlocks = t_ParIdx * t_Latency;

    unsigned int l_cols = p_inParamStr.read();
    unsigned int l_nnzs = p_inParamStr.read();
    p_outParamStr1.write(l_cols);
    p_outParamStr1.write(l_nnzs);

    unsigned int l_nnzBks = l_nnzs / t_ParEntries;
    p_outParamStr2.write(l_nnzBks);
    unsigned int l_count = 0;
    bool l_isRowIdx = true;
    bool l_isColIdx = true;
    while (l_count < l_nnzBks) {
#pragma HLS PIPELINE
        ap_uint<t_DataBits* t_ParEntries> l_val = p_nnzStr.read();
        if (l_isRowIdx && (l_count % t_RowBlocks == 0)) {
            p_valRowIdxStr.write(l_val);
            l_isRowIdx = false;
        } else if (l_isColIdx && (l_count % t_ParIdx == 0)) {
            p_wideColIdxStr.write(l_val);
            l_isColIdx = false;
        } else {
            p_valRowIdxStr.write(l_val);
            l_isRowIdx = true;
            l_isColIdx = true;
            l_count++;
        }
    }
}

template <typename t_IndexType, unsigned int t_ParIdx, unsigned int t_DataBits, unsigned int t_IndexBits>
void shiftIdx(const uint32_t p_bks,
              hls::stream<ap_uint<t_DataBits> >& p_wideIdxStr,
              hls::stream<ap_uint<t_IndexBits> >& p_idxStr) {
#pragma HLS INLINE
    unsigned int l_idxBks = p_bks / t_ParIdx;
    uint8_t l_res = p_bks % t_ParIdx;

    for (unsigned int i = 0; i < l_idxBks; ++i) {
#pragma HLS PIPELINE II = t_ParIdx
        WideType<t_IndexType, t_ParIdx> l_idxArr = p_wideIdxStr.read();
#pragma HLS ARRAY_PARTITION variable = l_idxArr complete dim = 1
        for (unsigned int j = 0; j < t_ParIdx; ++j) {
            p_idxStr.write(l_idxArr[j]);
        }
    }

    if (l_res != 0) {
        WideType<t_IndexType, t_ParIdx> l_idxArr = p_wideIdxStr.read();
#pragma HLS ARRAY_PARTITION variable = l_idxArr complete dim = 1
        for (unsigned int i = 0; i < l_res; ++i) {
#pragma HLS PIPELINE
            p_idxStr.write(l_idxArr[i]);
        }
    }
}

template <typename t_IndexType, unsigned int t_ParEntries, unsigned int t_DataBits, unsigned int t_IndexBits>
void shiftColIdx(hls::stream<uint32_t>& p_inParamStr,
                 hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_wideColIdxStr,
                 hls::stream<uint32_t>& p_outParamStr,
                 hls::stream<ap_uint<t_IndexBits> >& p_colIdxStr) {
    constexpr unsigned int t_ParIdx = t_DataBits * t_ParEntries / t_IndexBits;

    unsigned int l_cols = p_inParamStr.read();
    unsigned int l_nnzs = p_inParamStr.read();
    p_outParamStr.write(l_cols);
    p_outParamStr.write(l_nnzs);

    unsigned int l_nnzBks = l_nnzs / t_ParEntries;
    shiftIdx<t_IndexType, t_ParIdx, t_DataBits * t_ParEntries, t_IndexBits>(l_nnzBks, p_wideColIdxStr, p_colIdxStr);
}

template <unsigned int t_ParEntries, unsigned int t_Latency, unsigned int t_DataBits, unsigned int t_IndexBits>
void splitValRowIdx(const uint32_t p_nnzBks,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_nnzStr,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_wideRowIdxStr,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_valStr) {
#pragma HLS INLINE
    constexpr unsigned int t_ParIdx = t_DataBits * t_ParEntries / t_IndexBits;
    constexpr unsigned int t_RowBlocks = t_ParIdx * t_Latency;

    unsigned int l_count = 0;
    bool l_isRowIdx = true;
    while (l_count < p_nnzBks) {
#pragma HLS PIPELINE
        ap_uint<t_DataBits* t_ParEntries> l_val = p_nnzStr.read();
        if (l_isRowIdx && (l_count % t_RowBlocks == 0)) {
            p_wideRowIdxStr.write(l_val);
            l_isRowIdx = false;
        } else {
            p_valStr.write(l_val);
            l_isRowIdx = true;
            l_count++;
        }
    }
}

template <unsigned int t_ParEntries, unsigned int t_Latency, unsigned int t_DataBits, unsigned int t_IndexBits>
void splitValRowIdxStr(hls::stream<uint32_t>& p_inParamStr,
                       hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_nnzStr,
                       hls::stream<uint32_t>& p_outParamStr1,
                       hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_wideRowIdxStr,
                       hls::stream<uint32_t>& p_outParamStr2,
                       hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_valStr) {
    unsigned int l_nnzBks = p_inParamStr.read();
    p_outParamStr1.write(l_nnzBks);
    p_outParamStr2.write(l_nnzBks);

    splitValRowIdx<t_ParEntries, t_Latency, t_DataBits, t_IndexBits>(l_nnzBks, p_nnzStr, p_wideRowIdxStr, p_valStr);
}

template <typename t_IndexType,
          unsigned int t_ParEntries,
          unsigned int t_Latency,
          unsigned int t_DataBits,
          unsigned int t_IndexBits>
void shiftRowIdx(hls::stream<uint32_t>& p_inParamStr,
                 hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_wideRowIdxStr,
                 hls::stream<ap_uint<t_IndexBits> >& p_rowIdxStr) {
    constexpr unsigned int t_ParIdx = t_DataBits * t_ParEntries / t_IndexBits;

    unsigned int l_nnzBks = p_inParamStr.read();
    unsigned int l_bks = l_nnzBks / t_Latency;
    shiftIdx<t_IndexType, t_ParIdx, t_DataBits * t_ParEntries, t_IndexBits>(l_bks, p_wideRowIdxStr, p_rowIdxStr);
}

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxCols,
          unsigned int t_ParEntries,
          unsigned int t_Latency,
          unsigned int t_DataBits,
          unsigned int t_IndexBits>
void selMultAddX(hls::stream<uint32_t>& p_inParamStr,
                 hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_inXstr,
                 hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_nnzStr,
                 hls::stream<ap_uint<t_DataBits> >& p_outDatStr,
                 hls::stream<ap_uint<t_IndexBits> >& p_outIdxStr) {
#pragma HLS DATAFLOW
    hls::stream<uint32_t> l_paramStr1, l_paramStr2, l_paramStr3, l_paramStr4, l_paramStr5, l_paramStr6;
    hls::stream<ap_uint<t_DataBits * t_ParEntries> > l_valRowIdxStr, l_wideColIdxStr, l_wideRowIdxStr;
    hls::stream<ap_uint<t_IndexBits> > l_colIdxStr;
    hls::stream<ap_uint<t_DataBits * t_ParEntries> > l_xStr, l_nnzStr, l_nnzStr1;
#pragma HLS STREAM variable = l_paramStr1 depth = 32
#pragma HLS STREAM variable = l_paramStr2 depth = 32
#pragma HLS STREAM variable = l_paramStr3 depth = 32
#pragma HLS STREAM variable = l_paramStr4 depth = 32
#pragma HLS STREAM variable = l_wideColIdxStr depth = 32
#pragma HLS STREAM variable = l_valRowIdxStr depth = 32
#pragma HLS STREAM variable = l_colIdxStr depth = 32
#pragma HLS STREAM variable = l_wideRowIdxStr depth = 32
#pragma HLS STREAM variable = l_nnzStr depth = 32

    splitValColIdx<t_ParEntries, t_Latency, t_DataBits, t_IndexBits>(p_inParamStr, p_nnzStr, l_paramStr1,
                                                                     l_wideColIdxStr, l_paramStr2, l_valRowIdxStr);
    shiftColIdx<t_IndexType, t_ParEntries, t_DataBits, t_IndexBits>(l_paramStr1, l_wideColIdxStr, l_paramStr3,
                                                                    l_colIdxStr);
    splitValRowIdxStr<t_ParEntries, t_Latency, t_DataBits, t_IndexBits>(l_paramStr2, l_valRowIdxStr, l_paramStr4,
                                                                        l_wideRowIdxStr, l_paramStr5, l_nnzStr);
    selVecXstr<t_DataType, t_IndexType, t_MaxCols, t_ParEntries, t_DataBits, t_IndexBits>(l_paramStr3, p_inXstr,
                                                                                          l_colIdxStr, l_xStr);
    shiftRowIdx<t_IndexType, t_ParEntries, t_Latency, t_DataBits, t_IndexBits>(l_paramStr4, l_wideRowIdxStr,
                                                                               p_outIdxStr);
    fwdNnzStr<t_ParEntries, t_DataBits>(l_paramStr5, l_nnzStr, l_paramStr6, l_nnzStr1);
    multAddXstr<t_DataType, t_ParEntries, t_Latency, t_DataBits, t_IndexBits>(l_paramStr6, l_nnzStr1, l_xStr,
                                                                              p_outDatStr);
}

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_ParEntries,
          unsigned int t_DataBits,
          unsigned int t_IndexBits,
          unsigned int t_Latency>
void splitDatIdxStr(hls::stream<uint32_t>& p_inParamStr,
                    hls::stream<ap_uint<t_DataBits> >& p_inDatStr,
                    hls::stream<ap_uint<t_IndexBits> >& p_inIdxStr,
                    hls::stream<ap_uint<t_DataBits> >& p_leftDatStr,
                    hls::stream<ap_uint<t_IndexBits> >& p_leftIdxStr,
                    hls::stream<ap_uint<t_DataBits> >& p_rightDatStr,
                    hls::stream<ap_uint<t_IndexBits> >& p_rightIdxStr) {
    constexpr unsigned int t_ParamEntries = t_DataBits / 16;
#ifndef __SYNTHESIS__
    assert(t_ParamEntries > 2);
#endif

    unsigned int l_numRbs = p_inParamStr.read();
    unsigned int l_leftNumRbs = l_numRbs / 2 + l_numRbs % 2;
    unsigned int l_rightNumRbs = l_numRbs / 2;
    p_leftDatStr.write(l_leftNumRbs);
    p_rightDatStr.write(l_rightNumRbs);

    for (unsigned int r = 0; r < l_numRbs; ++r) {
        WideType<uint16_t, t_ParamEntries> l_param;
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1

        l_param[0] = p_inParamStr.read();
        l_param[1] = p_inParamStr.read();
        l_param[2] = p_inParamStr.read();
        if (r % 2 == 0) {
            p_leftDatStr.write(l_param);
        } else {
            p_rightDatStr.write(l_param);
        }
        unsigned int l_n = p_inParamStr.read();
        unsigned int l_bks = l_n / (t_ParEntries * t_Latency);
        if (r % 2 == 0) {
            p_leftDatStr.write(l_bks);
        } else {
            p_rightDatStr.write(l_bks);
        }
        for (unsigned int i = 0; i < l_bks; ++i) {
#pragma HLS PIPELINE
            ap_uint<t_DataBits> l_val = p_inDatStr.read();
            ap_uint<t_IndexBits> l_idx = p_inIdxStr.read();
            if (r % 2 == 0) {
                p_leftDatStr.write(l_val);
                p_leftIdxStr.write(l_idx);
            } else {
                p_rightDatStr.write(l_val);
                p_rightIdxStr.write(l_idx);
            }
        }
    }
}

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxRows,
          unsigned int t_ParEntries,
          unsigned int t_DataBits = 32,
          unsigned int t_IndexBits = 16,
          unsigned int t_Latency = 8>
void rowMemAcc(const unsigned int p_n,
               const unsigned int p_rows,
               t_DataType p_rowStore[t_MaxRows],
               hls::stream<ap_uint<t_DataBits> >& p_rowEntryStr,
               hls::stream<ap_uint<t_IndexBits> >& p_idxStr,
               hls::stream<ap_uint<t_DataBits> >& p_rowValStr) {
#pragma HLS INLINE

accumulate:
    for (unsigned int i = 0; i < p_n; ++i) {
#pragma HLS PIPELINE II = t_Latency
        t_IndexType l_idx = p_idxStr.read();
        ap_uint<t_DataBits> l_valBits = p_rowEntryStr.read();
        t_DataType l_val = *reinterpret_cast<t_DataType*>(&l_valBits);
        p_rowStore[l_idx] += l_val;
    }

    for (unsigned int i = 0; i < p_rows; ++i) {
#pragma HLS PIPELINE
        t_DataType l_val = p_rowStore[i];
        p_rowStore[i] = 0;
        ap_uint<t_DataBits> l_valBits = *reinterpret_cast<ap_uint<t_DataBits>*>(&l_val);
        p_rowValStr.write(l_valBits);
    }
}

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxRows,
          unsigned int t_ParEntries,
          unsigned int t_DataBits = 64,
          unsigned int t_IndexBits = 16,
          unsigned int t_Latency = 8>
void rowMemAccStr(hls::stream<ap_uint<t_DataBits> >& p_inDatStr,
                  hls::stream<ap_uint<t_IndexBits> >& p_idxStr,
                  hls::stream<ap_uint<t_DataBits> >& p_outDatStr) {
    constexpr unsigned int t_ParamEntries = t_DataBits / 16;
#ifndef __SYNTHESIS__
    assert(t_ParamEntries > 2);
#endif

    t_DataType l_rowStore[t_MaxRows];
#pragma HLS BIND_STORAGE variable = l_rowStore type = ram_t2p impl = uram

init_mem:
    for (unsigned int i = 0; i < t_MaxRows; ++i) {
#pragma HLS PIPELINE
        l_rowStore[i] = 0;
    }

    unsigned int l_numRbs = p_inDatStr.read();
    p_outDatStr.write(l_numRbs);

    for (unsigned int r = 0; r < l_numRbs; ++r) {
        WideType<uint16_t, t_ParamEntries> l_param;
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1

        l_param = p_inDatStr.read();
        uint16_t l_rbRows = l_param[0];
        uint16_t l_chRbStartId = l_param[1];
        uint16_t l_chRbRows = l_param[2];
        p_outDatStr.write(l_param);
        unsigned int l_n = p_inDatStr.read();
        rowMemAcc<t_DataType, t_IndexType, t_MaxRows, t_ParEntries, t_DataBits, t_IndexBits, t_Latency>(
            l_n, l_chRbRows, l_rowStore, p_inDatStr, p_idxStr, p_outDatStr);
    }
}

template <unsigned int t_DataBits>
void mergeDatStr(hls::stream<ap_uint<t_DataBits> >& p_leftDatStr,
                 hls::stream<ap_uint<t_DataBits> >& p_rightDatStr,
                 hls::stream<ap_uint<t_DataBits> >& p_outDatStr) {
    constexpr unsigned int t_ParamEntries = t_DataBits / 16;
#ifndef __SYNTHESIS__
    assert(t_ParamEntries > 2);
#endif

    unsigned int l_leftNumRbs = p_leftDatStr.read();
    unsigned int l_rightNumRbs = p_rightDatStr.read();
    unsigned int l_numRbs = l_leftNumRbs + l_rightNumRbs;
    p_outDatStr.write(l_numRbs);

    for (unsigned int r = 0; r < l_numRbs; ++r) {
        WideType<uint16_t, t_ParamEntries> l_param;
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1
        if (r % 2 == 0) {
            l_param = p_leftDatStr.read();
        } else {
            l_param = p_rightDatStr.read();
        }
        uint16_t l_rbRows = l_param[0];
        uint16_t l_chRbStartId = l_param[1];
        uint16_t l_chRbRows = l_param[2];
        p_outDatStr.write(l_param);
        for (unsigned int i = 0; i < l_chRbRows; ++i) {
#pragma HLS PIPELINE
            if (r % 2 == 0) {
                p_outDatStr.write(p_leftDatStr.read());
            } else {
                p_outDatStr.write(p_rightDatStr.read());
            }
        }
    }
}

template <typename t_DataType, unsigned int t_DataBits>
void genChRowStr(hls::stream<ap_uint<t_DataBits> >& p_inDatStr, hls::stream<ap_uint<t_DataBits> >& p_outDatStr) {
    constexpr unsigned int t_ParamEntries = t_DataBits / 16;
#ifndef __SYNTHESIS__
    assert(t_ParamEntries > 2);
#endif

    unsigned int l_numRbs = p_inDatStr.read();
    p_outDatStr.write(l_numRbs);

    for (unsigned int r = 0; r < l_numRbs; ++r) {
        WideType<uint16_t, t_ParamEntries> l_param;
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1

        l_param = p_inDatStr.read();
        uint16_t l_rbRows = l_param[0];
        uint16_t l_chRbStartId = l_param[1];
        uint16_t l_chRbRows = l_param[2];
        p_outDatStr.write(l_rbRows);

        t_DataType l_val = 0;
        uint16_t l_chRbMaxRow = l_chRbStartId + l_chRbRows;

        for (unsigned int i = 0; i < l_chRbStartId; ++i) {
#pragma HLS PIPELINE
            p_outDatStr.write(l_val);
        }
        for (unsigned int i = l_chRbStartId; i < l_chRbMaxRow; ++i) {
#pragma HLS PIPELINE
            p_outDatStr.write(p_inDatStr.read());
        }
        for (unsigned int i = l_chRbMaxRow; i < l_rbRows; ++i) {
#pragma HLS PIPELINE
            p_outDatStr.write(l_val);
        }
    }
}

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxRows,
          unsigned int t_ParEntries,
          unsigned int t_DataBits,
          unsigned int t_IndexBits,
          unsigned int t_Latency>
void rowAcc(hls::stream<uint32_t>& p_inParamStr,
            hls::stream<ap_uint<t_DataBits> >& p_inDatStr,
            hls::stream<ap_uint<t_IndexBits> >& p_inIdxStr,
            hls::stream<ap_uint<t_DataBits> >& p_outDatStr) {
    hls::stream<ap_uint<t_DataBits> > l_leftDatStr;
    hls::stream<ap_uint<t_IndexBits> > l_leftIdxStr;
    hls::stream<ap_uint<t_DataBits> > l_rightDatStr;
    hls::stream<ap_uint<t_IndexBits> > l_rightIdxStr;
    hls::stream<ap_uint<t_DataBits> > l_leftRowAccStr;
    hls::stream<ap_uint<t_DataBits> > l_rightRowAccStr;
    hls::stream<ap_uint<t_DataBits> > l_mergeDatStr;
#pragma HLS DATAFLOW
    splitDatIdxStr<t_DataType, t_IndexType, t_ParEntries, t_DataBits, t_IndexBits, t_Latency>(
        p_inParamStr, p_inDatStr, p_inIdxStr, l_leftDatStr, l_leftIdxStr, l_rightDatStr, l_rightIdxStr);
    rowMemAccStr<t_DataType, t_IndexType, t_MaxRows, t_ParEntries, t_DataBits, t_IndexBits, t_Latency>(
        l_leftDatStr, l_leftIdxStr, l_leftRowAccStr);
    rowMemAccStr<t_DataType, t_IndexType, t_MaxRows, t_ParEntries, t_DataBits, t_IndexBits, t_Latency>(
        l_rightDatStr, l_rightIdxStr, l_rightRowAccStr);
    mergeDatStr<t_DataBits>(l_leftRowAccStr, l_rightRowAccStr, l_mergeDatStr);
    genChRowStr<t_DataType, t_DataBits>(l_mergeDatStr, p_outDatStr);
}

} // end namespace sparse
} // end namespace xf

#endif
