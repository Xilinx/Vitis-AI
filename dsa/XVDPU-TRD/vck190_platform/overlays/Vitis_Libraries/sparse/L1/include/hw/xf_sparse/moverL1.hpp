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
 * @file moverL1.hpp
 * @brief SPARSE Level 1 template function implementation for moving and buffering data
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_MOVERL1_HPP
#define XF_SPARSE_MOVERL1_HPP

#ifndef __cplusplus
#error "SPARSE Library only works with C++."
#endif

#include <cstdint>
#include "ap_int.h"
#include "hls_stream.h"
#include "xf_blas.hpp"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

using namespace xf::blas;
namespace xf {
namespace sparse {

template <unsigned int t_MaxColParBlocks,
          unsigned int t_ParBlocks4Param,
          unsigned int t_HbmChannels,
          unsigned int t_ParEntries,
          unsigned int t_DataBits>
void decFwdColParam(uint32_t& p_param0,
                    uint32_t& p_param1,
                    uint32_t p_param1Hbm[t_HbmChannels],
                    uint32_t p_param2Hbm[t_HbmChannels],
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datStr,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datFwdStr) {
    static const unsigned int t_ParBits = t_DataBits * t_ParEntries;
    static const unsigned int t_ParamsPerPar = t_ParBits / 32;

    static const unsigned int t_Width = t_ParBlocks4Param * t_ParamsPerPar;
#pragma HLS INLINE
    WideType<uint32_t, t_Width> l_param;
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1

    for (unsigned int i = 0; i < t_ParBlocks4Param; ++i) {
#pragma HLS PIPELINE rewind
        ap_uint<t_ParBits> l_val = p_datStr.read();
        WideType<uint32_t, t_ParamsPerPar> l_wideVal(l_val);
        for (unsigned int j = 0; j < t_ParamsPerPar; ++j) {
#pragma HLS UNROLL
            l_param[i * t_ParamsPerPar + j] = l_wideVal[j];
        }
        p_datFwdStr.write(l_val);
    }

    p_param0 = l_param[0];
    p_param1 = l_param[1];
    for (unsigned int i = 0; i < t_HbmChannels; ++i) {
#pragma HLS UNROLL
        p_param1Hbm[i] = l_param[2 + i];
        p_param2Hbm[i] = l_param[2 + t_HbmChannels + i];
    }
}

template <unsigned int t_MaxColParBlocks,
          unsigned int t_ParBlocks4Param,
          unsigned int t_HbmChannels,
          unsigned int t_ParEntries,
          unsigned int t_DataBits>
void decColParam(uint32_t& p_param0,
                 uint32_t& p_param1,
                 uint32_t p_param1Hbm[t_HbmChannels],
                 uint32_t p_param2Hbm[t_HbmChannels],
                 hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datStr) {
    static const unsigned int t_ParBits = t_DataBits * t_ParEntries;
    static const unsigned int t_ParamsPerPar = t_ParBits / 32;

    static const unsigned int t_Width = t_ParBlocks4Param * t_ParamsPerPar;
#pragma HLS INLINE
    WideType<uint32_t, t_Width> l_param;
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1

    for (unsigned int i = 0; i < t_ParBlocks4Param; ++i) {
#pragma HLS PIPELINE
        ap_uint<t_ParBits> l_val = p_datStr.read();
        WideType<uint32_t, t_ParamsPerPar> l_wideVal(l_val);
        for (unsigned int j = 0; j < t_ParamsPerPar; ++j) {
            l_param[i * t_ParamsPerPar + j] = l_wideVal[j];
        }
    }

    p_param0 = l_param[0];
    p_param1 = l_param[1];
    for (unsigned int i = 0; i < t_HbmChannels; ++i) {
#pragma HLS PIPELINE
        p_param1Hbm[i] = l_param[2 + i];
        p_param2Hbm[i] = l_param[2 + t_HbmChannels + i];
    }
}

/**
 * @brief dispColVec function that forward and copy input column vector and parameters
 *
 * @tparam t_MaxColParBlocks the maximum number of parallel processed column blocks buffered in on-chip memory
 * @tparam t_ParBlocks4Param the number of parallelly processed parameter blocks
 * @tparam t_HbmChannels number of HBM channels
 * @tparam t_ParEntries parallelly processed entries
 * @tparam t_DataBits number of bits used to store each entry
 *
 * @param t_chId constant HBM channel ID
 * @param p_datStr input vector stream
 * @param p_datFwdStr an forwarded parameter and column vector streams
 * @param p_datOutStr an copied parameter and column vector streams
 */
// parallel cscmv related data movers
template <unsigned int t_MaxColParBlocks,
          unsigned int t_ParBlocks4Param,
          unsigned int t_HbmChannels,
          unsigned int t_ParEntries,
          unsigned int t_DataBits>
void dispColVec(const unsigned int t_chId,
                hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datStr,
                hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datFwdStr,
                hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datOutStr) {
    static const unsigned int t_ParBits = t_DataBits * t_ParEntries;
    static const unsigned int t_ParamsPerPar = t_ParBits / 32;
    // local buffer for colVec
    ap_uint<t_DataBits * t_ParEntries> l_vecStore[t_MaxColParBlocks];
#pragma HLS RESOURCE variable = l_vecStore core = RAM_1P_URAM uram
    unsigned int l_offset, l_vecBlocks;
    unsigned int l_minIdx[t_HbmChannels];
#pragma HLS ARRAY_PARTITION variable = l_minIdx complete dim = 1
    unsigned int l_maxIdx[t_HbmChannels];
#pragma HLS ARRAY_PARTITION variable = l_maxIdx complete dim = 1

    decFwdColParam<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
        l_offset, l_vecBlocks, l_minIdx, l_maxIdx, p_datStr, p_datFwdStr);

    uint16_t l_minIdxBlock = l_minIdx[t_chId] / t_ParEntries;
    uint8_t l_minIdxMod = l_minIdx[t_chId] % t_ParEntries;
    uint16_t l_maxIdxBlock = l_maxIdx[t_chId] / t_ParEntries;
    uint8_t l_maxIdMod = l_maxIdx[t_chId] % t_ParEntries;

    unsigned int l_chBlocks = (l_maxIdx[t_chId] - l_minIdx[t_chId] + t_ParEntries) / t_ParEntries;
    WideType<uint32_t, t_ParamsPerPar> l_outParam;
#pragma HLS ARRAY_PARTITION variable = l_outParam complete dim = 1
    l_outParam[0] = l_chBlocks;
    l_outParam[1] = l_chBlocks;
    p_datOutStr.write(l_outParam);
    // store and forward vector
    for (unsigned int i = 0; i < l_vecBlocks; ++i) {
#pragma HLS PIPELINE rewind
        ap_uint<t_DataBits* t_ParEntries> l_dat = p_datStr.read();
        p_datFwdStr.write(l_dat);
        if ((i >= l_minIdxBlock) && (i <= l_maxIdxBlock)) {
            l_vecStore[i - l_minIdxBlock] = l_dat;
        }
    }

    // select vector values for this channel
    ap_uint<t_DataBits * t_ParEntries> l_valFirst;
    ap_uint<t_DataBits * t_ParEntries> l_valNext;
    ap_uint<t_DataBits * t_ParEntries> l_valOut;
    uint16_t l_curBlock = 0;

    l_valNext = l_vecStore[l_curBlock];
    l_curBlock++;

    while (l_chBlocks != 0) {
#pragma HLS PIPELINE rewind
        l_valFirst = l_valNext;
        uint16_t l_blockId = l_curBlock + l_minIdxBlock;
        l_valNext = (!(l_blockId > l_maxIdxBlock)) ? l_vecStore[l_curBlock] : (ap_uint<t_DataBits * t_ParEntries>)0;
        for (unsigned int i = 0; i < t_ParEntries; ++i) {
#pragma HLS UNROLL
            uint8_t l_id = l_minIdxMod + i;
            l_valOut.range((i + 1) * t_DataBits - 1, i * t_DataBits) =
                (l_id < t_ParEntries)
                    ? l_valFirst.range((l_id + 1) * t_DataBits - 1, l_id * t_DataBits)
                    : l_valNext.range((l_id + 1 - t_ParEntries) * t_DataBits - 1, (l_id - t_ParEntries) * t_DataBits);
        }
        p_datOutStr.write(l_valOut);
        l_chBlocks--;
        l_curBlock++;
    }
}

template <unsigned int t_MaxColParBlocks,
          unsigned int t_ParBlocks4Param,
          unsigned int t_HbmChannels,
          unsigned int t_ParEntries,
          unsigned int t_DataBits>
void dispColVecSink(hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datStr,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datOutStr) {
    static const unsigned int t_ParBits = t_DataBits * t_ParEntries;
    static const unsigned int t_ParamsPerPar = t_ParBits / 32;
    // local buffer for colVec
    ap_uint<t_DataBits * t_ParEntries> l_vecStore[t_MaxColParBlocks];
#pragma HLS RESOURCE variable = l_vecStore core = RAM_1P_URAM uram

    unsigned int l_offset, l_vecBlocks;
    unsigned int l_minIdx[t_HbmChannels];
#pragma HLS ARRAY_PARTITION variable = l_minIdx complete dim = 1
    unsigned int l_maxIdx[t_HbmChannels];
#pragma HLS ARRAY_PARTITION variable = l_maxIdx complete dim = 1

    decColParam<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
        l_offset, l_vecBlocks, l_minIdx, l_maxIdx, p_datStr);

    uint16_t l_minIdxBlock = l_minIdx[t_HbmChannels - 1] / t_ParEntries;
    uint8_t l_minIdxMod = l_minIdx[t_HbmChannels - 1] % t_ParEntries;
    uint16_t l_maxIdxBlock = l_maxIdx[t_HbmChannels - 1] / t_ParEntries;
    uint8_t l_maxIdMod = l_maxIdx[t_HbmChannels - 1] % t_ParEntries;

    unsigned int l_chBlocks = (l_maxIdx[t_HbmChannels - 1] - l_minIdx[t_HbmChannels - 1] + t_ParEntries) / t_ParEntries;
    WideType<uint32_t, t_ParamsPerPar> l_outParam;
    l_outParam[0] = l_chBlocks;
    l_outParam[1] = l_chBlocks;
    p_datOutStr.write(l_outParam);
    // store vector
    for (unsigned int i = 0; i < l_vecBlocks; ++i) {
#pragma HLS PIPELINE
        ap_uint<t_DataBits* t_ParEntries> l_dat = p_datStr.read();
        if ((i >= l_minIdxBlock) && (i <= l_maxIdxBlock)) {
            l_vecStore[i - l_minIdxBlock] = l_dat;
        }
    }
    // select vector values for this channel

    ap_uint<t_DataBits * t_ParEntries> l_valFirst;
    ap_uint<t_DataBits * t_ParEntries> l_valNext;
    ap_uint<t_DataBits * t_ParEntries> l_valOut;
    uint16_t l_curBlock = 0;

    l_valNext = l_vecStore[l_curBlock];
    l_curBlock++;

    while (l_chBlocks != 0) {
#pragma HLS PIPELINE
        l_valFirst = l_valNext;
        uint16_t l_blockId = l_curBlock + l_minIdxBlock;
        l_valNext = (!(l_blockId > l_maxIdxBlock)) ? l_vecStore[l_curBlock] : (ap_uint<t_DataBits * t_ParEntries>)0;
        for (unsigned int i = 0; i < t_ParEntries; ++i) {
#pragma HLS UNROLL
            uint8_t l_id = l_minIdxMod + i;
            l_valOut.range((i + 1) * t_DataBits - 1, i * t_DataBits) =
                (l_id < t_ParEntries)
                    ? l_valFirst.range((l_id + 1) * t_DataBits - 1, l_id * t_DataBits)
                    : l_valNext.range((l_id + 1 - t_ParEntries) * t_DataBits - 1, (l_id - t_ParEntries) * t_DataBits);
        }
        p_datOutStr.write(l_valOut);
        l_chBlocks--;
        l_curBlock++;
    }
}

/**
 * @brief dispCol function that dispatchs input column vectors accross parallel CUs for computing SpMV simultaneously
 *
 * @tparam t_MaxColParBlocks the maximum number of parallelly processed column vector entries in the on-chip buffer
 * @tparam t_ParBlocks4Param the number of parallelly processed parameter blocks
 * @tparam t_HbmChannels number of HBM channels
 * @tparam t_ParEntries parallelly processed entries
 * @tparam t_DataBits number of bits used to store each entry
 *
 * @param p_datStr input vector stream
 * @param p_datOutStr an output array of column vector streams
 */
template <unsigned int t_MaxColParBlocks,
          unsigned int t_ParBlocks4Param,
          unsigned int t_HbmChannels,
          unsigned int t_ParEntries,
          unsigned int t_DataBits>
void dispCol(hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datStr,
             hls::stream<ap_uint<t_DataBits * t_ParEntries> > p_datOutStr[t_HbmChannels]) {
#if SPARSE_hbmChannels == 1
    dispColVecSink<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(p_datStr,
                                                                                                  p_datOutStr[0]);
#else
    const unsigned int t_FwdChannels = t_HbmChannels - 1;
    hls::stream<ap_uint<t_DataBits * t_ParEntries> > l_datFwdStr[t_FwdChannels];
#pragma HLS DATAFLOW
    dispColVec<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
        0, p_datStr, l_datFwdStr[0], p_datOutStr[0]);
    for (unsigned int i = 1; i < t_FwdChannels; ++i) {
#pragma HLS UNROLL
        dispColVec<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            i, l_datFwdStr[i - 1], l_datFwdStr[i], p_datOutStr[i]);
    }
    dispColVecSink<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
        l_datFwdStr[t_FwdChannels - 1], p_datOutStr[t_FwdChannels]);

#endif
}

template <unsigned int t_MaxColParBlocks,
          unsigned int t_ParBlocks4Param,
          unsigned int t_HbmChannels,
          unsigned int t_ParEntries,
          unsigned int t_DataBits>
void dispNnzColStep(const unsigned int t_ChId,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datStr,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datFwdStr,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datOutStr) {
    static const unsigned int t_ParBits = t_DataBits * t_ParEntries;
    static const unsigned int t_ParamsPerPar = t_ParBits / 32;

    ap_uint<t_DataBits * t_ParEntries> l_nnzColStore[t_MaxColParBlocks];
#pragma HLS RESOURCE variable = l_nnzColStore core = RAM_1P_URAM uram
    unsigned int l_offset, l_vecBlocks;
    unsigned int l_parBlocks[t_HbmChannels];
#pragma HLS ARRAY_PARTITION variable = l_parBlocks complete dim = 1
    unsigned int l_nnzBlocks[t_HbmChannels];
#pragma HLS ARRAY_PARTITION variable = l_nnzBlocks complete dim = 1

    decFwdColParam<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
        l_offset, l_vecBlocks, l_parBlocks, l_nnzBlocks, p_datStr, p_datFwdStr);
    // store and forward vector
    for (unsigned int i = 0; i < t_ChId; ++i) {
#pragma HLS PIPELINE
        l_vecBlocks -= l_parBlocks[i];
    }
    unsigned int l_blocks = l_parBlocks[t_ChId];

    for (unsigned int i = 0; i < l_blocks; ++i) {
#pragma HLS PIPELINE
        ap_uint<t_DataBits* t_ParEntries> l_dat = p_datStr.read();
        l_nnzColStore[i] = l_dat;
    }
    for (unsigned int b = l_blocks; b < l_vecBlocks; ++b) {
#pragma HLS PIPELINE
        ap_uint<t_DataBits* t_ParEntries> l_dat = p_datStr.read();
        p_datFwdStr.write(l_dat);
    }
    WideType<uint32_t, t_ParamsPerPar> l_outParam;
    l_outParam[0] = l_blocks;
    l_outParam[1] = l_nnzBlocks[t_ChId];
    p_datOutStr.write(l_outParam);
    for (unsigned int i = 0; i < l_blocks; ++i) {
#pragma HLS PIPELINE
        ap_uint<t_DataBits * t_ParEntries> l_val;
        l_val = l_nnzColStore[i];
        p_datOutStr.write(l_val);
    }
}

template <unsigned int t_MaxColParBlocks,
          unsigned int t_ParBlocks4Param,
          unsigned int t_HbmChannels,
          unsigned int t_ParEntries,
          unsigned int t_DataBits>
void dispNnzColStepDup(const unsigned int t_ChId,
                       hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datStr,
                       hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datFwdStr,
                       hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datOutStr) {
    static const unsigned int t_ParBits = t_DataBits * t_ParEntries;
    static const unsigned int t_ParamsPerPar = t_ParBits / 32;

    ap_uint<t_DataBits * t_ParEntries> l_nnzColStore[t_MaxColParBlocks];
#pragma HLS RESOURCE variable = l_nnzColStore core = RAM_1P_URAM uram
    unsigned int l_offset, l_vecBlocks;
    unsigned int l_parBlocks[t_HbmChannels];
#pragma HLS ARRAY_PARTITION variable = l_parBlocks complete dim = 1
    unsigned int l_nnzBlocks[t_HbmChannels];
#pragma HLS ARRAY_PARTITION variable = l_nnzBlocks complete dim = 1

    decFwdColParam<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
        l_offset, l_vecBlocks, l_parBlocks, l_nnzBlocks, p_datStr, p_datFwdStr);
    // store and forward vector
    for (unsigned int i = 0; i < t_ChId; ++i) {
#pragma HLS PIPELINE
        l_vecBlocks -= l_parBlocks[i];
    }
    unsigned int l_blocks = l_parBlocks[t_ChId];

    for (unsigned int i = 0; i < l_blocks; ++i) {
#pragma HLS PIPELINE
        ap_uint<t_DataBits* t_ParEntries> l_dat = p_datStr.read();
        l_nnzColStore[i] = l_dat;
    }
    for (unsigned int b = l_blocks; b < l_vecBlocks; ++b) {
#pragma HLS PIPELINE
        ap_uint<t_DataBits* t_ParEntries> l_dat = p_datStr.read();
        p_datFwdStr.write(l_dat);
    }
    WideType<uint32_t, t_ParamsPerPar> l_outParam;
    l_outParam[0] = l_blocks;
    l_outParam[1] = l_nnzBlocks[t_ChId];
    p_datOutStr.write(l_outParam);
    for (unsigned int i = 0; i < l_blocks; ++i) {
#pragma HLS PIPELINE
        ap_uint<t_DataBits * t_ParEntries> l_val;
        l_val = l_nnzColStore[i];
        p_datOutStr.write(l_val);
    }
}

template <unsigned int t_MaxColParBlocks,
          unsigned int t_ParBlocks4Param,
          unsigned int t_HbmChannels,
          unsigned int t_ParEntries,
          unsigned int t_DataBits>
void dispNnzColSink(hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datStr,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datOutStr) {
    // local buffer for colVec
    static const unsigned int t_ParBits = t_DataBits * t_ParEntries;
    static const unsigned int t_ParamsPerPar = t_ParBits / 32;

    ap_uint<t_DataBits * t_ParEntries> l_nnzColStore[t_MaxColParBlocks];
#pragma HLS RESOURCE variable = l_nnzColStore core = RAM_1P_URAM uram
    unsigned int l_offset, l_vecBlocks;
    unsigned int l_parBlocks[t_HbmChannels];
#pragma HLS ARRAY_PARTITION variable = l_parBlocks complete dim = 1
    unsigned int l_nnzBlocks[t_HbmChannels];
#pragma HLS ARRAY_PARTITION variable = l_nnzBlocks complete dim = 1

    decColParam<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
        l_offset, l_vecBlocks, l_parBlocks, l_nnzBlocks, p_datStr);
    // store and forward vector
    for (unsigned int i = 0; i < t_HbmChannels - 1; ++i) {
#pragma HLS PIPELINE
        l_vecBlocks -= l_parBlocks[i];
    }
    unsigned int l_blocks = l_parBlocks[t_HbmChannels - 1];
    for (unsigned int i = 0; i < l_blocks; ++i) {
#pragma HLS PIPELINE rewind
        ap_uint<t_DataBits* t_ParEntries> l_dat = p_datStr.read();
        l_nnzColStore[i] = l_dat;
    }
    for (unsigned int i = l_blocks; i < l_vecBlocks; ++i) {
#pragma HLS PIPELINE rewind
        ap_uint<t_DataBits* t_ParEntries> l_dat = p_datStr.read();
    }

    WideType<uint32_t, t_ParamsPerPar> l_outParam;
    l_outParam[0] = l_blocks;
    l_outParam[1] = l_nnzBlocks[t_HbmChannels - 1];
    p_datOutStr.write(l_outParam);
    for (unsigned int i = 0; i < l_blocks; ++i) {
#pragma HLS PIPELINE rewind
        ap_uint<t_DataBits * t_ParEntries> l_val;
        l_val = l_nnzColStore[i];
        p_datOutStr.write(l_val);
    }
}

/**
 * @brief dispNnzCol function that dispatchs NNZ Col pointer entries accross parallel compute CUs
 *
 * @tparam t_MaxColParBlocks the maximum number of parallelly processed column entries in the on-chip buffer
 * @tparam t_ParBlocks4Param the number of parallelly processed parameter blocks
 * @tparam t_HbmChannels number of HBM channels
 * @tparam t_ParEntries parallelly processed entries
 * @tparam t_DataBits number of bits used to store each entry
 *
 * @param p_datStr input vector stream
 * @param p_datOutStr an output array of vector streams
 */
template <unsigned int t_MaxColParBlocks,
          unsigned int t_ParBlocks4Param,
          unsigned int t_HbmChannels,
          unsigned int t_ParEntries,
          unsigned int t_DataBits>
void dispNnzCol(hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datStr,
                hls::stream<ap_uint<t_DataBits * t_ParEntries> > p_datOutStr[t_HbmChannels]) {
#if SPARSE_hbmChannels == 1
    dispNnzColSink<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(p_datStr,
                                                                                                  p_datOutStr[0]);
#else
    static const unsigned int t_FwdChannels = t_HbmChannels - 1;
    hls::stream<ap_uint<t_DataBits * t_ParEntries> > l_datFwdStr[t_FwdChannels];
#pragma HLS ARRAY_PARTITION variable = l_datFwdStr complete dim = 1
#pragma HLS DATAFLOW
    dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
        0, p_datStr, l_datFwdStr[0], p_datOutStr[0]);
    /*
        dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            1, l_datFwdStr[0], l_datFwdStr[1], p_datOutStr[1]);
        dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            2, l_datFwdStr[1], l_datFwdStr[2], p_datOutStr[2]);
        dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            3, l_datFwdStr[2], l_datFwdStr[3], p_datOutStr[3]);
        dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            4, l_datFwdStr[3], l_datFwdStr[4], p_datOutStr[4]);
        dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            5, l_datFwdStr[4], l_datFwdStr[5], p_datOutStr[5]);
        dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            6, l_datFwdStr[5], l_datFwdStr[6], p_datOutStr[6]);
        dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            7, l_datFwdStr[6], l_datFwdStr[7], p_datOutStr[7]);
        dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            8, l_datFwdStr[7], l_datFwdStr[8], p_datOutStr[8]);
        dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            9, l_datFwdStr[8], l_datFwdStr[9], p_datOutStr[9]);
        dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            10, l_datFwdStr[9], l_datFwdStr[10], p_datOutStr[10]);
        dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            11, l_datFwdStr[10], l_datFwdStr[11], p_datOutStr[11]);
        dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            12, l_datFwdStr[11], l_datFwdStr[12], p_datOutStr[12]);
        dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            13, l_datFwdStr[12], l_datFwdStr[13], p_datOutStr[13]);
        dispNnzColStepDup<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            14, l_datFwdStr[13], l_datFwdStr[14], p_datOutStr[14]);*/
    for (unsigned int i = 1; i < t_FwdChannels; ++i) {
#pragma HLS UNROLL
        dispNnzColStep<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
            i, l_datFwdStr[i - 1], l_datFwdStr[i], p_datOutStr[i]);
    }
    dispNnzColSink<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(
        l_datFwdStr[t_FwdChannels - 1], p_datOutStr[t_FwdChannels]);

#endif
}
} // end namespace sparse
} // end namespace xf
#endif
