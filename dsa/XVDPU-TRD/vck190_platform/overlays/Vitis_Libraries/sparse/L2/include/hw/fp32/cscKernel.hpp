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
 * @file cscKernel.hpp
 * @brief SPARSE Level 2 template functions for building kernels.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_CSCKERNEL_HPP
#define XF_SPARSE_CSCKERNEL_HPP

#ifndef __cplusplus
#error "SPARSE Library only works with C++."
#endif

#include <cstdint>
#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#ifndef __SYNTHESIS__
#include <iostream>
#include <fstream>
#include <cstring>
#endif
#include "xf_sparse.hpp"
#include "cscMatMoverL2.hpp"

namespace xf {
namespace sparse {
using namespace xf::blas;

#if DEBUG_dumpData
template <typename Func>
void openOutputFile(const std::string& p_filename, const Func& func) {
    std::ofstream l_of(p_filename.c_str(), std::ios::binary);
    if (!l_of.is_open()) {
        std::cout << "ERROR: Open " << p_filename << std::endl;
    }
    func(l_of);
    l_of.close();
}

template <int t_Bits>
void saveStream(hls::stream<ap_uint<t_Bits> >& p_stream, std::ofstream& p_of) {
    hls::stream<ap_uint<t_Bits> > l_stream;

    size_t l_size = 0;

    while (!p_stream.empty()) {
        auto l_dat = p_stream.read();
        l_stream.write(l_dat);
        l_size++;
    }

    p_of.write(reinterpret_cast<char*>(&l_size), sizeof(size_t));

    while (!l_stream.empty()) {
        auto l_dat = l_stream.read();
        p_stream.write(l_dat);
        p_of.write(reinterpret_cast<char*>(&l_dat), (t_Bits + 7) / 8);
    }
}
#endif

template <unsigned int t_MemBits>
void readMem(ap_uint<t_MemBits>* p_memRdPtr, unsigned int p_memRdBlocks, hls::stream<ap_uint<t_MemBits> >& p_memStr) {
#pragma HLS INLINE
    for (unsigned int i = 0; i < p_memRdBlocks; ++i) {
#pragma HLS PIPELINE rewind
        ap_uint<t_MemBits> l_memVal = p_memRdPtr[i];
        p_memStr.write(l_memVal);
    }
}

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_ParEntries,
          unsigned int t_MemBits,
          unsigned int t_DataBits,
          unsigned int t_IndexBits>
void nnzIdx2DatStr(hls::stream<ap_uint<t_MemBits> >& p_memStr,
                   const unsigned int p_blocks,
                   hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_nnzStr,
#if DEBUG_dumpData
                   hls::stream<ap_uint<t_IndexBits * t_ParEntries> >& p_idxStr,
                   std::ofstream& p_of) {
#else
                   hls::stream<ap_uint<t_IndexBits * t_ParEntries> >& p_idxStr) {
#endif
#ifndef __SYNTHESIS__
    assert(t_MemBits / 2 == t_DataBits * t_ParEntries);
    assert(t_MemBits / 2 == t_IndexBits * t_ParEntries);
#endif

    for (unsigned int i = 0; i < p_blocks; ++i) {
#pragma HLS PIPELINE
        ap_uint<t_MemBits> l_memVal = p_memStr.read();
        ap_uint<t_DataBits* t_ParEntries> l_datVal = l_memVal.range(t_MemBits - 1, t_MemBits / 2);
        ap_uint<t_IndexBits* t_ParEntries> l_idxVal = l_memVal.range(t_MemBits / 2 - 1, 0);
        p_nnzStr.write(l_datVal);
        p_idxStr.write(l_idxVal);
#if DEBUG_dumpData
        WideType<t_DataType, t_ParEntries> l_valDump(l_datVal);
        WideType<t_IndexType, t_ParEntries> l_idxDump(l_idxVal);
        l_idxDump.write(p_of);
        l_valDump.write(p_of);
#endif
    }
}

template <unsigned int t_MemBits, unsigned int t_ParEntries, unsigned int t_DataBits>
void loadDat2Str(const ap_uint<t_MemBits>* p_memPtr,
                 const unsigned int p_memBlocks,
                 hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datStr) {
    hls::stream<ap_uint<t_MemBits> > l_memStr;
#pragma HLS DATAFLOW
    loadMemBlocks<t_MemBits>(p_memPtr, p_memBlocks, l_memStr);
    memStr2DatStr<t_ParEntries, t_MemBits, t_DataBits>(l_memStr, p_memBlocks, p_datStr);
}

template <unsigned int t_MaxRowBlocks,
          unsigned int t_LogParEntries,
          unsigned int t_LogParGroups,
          typename t_DataType,
          typename t_IndexType,
          unsigned int t_DataBits,
          unsigned int t_IndexBits,
          unsigned int t_MemBits>
void cscRowUnit(hls::stream<ap_uint<t_MemBits> >& p_aNnzIdxStr,
                hls::stream<ap_uint<t_DataBits*(1 << t_LogParEntries)> >& p_nnzColValStr,
                unsigned int p_nnzBlocks,
                unsigned int p_rowBlocks,
#if DEBUG_dumpData
                hls::stream<ap_uint<t_DataBits*(1 << t_LogParEntries)> >& p_rowAggStr,
                unsigned int p_cuId) {
#else
                hls::stream<ap_uint<t_DataBits*(1 << t_LogParEntries)> >& p_rowAggStr) {
#endif
    static const unsigned int t_ParEntries = 1 << t_LogParEntries;
    static const unsigned int t_ParGroups = 1 << t_LogParGroups;
    hls::stream<ap_uint<t_DataBits * t_ParEntries> > l_nnzStr;
    hls::stream<ap_uint<t_IndexBits * t_ParEntries> > l_idxStr;
    hls::stream<ap_uint<t_DataBits * t_ParEntries> > l_colValStr;
    hls::stream<ap_uint<t_DataBits * t_ParEntries> > l_rowAggStr;
#pragma HLS DATAFLOW

#if DEBUG_dumpData
    std::string l_toCscRowFileName = "toCscRow_" + std::to_string(p_cuId) + ".dat";
    std::ofstream l_of2CscRow(l_toCscRowFileName.c_str(), std::ios::binary);
    if (!l_of2CscRow.is_open()) {
        std::cout << "ERROR: Open " << l_toCscRowFileName << std::endl;
    }
    l_of2CscRow.write(reinterpret_cast<char*>(&p_nnzBlocks), sizeof(uint32_t));
    l_of2CscRow.write(reinterpret_cast<char*>(&p_rowBlocks), sizeof(uint32_t));
    nnzIdx2DatStr<t_DataType, t_IndexType, t_ParEntries, t_MemBits, t_DataBits, t_IndexBits>(
        p_aNnzIdxStr, p_nnzBlocks, l_nnzStr, l_idxStr, l_of2CscRow);
    l_of2CscRow.close();
#else
    nnzIdx2DatStr<t_DataType, t_IndexType, t_ParEntries, t_MemBits, t_DataBits, t_IndexBits>(p_aNnzIdxStr, p_nnzBlocks,
                                                                                             l_nnzStr, l_idxStr);
#endif
    cscRow<t_MaxRowBlocks, t_LogParEntries, t_LogParGroups, t_DataType, t_IndexType, t_DataBits, t_IndexBits>(
        p_nnzBlocks, p_rowBlocks, l_nnzStr, p_nnzColValStr, l_idxStr, p_rowAggStr);
}

template <unsigned int t_ParEntries, unsigned int t_MemBits, unsigned int t_DataBits>
void storeDa(hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datStr,
             const unsigned int p_memBlocks,
             ap_uint<t_MemBits>* p_memPtr) {
    hls::stream<ap_uint<t_MemBits> > l_memStr;

#pragma HLS DATAFLOW

    datStr2MemStr<t_ParEntries, t_MemBits, t_DataBits>(p_datStr, p_memBlocks, l_memStr);
    storeMemBlocks<t_MemBits>(l_memStr, p_memBlocks, p_memPtr);
}

template <unsigned int t_MaxParamMemBlocks, unsigned int t_ParamsPerMem, unsigned int t_MemBlocks4Param>
void getHbmParam(uint32_t p_paramStore[t_MaxParamMemBlocks][t_ParamsPerMem],
                 unsigned int p_blockId,
                 uint32_t& p_rdOffset,
                 uint32_t& p_wrOffset,
                 uint32_t& p_nnzBlocks,
                 uint32_t& p_rowBlocks) {
    static const unsigned int t_Width = t_MemBlocks4Param * t_ParamsPerMem;
#pragma HLS INLINE
    WideType<uint32_t, t_Width> l_param;
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1

    for (unsigned int i = 0; i < t_MemBlocks4Param; ++i) {
#pragma HLS PIPELINE
        for (unsigned int j = 0; j < t_ParamsPerMem; ++j) {
            l_param[i * t_ParamsPerMem + j] = p_paramStore[p_blockId + i][j];
        }
    }
    p_rdOffset = l_param[0];
    p_wrOffset = l_param[1];
    p_nnzBlocks = l_param[2];
    p_rowBlocks = l_param[3];
}

template <unsigned int t_MaxParamMemBlocks,
          unsigned int t_ParamsPerMem,
          unsigned int t_MemBits,
          unsigned int t_ParamMemOffset>
void readMemParamBlocks(ap_uint<t_MemBits>* p_ptr,
                        unsigned int& p_paramBlocks,
                        unsigned int& p_iterations,
                        uint32_t p_paramStore[t_MaxParamMemBlocks][t_ParamsPerMem]) {
#pragma HLS INLINE
    ap_uint<t_MemBits>* l_ptr = p_ptr;
    ap_uint<t_MemBits> l_val = l_ptr[0];
    WideType<uint32_t, t_ParamsPerMem> l_param(l_val);
    p_paramBlocks = l_param[0];
    p_iterations = l_param[1];
    l_ptr += t_ParamMemOffset;
    for (unsigned int i = 0; i < p_paramBlocks; ++i) {
#pragma HLS PIPELINE
        l_val = l_ptr[i];
        l_param = l_val;
        for (unsigned int j = 0; j < t_ParamsPerMem; ++j) {
            p_paramStore[i][j] = l_param[j];
        }
    }
}

template <unsigned int t_MaxParamHbmBlocks, unsigned int t_MemBits, unsigned int t_ParamOffset>
void readHbm(ap_uint<t_MemBits>* p_memRdPtr,
             hls::stream<uint32_t>& p_paramStr,
             hls::stream<ap_uint<t_MemBits> >& p_memStr) {
    static const unsigned int t_ParamsPerHbm = t_MemBits / 32;
    static const unsigned int t_ParamMemOffset = t_ParamOffset * 8 / t_MemBits;
    static const unsigned int t_HbmBlocks4Param = (4 + t_ParamsPerHbm - 1) / t_ParamsPerHbm;

    uint32_t l_paramStore[t_MaxParamHbmBlocks][t_ParamsPerHbm];
#pragma HLS ARRAY_PARTITION variable = l_paramStore complete dim = 2

    unsigned int l_iterations = 0;
    unsigned int l_totalParamBlocks = 0;
    readMemParamBlocks<t_MaxParamHbmBlocks, t_ParamsPerHbm, t_MemBits, t_ParamMemOffset>(p_memRdPtr, l_totalParamBlocks,
                                                                                         l_iterations, l_paramStore);

    unsigned int l_totalParams = l_totalParamBlocks / t_HbmBlocks4Param;
    p_paramStr.write(l_totalParams * l_iterations);

    unsigned int l_paramBlockId = 0;
    WideType<uint32_t, t_ParamsPerHbm> l_param;
    while (l_iterations > 0) {
        if (l_paramBlockId == 0) {
            l_param[0] = l_totalParams;
            p_memStr.write(l_param);
        }
        uint32_t l_rdOffset;
        uint32_t l_wrOffset;
        uint32_t l_nnzBlocks;
        uint32_t l_rowBlocks;

        getHbmParam<t_MaxParamHbmBlocks, t_ParamsPerHbm, t_HbmBlocks4Param>(l_paramStore, l_paramBlockId, l_rdOffset,
                                                                            l_wrOffset, l_nnzBlocks, l_rowBlocks);
        p_paramStr.write(l_wrOffset);
        p_paramStr.write(l_rowBlocks);

        l_param[0] = l_nnzBlocks;
        l_param[1] = l_rowBlocks;
        p_memStr.write(l_param);

        ap_uint<t_MemBits>* l_datPtr = p_memRdPtr + l_rdOffset;
        readMem<t_MemBits>(l_datPtr, l_nnzBlocks, p_memStr);
        l_paramBlockId += t_HbmBlocks4Param;
        if (l_paramBlockId == l_totalParamBlocks) {
            l_paramBlockId = 0;
            l_iterations--;
        }
    }
}
template <unsigned int t_MaxRowBlocks,
          unsigned int t_ParEntries,
          unsigned int t_ParGroups,
          unsigned int t_MemBits,
          unsigned int t_DataBits>
void writeHbm(hls::stream<uint32_t>& p_paramStr,
              hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_datStr,
              ap_uint<t_MemBits>* p_memPtr) {
    static const unsigned int t_ParDataBits = t_DataBits * t_ParEntries;
    static const unsigned int t_DataWords = t_MemBits / t_ParDataBits;

    ap_uint<t_MemBits> l_rowStore[t_MaxRowBlocks][t_DataWords];
#pragma HLS ARRAY_PARTITION variable = l_rowStore complete dim = 2
#pragma HLS RESOURCE variable = l_rowStore core = RAM_1P_URAM uram

    hls::stream<ap_uint<t_MemBits> > l_memStr;
    ap_uint<t_MemBits>* l_wrPtr;
#ifndef __SYNTHESIS__
    assert(t_MemBits >= t_ParDataBits);
    assert(t_MemBits % t_ParDataBits == 0);
#endif

    uint32_t l_totalParams = p_paramStr.read();
    for (unsigned int r = 0; r < l_totalParams; ++r) {
        uint32_t l_wrOffset = p_paramStr.read();
        uint32_t l_rowBlocks = p_paramStr.read();
        uint32_t l_rowResBlocks = l_rowBlocks * t_ParGroups;
        l_wrPtr = p_memPtr + l_wrOffset;

        unsigned int l_memBlocks = l_rowResBlocks / t_DataWords;
        for (unsigned int i = 0; i < l_memBlocks; ++i) {
            for (unsigned int j = 0; j < t_DataWords; ++j) {
#pragma HLS PIPELINE II = 1
                ap_uint<t_MemBits> l_dat = p_datStr.read();
                l_rowStore[i][j] = l_dat;
            }
        }

        if ((l_rowResBlocks % t_DataWords) != 0) {
            ap_uint<t_MemBits> l_dat = p_datStr.read();
            l_rowStore[l_memBlocks][0] = l_dat;
            l_memBlocks++;
        }

        for (unsigned int i = 0; i < l_memBlocks; ++i) {
#pragma HLS PIPELINE II = 1
            ap_uint<t_MemBits> l_memVal;
            for (unsigned int j = 0; j < t_DataWords; ++j) {
                l_memVal.range((j + 1) * t_ParDataBits - 1, j * t_ParDataBits) = l_rowStore[i][j];
            }
            l_wrPtr[i] = l_memVal;
        }
    }
}

template <unsigned int t_MaxParamHbmBlocks,
          unsigned int t_MaxRowBlocks,
          unsigned int t_ParEntries,
          unsigned int t_ParGroups,
          unsigned int t_DataBits,
          unsigned int t_HbmMemBits,
          unsigned int t_ParamOffset>
void rdWrHbmChannel(ap_uint<t_HbmMemBits>* p_rdPtr,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_inStr,
                    ap_uint<t_HbmMemBits>* p_wrPtr,
                    hls::stream<ap_uint<t_HbmMemBits> >& p_outStr) {
    hls::stream<uint32_t> l_paramStr;
#pragma HLS STREAM variable = l_paramStr depth = 4

#pragma HLS DATAFLOW
    readHbm<t_MaxParamHbmBlocks, t_HbmMemBits, t_ParamOffset>(p_rdPtr, l_paramStr, p_outStr);
    writeHbm<t_MaxRowBlocks, t_ParEntries, t_ParGroups, t_HbmMemBits, t_DataBits>(l_paramStr, p_inStr, p_wrPtr);
}

template <unsigned int t_LogParEntries,
          typename t_DataType,
          typename t_IndexType,
          unsigned int t_DataBits,
          unsigned int t_IndexBits>
void xBarColUnit(const unsigned int p_colPtrBlocks,
                 const unsigned int p_nnzBlocks,
                 hls::stream<ap_uint<t_DataBits*(1 << t_LogParEntries)> >& p_colValStr,
                 hls::stream<ap_uint<t_IndexBits*(1 << t_LogParEntries)> >& p_colPtrStr,
                 hls::stream<ap_uint<t_DataBits*(1 << t_LogParEntries)> >& p_nnzColValStr
#if DEBUG_dumpData
                 ,
                 unsigned int p_cuId
#endif
                 ) {
    const unsigned int t_ParEntries = 1 << t_LogParEntries;
    const unsigned int t_IndexBusBits = t_IndexBits * t_ParEntries;
    const unsigned int t_DataBusBits = t_DataBits * t_ParEntries;

    hls::stream<ap_uint<t_IndexBusBits> > l_colPtrStr;
    hls::stream<ap_uint<t_DataBusBits> > l_colValStr;
    hls::stream<ap_uint<t_DataBusBits> > l_nnzColValStr;

#pragma HLS DATAFLOW

#if DEBUG_dumpData
    openOutputFile("toXBarCol_" + std::to_string(p_cuId) + ".dat", [&](std::ofstream& of) {
        of.write(reinterpret_cast<const char*>(&p_colPtrBlocks), sizeof(unsigned int));
        of.write(reinterpret_cast<const char*>(&p_nnzBlocks), sizeof(unsigned int));
        saveStream(p_colPtrStr, of);
        saveStream(p_colValStr, of);
    });
#endif
    xBarCol<t_LogParEntries, t_DataType, t_IndexType, t_DataBits, t_IndexBits>(p_colPtrBlocks, p_nnzBlocks, p_colPtrStr,
                                                                               p_colValStr, p_nnzColValStr);
}

template <unsigned int t_MaxParamDdrBlocks,
          unsigned int t_ParamsPerDdr,
          unsigned int t_DdrBlocks4Param,
          unsigned int t_HbmChannels>
void getColParam(uint32_t p_paramStore[t_MaxParamDdrBlocks][t_ParamsPerDdr],
                 unsigned int p_id,
                 uint32_t& p_offset,
                 uint32_t& p_memBlocks,
                 uint32_t p_param1Hbm[t_HbmChannels],
                 uint32_t p_param2Hbm[t_HbmChannels]) {
    static const unsigned int t_Width = t_DdrBlocks4Param * t_ParamsPerDdr;
#pragma HLS INLINE
    WideType<uint32_t, t_Width> l_param;
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1

    for (unsigned int i = 0; i < t_DdrBlocks4Param; ++i) {
#pragma HLS PIPELINE rewind
        for (unsigned int j = 0; j < t_ParamsPerDdr; ++j) {
            l_param[i * t_ParamsPerDdr + j] = p_paramStore[p_id + i][j];
        }
    }
    p_offset = l_param[0];
    p_memBlocks = l_param[1];
    for (unsigned int i = 0; i < t_HbmChannels; ++i) {
#pragma HLS UNROLL
        p_param1Hbm[i] = l_param[2 + i];
        p_param2Hbm[i] = l_param[2 + t_HbmChannels + i];
    }
}

template <unsigned int t_ParamsPerDdr, unsigned int t_DdrBlocks4Param, unsigned int t_HbmChannels>
void colParam2Str(uint32_t p_offset,
                  uint32_t p_memBlocks,
                  uint32_t p_param1Hbm[t_HbmChannels],
                  uint32_t p_param2Hbm[t_HbmChannels],
                  hls::stream<ap_uint<32 * t_ParamsPerDdr> >& p_datStr) {
    static const unsigned int t_Width = t_DdrBlocks4Param * t_ParamsPerDdr;
#pragma HLS INLINE
    WideType<uint32_t, t_Width> l_param;
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1

    l_param[0] = p_offset;
    l_param[1] = p_memBlocks;
    for (unsigned int i = 0; i < t_HbmChannels; ++i) {
#pragma HLS UNROLL
        l_param[2 + i] = p_param1Hbm[i];
        l_param[2 + t_HbmChannels + i] = p_param2Hbm[i];
    }
    WideType<uint32_t, t_ParamsPerDdr> l_outParam;
#pragma HLS ARRAY_PARTITION variable = l_outParam complete dim = 1

    for (unsigned int i = 0; i < t_DdrBlocks4Param; ++i) {
#pragma HLS PIPELINE rewind
        for (unsigned int j = 0; j < t_ParamsPerDdr; ++j) {
            l_outParam[j] = l_param[i * t_ParamsPerDdr + j];
        }
        p_datStr.write(l_outParam);
    }
}

template <unsigned int t_MaxParamDdrBlocks,
          unsigned int t_HbmChannels,
          unsigned int t_MemBits,
          unsigned int t_DataBits,
          unsigned int t_ParEntries,
          unsigned int t_ParamOffset>
void loadColVec(ap_uint<t_MemBits>* p_colValPtr, hls::stream<ap_uint<t_MemBits> >& p_colVecStr) {
    static const unsigned int t_ParamsPerDdr = t_MemBits / 32;
    static const unsigned int t_ParamDdrOffset = t_ParamOffset * 8 / t_MemBits;
    static const unsigned int t_IntsPerDdr = t_MemBits / (8 * sizeof(uint32_t));
    static const unsigned int t_IntsPerColParam = 2 + t_HbmChannels * 2;
    static const unsigned int t_DdrBlocks4Param = (t_IntsPerColParam + t_IntsPerDdr - 1) / t_IntsPerDdr;

    uint32_t l_paramStore[t_MaxParamDdrBlocks][t_ParamsPerDdr];
#pragma HLS ARRAY_PARTITION variable = l_paramStore complete dim = 2
    unsigned int l_iterations = 0;
    unsigned int l_paramBlocks = 0;
    readMemParamBlocks<t_MaxParamDdrBlocks, t_ParamsPerDdr, t_MemBits, t_ParamDdrOffset>(p_colValPtr, l_paramBlocks,
                                                                                         l_iterations, l_paramStore);

    uint32_t l_offset, l_memBlocks;
    uint32_t l_minColId[t_HbmChannels];
#pragma HLS ARRAY_PARTITION variable = l_minColId complete dim = 1
    uint32_t l_maxColId[t_HbmChannels];
#pragma HLS ARRAY_PARTITION variable = l_maxColId complete dim = 1
    ap_uint<t_MemBits>* l_datPtr;

    uint32_t l_totalParams = l_paramBlocks / t_DdrBlocks4Param;
    WideType<uint32_t, t_ParamsPerDdr> l_param;
#if SEQ_KERNEL
    l_param[0] = l_totalParams;
    l_param[1] = l_iterations;
    p_colVecStr.write(l_param);
#endif

    unsigned int l_i = 0;
    while (l_iterations > 0) {
        getColParam<t_MaxParamDdrBlocks, t_ParamsPerDdr, t_DdrBlocks4Param, t_HbmChannels>(
            l_paramStore, l_i, l_offset, l_memBlocks, l_minColId, l_maxColId);
        colParam2Str<t_ParamsPerDdr, t_DdrBlocks4Param, t_HbmChannels>(l_offset, l_memBlocks, l_minColId, l_maxColId,
                                                                       p_colVecStr);
        l_datPtr = p_colValPtr + l_offset;
        readMem<t_MemBits>(l_datPtr, l_memBlocks, p_colVecStr);
        l_i += t_DdrBlocks4Param;
        if (l_i == l_paramBlocks) {
            l_i = 0;
            l_iterations--;
        }
    }
}

template <unsigned int t_MaxParamDdrBlocks,
          unsigned int t_HbmChannels,
          unsigned int t_MemBits,
          unsigned int t_DataBits,
          unsigned int t_ParEntries,
          unsigned int t_ParamOffset>
void loadColPtr(ap_uint<t_MemBits>* p_nnzColPtr, hls::stream<ap_uint<t_MemBits> >& p_nnzColStr) {
    static const unsigned int t_ParamsPerDdr = t_MemBits / 32;
    static const unsigned int t_ParamDdrOffset = t_ParamOffset * 8 / t_MemBits;
    static const unsigned int t_IntsPerDdr = t_MemBits / (8 * sizeof(uint32_t));
    static const unsigned int t_IntsPerColParam = 2 + t_HbmChannels * 2;
    static const unsigned int t_DdrBlocks4Param = (t_IntsPerColParam + t_IntsPerDdr - 1) / t_IntsPerDdr;

    uint32_t l_paramStore[t_MaxParamDdrBlocks][t_ParamsPerDdr];
#pragma HLS ARRAY_PARTITION variable = l_paramStore complete dim = 2
    unsigned int l_iterations = 0;
    unsigned int l_paramBlocks = 0;
    readMemParamBlocks<t_MaxParamDdrBlocks, t_ParamsPerDdr, t_MemBits, t_ParamDdrOffset>(p_nnzColPtr, l_paramBlocks,
                                                                                         l_iterations, l_paramStore);

    uint32_t l_offset, l_memBlocks;
    uint32_t l_parBlocks[t_HbmChannels];
#pragma HLS ARRAY_PARTITION variable = l_parBlocks complete dim = 1
    uint32_t l_nnzBlocks[t_HbmChannels];
#pragma HLS ARRAY_PARTITION variable = l_nnzBlocks complete dim = 1
    ap_uint<t_MemBits>* l_datPtr;

    uint32_t l_totalParams = l_paramBlocks / t_DdrBlocks4Param;
    WideType<uint32_t, t_ParamsPerDdr> l_param;
#if SEQ_KERNEL
    l_param[0] = l_totalParams;
    l_param[1] = l_iterations;
    p_nnzColStr.write(l_param);
#endif

    unsigned int l_i = 0;
    while (l_iterations > 0) {
        getColParam<t_MaxParamDdrBlocks, t_ParamsPerDdr, t_DdrBlocks4Param, t_HbmChannels>(
            l_paramStore, l_i, l_offset, l_memBlocks, l_parBlocks, l_nnzBlocks);
        colParam2Str<t_ParamsPerDdr, t_DdrBlocks4Param, t_HbmChannels>(l_offset, l_memBlocks, l_parBlocks, l_nnzBlocks,
                                                                       p_nnzColStr);
        l_datPtr = p_nnzColPtr + l_offset;
        readMem<t_MemBits>(l_datPtr, l_memBlocks, p_nnzColStr);
        l_i += t_DdrBlocks4Param;
        if (l_i == l_paramBlocks) {
            l_i = 0;
            l_iterations--;
        }
    }
}
template <unsigned int t_MaxParamDdrBlocks,
          unsigned int t_HbmChannels,
          unsigned int t_MemBits,
          unsigned int t_DataBits,
          unsigned int t_ParEntries,
          unsigned int t_ParamOffset>
void loadCol(ap_uint<t_MemBits>* p_colValPtr,
             ap_uint<t_MemBits>* p_nnzColPtr,
             hls::stream<ap_uint<t_MemBits> >& p_colVecStr,
             hls::stream<ap_uint<t_MemBits> >& p_nnzColStr) {
#pragma HLS DATAFLOW
    loadColVec<t_MaxParamDdrBlocks, t_HbmChannels, t_MemBits, t_DataBits, t_ParEntries, t_ParamOffset>(p_colValPtr,
                                                                                                       p_colVecStr);

    loadColPtr<t_MaxParamDdrBlocks, t_HbmChannels, t_MemBits, t_DataBits, t_ParEntries, t_ParamOffset>(p_nnzColPtr,
                                                                                                       p_nnzColStr);
}

template <unsigned int t_MaxColBlocks,
          unsigned int t_HbmChannels,
          unsigned int t_ParEntries,
          unsigned int t_MemBits,
          unsigned int t_DataBits>
void bufTransColVec(hls::stream<ap_uint<t_MemBits> >& p_colVecStr,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> > p_datOutStr[t_HbmChannels]) {
    static const unsigned int t_ParWords = t_MemBits / (t_DataBits * t_ParEntries);
    static const unsigned int t_MaxColParBlocks = t_MaxColBlocks * t_ParWords;
    static const unsigned int t_IntsPerColParam = 2 + t_HbmChannels * 2;
    static const unsigned int t_ParamsPerPar = t_DataBits * t_ParEntries / 32;
    static const unsigned int t_ParBlocks4Param = (t_IntsPerColParam + t_ParamsPerPar - 1) / t_ParamsPerPar;

    hls::stream<ap_uint<t_DataBits * t_ParEntries> > l_datStr;
#pragma HLS STREAM variable = l_datStr depth = t_MaxColParBlocks
#pragma HLS DATAFLOW
    readCol<t_HbmChannels, t_ParEntries, t_MemBits, t_DataBits>(p_colVecStr, l_datStr);

    dispCol<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(l_datStr, p_datOutStr);
}

template <unsigned int t_MaxColBlocks,
          unsigned int t_HbmChannels,
          unsigned int t_ParEntries,
          unsigned int t_MemBits,
          unsigned int t_DataBits>
void bufTransNnzCol(hls::stream<ap_uint<t_MemBits> >& p_nnzColStr,
                    hls::stream<ap_uint<t_DataBits * t_ParEntries> > p_datOutStr[t_HbmChannels]) {
    static const unsigned int t_ParWords = t_MemBits / (t_DataBits * t_ParEntries);
    static const unsigned int t_MaxColParBlocks = t_MaxColBlocks * t_ParWords;
    static const unsigned int t_IntsPerColParam = 2 + t_HbmChannels * 2;
    static const unsigned int t_ParamsPerPar = t_DataBits * t_ParEntries / 32;
    static const unsigned int t_ParBlocks4Param = (t_IntsPerColParam + t_ParamsPerPar - 1) / t_ParamsPerPar;
    static const unsigned int t_DatStreamDepth = t_MaxColParBlocks * t_HbmChannels;

    hls::stream<ap_uint<t_DataBits * t_ParEntries> > l_datStr;
#pragma HLS STREAM variable = l_datStr depth = t_DatStreamDepth

#pragma HLS DATAFLOW
    readCol<t_HbmChannels, t_ParEntries, t_MemBits, t_DataBits>(p_nnzColStr, l_datStr);

    dispNnzCol<t_MaxColParBlocks, t_ParBlocks4Param, t_HbmChannels, t_ParEntries, t_DataBits>(l_datStr, p_datOutStr);
}
} // end namespace sparse
} // end namespace xf
#endif
