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
 * @file cscmv.hpp
 * @brief SPARSE Level 1 cscmv template function implementation.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_CSCMV_HPP
#define XF_SPARSE_CSCMV_HPP

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

template <typename t_IndexType = unsigned int>
class ColPtrPair {
   public:
    ColPtrPair() {}
    ColPtrPair(unsigned int p_start, unsigned int p_end) : m_start(p_start), m_end(p_end) {}
    ColPtrPair& operator=(ColPtrPair p_src) {
        m_start = p_src.getStart();
        m_end = p_src.getEnd();
        return (*this);
    }
    inline bool operator==(ColPtrPair& p_src) { return ((p_src.getStart() == m_start) && (p_src.getEnd() == m_end)); }
    inline t_IndexType& getStart() { return m_start; }
    inline t_IndexType& getEnd() { return m_end; }

   private:
    t_IndexType m_start;
    t_IndexType m_end;
};

template <unsigned int t_LogParEntries, typename t_IndexType = unsigned int, unsigned int t_IndexBits = 32>
class ColPtrPairDist {
   public:
    ColPtrPairDist() {}
    ColPtrPairDist(t_IndexType p_dist,
                   ap_uint<t_LogParEntries> p_startMod,
                   ap_uint<t_LogParEntries> p_endMod,
                   ap_uint<1> p_end)
        : m_dist(p_dist), m_startMod(p_startMod), m_endMod(p_endMod), m_end(p_end) {}

    inline t_IndexType& getDist() { return m_dist; }
    inline ap_uint<t_LogParEntries>& getStartMod() { return m_startMod; }
    inline ap_uint<t_LogParEntries>& getEndMod() { return m_endMod; }
    inline ap_uint<1>& getEnd() { return m_end; }

   private:
    t_IndexType m_dist;
    ap_uint<t_LogParEntries> m_startMod;
    ap_uint<t_LogParEntries> m_endMod;
    ap_uint<1> m_end;
};

template <typename t_DataType,
          typename t_IndexType = unsigned int,
          unsigned int t_DataBits = 32,
          unsigned int t_IndexBits = 32>
class RowEntry {
   public:
    RowEntry() {}
    RowEntry(t_DataType p_val, t_IndexType p_row) : m_val(p_val), m_row(p_row) {}

   public:
    t_IndexType& getRow() {
#pragma HLS INLINE
        return m_row;
    }
    t_DataType& getVal() {
#pragma HLS INLINE
        return m_val;
    }
    ap_uint<t_DataBits + t_IndexBits> toBits() {
#pragma HLS INLINE
        BitConv<t_DataType> l_conVal;
        ap_uint<t_DataBits> l_valBits;
        ap_uint<t_IndexBits> l_indexBits;
        ap_uint<t_DataBits + t_IndexBits> l_res;
        l_valBits = l_conVal.toBits(m_val);
        l_indexBits = (ap_uint<t_IndexBits>)(m_row);
        l_res.range(t_IndexBits - 1, 0) = l_indexBits;
        l_res.range(t_DataBits + t_IndexBits - 1, t_IndexBits) = l_valBits;
        return l_res;
    }
    void toVal(ap_uint<t_DataBits + t_IndexBits> p_bits) {
#pragma HLS INLINE
        ap_uint<t_DataBits> l_valBits = p_bits.range(t_DataBits + t_IndexBits - 1, t_IndexBits);
        ap_uint<t_IndexBits> l_indexBits = p_bits.range(t_IndexBits - 1, 0);
        BitConv<t_DataType> l_conVal;
        m_val = l_conVal.toType(l_valBits);
        m_row = (t_IndexType)(l_indexBits);
    }
    void print(std::ostream& os) {
        os << std::setw(SPARSE_printWidth) << int(getRow()) << " " << std::setw(SPARSE_printWidth) << getVal() << " ";
    }

   private:
    t_DataType m_val;
    t_IndexType m_row;
};

template <typename t_DataType, typename t_IndexType, unsigned int t_DataBits = 32, unsigned int t_IndexBits = 32>
std::ostream& operator<<(std::ostream& os, RowEntry<t_DataType, t_IndexType, t_DataBits, t_IndexBits>& p_val) {
    p_val.print(os);
    return (os);
}

template <unsigned int t_LogParEntries, typename t_IndexType = unsigned int>
ap_uint<t_LogParEntries> getRowBank(t_IndexType p_row) {
#pragma HLS inline
    const unsigned int t_ParEntries = 1 << t_LogParEntries;
    return (p_row % t_ParEntries);
}

template <unsigned int t_LogParEntries,
          unsigned int t_LogParGroups,
          typename t_IndexType = unsigned int,
          unsigned int t_IndexBits = 32>
ap_uint<t_IndexBits - t_LogParEntries - t_LogParGroups> getRowOffset(t_IndexType p_row) {
#pragma HLS inline
    const unsigned int t_ParEntries = 1 << t_LogParEntries;
    const unsigned int t_ParGroups = 1 << t_LogParGroups;
    return (p_row / (t_ParEntries * t_ParGroups));
}

template <unsigned int t_ParEntries, typename t_DataType, unsigned int t_DataBits, unsigned int t_NumCopys>
void duplicateStream(const unsigned int p_blocks,
                     hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_inStr,
                     hls::stream<ap_uint<t_DataBits * t_ParEntries> > p_outStr[t_NumCopys]) {
    for (unsigned int i = 0; i < p_blocks; ++i) {
#pragma HLS PIPELINE
        ap_uint<t_DataBits * t_ParEntries> l_valBits;
        l_valBits = p_inStr.read();
        for (unsigned int j = 0; j < t_NumCopys; ++j) {
            p_outStr[j].write(l_valBits);
        }
    }
}
template <unsigned int t_LogParEntries, typename t_IndexType = unsigned int, unsigned int t_IndexBits = 32>
void getColPtrPair(const unsigned int p_colPtrBlocks,
                   hls::stream<ap_uint<t_IndexBits*(1 << t_LogParEntries)> >& p_colPtrStr,
                   hls::stream<ColPtrPair<t_IndexType> > p_colPtrPairStr[1 << t_LogParEntries]) {
    const unsigned int t_ParEntries = 1 << t_LogParEntries;

    ap_uint<t_IndexBits * t_ParEntries> l_colPtrArrBits;
    WideType<ColPtrPair<t_IndexType>, t_ParEntries> l_colPtrPairArr;
#pragma HLS ARRAY_PARTITION variable = l_colPtrPairArr complete
    WideType<t_IndexType, t_ParEntries> l_colPtrArr(0);
#pragma HLS ARRAY_PARTITION variable = l_colPtrArr complete

    for (unsigned int i = 0; i < p_colPtrBlocks; ++i) {
#pragma HLS PIPELINE
        l_colPtrPairArr[0].getStart() = l_colPtrArr[t_ParEntries - 1];
        l_colPtrArrBits = p_colPtrStr.read();
        WideType<t_IndexType, t_ParEntries> l_colPtrArrTmp(l_colPtrArrBits);
#pragma HLS ARRAY_PARTITION variable = l_colPtrArrTmp complete
        l_colPtrArr = l_colPtrArrTmp;
        for (unsigned int j = 1; j < t_ParEntries; ++j) {
            l_colPtrPairArr[j].getStart() = l_colPtrArr[j - 1];
        }
        for (unsigned int j = 0; j < t_ParEntries; ++j) {
            l_colPtrPairArr[j].getEnd() = l_colPtrArr[j];
        }
        for (unsigned int j = 0; j < t_ParEntries; ++j) {
            if (l_colPtrPairArr[j].getEnd() > l_colPtrPairArr[j].getStart()) {
                p_colPtrPairStr[j].write(l_colPtrPairArr[j]);
            }
        }
    }

    ColPtrPair<t_IndexType> l_endPtrPair(0, 0);
    for (unsigned int i = 0; i < t_ParEntries; ++i) {
#pragma HLS UNROLL
        p_colPtrPairStr[i].write(l_endPtrPair);
    }
}

template <unsigned int t_LogParEntries, typename t_IndexType = unsigned int, unsigned int t_IndexBits = 32>
void getColPtrPairDist(
    const unsigned int p_colPtrBlocks,
    hls::stream<ap_uint<t_IndexBits*(1 << t_LogParEntries)> >& p_colPtrStr,
    hls::stream<ColPtrPairDist<t_LogParEntries, t_IndexType, t_IndexBits> > p_colPtrPairDistStr[1 << t_LogParEntries]) {
    const unsigned int t_ParEntries = 1 << t_LogParEntries;

    ap_uint<t_IndexBits * t_ParEntries> l_colPtrArrBits;
    WideType<ColPtrPair<t_IndexType>, t_ParEntries> l_colPtrPairArr;
#pragma HLS ARRAY_PARTITION variable = l_colPtrPairArr complete
    WideType<t_IndexType, t_ParEntries> l_colPtrArr(0);
#pragma HLS ARRAY_PARTITION variable = l_colPtrArr complete

    for (unsigned int i = 0; i < p_colPtrBlocks; ++i) {
#pragma HLS PIPELINE
        l_colPtrPairArr[0].getStart() = l_colPtrArr[t_ParEntries - 1];
        l_colPtrArrBits = p_colPtrStr.read();
        WideType<t_IndexType, t_ParEntries> l_colPtrArrTmp(l_colPtrArrBits);
#pragma HLS ARRAY_PARTITION variable = l_colPtrArrTmp complete
        l_colPtrArr = l_colPtrArrTmp;
        for (unsigned int j = 1; j < t_ParEntries; ++j) {
            l_colPtrPairArr[j].getStart() = l_colPtrArr[j - 1];
        }
        for (unsigned int j = 0; j < t_ParEntries; ++j) {
            l_colPtrPairArr[j].getEnd() = l_colPtrArr[j];
        }
        for (unsigned int j = 0; j < t_ParEntries; ++j) {
#ifndef __SYNTHESIS__
            if (l_colPtrPairArr[j].getEnd() < l_colPtrPairArr[j].getStart()) {
                std::cout << "ERROR: colPtr at " << i * t_ParEntries + j << " start=" << l_colPtrPairArr[j].getStart()
                          << " end=" << l_colPtrPairArr[j].getEnd() << std::endl;
            }
            assert(l_colPtrPairArr[j].getEnd() >= l_colPtrPairArr[j].getStart());
#endif
            t_IndexType l_dist = l_colPtrPairArr[j].getEnd() - l_colPtrPairArr[j].getStart();
            ap_uint<t_LogParEntries> l_startMod = l_colPtrPairArr[j].getStart() % t_ParEntries;
            ap_uint<t_LogParEntries> l_endMod = l_colPtrPairArr[j].getEnd() % t_ParEntries;
            ColPtrPairDist<t_LogParEntries, t_IndexType, t_IndexBits> l_colPtrPairDist(l_dist, l_startMod, l_endMod, 0);
            p_colPtrPairDistStr[j].write(l_colPtrPairDist);
        }
    }

    for (unsigned int i = 0; i < t_ParEntries; ++i) {
#pragma HLS UNROLL
        ColPtrPairDist<t_LogParEntries, t_IndexType, t_IndexBits> l_colPtrPairDist(0, 0, 0, 1);
        p_colPtrPairDistStr[i].write(l_colPtrPairDist);
    }
}
template <unsigned int t_ParEntries, typename t_DataType, unsigned int t_DataBits = 32>
void splitStream(const unsigned int p_colBlocks,
                 hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_inStr,
                 hls::stream<t_DataType> p_outStr[t_ParEntries]) {
    for (unsigned int i = 0; i < p_colBlocks; ++i) {
#pragma HLS PIPELINE
        ap_uint<t_DataBits * t_ParEntries> l_valBits;
        l_valBits = p_inStr.read();
        WideType<t_DataType, t_ParEntries> l_val(l_valBits);
#pragma HLS ARRAY_PARTITION variable = l_val complete
        for (unsigned int j = 0; j < t_ParEntries; ++j) {
            p_outStr[j].write(l_val[j]);
        }
    }

    for (unsigned int j = 0; j < t_ParEntries; ++j) {
#pragma HLS UNROLL
        p_outStr[j].write(0);
    }
}

template <unsigned int t_LogParEntries, typename t_IndexType = unsigned int, unsigned int t_IndexBits = 32>
void genColSelContr(const unsigned int p_nnzBlocks,
                    hls::stream<ColPtrPair<t_IndexType> >& p_colPtrPairStr,
                    hls::stream<BoolArr<1 << t_LogParEntries> >& p_colSelStr) {
    const unsigned int t_ParEntries = 1 << t_LogParEntries;
    const unsigned int t_BlockBits = t_IndexBits - t_LogParEntries;

    ap_uint<t_BlockBits> l_nnzBlocks = 0;
    bool l_stop = false;
    bool l_end = false;
    BoolArr<t_ParEntries> l_selVal(false);
    ColPtrPair<t_IndexType> l_zeroPtrPair(0, 0);
    ColPtrPair<t_IndexType> l_colPtrPair = p_colPtrPairStr.read();

    ap_uint<t_BlockBits> l_colPtrStartBlock = l_colPtrPair.getStart() / t_ParEntries;
    ap_uint<t_LogParEntries> l_colPtrStartMod = l_colPtrPair.getStart() % t_ParEntries;
    ap_uint<t_BlockBits> l_colPtrEndBlock = l_colPtrPair.getEnd() / t_ParEntries;
    ap_uint<t_LogParEntries> l_colPtrEndMod = l_colPtrPair.getEnd() % t_ParEntries;

    while (l_nnzBlocks < p_nnzBlocks) {
#pragma HLS PIPELINE
        l_end = (l_colPtrPair == l_zeroPtrPair);
        l_stop = l_colPtrEndBlock > l_nnzBlocks;
        bool l_afterStart = l_nnzBlocks > l_colPtrStartBlock;
        bool l_equalStart = l_nnzBlocks == l_colPtrStartBlock;
        bool l_beforeEnd = l_nnzBlocks < l_colPtrEndBlock;
        bool l_equalEnd = l_nnzBlocks == l_colPtrEndBlock;
        for (unsigned int i = 0; i < t_ParEntries; ++i) {
            if ((l_afterStart || (l_equalStart && i >= l_colPtrStartMod)) &&
                (l_beforeEnd || (l_equalEnd && i < l_colPtrEndMod))) {
                l_selVal[i] = true;
            }
        }
        if (l_stop || l_end) {
            p_colSelStr.write(l_selVal);
            l_selVal.Reset();
            l_nnzBlocks++;
        }
        if (!l_stop && !l_end) {
            l_colPtrPair = p_colPtrPairStr.read();
            l_colPtrStartBlock = l_colPtrPair.getStart() / t_ParEntries;
            l_colPtrStartMod = l_colPtrPair.getStart() % t_ParEntries;
            l_colPtrEndBlock = l_colPtrPair.getEnd() / t_ParEntries;
            l_colPtrEndMod = l_colPtrPair.getEnd() % t_ParEntries;
        }
    }
    if (!l_end) {
        l_colPtrPair = p_colPtrPairStr.read();
    }
    p_colSelStr.write(l_selVal);
}

template <unsigned int t_LogParEntries,
          typename t_DataType,
          typename t_IndexType = unsigned int,
          unsigned int t_IndexBits = 32>
void selColVal(hls::stream<ColPtrPairDist<t_LogParEntries, t_IndexType, t_IndexBits> >& p_colPtrPairDistStr,
               hls::stream<t_DataType>& p_colValStr,
               hls::stream<t_DataType> p_colValSplitStr[1 << t_LogParEntries]) {
    const unsigned int t_ParEntries = 1 << t_LogParEntries;
    const unsigned int t_BlockBits = t_IndexBits - t_LogParEntries;

    WideType<t_DataType, t_ParEntries> l_colValArr;
#pragma HLS ARRAY_PARTITION variable = l_colValArr complete
    ColPtrPairDist<t_LogParEntries, t_IndexType, t_IndexBits> l_colPtrPairDist = p_colPtrPairDistStr.read();
    t_DataType l_colVal = p_colValStr.read();
    ap_uint<t_LogParEntries> l_colPtrStartMod = l_colPtrPairDist.getStartMod();
    ap_uint<t_LogParEntries> l_colPtrEndMod = l_colPtrPairDist.getEndMod();
    ap_uint<1> l_end = l_colPtrPairDist.getEnd();
    t_IndexType l_dist = l_colPtrPairDist.getDist();
    unsigned int l_blocks = 1;

    while (!l_end) {
#pragma HLS PIPELINE
        bool l_stop = l_dist > (l_blocks << t_LogParEntries);
        bool l_valid = (l_dist != 0);

        for (unsigned int i = 0; i < t_ParEntries; ++i) {
            if (l_stop || (l_valid && (l_colPtrEndMod <= l_colPtrStartMod) &&
                           ((i >= l_colPtrStartMod) || (i < l_colPtrEndMod))) ||
                (l_valid && (l_colPtrEndMod > l_colPtrStartMod) && (i >= l_colPtrStartMod) && (i < l_colPtrEndMod))) {
                p_colValSplitStr[i].write(l_colVal);
            }
        }
        if (!l_stop && !l_end) {
            l_colPtrPairDist = p_colPtrPairDistStr.read();
            l_colVal = p_colValStr.read();
            l_colPtrStartMod = l_colPtrPairDist.getStartMod();
            l_colPtrEndMod = l_colPtrPairDist.getEndMod();
            l_dist = l_colPtrPairDist.getDist();
            l_end = l_colPtrPairDist.getEnd();
            l_blocks = 1;
        } else {
            l_colPtrStartMod += t_ParEntries;
            l_blocks++;
        }
    }
}

template <unsigned int t_ParEntries,
          typename t_DataType,
          typename t_IndexType = unsigned int,
          unsigned int t_DataBits = 32>
void xBarMergeCol(hls::stream<BoolArr<t_ParEntries> > p_colSelStr[t_ParEntries],
                  hls::stream<t_DataType> p_colValStr[t_ParEntries][t_ParEntries],
                  hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_nnzColStr) {
    BoolArr<t_ParEntries> l_outValidArr(false);
    bool l_exit = false;
    WideType<t_DataType, t_ParEntries> l_outVal;
    ap_uint<t_DataBits * t_ParEntries> l_outValBits;

#pragma HLS ARRAY_PARTITION variable = l_outValidArr complete
#pragma HLS ARRAY_PARTITION variable = l_outVal complete
    while (!l_exit) {
#pragma HLS PIPELINE
        bool l_continue = false;
        BoolArr<t_ParEntries> l_colSelArr[t_ParEntries];
#pragma HLS ARRAY_PARTITION variable = l_colSelArr complete dim = 0
        for (unsigned int i = 0; i < t_ParEntries; ++i) {
            l_colSelArr[i] = p_colSelStr[i].read();
        }
        for (unsigned int i = 0; i < t_ParEntries; ++i) {
            l_continue = l_continue || l_colSelArr[i].Or();
        }
        l_exit = !l_continue;
        for (unsigned int i = 0; i < t_ParEntries; ++i) {
            for (unsigned int j = 0; j < t_ParEntries; ++j) {
                if (l_colSelArr[j][i]) {
                    l_outValidArr[i] = true;
                    l_outVal[i] = p_colValStr[j][i].read();
                    break;
                }
            }
        }

        if (l_outValidArr.And()) {
            l_outValBits = l_outVal;
            p_nnzColStr.write(l_outValBits);
            l_outValidArr.Reset();
        }
    }
}

/**
 * @brief xBarCol function that distributes input col values to the dedicated banks according to their col index
 * pointers
 *
 * @tparam t_LogParEntries log2 of the parallelly processed entries in the input/output vector stream
 * @tparam t_DataType the data type of the matrix and vector entries
 * @tparam t_IndexType the data type of the indicies
 * @tparam t_DataBits the number of bits for storing the data
 * @tparam t_IndexBits the number of bits for storing the indices
 *
 * @param p_colPtrBlocks the number of col index pointer blocks
 * @param p_nnzBlocks the number of NNZ blocks
 * @param p_colPtrStr the input col pointer vector stream
 * @param p_colValStr the input col value vector stream
 * @param p_nnzColValStr the output banked col value vector stream
 */
template <unsigned int t_LogParEntries,
          typename t_DataType,
          typename t_IndexType = unsigned int,
          unsigned int t_DataBits = 32,
          unsigned int t_IndexBits = 32>
void xBarCol(const unsigned int p_colPtrBlocks,
             const unsigned int p_nnzBlocks,
             hls::stream<ap_uint<t_IndexBits*(1 << t_LogParEntries)> >& p_colPtrStr,
             hls::stream<ap_uint<t_DataBits*(1 << t_LogParEntries)> >& p_colValStr,
             hls::stream<ap_uint<t_DataBits*(1 << t_LogParEntries)> >& p_nnzColValStr) {
    const unsigned int t_ParEntries = 1 << t_LogParEntries;

    hls::stream<ap_uint<t_IndexBits * t_ParEntries> > l_colPtrStr[2];
    hls::stream<ColPtrPair<t_IndexType> > l_colPtrPairStr[t_ParEntries];
    hls::stream<ColPtrPairDist<t_LogParEntries, t_IndexType, t_IndexBits> > l_colPtrPairDistStr[t_ParEntries];
    hls::stream<BoolArr<t_ParEntries> > l_colSelStr[t_ParEntries];
    hls::stream<t_DataType> l_colValStr[t_ParEntries];
    hls::stream<t_DataType> l_colValSplitStr[t_ParEntries][t_ParEntries];
#pragma HLS ARRAY_PARTITION variable = l_colValSplitStr complete dim = 0
#pragma HLS STREAM variable = l_colPtrStr depth = 4
#pragma HLS STREAM variable = l_colPtrPairStr depth = 4
#pragma HLS STREAM variable = l_colPtrPairDistStr depth = 4
#pragma HLS STREAM variable = l_colSelStr depth = 4
#pragma HLS STREAM variable = l_colValStr depth = 4
#pragma HLS STREAM variable = l_colValSplitStr depth = 16
#pragma HLS DATAFLOW
    duplicateStream<t_ParEntries, t_IndexType, t_IndexBits, 2>(p_colPtrBlocks, p_colPtrStr, l_colPtrStr);
    getColPtrPair<t_LogParEntries, t_IndexType, t_IndexBits>(p_colPtrBlocks, l_colPtrStr[0], l_colPtrPairStr);
    getColPtrPairDist<t_LogParEntries, t_IndexType, t_IndexBits>(p_colPtrBlocks, l_colPtrStr[1], l_colPtrPairDistStr);
    splitStream<t_ParEntries, t_DataType, t_DataBits>(p_colPtrBlocks, p_colValStr, l_colValStr);
    for (unsigned int i = 0; i < t_ParEntries; ++i) {
#pragma HLS UNROLL
        genColSelContr<t_LogParEntries, t_IndexType, t_IndexBits>(p_nnzBlocks, l_colPtrPairStr[i], l_colSelStr[i]);
        selColVal<t_LogParEntries, t_DataType, t_IndexType, t_IndexBits>(l_colPtrPairDistStr[i], l_colValStr[i],
                                                                         l_colValSplitStr[i]);
    }
    xBarMergeCol<t_ParEntries, t_DataType, t_IndexType, t_DataBits>(l_colSelStr, l_colValSplitStr, p_nnzColValStr);
#ifndef __SYNTHESIS__
    for (unsigned int j = 0; j < t_ParEntries; ++j) {
        if (!l_colPtrPairStr[j].empty()) {
            std::cout << "ERROR: l_colPtrPairStr[" << j << "] not empty" << std::endl;
        }
    }
    for (unsigned int i = 0; i < t_ParEntries; ++i) {
        if (!l_colSelStr[i].empty()) {
            std::cout << "ERROR: l_colSelStr[" << i << "]"
                      << " not empty" << std::endl;
        }
        if (!l_colValStr[i].empty()) {
            std::cout << "ERROR: l_colValStr[" << i << "]"
                      << " not empty" << std::endl;
        }
        for (unsigned int j = 0; j < t_ParEntries; ++j) {
            if (!l_colValSplitStr[i][j].empty()) {
                std::cout << "ERROR: l_colValSplitStr[" << i << "][" << j << "] not empty" << std::endl;
                while (!l_colValSplitStr[i][j].empty()) {
                    t_DataType l_val = l_colValSplitStr[i][j].read();
                    std::cout << "      read out " << l_val << " from  l_colValSplitStr[" << i << "][" << j << "]"
                              << std::endl;
                }
            }
        }
    }
#endif
}

template <unsigned int t_LogParEntries,
          typename t_DataType,
          typename t_IndexType = unsigned int,
          unsigned int t_DataBits = 32,
          unsigned int t_IndexBits = 32>
void formRowEntries(const unsigned int p_nnzBlocks,
                    hls::stream<ap_uint<t_DataBits*(1 << t_LogParEntries)> >& p_nnzValStr,
                    hls::stream<ap_uint<t_DataBits*(1 << t_LogParEntries)> >& p_nnzColValStr,
                    hls::stream<ap_uint<t_IndexBits*(1 << t_LogParEntries)> >& p_rowIndexStr,
                    hls::stream<ap_uint<t_DataBits + t_IndexBits> > p_rowEntryStr[1 << t_LogParEntries],
                    hls::stream<ap_uint<1> > p_isEndStr[1 << t_LogParEntries]) {
    const unsigned int t_ParEntries = 1 << t_LogParEntries;
    for (unsigned int i = 0; i < p_nnzBlocks; ++i) {
#pragma HLS PIPELINE
        ap_uint<t_DataBits * t_ParEntries> l_nnzBits;
        ap_uint<t_DataBits * t_ParEntries> l_nnzColBits;
        ap_uint<t_DataBits * t_ParEntries> l_rowIndexBits;

        RowEntry<t_DataType, t_IndexType> l_rowEntry[t_ParEntries];
#pragma HLS ARRAY_PARTITION variable = l_rowEntry complete
        l_nnzBits = p_nnzValStr.read();
        l_nnzColBits = p_nnzColValStr.read();
        l_rowIndexBits = p_rowIndexStr.read();
        WideType<t_DataType, t_ParEntries> l_nnzVal(l_nnzBits);
        WideType<t_DataType, t_ParEntries> l_nnzColVal(l_nnzColBits);
        WideType<t_IndexType, t_ParEntries> l_rowIndex(l_rowIndexBits);
#pragma HLS ARRAY_PARTITION variable = l_nnzVal complete
#pragma HLS ARRAY_PARTITION variable = l_nnzColVal complete
#pragma HLS ARRAY_PARTITION variable = l_rowIndex complete

        for (unsigned int j = 0; j < t_ParEntries; ++j) {
            l_rowEntry[j].getVal() = l_nnzVal[j] * l_nnzColVal[j];
            l_rowEntry[j].getRow() = l_rowIndex[j];
            if (l_rowEntry[j].getVal() != 0) {
                p_rowEntryStr[j].write(l_rowEntry[j].toBits());
                p_isEndStr[j].write(0);
            }
        }
    }
    for (unsigned int j = 0; j < t_ParEntries; ++j) {
#pragma HLS UNROLL
        p_isEndStr[j].write(1);
    }
}

template <unsigned int t_LogParEntries,
          typename t_DataType,
          typename t_IndexType = unsigned int,
          unsigned int t_DataBits = 32,
          unsigned int t_IndexBits = 32>
void xBarRowSplit(hls::stream<ap_uint<t_DataBits + t_IndexBits> >& p_rowEntryStr,
                  hls::stream<ap_uint<1> >& p_isEndStr,
                  hls::stream<ap_uint<t_DataBits + t_IndexBits> > p_splittedRowEntryStr[1 << t_LogParEntries],
                  hls::stream<ap_uint<1> >& p_isEndOutStr) {
    const unsigned int t_ParEntries = 1 << t_LogParEntries;
    ap_uint<1> l_end = p_isEndStr.read();
    while (!l_end) {
#pragma HLS PIPELINE
        ap_uint<t_DataBits + t_IndexBits> l_rowEntryBits = p_rowEntryStr.read();
        RowEntry<t_DataType, t_IndexType> l_rowEntry;
        l_rowEntry.toVal(l_rowEntryBits);
        l_end = p_isEndStr.read();
        ap_uint<t_LogParEntries> l_bank = getRowBank<t_LogParEntries, t_IndexType>(l_rowEntry.getRow());
        p_splittedRowEntryStr[l_bank].write(l_rowEntryBits);
    }
    // for (unsigned int i = 0; i < t_ParEntries; ++i) {
    p_isEndOutStr.write(1);
    //}
}

template <unsigned int t_LogParEntries,
          typename t_DataType,
          typename t_IndexType = unsigned int,
          unsigned int t_DataBits = 32,
          unsigned int t_IndexBits = 32>
void xBarRowMerge(
    hls::stream<ap_uint<t_DataBits + t_IndexBits> > p_splittedRowEntryStr[1 << t_LogParEntries][1 << t_LogParEntries],
    hls::stream<ap_uint<1> > p_isEndStr[1 << t_LogParEntries],
    hls::stream<ap_uint<t_DataBits + t_IndexBits> > p_rowEntryStr[1 << t_LogParEntries],
    hls::stream<ap_uint<1> > p_isEndOutStr[1 << t_LogParEntries]) {
    const unsigned int t_ParEntries = 1 << t_LogParEntries;
    BoolArr<t_ParEntries> l_preDone(false);
    BoolArr<t_ParEntries> l_activity(false);
    bool l_exit = false;

    while (!l_exit) {
#pragma HLS PIPELINE
        if (l_preDone.And() && !l_activity.Or()) {
            l_exit = true;
        }
        for (unsigned int i = 0; i < t_ParEntries; ++i) {
            ap_uint<1> l_unused;
            if (p_isEndStr[i].read_nb(l_unused)) {
                l_preDone[i] = true;
            }
        }
        l_activity.Reset();

        for (unsigned int b = 0; b < t_ParEntries; ++b) {
#pragma HLS UNROLL
            ap_uint<t_LogParEntries> l_ch = 0;
            for (unsigned int c = 0; c < t_ParEntries; ++c) {
#pragma HLS UNROLL
                ap_uint<t_LogParEntries> l_bank = (b + c) % t_ParEntries;
                if (!p_splittedRowEntryStr[l_bank][b].empty()) {
                    l_ch = l_bank;
                    l_activity[b] = true;
                    break;
                }
            }
            ap_uint<t_DataBits + t_IndexBits> l_valBits;
            if (p_splittedRowEntryStr[l_ch][b].read_nb(l_valBits)) {
                p_rowEntryStr[b].write(l_valBits);
            }
        }
    }
    for (unsigned int i = 0; i < t_ParEntries; ++i) {
#pragma HLS UNROLL
        p_isEndOutStr[i].write(1);
    }
}

/**
 * @brief xBarRow function that multiplies input NNZs' values with input vectors and distributes the results to the
 * dedicated banks according to their row indices
 *
 * @tparam t_LogParEntries log2 of the parallelly processed entries in the input/output vector stream
 * @tparam t_DataType the data type of the matrix and vector entries
 * @tparam t_IndexType the data type of the indicies
 * @tparam t_DataBits the number of bits for storing the data
 * @tparam t_IndexBits the number of bits for storing the indices
 *
 * @param p_nnzBlocks the number of NNZ blocks
 * @param p_nnzValStr the input NNZ value stream
 * @param p_nnzColValStr the input col value stream
 * @param p_rowIndexStr the inpuut NNZ row index stream
 * @param p_rowEntryStr the output banked multiplication results stream array
 * @param p_isEndStr the output control stream
 */
template <unsigned int t_LogParEntries,
          typename t_DataType,
          typename t_IndexType = unsigned int,
          unsigned int t_DataBits = 32,
          unsigned int t_IndexBits = 32>
void xBarRow(const unsigned int p_nnzBlocks,
             hls::stream<ap_uint<t_DataBits*(1 << t_LogParEntries)> >& p_nnzValStr,
             hls::stream<ap_uint<t_DataBits*(1 << t_LogParEntries)> >& p_nnzColValStr,
             hls::stream<ap_uint<t_IndexBits*(1 << t_LogParEntries)> >& p_rowIndexStr,
             hls::stream<ap_uint<t_DataBits + t_IndexBits> > p_rowEntryStr[1 << t_LogParEntries],
             hls::stream<ap_uint<1> > p_isEndStr[1 << t_LogParEntries]) {
    const unsigned int t_ParEntries = 1 << t_LogParEntries;
    hls::stream<ap_uint<t_DataBits + t_IndexBits> > l_rowEntryStr[t_ParEntries];
#pragma HLS STREAM variable = l_rowEntryStr depth = 16
    hls::stream<ap_uint<1> > l_isEndStr[t_ParEntries];
#pragma HLS STREAM variable = l_isEndStr depth = 16
    hls::stream<ap_uint<t_DataBits + t_IndexBits> > l_splittedRowEntryStr[t_ParEntries][t_ParEntries];
    hls::stream<ap_uint<1> > l_isEndSplitStr[t_ParEntries];
#pragma HLS DATAFLOW
    formRowEntries<t_LogParEntries, t_DataType, t_IndexType, t_DataBits, t_IndexBits>(
        p_nnzBlocks, p_nnzValStr, p_nnzColValStr, p_rowIndexStr, l_rowEntryStr, l_isEndStr);
    for (unsigned int i = 0; i < t_ParEntries; ++i) {
#pragma HLS UNROLL
        xBarRowSplit<t_LogParEntries, t_DataType, t_IndexType, t_DataBits, t_IndexBits>(
            l_rowEntryStr[i], l_isEndStr[i], l_splittedRowEntryStr[i], l_isEndSplitStr[i]);
    }
    xBarRowMerge<t_LogParEntries, t_DataType, t_IndexType, t_DataBits, t_IndexBits>(
        l_splittedRowEntryStr, l_isEndSplitStr, p_rowEntryStr, p_isEndStr);
}

template <unsigned int t_MaxRowBlocks,
          unsigned int t_ParEntries,
          typename t_DataType,
          typename t_IndexType = unsigned int,
          unsigned int t_DataBits = 32,
          unsigned int t_RowOffsetBits = 32>
void rowMemAcc(const unsigned int p_rowBlocks,
               hls::stream<ap_uint<t_DataBits + t_RowOffsetBits> >& p_rowEntryStr,
               hls::stream<ap_uint<1> >& p_isEndStr,
               hls::stream<ap_uint<t_DataBits> >& p_rowValStr) {
    static const unsigned int t_Latency = 6;

    t_DataType l_rowStore[t_MaxRowBlocks][t_Latency];
#pragma HLS RESOURCE variable = l_rowStore core = RAM_T2P
#pragma HLS ARRAY_PARTITION variable = l_rowStore complete dim = 2

init_mem:
    for (unsigned int i = 0; i < p_rowBlocks; ++i) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT max = t_MaxRowBlocks
        for (unsigned int j = 0; j < t_Latency; j++) {
            l_rowStore[i][j] = 0;
        }
    }

    ap_uint<1> l_exit = 0;
    ap_uint<1> l_preDone = 0;
    BoolArr<t_Latency> l_activity(true);
#pragma HLS ARRAY_PARTITION variable = l_activity complete dim = 1

accumulate:
    while (!l_exit) {
#pragma HLS PIPELINE II = t_Latency
        //#pragma HLS DEPENDENCE variable = l_rowStore array inter false

        if (l_preDone && !l_activity.Or() && p_rowEntryStr.empty()) {
            l_exit = 1;
        }
        ap_uint<1> l_unused;
        if (p_isEndStr.read_nb(l_unused)) {
            l_preDone = 1;
        }

        ap_uint<t_DataBits + t_RowOffsetBits> l_val;
        for (unsigned int l_mod = 0; l_mod < t_Latency; ++l_mod) {
            l_activity[l_mod] = p_rowEntryStr.read_nb(l_val);
            RowEntry<t_DataType, t_IndexType, t_DataBits, t_RowOffsetBits> l_rowEntry;
            l_rowEntry.toVal(l_val);
            t_IndexType l_rowIndex = (l_activity[l_mod]) ? l_rowEntry.getRow() / t_ParEntries : 0;
#ifndef __SYNTHESIS__
            assert(l_rowIndex < t_MaxRowBlocks);
#endif
            t_DataType l_rowVal = l_rowStore[l_rowIndex][l_mod];
            l_rowVal += (l_activity[l_mod]) ? l_rowEntry.getVal() : 0;
            l_rowStore[l_rowIndex][l_mod] = l_rowVal;
        }
    }

    for (unsigned int i = 0; i < p_rowBlocks; ++i) {
#pragma HLS PIPELINE II = 1
        t_DataType l_addRegs[t_Latency];
#pragma HLS ARRAY_PARTITION variable = l_addRegs complete dim = 1
#pragma HLS LOOP_TRIPCOUNT max = t_MaxRowBlocks
        for (unsigned int b = 0; b < t_Latency; ++b) {
#pragma HLS UNROLL
            l_addRegs[b] = l_rowStore[i][b];
        }
        t_DataType l_sum = 0;
        //#pragma HLS BIND_OP variable = l_sum op = fadd impl = fabric
        for (unsigned int b = 0; b < t_Latency; ++b) {
#pragma HLS UNROLL
            l_sum += l_addRegs[b];
        }
        BitConv<t_DataType> l_conVal;
        ap_uint<t_DataBits> l_rowBits = l_conVal.toBits(l_sum);
        p_rowValStr.write(l_rowBits);
    }
}

/**
 * @brief rowAgg function that aggregates multiple row entry streams into one row entry stream
 *
 * @tparam t_ParEntries the parallelly processed entries in the input/output vector stream
 * @tparam t_ParGroups  the number of parallel accumulation paths
 * @tparam t_DataType the data type of the matrix and vector entries
 * @tparam t_IndexType the data type of the indicies
 * @tparam t_DataBits the number of bits for storing the data
 *
 * @param p_rowBlocks the number of row blocks
 * @param p_rowValStr the iutput row entry stream array
 * @param p_rowAggStr the output aggregated row entry stream
 */
template <unsigned int t_ParEntries,
          unsigned int t_ParGroups,
          typename t_DataType,
          typename t_IndexType,
          unsigned int t_DataBits = 32>
void rowAgg(const unsigned int p_rowBlocks,
            hls::stream<ap_uint<t_DataBits> > p_rowValStr[t_ParEntries],
            hls::stream<ap_uint<t_DataBits * t_ParEntries> >& p_rowAggStr) {
    for (unsigned int i = 0; i < p_rowBlocks; ++i) {
#pragma HLS PIPELINE
        ap_uint<t_DataBits * t_ParEntries> l_valOut;
        for (unsigned int b = 0; b < t_ParEntries; ++b) {
            ap_uint<t_DataBits> l_val = p_rowValStr[b].read();
            l_valOut.range((b + 1) * t_DataBits - 1, b * t_DataBits) = l_val;
        }
        p_rowAggStr.write(l_valOut);
    }
}

/**
 * @brief cscRow function that returns the multiplication results of a sparse matrix and a dense vector
 *
 * @tparam t_MaxRowBlocks the maximum number of row entrie blocks buffered onchip per PE
 * @tparam t_LogParEntries log2 of the parallelly processed entries in the input/output vector stream
 * @tparam t_LogParGroups log2 of the number of parallel accumulation paths
 * @tparam t_DataType the data type of the matrix and vector entries
 * @tparam t_IndexType the data type of the indicies
 * @tparam t_DataBits the number of bits for storing the data
 * @tparam t_IndexBits the number of bits for storing the indices
 *
 * @param p_nnzBlocks the number of NNZ vector blocks
 * @param p_rowBlocks the number of result row vector blocks
 * @param p_nnzValStr the input NNZ value vector stream
 * @param p_nnzColValStr the input col vector stream
 * @param p_rowIndexStr the input NNZ index vector stream
 * @param p_rowAggStr the output row vector stream
 */
template <unsigned int t_MaxRowBlocks,
          unsigned int t_LogParEntries,
          unsigned int t_LogParGroups,
          typename t_DataType,
          typename t_IndexType = unsigned int,
          unsigned int t_DataBits = 32,
          unsigned int t_IndexBits = 32>
void cscRow(const unsigned int p_nnzBlocks,
            const unsigned int p_rowBlocks,
            hls::stream<ap_uint<t_DataBits*(1 << t_LogParEntries)> >& p_nnzValStr,
            hls::stream<ap_uint<t_DataBits*(1 << t_LogParEntries)> >& p_nnzColValStr,
            hls::stream<ap_uint<t_IndexBits*(1 << t_LogParEntries)> >& p_rowIndexStr,
            hls::stream<ap_uint<t_DataBits*(1 << t_LogParEntries)> >& p_rowAggStr) {
    const unsigned int t_ParEntries = 1 << t_LogParEntries;
    const unsigned int t_ParGroups = 1 << t_LogParGroups;
    const unsigned int t_RowOffsetBits = t_IndexBits;

    hls::stream<ap_uint<t_DataBits + t_IndexBits> > l_xBarRowDatStr[t_ParEntries];
#pragma HLS STREAM variable = l_xBarRowDatStr depth = 16
    hls::stream<ap_uint<1> > l_xBarRowContStr[t_ParEntries];
    hls::stream<ap_uint<t_DataBits> > l_rowValStr[t_ParEntries];

#pragma HLS DATAFLOW
    xBarRow<t_LogParEntries, t_DataType, t_IndexType, t_DataBits, t_IndexBits>(
        p_nnzBlocks, p_nnzValStr, p_nnzColValStr, p_rowIndexStr, l_xBarRowDatStr, l_xBarRowContStr);
    for (unsigned int i = 0; i < t_ParEntries; ++i) {
#pragma HLS UNROLL
        rowMemAcc<t_MaxRowBlocks, t_ParEntries, t_DataType, t_IndexType, t_DataBits, t_RowOffsetBits>(
            p_rowBlocks, l_xBarRowDatStr[i], l_xBarRowContStr[i], l_rowValStr[i]);
    }

    rowAgg<t_ParEntries, t_ParGroups, t_DataType, t_IndexType, t_DataBits>(p_rowBlocks, l_rowValStr, p_rowAggStr);
}
} // end namespace sparse
} // end namespace xf

#endif
