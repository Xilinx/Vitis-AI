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
 *  @brief Sparse matrix vector multiply  C = A * B
 *
 *  $DateTime: 2019/03/05 09:20:31 $
 *  $Author: Xilinx $
 */

#ifndef XF_FINTECH_BLAS_SPMV_HPP
#define XF_FINTECH_BLAS_SPMV_HPP

#include "assert.h"
#include "hls_stream.h"
#include "blas_types.hpp"

namespace xf {
namespace fintech {
namespace blas {
template <typename t_DataType>
class SpmA {
   private:
    t_DataType m_Val;
    unsigned int m_Col;
    unsigned int m_Row;

   public:
    SpmA() {
#pragma HLS inline
    }
    SpmA(t_DataType p_Val, unsigned int p_Col, unsigned int p_Row) : m_Val(p_Val), m_Col(p_Col), m_Row(p_Row) {
#pragma HLS inline
    }
    t_DataType& getVal() { return m_Val; }
    unsigned int& getCol() { return m_Col; }
    unsigned int& getRow() { return m_Row; }
};

template <typename t_DataType>
class SpmC {
   private:
    t_DataType m_Val;
    unsigned int m_Row;

   public:
    SpmC() {
#pragma HLS inline
    }
    SpmC(t_DataType p_Val, unsigned int p_Row) : m_Val(p_Val), m_Row(p_Row) {
#pragma HLS inline
    }
    t_DataType& getVal() { return m_Val; }
    unsigned int& getRow() { return m_Row; }
};

template <typename t_DataType,
          unsigned int t_MemWidth,   // number of matrix elements in one mem access
          unsigned int t_IndexWidth, // number of indices in one mem access, index type is always unsigned int
          unsigned int t_MaxK,       // maximum number of input vector elements
          unsigned int t_MaxM,       // maximum number of output vector elements
          unsigned int t_MaxNnz      // maximum number of Nnzs
          >
class Spmv {
   private:
    static const unsigned int t_IdxWidthOverMemWidth = t_IndexWidth / t_MemWidth;
    static const unsigned int t_AccWidth = 3; // width for accumulator
    static const unsigned int t_FifoDepth = 2;

   public:
    typedef WideType<t_DataType, t_MemWidth> DataWideType;
    typedef hls::stream<DataWideType> DataWideStreamType;
    typedef WideType<unsigned int, t_MemWidth> IdxWideType;
    typedef hls::stream<IdxWideType> IdxWideStreamType;

    typedef SpmA<t_DataType> SpmAtype;
    typedef WideType<SpmAtype, t_MemWidth> SpmAwideType;
    typedef hls::stream<SpmAwideType> SpmAwideStreamType;
    typedef hls::stream<SpmAtype> SpmAstreamType;

    typedef SpmC<t_DataType> SpmCtype;
    typedef WideType<SpmCtype, t_MemWidth> SpmCwideType;
    typedef hls::stream<SpmCwideType> SpmCwideStreamType;
    typedef hls::stream<SpmCtype> SpmCstreamType;

    typedef WideType<t_DataType, t_AccWidth> AccDataType;
    typedef SpmC<AccDataType> SpmCaccType;
    typedef hls::stream<SpmCaccType> SpmCaccStreamType;

    typedef hls::stream<bool> ControlStreamType;

   private:
    void readA(t_DataType p_A[t_MaxNnz], unsigned int p_NnzWords, DataWideStreamType& p_ValStr) {
        for (unsigned int i = 0; i < p_NnzWords; ++i) {
#pragma HLS PIPELINE
            DataWideType l_val;
            for (unsigned int j = 0; j < t_MemWidth; ++j) {
#pragma HLS UNROLL
                l_val[j] = p_A[i * t_MemWidth + j];
            }
            p_ValStr.write(l_val);
        }
    }
    void readCol(unsigned int p_Ac[t_MaxNnz], unsigned int p_NnzWords, IdxWideStreamType& p_AcStr) {
        unsigned int l_idxWords = p_NnzWords / t_IdxWidthOverMemWidth;
        if (l_idxWords * t_IdxWidthOverMemWidth < p_NnzWords) {
            l_idxWords++;
        }
        for (unsigned int i = 0; i < l_idxWords; ++i) {
#pragma HLS PIPELINE II = t_IdxWidthOverMemWidth
            for (unsigned int j = 0; j < t_IdxWidthOverMemWidth; ++j) {
                IdxWideType l_val;
                for (unsigned int k = 0; k < t_MemWidth; ++k) {
                    l_val[k] = p_Ac[i * t_IndexWidth + j * t_MemWidth + k];
                }
                if ((i * t_IdxWidthOverMemWidth + j) < p_NnzWords) {
                    p_AcStr.write(l_val);
                }
            }
        }
    }
    void readRow(unsigned int p_Ar[t_MaxNnz], unsigned int p_NnzWords, IdxWideStreamType& p_ArStr) {
        unsigned int l_idxWords = p_NnzWords / t_IdxWidthOverMemWidth;
        if (l_idxWords * t_IdxWidthOverMemWidth < p_NnzWords) {
            l_idxWords++;
        }
        for (unsigned int i = 0; i < l_idxWords; ++i) {
#pragma HLS PIPELINE II = t_IdxWidthOverMemWidth
            for (unsigned int j = 0; j < t_IdxWidthOverMemWidth; ++j) {
                IdxWideType l_val;
                for (unsigned int k = 0; k < t_MemWidth; ++k) {
                    l_val[k] = p_Ar[i * t_IndexWidth + j * t_MemWidth + k];
                }
                if ((i * t_IdxWidthOverMemWidth + j) < p_NnzWords) {
                    p_ArStr.write(l_val);
                }
            }
        }
    }
    void formA(DataWideStreamType& p_ValStr,
               IdxWideStreamType& p_AcStr,
               IdxWideStreamType& p_ArStr,
               unsigned int p_NnzWords,
               SpmAwideStreamType& p_Astr) {
        for (unsigned int i = 0; i < p_NnzWords; ++i) {
#pragma HLS PIPELINE
            DataWideType l_val;
            IdxWideType l_col, l_row;
            l_val = p_ValStr.read();
            l_col = p_AcStr.read();
            l_row = p_ArStr.read();
            SpmAwideType l_aWide;
            for (unsigned int j = 0; j < t_MemWidth; ++j) {
#pragma HLS UNROLL
                SpmAtype l_a(l_val[j], l_col[j], l_row[j]);
                l_aWide[j] = l_a;
            }
            p_Astr.write(l_aWide);
        }
    }
    void xBarColSplit(SpmAwideStreamType& p_StrIn,
                      unsigned int p_NnzWords,
                      SpmAstreamType p_StrOut[t_MemWidth][t_MemWidth],
                      ControlStreamType& p_CtrOut) {
    LOOP_XC_WORDS:
        for (unsigned int l_idxA = 0; l_idxA < p_NnzWords; ++l_idxA) {
#pragma HLS PIPELINE
            SpmAwideType l_val = p_StrIn.read();
#pragma HLS array_partition variable = l_val COMPLETE
        LOOP_XC_W:
            for (int w = 0; w < t_MemWidth; ++w) {
#pragma HLS UNROLL
                unsigned int l_colBank = l_val[w].getCol() % t_MemWidth;
                p_StrOut[w][l_colBank].write(l_val[w]);
            }
        }
        p_CtrOut.write(true);
    }
    void xBarColMerge(SpmAstreamType p_StrIn[t_MemWidth][t_MemWidth],
                      ControlStreamType& p_CtrIn,
                      SpmAstreamType p_StrOut[t_MemWidth],
                      ControlStreamType p_CtrOut[t_MemWidth]) {
        bool l_exit = false, l_preDone = false;
        BoolArr<t_MemWidth> l_activity(true);
#pragma HLS array_partition variable = l_activity COMPLETE
    LOOP_XCM_WHILE:
        while (!l_exit) {
#pragma HLS PIPELINE
            if (l_preDone && !l_activity.Or()) {
                l_exit = true;
            }
            bool l_unused;
            if (p_CtrIn.read_nb(l_unused)) {
                l_preDone = true;
            }
            l_activity.Reset();

        LOOP_XCM_BANK_MERGE:
            for (int b = 0; b < t_MemWidth; ++b) {
#pragma HLS UNROLL
                unsigned int l_idx = 0;
            LOOP_XCM_IDX:
                for (int bb = 0; bb < t_MemWidth; ++bb) {
#pragma HLS UNROLL
                    unsigned int l_bank = (bb + b) % t_MemWidth;
                    if (!p_StrIn[l_bank][b].empty()) {
                        l_idx = l_bank;
                        break;
                    }
                }

                SpmAtype l_val;
                if (p_StrIn[l_idx][b].read_nb(l_val)) {
                    p_StrOut[b].write(l_val);
                    l_activity[b] = true;
                }
            }
        }
    LOOP_XCM_SEND_EXIT:
        for (int b = 0; b < t_MemWidth; ++b) {
#pragma HLS UNROLL
            p_CtrOut[b].write(true);
        }
    }
    void colUnit(t_DataType p_Vin[t_MaxK],
                 SpmAstreamType p_StrIn[t_MemWidth],
                 ControlStreamType p_CtrIn[t_MemWidth],
                 SpmCstreamType p_StrOut[t_MemWidth],
                 ControlStreamType p_CtrOut[t_MemWidth]) {
        SpmAtype l_val;
        bool l_exit = false;
        BoolArr<t_MemWidth> l_preDone(false);
        BoolArr<t_MemWidth> l_activity(true);
    LOOP_CU_WHILE:
        while (!l_exit) {
#pragma HLS PIPELINE
            if (l_preDone.And() && (!l_activity.Or())) {
                l_exit = true;
            }
        LOOP_CU_CALC:
            for (unsigned int b = 0; b < t_MemWidth; ++b) {
#pragma HLS UNROLL
                SpmAtype l_val;
                if (p_StrIn[b].read_nb(l_val)) {
                    unsigned int l_colOffset = l_val.getCol() / t_MemWidth;
                    t_DataType l_valB = p_Vin[l_colOffset * t_MemWidth + b];
                    t_DataType l_valC = l_val.getVal() * l_valB;
                    SpmCtype l_valOut(l_valC, l_val.getRow());
                    p_StrOut[b].write(l_valOut);
                    l_activity[b] = true;
                } else {
                    l_activity[b] = false;
                }
                bool l_unused = false;
                if (p_CtrIn[b].read_nb(l_unused)) {
                    l_preDone[b] = true;
                }
            }
        }
        for (unsigned int b = 0; b < t_MemWidth; ++b) {
            p_CtrOut[b].write(true);
        }
    }
    void xBarRowSplit(SpmCstreamType p_StrIn[t_MemWidth],
                      ControlStreamType p_CtrIn[t_MemWidth],
                      SpmCstreamType p_StrOut[t_MemWidth][t_MemWidth],
                      ControlStreamType& p_CtrOut) {
        bool l_exit = false;
        bool l_unused = false;
        bool l_preDone = false;
        BoolArr<t_MemWidth> l_activity(true);
        BoolArr<t_MemWidth> l_preActive(true);
#pragma HLS array_partition variable = l_activity COMPLETE
#pragma HLS array_partition variable = l_preActive COMPLETE
    LOOP_XRS_WHILE:
        while (!l_exit) {
#pragma HLS PIPELINE
            if (!l_preActive.Or() && !l_activity.Or()) {
                l_exit = true;
            }
            l_activity.Reset();
        LOOP_XRS_W:
            for (int w = 0; w < t_MemWidth; ++w) {
#pragma HLS UNROLL
                SpmCtype l_val;
                if (p_StrIn[w].read_nb(l_val)) {
                    unsigned int l_rowBank = l_val.getRow() % t_MemWidth;
                    p_StrOut[w][l_rowBank].write(l_val);
                    l_activity[w] = true;
                }

                bool l_unused;
                if (p_CtrIn[w].read_nb(l_unused)) {
                    l_preActive[w] = false;
                }
            }
        }
        p_CtrOut.write(true);
    }
    void xBarRowMerge(SpmCstreamType p_StrIn[t_MemWidth][t_MemWidth],
                      ControlStreamType& p_CtrIn,
                      SpmCstreamType p_StrOut[t_MemWidth],
                      ControlStreamType p_CtrOut[t_MemWidth]) {
        bool l_exit = false, l_preDone = false;
        BoolArr<t_MemWidth> l_activity(true);
    LOOP_XRM_WHILE:
        while (!l_exit) {
#pragma HLS PIPELINE
            if (l_preDone && !l_activity.Or()) {
                l_exit = true;
            }
            bool l_unused;
            if (p_CtrIn.read_nb(l_unused)) {
                l_preDone = true;
            }
            l_activity.Reset();
        LOOP_XRM_BANK_MERGE:
            for (int b = 0; b < t_MemWidth; ++b) {
#pragma HLS UNROLL
                unsigned int l_idx = 0;
            LOOP_XRM_IDX:
                for (int bb = 0; bb < t_MemWidth; ++bb) {
#pragma HLS UNROLL
                    unsigned int l_bank = (bb + b) % t_MemWidth;
                    if (!p_StrIn[l_bank][b].empty()) {
                        l_idx = l_bank;
                        break;
                    }
                }
                SpmCtype l_val;
                if (p_StrIn[l_idx][b].read_nb(l_val)) {
                    p_StrOut[b].write(l_val);
                    l_activity[b] = true;
                }
            }
        }
    LOOP_XRM_SEND_EXIT:
        for (int b = 0; b < t_MemWidth; ++b) {
#pragma HLS UNROLL
            p_CtrOut[b].write(true);
        }
    }
    void rowUnit(SpmCstreamType& p_StrIn,
                 ControlStreamType& p_CtrIn,
                 SpmCaccStreamType& p_StrOut,
                 ControlStreamType& p_CtrOut,
                 unsigned int t_BankId) {
        SpmCtype l_val(0, t_BankId);
        unsigned int l_curRow = t_BankId;
        AccDataType l_accVal(0);
#pragma HLS ARRAY_PARTITION variable = l_accVal dim = 0 complete
        uint8_t l_count = 0;

        bool l_exit = false;
        bool l_preDone = false;
        bool l_activity = true;
    LOOP_RU_WHILE:
        while (!l_exit) {
#pragma HLS PIPELINE
            if (l_preDone && !l_activity) {
                l_exit = true;
            }
            bool l_unused = false;
            if (p_CtrIn.read_nb(l_unused)) {
                l_preDone = true;
            }
            l_activity = false;

            if (p_StrIn.read_nb(l_val)) {
                if ((l_val.getRow() != l_curRow) || (l_count == (t_AccWidth - 1))) {
                    SpmCaccType l_acc(l_accVal, l_curRow);
                    p_StrOut.write(l_acc);
                    l_curRow = l_val.getRow();
                    l_accVal[0] = l_val.getVal();
                    for (unsigned int i = 1; i < t_AccWidth; ++i) {
                        l_accVal[i] = 0;
                    }
                    l_count = 0;
                } else {
                    (void)l_accVal.shift(l_val.getVal());
                    l_count++;
                }
                l_activity = true;
            }
        }
        SpmCaccType l_acc(l_accVal, l_curRow);
        p_StrOut.write(l_acc);
        p_CtrOut.write(true);
    }
    void reduceUnit(SpmCaccStreamType& p_StrIn,
                    ControlStreamType& p_CtrIn,
                    SpmCstreamType& p_StrOut,
                    ControlStreamType& p_CtrOut,
                    unsigned int t_BankId) {
        SpmCaccType l_val;
        unsigned int l_accRow;
        AccDataType l_accVal(0);
#pragma HLS ARRAY_PARTITION variable = l_accVal dim = 0 complete
        WideType<SpmCtype, t_AccWidth> l_cArr;
#pragma HLS ARRAY_PARTITION variable = l_cArr dim = 1 complete

        bool l_exit = false;
        bool l_preDone = false;
        bool l_activity = true;
    LOOP_REU_WHILE:
        while (!l_exit) {
#pragma HLS PIPELINE
            if (l_preDone && !l_activity) {
                l_exit = true;
            }
            bool l_unused = false;
            if (p_CtrIn.read_nb(l_unused)) {
                l_preDone = true;
            }
            l_activity = false;

            if (p_StrIn.read_nb(l_val)) {
                l_accVal = l_val.getVal();
                l_accRow = l_val.getRow();
                for (unsigned int i = 0; i < t_AccWidth; ++i) {
#pragma HLS UNROLL
                    l_cArr[i].getVal() = l_accVal[i];
                    l_cArr[i].getRow() = l_accRow;
                }
                SpmCtype l_sum(0, l_accRow);
                for (unsigned int i = 0; i < t_AccWidth; ++i) {
                    l_sum.getVal() += l_cArr[i].getVal();
                    l_sum.getRow() = l_cArr[i].getRow();
                }
                if (l_sum.getVal() != 0) {
                    p_StrOut.write(l_sum);
                }
                l_activity = true;
            }
        }
        p_CtrOut.write(true);
    }
    void aggUnit(t_DataType p_Vout[t_MaxM],
                 SpmCstreamType p_StrIn[t_MemWidth],
                 ControlStreamType p_CtrIn[t_MemWidth],
                 unsigned int p_Mwords) {
        for (unsigned int i = 0; i < p_Mwords; ++i) {
#pragma HLS PIPELINE
            for (unsigned int j = 0; j < t_MemWidth; ++j) {
#pragma HLS UNROLL
                p_Vout[i * t_MemWidth + j] = 0;
            }
        }
        bool l_exit = false;
        BoolArr<t_MemWidth> l_preDone(false);
        BoolArr<t_MemWidth> l_activity(true);
    LOOP_AU_WHILE:
        while (!l_exit) {
#pragma HLS PIPELINE
            if (l_preDone.And() && !l_activity.Or()) {
                l_exit = true;
            }
            for (unsigned int b = 0; b < t_MemWidth; ++b) {
#pragma HLS UNROLL
                bool l_unused = false;
                if (p_CtrIn[b].read_nb(l_unused)) {
                    l_preDone[b] = true;
                }
                l_activity[b] = false;

                SpmCtype l_val;
                if (p_StrIn[b].read_nb(l_val)) {
                    unsigned int l_rowOffset = l_val.getRow() / t_MemWidth;
                    p_Vout[l_rowOffset * t_MemWidth + b] += l_val.getVal();
                    l_activity = true;
                }
            }
        }
    }

    void aggAddUnit(t_DataType p_Vin[t_MaxM],
                    t_DataType p_Vout[t_MaxM],
                    SpmCstreamType p_StrIn[t_MemWidth],
                    ControlStreamType p_CtrIn[t_MemWidth],
                    unsigned int p_Mwords) {
        for (unsigned int i = 0; i < p_Mwords; ++i) {
#pragma HLS PIPELINE
            for (unsigned int j = 0; j < t_MemWidth; ++j) {
#pragma HLS UNROLL
                p_Vout[i * t_MemWidth + j] = p_Vin[i * t_MemWidth + j];
            }
        }
        bool l_exit = false;
        BoolArr<t_MemWidth> l_preDone(false);
        BoolArr<t_MemWidth> l_activity(true);
    LOOP_AU_WHILE:
        while (!l_exit) {
#pragma HLS PIPELINE
            if (l_preDone.And() && !l_activity.Or()) {
                l_exit = true;
            }
            for (unsigned int b = 0; b < t_MemWidth; ++b) {
#pragma HLS UNROLL
                bool l_unused = false;
                if (p_CtrIn[b].read_nb(l_unused)) {
                    l_preDone[b] = true;
                }
                l_activity[b] = false;

                SpmCtype l_val;
                if (p_StrIn[b].read_nb(l_val)) {
                    unsigned int l_rowOffset = l_val.getRow() / t_MemWidth;
                    p_Vout[l_rowOffset * t_MemWidth + b] += l_val.getVal();
                    l_activity = true;
                }
            }
        }
    }
    void multAB(t_DataType p_A[t_MaxNnz],
                unsigned int p_Ar[t_MaxNnz],
                unsigned int p_Ac[t_MaxNnz],
                t_DataType p_Vin[t_MaxK],
                unsigned int p_NnzWords,
                SpmCstreamType p_StrOut[t_MemWidth],
                ControlStreamType p_CtrOut[t_MemWidth]) {
        DataWideStreamType l_valAstr;
#pragma HLS DATA_PACK variable = l_valAstr
#pragma HLS STREAM variable = l_valAstr depth = t_FifoDepth
        IdxWideStreamType l_colAstr;
#pragma HLS DATA_PACK variable = l_colAstr
#pragma HLS STREAM variable = l_colAstr depth = t_FifoDepth
        IdxWideStreamType l_rowAstr;
#pragma HLS DATA_PACK variable = l_rowAstr
#pragma HLS STREAM variable = l_rowAstr depth = t_FifoDepth
        SpmAwideStreamType l_spmAstr;
#pragma HLS DATA_PACK variable = l_spmAstr
#pragma HLS STREAM variable = l_spmAstr depth = t_FifoDepth
        SpmAstreamType l_str2colMerge[t_MemWidth][t_MemWidth];
#pragma HLS DATA_PACK variable = l_str2colMerge
#pragma HLS STREAM variable = l_str2colMerge depth = t_FifoDepth
        ControlStreamType l_ctr2colMerge;
#pragma HLS DATA_PACK variable = l_ctr2colMerge
#pragma HLS STREAM variable = l_ctr2colMerge depth = t_FifoDepth
        SpmAstreamType l_str2colUnit[t_MemWidth];
#pragma HLS DATA_PACK variable = l_str2colUnit
#pragma HLS STREAM variable = l_str2colUnit depth = t_FifoDepth
        ControlStreamType l_ctr2colUnit[t_MemWidth];
#pragma HLS DATA_PACK variable = l_ctr2colUnit
#pragma HLS STREAM variable = l_ctr2colUnit depth = t_FifoDepth
        SpmCstreamType l_str2rowSplit[t_MemWidth];
#pragma HLS DATA_PACK variable = l_str2rowSplit
#pragma HLS STREAM variable = l_str2rowSplit depth = t_FifoDepth
        ControlStreamType l_ctr2rowSplit[t_MemWidth];
#pragma HLS DATA_PACK variable = l_ctr2rowSplit
#pragma HLS STREAM variable = l_ctr2rowSplit depth = t_FifoDepth
        SpmCstreamType l_str2rowMerge[t_MemWidth][t_MemWidth];
#pragma HLS DATA_PACK variable = l_str2rowMerge
#pragma HLS STREAM variable = l_str2rowMerge depth = t_FifoDepth
        ControlStreamType l_ctr2rowMerge;
#pragma HLS DATA_PACK variable = l_ctr2rowMerge
#pragma HLS STREAM variable = l_ctr2rowMerge depth = t_FifoDepth
        SpmCstreamType l_str2rowUnit[t_MemWidth];
#pragma HLS DATA_PACK variable = l_str2rowUnit
#pragma HLS STREAM variable = l_str2rowUnit depth = t_FifoDepth
        ControlStreamType l_ctr2rowUnit[t_MemWidth];
#pragma HLS DATA_PACK variable = l_ctr2rowUnit
#pragma HLS STREAM variable = l_ctr2rowUnit depth = t_FifoDepth
        SpmCaccStreamType l_str2reduceUnit[t_MemWidth];
#pragma HLS DATA_PACK variable = l_str2reduceUnit
#pragma HLS STREAM variable = l_str2reduceUnit depth = t_FifoDepth
        ControlStreamType l_ctr2reduceUnit[t_MemWidth];
#pragma HLS DATA_PACK variable = l_ctr2reduceUnit
#pragma HLS STREAM variable = l_ctr2reduceUnit depth = t_FifoDepth
        SpmCstreamType l_str2rowUnit1[t_MemWidth];
#pragma HLS DATA_PACK variable = l_str2rowUnit1
#pragma HLS STREAM variable = l_str2rowUnit1 depth = t_FifoDepth
        ControlStreamType l_ctr2rowUnit1[t_MemWidth];
#pragma HLS DATA_PACK variable = l_ctr2rowUnit1
#pragma HLS STREAM variable = l_ctr2rowUnit1 depth = t_FifoDepth
        SpmCaccStreamType l_str2reduceUnit1[t_MemWidth];
#pragma HLS DATA_PACK variable = l_str2reduceUnit1
#pragma HLS STREAM variable = l_str2reduceUnit1 depth = t_FifoDepth
        ControlStreamType l_ctr2reduceUnit1[t_MemWidth];
#pragma HLS DATA_PACK variable = l_ctr2reduceUnit1
#pragma HLS STREAM variable = l_ctr2reduceUnit1 depth = t_FifoDepth
        SpmCstreamType l_str2aggUnit[t_MemWidth];
#pragma HLS DATA_PACK variable = l_str2aggUnit
#pragma HLS STREAM variable = l_str2aggUnit depth = t_FifoDepth
        ControlStreamType l_ctr2aggUnit[t_MemWidth];
#pragma HLS DATA_PACK variable = l_ctr2aggUnit
#pragma HLS STREAM variable = l_ctr2aggUnit depth = t_FifoDepth

#pragma HLS DATAFLOW
        readA(p_A, p_NnzWords, l_valAstr);
        readCol(p_Ac, p_NnzWords, l_colAstr);
        readRow(p_Ar, p_NnzWords, l_rowAstr);
        formA(l_valAstr, l_colAstr, l_rowAstr, p_NnzWords, l_spmAstr);
        xBarColSplit(l_spmAstr, p_NnzWords, l_str2colMerge, l_ctr2colMerge);
        xBarColMerge(l_str2colMerge, l_ctr2colMerge, l_str2colUnit, l_ctr2colUnit);
        colUnit(p_Vin, l_str2colUnit, l_ctr2colUnit, l_str2rowSplit, l_ctr2rowSplit);
        xBarRowSplit(l_str2rowSplit, l_ctr2rowSplit, l_str2rowMerge, l_ctr2rowMerge);
        xBarRowMerge(l_str2rowMerge, l_ctr2rowMerge, l_str2rowUnit, l_ctr2rowUnit);
        for (unsigned int b = 0; b < t_MemWidth; ++b) {
#pragma HLS UNROLL
            rowUnit(l_str2rowUnit[b], l_ctr2rowUnit[b], l_str2reduceUnit[b], l_ctr2reduceUnit[b], b);
            reduceUnit(l_str2reduceUnit[b], l_ctr2reduceUnit[b], l_str2rowUnit1[b], l_ctr2rowUnit1[b], b);
            rowUnit(l_str2rowUnit1[b], l_ctr2rowUnit1[b], l_str2reduceUnit1[b], l_ctr2reduceUnit1[b], b);
            reduceUnit(l_str2reduceUnit1[b], l_ctr2reduceUnit1[b], p_StrOut[b], p_CtrOut[b], b);
        }
    }

   public:
    Spmv() {
#pragma HLS inline
    }

    void spmvFlow(t_DataType p_A[t_MaxNnz],
                  unsigned int p_Ar[t_MaxNnz],
                  unsigned int p_Ac[t_MaxNnz],
                  t_DataType p_Vin[t_MaxK],
                  t_DataType p_Vout[t_MaxM],
                  unsigned int p_NnzWords,
                  unsigned int p_Mwords) {
        SpmCstreamType l_str2aggUnit[t_MemWidth];
#pragma HLS DATA_PACK variable = l_str2aggUnit
#pragma HLS STREAM variable = l_str2aggUnit depth = t_FifoDepth
        ControlStreamType l_ctr2aggUnit[t_MemWidth];
#pragma HLS DATA_PACK variable = l_ctr2aggUnit
#pragma HLS STREAM variable = l_ctr2aggUnit depth = t_FifoDepth

#pragma HLS DATAFLOW
        multAB(p_A, p_Ar, p_Ac, p_Vin, p_NnzWords, l_str2aggUnit, l_ctr2aggUnit);
        aggUnit(p_Vout, l_str2aggUnit, l_ctr2aggUnit, p_Mwords);
    }

    void spmvAddFlow(t_DataType p_A[t_MaxNnz],
                     unsigned int p_Ar[t_MaxNnz],
                     unsigned int p_Ac[t_MaxNnz],
                     t_DataType p_B[t_MaxK],
                     t_DataType p_Vin[t_MaxM],
                     t_DataType p_Vout[t_MaxM],
                     unsigned int p_NnzWords,
                     unsigned int p_Mwords) {
        SpmCstreamType l_str2aggUnit[t_MemWidth];
#pragma HLS DATA_PACK variable = l_str2aggUnit
#pragma HLS STREAM variable = l_str2aggUnit depth = t_FifoDepth
        ControlStreamType l_ctr2aggUnit[t_MemWidth];
#pragma HLS DATA_PACK variable = l_ctr2aggUnit
#pragma HLS STREAM variable = l_ctr2aggUnit depth = t_FifoDepth

#pragma HLS DATAFLOW
        multAB(p_A, p_Ar, p_Ac, p_B, p_NnzWords, l_str2aggUnit, l_ctr2aggUnit);
        aggAddUnit(p_Vin, p_Vout, l_str2aggUnit, l_ctr2aggUnit, p_Mwords);
    }
    void sparseMult(t_DataType p_A[t_MaxNnz],
                    unsigned int p_Ar[t_MaxNnz],
                    unsigned int p_Ac[t_MaxNnz],
                    t_DataType p_Vin[t_MaxK],
                    t_DataType p_Vout[t_MaxM],
                    unsigned int p_NNZs,
                    unsigned int p_Ms) {
        // initialize p_Vout with 0S
        unsigned int l_mWords = p_Ms / t_MemWidth;
        unsigned int l_nnzWords = p_NNZs / t_MemWidth;
        spmvFlow(p_A, p_Ar, p_Ac, p_Vin, p_Vout, l_nnzWords, l_mWords);
    }

    void sparseMultAdd(t_DataType p_A[t_MaxNnz],
                       unsigned int p_Ar[t_MaxNnz],
                       unsigned int p_Ac[t_MaxNnz],
                       t_DataType p_B[t_MaxK],
                       t_DataType p_Vin[t_MaxM],
                       t_DataType p_Vout[t_MaxM],
                       unsigned int p_NNZs,
                       unsigned int p_Ms) {
        // initialize p_Vout with 0S
        unsigned int l_mWords = p_Ms / t_MemWidth;
        unsigned int l_nnzWords = p_NNZs / t_MemWidth;
        spmvAddFlow(p_A, p_Ar, p_Ac, p_B, p_Vin, p_Vout, l_nnzWords, l_mWords);
    }
};
} // namespace blas
} // namespace linear_algebra
} // namespace xf

#endif
