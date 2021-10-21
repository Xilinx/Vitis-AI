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
 * @file gen_cscmv.hpp
 * @brief header file for generating data images of cscmv operation.
 *
 * This file is part of Vitis SPARSE Library.
 */
#ifndef XF_SPARSE_GEN_CSCMV_HPP
#define XF_SPARSE_GEN_CSCMV_HPP

#include <ctime>
#include <cstdlib>
#include <vector>
#include "L2_types.hpp"
#include "program.hpp"
#include "mtxFile.hpp"

using namespace std;
namespace xf {
namespace sparse {

template <typename t_DataType, unsigned int t_MemBits, unsigned int t_PageSize = 4096>
class GenVec {
   public:
    const unsigned int t_MemWords = t_MemBits / (8 * sizeof(t_DataType));

   public:
    GenVec() {}
    void genEntVecFromRnd(unsigned int p_entries,
                          t_DataType p_maxVal,
                          t_DataType p_valStep,
                          vector<t_DataType>& p_entryVec) {
        t_DataType l_val = 0;
        for (unsigned int i = 0; i < p_entries; ++i) {
            p_entryVec.push_back(l_val);
            l_val += p_valStep;
            if (l_val > p_maxVal) {
                l_val = 0;
            }
        }
    }
    bool genColVecFromEnt(vector<t_DataType>& p_entryVec,
                          Program<t_PageSize>& p_program,
                          ColVec<t_DataType, t_MemBits>& p_colVec) {
        unsigned int l_entries = p_entryVec.size();
        while (l_entries % t_MemWords != 0) {
            cout << "INFO: padding col vector with 0 entry" << endl;
            p_entryVec.push_back(0);
            l_entries++;
        }
        unsigned long long l_vecSz = l_entries * sizeof(t_DataType);
        void* l_valAddr = p_program.allocMem(l_vecSz);
        if (l_valAddr == nullptr) {
            return false;
        }
        p_colVec.setEntries(l_entries);
        p_colVec.setValAddr(reinterpret_cast<uint8_t*>(l_valAddr));
        p_colVec.storeVal(p_entryVec);
        return true;
    }
    bool genColVecFromRnd(unsigned int p_entries,
                          t_DataType p_maxVal,
                          t_DataType p_valStep,
                          Program<t_PageSize>& p_program,
                          ColVec<t_DataType, t_MemBits>& p_colVec) {
        vector<t_DataType> l_entryVec;
        p_colVec.setEntries(p_entries);
        genEntVecFromRnd(p_entries, p_maxVal, p_valStep, l_entryVec);
        bool l_res = false;
        l_res = genColVecFromEnt(l_entryVec, p_program, p_colVec);
        return l_res;
    }

    bool genEmptyColVec(unsigned int p_entries,
                        Program<t_PageSize>& p_program,
                        ColVec<t_DataType, t_MemBits>& p_colVec) {
        unsigned long long l_vecSz = p_entries * sizeof(t_DataType);
        void* l_valAddr = p_program.allocMem(l_vecSz);
        if (l_valAddr == nullptr) {
            return false;
        }
        p_colVec.setEntries(p_entries);
        p_colVec.setValAddr(reinterpret_cast<uint8_t*>(l_valAddr));
        for (unsigned int i = 0; i < p_entries; ++i) {
            p_colVec.setEntryVal(i, 0);
        }
        return true;
    }
};

template <typename t_DataType, typename t_IndexType, unsigned int t_PageSize = 4096>
class GenMatCsc {
   public:
    typedef NnzUnit<t_DataType, t_IndexType> t_NnzUnitType;
    typedef MatCsc<t_DataType, t_IndexType> t_MatCscType;

   public:
    GenMatCsc() {}
    bool genMatFromUnits(unsigned int p_rows,
                         unsigned int p_cols,
                         unsigned int p_nnzs,
                         vector<t_NnzUnitType>& p_nnzUnits,
                         Program<t_PageSize>& p_program,
                         t_MatCscType& p_matCsc) {
        // sort p_nnzUnits along cols
        sort(p_nnzUnits.begin(), p_nnzUnits.end());
        p_matCsc.setRows(p_rows);
        p_matCsc.setCols(p_cols);
        p_matCsc.setNnzs(p_nnzs);
        unsigned long long l_valSz = p_nnzs * sizeof(t_DataType);
        void* l_valAddr = p_program.allocMem(l_valSz);
        unsigned long long l_rowSz = p_nnzs * sizeof(t_IndexType);
        void* l_rowAddr = p_program.allocMem(l_rowSz);
        unsigned long long l_colPtrSz = p_cols * sizeof(t_IndexType);
        void* l_colPtrAddr = p_program.allocMem(l_colPtrSz);
        if (l_valAddr == nullptr) {
            return false;
        }
        if (l_rowAddr == nullptr) {
            return false;
        }
        if (l_colPtrAddr == nullptr) {
            return false;
        }
        p_matCsc.setValAddr(l_valAddr);
        p_matCsc.setRowAddr(l_rowAddr);
        p_matCsc.setColPtrAddr(l_colPtrAddr);
        // populate data
        t_DataType* l_val = reinterpret_cast<t_DataType*>(l_valAddr);
        for (unsigned int i = 0; i < p_nnzs; ++i) {
            l_val[i] = p_nnzUnits[i].getVal();
        }
        t_IndexType* l_rowIdx = reinterpret_cast<t_IndexType*>(l_rowAddr);
        for (unsigned int i = 0; i < p_nnzs; ++i) {
            l_rowIdx[i] = p_nnzUnits[i].getRow();
        }
        t_IndexType* l_colPtr = reinterpret_cast<t_IndexType*>(l_colPtrAddr);
        unsigned int l_id = 0;
        l_colPtr[l_id] = 0;
        unsigned int l_nnzId = 0;
        while (l_nnzId < p_nnzs) {
            if (p_nnzUnits[l_nnzId].getCol() > l_id) {
                l_id++;
                l_colPtr[l_id] = l_colPtr[l_id - 1];
            } else if (p_nnzUnits[l_nnzId].getCol() == l_id) {
                l_colPtr[l_id]++;
                l_nnzId++;
            }
        }
        return true;
    }

    bool genMatFromCscFiles(string p_valFileName,
                            string p_rowFileName,
                            string p_colPtrFileName,
                            Program<t_PageSize>& p_program,
                            t_MatCscType& p_matCsc) {
        size_t l_valSz = getFileSize(p_valFileName);
        size_t l_rowSz = getFileSize(p_rowFileName);
        size_t l_colPtrSz = getFileSize(p_colPtrFileName);
        assert(l_valSz == l_rowSz);
        assert(l_valSz % (sizeof(t_DataType)) == 0);
        unsigned int l_nnzs = l_valSz / sizeof(t_DataType);
        unsigned int l_cols = l_colPtrSz / sizeof(t_IndexType);
        ifstream l_ifVal(p_valFileName.c_str(), ios::binary);
        if (!l_ifVal.is_open()) {
            cout << "ERROR: Open " << p_valFileName << endl;
            return false;
        }
        ifstream l_ifRow(p_rowFileName.c_str(), ios::binary);
        if (!l_ifRow.is_open()) {
            cout << "ERROR: Open " << p_rowFileName << endl;
            return false;
        }
        ifstream l_ifColPtr(p_colPtrFileName.c_str(), ios::binary);
        if (!l_ifColPtr.is_open()) {
            cout << "ERROR: Open " << p_colPtrFileName << endl;
            return false;
        }
        void* l_valAddr = p_program.allocMem(l_valSz);
        void* l_rowAddr = p_program.allocMem(l_rowSz);
        void* l_colPtrAddr = p_program.allocMem(l_colPtrSz);
        l_ifVal.read(reinterpret_cast<char*>(l_valAddr), l_valSz);
        l_ifRow.read(reinterpret_cast<char*>(l_rowAddr), l_rowSz);
        l_ifColPtr.read(reinterpret_cast<char*>(l_colPtrAddr), l_colPtrSz);
        p_matCsc.setNnzs(l_nnzs);
        p_matCsc.setCols(l_cols);
        p_matCsc.setValAddr(l_valAddr);
        p_matCsc.setRowAddr(l_rowAddr);
        p_matCsc.setColPtrAddr(l_colPtrAddr);

        t_IndexType* l_rowIdx = reinterpret_cast<t_IndexType*>(l_rowAddr);
        t_IndexType l_rowIdxMax = 0;
        for (unsigned int i = 0; i < l_nnzs; ++i) {
            if (l_rowIdx[i] > l_rowIdxMax) {
                l_rowIdxMax = l_rowIdx[i];
            }
        }
        p_matCsc.setRows(l_rowIdxMax + 1);
        return true;
    }

    bool genMatFromMtxFile(string p_mtxFileName, Program<t_PageSize>& p_program, t_MatCscType& p_matCsc) {
        MtxFile<t_DataType, t_IndexType> l_mtxFile;
        l_mtxFile.loadFile(p_mtxFileName);
        vector<t_NnzUnitType> l_nnzUnits;
        if (!l_mtxFile.good()) {
            return false;
        }
        l_nnzUnits = l_mtxFile.getNnzUnits();
        unsigned int l_rows = l_mtxFile.rows();
        unsigned int l_cols = l_mtxFile.cols();
        unsigned int l_nnzs = l_mtxFile.nnzs();
        if (!genMatFromUnits(l_rows, l_cols, l_nnzs, l_nnzUnits, p_program, p_matCsc)) {
            return false;
        }
        return true;
    }
};

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxRowsPerPar,
          unsigned int t_MaxColsPerPar,
          unsigned int t_ParEntries,
          unsigned int t_ParGroups,
          unsigned int t_DdrMemBits,
          unsigned int t_HbmChannels>
class GenMatPar {
   public:
    typedef NnzUnit<t_DataType, t_IndexType> t_NnzUnitType;
    typedef struct ChBlockDesp t_ChBlockDespType;
    typedef struct ColParDesp<t_HbmChannels> t_ColParDespType;
    typedef MatPar<t_DataType, t_IndexType, t_MaxRowsPerPar, t_MaxColsPerPar, t_HbmChannels> t_MatParType;
    typedef RoCooPar<t_DataType, t_IndexType, t_MaxRowsPerPar, t_MaxColsPerPar, t_HbmChannels> t_RoCooParType;
    typedef KrnColParDesp<t_HbmChannels> t_KrnColParDespType;
    typedef KrnRowParDesp<t_HbmChannels> t_KrnRowParDespType;

    static const unsigned int t_ColsPerMem = t_DdrMemBits / (8 * sizeof(t_DataType));
    static const unsigned int t_ParWordsPerMem = t_ColsPerMem / t_ParEntries;

   public:
    GenMatPar() {}
    void genRoCooPar(vector<t_NnzUnitType>& p_nnzUnits, t_RoCooParType& p_roCooPar) {
        // assume memory has already been allocated for p_roCooPar
        unsigned int l_nnzs = p_nnzUnits.size();
        p_roCooPar.nnzs() = l_nnzs;
        unsigned int l_colPars = p_roCooPar.colPars();
        for (unsigned int i = 0; i < l_nnzs; ++i) {
            t_NnzUnitType l_nnzUnit = p_nnzUnits[i];
            unsigned int l_rowId = l_nnzUnit.getRow();
            unsigned int l_colId = l_nnzUnit.getCol();
            unsigned int l_rowParId = l_rowId / t_MaxRowsPerPar;
            unsigned int l_colParId = l_colId / t_MaxColsPerPar;
            unsigned int l_parId = l_rowParId * l_colPars + l_colParId;
            p_roCooPar.addNnzUnit(l_parId, l_nnzUnit);
        }
        unsigned int l_totalPars = p_roCooPar.totalPars();
        for (unsigned int i = 0; i < l_totalPars; ++i) {
            // sort(p_roCooPar[i].begin(), p_roCooPar[i].end(), compareRow<t_DataType, t_IndexType>());
            sort(p_roCooPar[i].begin(), p_roCooPar[i].end());
        }
    }

    void genNextParDesp(vector<t_NnzUnitType>& p_nnzUnits,
                        unsigned int p_nnzsPerCh,
                        unsigned int p_chId,
                        unsigned int& p_startId,
                        t_ColParDespType& p_colParDesp,
                        t_KrnRowParDespType& p_krnRowParDesp) {
        unsigned int l_nnzs = p_nnzUnits.size();
        unsigned int l_chNnzs =
            (p_startId >= l_nnzs) ? 0 : ((p_startId + p_nnzsPerCh) > l_nnzs) ? (l_nnzs - p_startId) : p_nnzsPerCh;
        t_ChBlockDespType l_chBlockDesp;
        if (l_nnzs == 0) {
            p_krnRowParDesp.addChBlockDesp(l_chBlockDesp);
            p_colParDesp.m_minColId[p_chId] = 1;
            p_colParDesp.m_maxColId[p_chId] = 0;
            p_colParDesp.m_cols[p_chId] = 0;
            p_colParDesp.m_colBlocks[p_chId] = 0;
            p_colParDesp.m_nnzs[p_chId] = 0;
            p_colParDesp.m_nnzBlocks[p_chId] = 0;
            p_colParDesp.m_parMinColId = 1;
            p_colParDesp.m_parMaxColId = 0;
            p_colParDesp.m_nnzColMemBlocks = 0;
            return;
        }
        if (l_chNnzs == 0) {
            p_krnRowParDesp.addChBlockDesp(l_chBlockDesp);
            p_colParDesp.m_minColId[p_chId] = 1;
            p_colParDesp.m_maxColId[p_chId] = 0;
            p_colParDesp.m_cols[p_chId] = 0;
            p_colParDesp.m_colBlocks[p_chId] = 0;
            p_colParDesp.m_nnzs[p_chId] = 0;
            p_colParDesp.m_nnzBlocks[p_chId] = 0;
            return;
        }
        l_chBlockDesp.m_startId = p_startId;
        t_NnzUnitType l_nnzUnit = p_nnzUnits[p_startId];
        l_chBlockDesp.m_minRowId = l_nnzUnit.getRow();
        l_chBlockDesp.m_maxRowId = l_chBlockDesp.m_minRowId;
        l_chBlockDesp.m_minColId = l_nnzUnit.getCol();
        l_chBlockDesp.m_maxColId = l_chBlockDesp.m_minColId;
        l_chBlockDesp.m_nnzs = l_chNnzs;
        for (unsigned int i = p_startId; i < p_startId + l_chNnzs; ++i) {
            l_nnzUnit = p_nnzUnits[i];
            unsigned int l_rowId = l_nnzUnit.getRow();
            unsigned int l_colId = l_nnzUnit.getCol();
            l_chBlockDesp.m_minRowId = (l_chBlockDesp.m_minRowId > l_rowId) ? l_rowId : l_chBlockDesp.m_minRowId;
            l_chBlockDesp.m_maxRowId = (l_chBlockDesp.m_maxRowId < l_rowId) ? l_rowId : l_chBlockDesp.m_maxRowId;
            l_chBlockDesp.m_minColId = (l_chBlockDesp.m_minColId > l_colId) ? l_colId : l_chBlockDesp.m_minColId;
            l_chBlockDesp.m_maxColId = (l_chBlockDesp.m_maxColId < l_colId) ? l_colId : l_chBlockDesp.m_maxColId;
        }
        l_chBlockDesp.m_rows = (l_chBlockDesp.m_maxRowId + 1) - l_chBlockDesp.m_minRowId;
        l_chBlockDesp.m_cols = (l_chBlockDesp.m_maxColId + 1) - l_chBlockDesp.m_minColId;
        l_chBlockDesp.m_nnzBlocks = alignedBlock(l_chBlockDesp.m_nnzs, t_ParEntries);
        l_chBlockDesp.m_colBlocks = alignedBlock(l_chBlockDesp.m_cols, t_ParEntries);
        l_chBlockDesp.m_rowBlocks = alignedBlock(l_chBlockDesp.m_rows, t_ParEntries * t_ParGroups);
        l_chBlockDesp.m_rowResBlocks = alignedNum(l_chBlockDesp.m_rowBlocks * t_ParGroups, 2);

        p_krnRowParDesp.minRowId() = (p_krnRowParDesp.minRowId() > l_chBlockDesp.m_minRowId)
                                         ? l_chBlockDesp.m_minRowId
                                         : p_krnRowParDesp.minRowId();
        p_krnRowParDesp.maxRowId() = (p_krnRowParDesp.maxRowId() < l_chBlockDesp.m_maxRowId)
                                         ? l_chBlockDesp.m_maxRowId
                                         : p_krnRowParDesp.maxRowId();
        p_krnRowParDesp.nnzs() = p_krnRowParDesp.nnzs() + l_chBlockDesp.m_nnzs;
        p_krnRowParDesp.nnzBlocks() = p_krnRowParDesp.nnzBlocks() + l_chBlockDesp.m_nnzBlocks;
        p_krnRowParDesp.addChBlockDesp(l_chBlockDesp);

        p_colParDesp.m_minColId[p_chId] = l_chBlockDesp.m_minColId;
        p_colParDesp.m_maxColId[p_chId] = l_chBlockDesp.m_maxColId;
        p_colParDesp.m_cols[p_chId] = l_chBlockDesp.m_cols;
        p_colParDesp.m_colBlocks[p_chId] = l_chBlockDesp.m_colBlocks;
        p_colParDesp.m_nnzs[p_chId] = l_chBlockDesp.m_nnzs;
        p_colParDesp.m_nnzBlocks[p_chId] = l_chBlockDesp.m_nnzBlocks;
        p_colParDesp.m_parMinColId = (p_colParDesp.m_parMinColId > l_chBlockDesp.m_minColId)
                                         ? l_chBlockDesp.m_minColId
                                         : p_colParDesp.m_parMinColId;
        p_colParDesp.m_parMaxColId = (p_colParDesp.m_parMaxColId < l_chBlockDesp.m_maxColId)
                                         ? l_chBlockDesp.m_maxColId
                                         : p_colParDesp.m_parMaxColId;
        p_colParDesp.m_nnzColMemBlocks += p_colParDesp.m_colBlocks[p_chId];

        p_startId = ((p_startId + p_nnzsPerCh) > l_nnzs) ? l_nnzs : (p_startId + p_nnzsPerCh);
    }
    void genMatPar(vector<t_NnzUnitType>& p_nnzUnits,
                   unsigned int p_rows,
                   unsigned int p_cols,
                   t_MatParType& p_matPar) {
        p_matPar.init(p_rows, p_cols);
        genRoCooPar(p_nnzUnits, p_matPar.roCooPar());
        unsigned int l_rowPars = p_matPar.roCooPar().rowPars();
        unsigned int l_colPars = p_matPar.roCooPar().colPars();
        unsigned int l_matRows = p_matPar.roCooPar().rows();
        unsigned int l_matCols = p_matPar.roCooPar().cols();
        for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
            for (unsigned int i = 0; i < l_rowPars; ++i) {
                p_matPar.krnRowParDesps(ch, i).minRowId() = l_matRows;
                p_matPar.krnRowParDesps(ch, i).maxRowId() = 0;
                p_matPar.krnRowParDesps(ch, i).rows() = 0;
                p_matPar.krnRowParDesps(ch, i).nnzs() = 0;
                p_matPar.krnRowParDesps(ch, i).nnzBlocks() = 0;
                p_matPar.krnRowParDesps(ch, i).rowBlocks() = 0;
                p_matPar.krnRowParDesps(ch, i).rowResBlocks() = 0;
            }
        }
        for (unsigned int i = 0; i < l_rowPars; ++i) {
            for (unsigned int j = 0; j < l_colPars; ++j) {
                unsigned int l_parId = i * l_colPars + j;
                unsigned int l_nnzs = p_matPar.roCooPar(l_parId).size();
                unsigned int l_alignedNnzs = alignedNum(l_nnzs, t_HbmChannels);
                unsigned int l_nnzsPerCh = l_alignedNnzs / t_HbmChannels;
                unsigned int l_startId = 0;
                p_matPar.krnColParDesp(l_parId).m_parMinColId = l_matCols;
                p_matPar.krnColParDesp(l_parId).m_parMaxColId = 0;
                p_matPar.krnColParDesp(l_parId).m_nnzColMemBlocks = 0;
                for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
                    genNextParDesp(p_matPar.roCooPar(l_parId), l_nnzsPerCh, ch, l_startId,
                                   p_matPar.krnColParDesp(l_parId), p_matPar.krnRowParDesps(ch, i));
                    p_matPar.nnzBlocks()[ch] += p_matPar.krnRowParDesps(ch, i).nnzBlocks();
                }
                p_matPar.krnColParDesp(l_parId).m_nnzColMemBlocks =
                    alignedBlock(p_matPar.krnColParDesp(l_parId).m_nnzColMemBlocks, t_ParWordsPerMem);
                unsigned int l_colBlocks = p_matPar.krnColParDesp(l_parId).m_parMinColId / t_ColsPerMem;
                unsigned int l_colsInPar =
                    (l_nnzs == 0) ? 0 : p_matPar.krnColParDesp(l_parId).m_parMaxColId + 1 - l_colBlocks * t_ColsPerMem;
                p_matPar.krnColParDesp(l_parId).m_colVecMemBlocks = alignedBlock(l_colsInPar, t_ColsPerMem);
                if (p_matPar.krnColParDesp(l_parId).m_nnzColMemBlocks != 0) {
                    p_matPar.validPars() += 1;
                }
            }
        }
        for (unsigned int i = 0; i < l_rowPars; ++i) {
            for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
                if (p_matPar.krnRowParDesps(ch, i).nnzs() != 0) {
                    p_matPar.validRowPars()[ch] += 1;
                }
                p_matPar.krnRowParDesps(ch, i).rows() =
                    (p_matPar.krnRowParDesps(ch, i).nnzs() == 0)
                        ? 0
                        : p_matPar.krnRowParDesps(ch, i).maxRowId() + 1 - p_matPar.krnRowParDesps(ch, i).minRowId();
                p_matPar.krnRowParDesps(ch, i).rowBlocks() =
                    alignedBlock(p_matPar.krnRowParDesps(ch, i).rows(), t_ParEntries * t_ParGroups);
                p_matPar.krnRowParDesps(ch, i).rowResBlocks() =
                    alignedNum(p_matPar.krnRowParDesps(ch, i).rowBlocks() * t_ParGroups, 2);
                p_matPar.rowResBlocks()[ch] += alignedNum(p_matPar.krnRowParDesps(ch, i).rowResBlocks(), 2);
            }
        }
    }
};

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxRowsPerPar,
          unsigned int t_MaxColsPerPar,
          unsigned int t_MaxParamDdrBlocks,
          unsigned int t_MaxParamHbmBlocks,
          unsigned int t_ParEntries,
          unsigned int t_ParGroups,
          unsigned int t_DdrMemBits,
          unsigned int t_HbmMemBits,
          unsigned int t_HbmChannels,
          unsigned int t_ParamOffset = 1024,
          unsigned int t_PageSize = 4096>
class GenRunConfig {
   public:
    static const unsigned int t_BytesPerHbmRd = t_HbmMemBits / 8;
    static const unsigned int t_BytesPerDdrRd = t_DdrMemBits / 8;
    static const unsigned int t_IntsPerDdrRd = t_DdrMemBits / (8 * sizeof(uint32_t));
    static const unsigned int t_IntsPerHbmRd = t_HbmMemBits / (8 * sizeof(uint32_t));
    static const unsigned int t_IntsPerColParam = 2 + t_HbmChannels * 2;
    static const unsigned int t_DdrBlocks4ColParam = (t_IntsPerColParam + t_IntsPerDdrRd - 1) / t_IntsPerDdrRd;
    static const unsigned int t_MaxColParams = t_MaxParamDdrBlocks / t_DdrBlocks4ColParam;
    static const unsigned int t_IntsPerRwHbmParam = 4;
    static const unsigned int t_HbmBlocks4HbmParam = (t_IntsPerRwHbmParam + t_IntsPerHbmRd - 1) / t_IntsPerHbmRd;
    static const unsigned int t_MaxHbmParams = t_MaxParamHbmBlocks / t_HbmBlocks4HbmParam;
    static const unsigned int t_DatasPerDdrRd = t_DdrMemBits / (8 * sizeof(t_DataType));

   public:
    typedef Program<t_PageSize> t_ProgramType;
    typedef NnzUnit<t_DataType, t_IndexType> t_NnzUnitType;
    typedef struct ParamColPtr<t_HbmChannels> t_ParamColPtrType;
    typedef struct ParamColVec<t_HbmChannels> t_ParamColVecType;
    typedef struct ParamRwHbm t_ParamRwHbmType;
    typedef struct ChBlockDesp t_ChBlockDespType;
    typedef struct ColParDesp<t_HbmChannels> t_ColParDespType;
    typedef RoCooPar<t_DataType, t_IndexType, t_MaxRowsPerPar, t_MaxColsPerPar, t_HbmChannels> t_RoCooParType;
    typedef KrnColParDesp<t_HbmChannels> t_KrnColParDespType;
    typedef KrnRowParDesp<t_HbmChannels> t_KrnRowParDespType;
    typedef MatPar<t_DataType, t_IndexType, t_MaxRowsPerPar, t_MaxColsPerPar, t_HbmChannels> t_MatParType;
    typedef GenMatPar<t_DataType,
                      t_IndexType,
                      t_MaxRowsPerPar,
                      t_MaxColsPerPar,
                      t_ParEntries,
                      t_ParGroups,
                      t_DdrMemBits,
                      t_HbmChannels>
        t_GenMatParType;
    typedef RunConfig<t_DataType,
                      t_IndexType,
                      t_ParEntries,
                      t_ParGroups,
                      t_DdrMemBits,
                      t_HbmMemBits,
                      t_HbmChannels,
                      t_ParamOffset,
                      t_PageSize>
        t_RunConfigType;

   public:
    GenRunConfig() {}
    void genConfigParams(t_MatParType& p_matPar, t_RunConfigType& p_config) {
        // generate host params
        t_KrnColParDespType l_krnColParDesp = p_matPar.krnColParDesp();
        p_config.krnColParDesp() = l_krnColParDesp;
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            vector<t_KrnRowParDespType> l_krnRowParDesps = p_matPar.krnRowParDesps(i);
            p_config.krnRowParDesps(i) = l_krnRowParDesps;
        }
        unsigned int l_rows = p_matPar.roCooPar().rows();
        unsigned int l_cols = p_matPar.roCooPar().cols();
        unsigned int l_nnzs = p_matPar.roCooPar().nnzs();
        unsigned int l_rowPars = p_matPar.roCooPar().rowPars();
        unsigned int l_colPars = p_matPar.roCooPar().colPars();
        unsigned int l_totalPars = p_matPar.roCooPar().totalPars();
        unsigned int l_validPars = p_matPar.validPars();
        unsigned int* l_validRowPars = p_matPar.validRowPars();
        unsigned int* l_nnzBlocks = p_matPar.nnzBlocks();
        unsigned int* l_rowResBlocks = p_matPar.rowResBlocks();
        p_config.rows() = l_rows;
        p_config.cols() = l_cols;
        p_config.nnzs() = l_nnzs;
        p_config.rowPars() = l_rowPars;
        p_config.colPars() = l_colPars;
        p_config.totalPars() = l_totalPars;
        p_config.validPars() = l_validPars;
        p_config.nnzColMemBlocks() = 0;

        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            p_config.validRowPars()[i] = l_validRowPars[i];
            p_config.nnzBlocks()[i] = l_nnzBlocks[i];
            p_config.rowResBlocks()[i] = l_rowResBlocks[i];
        }
        for (unsigned int i = 0; i < l_totalPars; ++i) {
            p_config.nnzColMemBlocks() += p_config.krnColParDesp()[i].m_nnzColMemBlocks;
        }
    }
    void genConfigMem(t_ProgramType& p_program, t_RunConfigType& p_config) {
        unsigned int l_validPars = p_config.validPars();
        unsigned int l_colPtrParamBlocks = l_validPars * t_DdrBlocks4ColParam;
        p_config.colPtrParamBlocks() = l_colPtrParamBlocks;
        p_config.colVecParamBlocks() = l_colPtrParamBlocks;

        unsigned long long l_colPtrSz =
            (l_colPtrParamBlocks + p_config.nnzColMemBlocks()) * t_BytesPerDdrRd + t_ParamOffset;
        unsigned long long l_colVecSz = t_ParamOffset + l_colPtrParamBlocks * t_BytesPerDdrRd +
                                        alignedNum(p_config.cols(), t_DatasPerDdrRd) * sizeof(t_DataType);
        p_config.setColPtrAddr(p_program.allocMem(l_colPtrSz));
        p_config.setColVecAddr(p_program.allocMem(l_colVecSz));

        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            unsigned int l_paramBlocks = p_config.validRowPars()[i] * t_HbmBlocks4HbmParam;
            p_config.paramBlocks()[i] = l_paramBlocks;
            unsigned long long l_rdHbmSz = t_ParamOffset + (l_paramBlocks + p_config.nnzBlocks()[i]) * t_BytesPerHbmRd;
            p_config.setRdHbmAddr(p_program.allocMem(l_rdHbmSz), i);

            unsigned long long l_wrHbmSz = p_config.rowResBlocks()[i] * t_ParEntries * sizeof(t_DataType);
            p_config.setWrHbmAddr(p_program.allocMem(l_wrHbmSz), i);
            p_config.setRefHbmAddr(p_program.allocMem(l_wrHbmSz), i);
        }
    }
    void genCscFromNnzUnits(vector<t_NnzUnitType>& p_nnzUnits,
                            unsigned int p_minRowId,
                            unsigned int p_minColId,
                            vector<t_DataType>& p_val,
                            vector<t_IndexType>& p_rowIdx,
                            vector<t_IndexType>& p_colPtr) {
        sort(p_nnzUnits.begin(), p_nnzUnits.end());
        unsigned l_nnzs = p_nnzUnits.size();
        for (unsigned int i = 0; i < l_nnzs; ++i) {
            p_val[i] = p_nnzUnits[i].getVal();
            p_rowIdx[i] = p_nnzUnits[i].getRow() - p_minRowId;
            unsigned int l_colId = p_nnzUnits[i].getCol();
            p_colPtr[l_colId - p_minColId]++;
        }
        for (unsigned int i = 1; i < p_colPtr.size(); ++i) {
            p_colPtr[i] += p_colPtr[i - 1];
        }
    }
    void genCsc(vector<t_NnzUnitType>& p_nnzUnits,
                t_ChBlockDespType& p_desp,
                unsigned int p_minRowId,
                vector<t_DataType>& p_val,
                vector<t_IndexType>& p_rowIdx,
                vector<t_IndexType>& p_colPtr) {
        unsigned int l_startId = p_desp.m_startId;
        unsigned int l_minColId = p_desp.m_minColId;
        unsigned int l_colBlocks = p_desp.m_colBlocks;
        unsigned int l_nnzs = p_desp.m_nnzs;
        unsigned int l_nnzBlocks = p_desp.m_nnzBlocks;
        unsigned int l_valIdxSize = l_nnzBlocks * t_ParEntries;
        unsigned int l_colPtrSize = l_colBlocks * t_ParEntries;

        vector<t_NnzUnitType> l_nnzUnits(p_nnzUnits.begin() + l_startId, p_nnzUnits.begin() + l_startId + l_nnzs);
        assert(l_nnzUnits.size() == l_nnzs);

        p_val.resize(l_valIdxSize);
        p_rowIdx.resize(l_valIdxSize);
        p_colPtr.resize(l_colPtrSize, 0);
        genCscFromNnzUnits(l_nnzUnits, p_minRowId, l_minColId, p_val, p_rowIdx, p_colPtr);
        for (unsigned int i = l_nnzs; i < l_valIdxSize; ++i) {
            p_val[i] = 0;
            p_rowIdx[i] = p_rowIdx[l_nnzs - 1];
        }
        unsigned int l_cols = p_desp.m_cols;
        for (unsigned int i = l_cols; i < l_colPtrSize; ++i) {
            p_colPtr[i] = l_valIdxSize;
        }
        if ((l_cols == l_colPtrSize) && (l_cols != 0)) {
            p_colPtr[l_cols - 1] = l_valIdxSize;
        }
    }
    void genRowRes(vector<t_NnzUnitType>& p_nnzUnits,
                   vector<t_DataType>& p_inVec,
                   t_ChBlockDespType& p_desp,
                   unsigned int p_minRowId,
                   vector<t_DataType>& p_res) {
        unsigned int l_startId = p_desp.m_startId;
        unsigned int l_nnzs = p_desp.m_nnzs;
        for (unsigned int i = 0; i < l_nnzs; ++i) {
            t_NnzUnitType l_nnzUnit = p_nnzUnits[i + l_startId];
            t_IndexType l_rowId = l_nnzUnit.getRow();
            t_IndexType l_colId = l_nnzUnit.getCol();
            t_DataType l_val = l_nnzUnit.getVal();
            t_DataType l_mulVal = l_val * p_inVec[l_colId];
            p_res[l_rowId - p_minRowId] += l_mulVal;
        }
    }
    void setColPtrParamMem(t_ColParDespType& p_desp,
                           unsigned int& p_offset,
                           vector<t_ParamColPtrType>& p_paramColPtr,
                           void*& p_colPtrParamAddr) {
        t_ParamColPtrType l_param;
        l_param.m_offset = p_offset;
        l_param.m_memBlocks = p_desp.m_nnzColMemBlocks;
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            l_param.m_parBlocks[i] = p_desp.m_colBlocks[i];
            l_param.m_nnzBlocks[i] = p_desp.m_nnzBlocks[i];
        }
        p_paramColPtr.push_back(l_param);
        memcpy(reinterpret_cast<char*>(p_colPtrParamAddr), reinterpret_cast<char*>(&l_param), sizeof(l_param));
        p_colPtrParamAddr = reinterpret_cast<char*>(p_colPtrParamAddr) + t_DdrBlocks4ColParam * t_BytesPerDdrRd;
        p_offset += p_desp.m_nnzColMemBlocks;
    }

    void setColVecParamMem(t_ColParDespType& p_desp,
                           unsigned int& p_offset,
                           vector<t_ParamColVecType>& p_paramColVec,
                           void*& p_colVecParamAddr) {
        unsigned int l_parMinBlockId = p_desp.m_parMinColId / t_DatasPerDdrRd;
        t_ParamColVecType l_param;
        l_param.m_offset = p_offset;
        l_param.m_memBlocks = p_desp.m_colVecMemBlocks;
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            l_param.m_minId[i] = (p_desp.m_minColId[i] < p_desp.m_parMinColId)
                                     ? p_desp.m_minColId[i]
                                     : p_desp.m_minColId[i] - (l_parMinBlockId * t_DatasPerDdrRd);
            l_param.m_maxId[i] = (p_desp.m_maxColId[i] < p_desp.m_parMinColId)
                                     ? p_desp.m_maxColId[i]
                                     : p_desp.m_maxColId[i] - (l_parMinBlockId * t_DatasPerDdrRd);
        }
        p_paramColVec.push_back(l_param);
        memcpy(reinterpret_cast<char*>(p_colVecParamAddr), reinterpret_cast<char*>(&l_param), sizeof(l_param));
        p_colVecParamAddr = reinterpret_cast<char*>(p_colVecParamAddr) + t_DdrBlocks4ColParam * t_BytesPerDdrRd;
        p_offset += p_desp.m_colVecMemBlocks;
    }

    void setColPtrDatMem(vector<t_IndexType>& p_colPtr, void*& p_colPtrDatAddr) {
        unsigned long long l_colPtrSz = p_colPtr.size() * sizeof(t_IndexType);
        memcpy(reinterpret_cast<char*>(p_colPtrDatAddr), reinterpret_cast<char*>(p_colPtr.data()), l_colPtrSz);
        p_colPtrDatAddr = reinterpret_cast<char*>(p_colPtrDatAddr) + l_colPtrSz;
    }
    void setHbmDatMem(vector<t_DataType>& p_nnzVal, vector<t_IndexType>& p_rowIdx, void*& p_rdHbmDatAddr) {
        unsigned int l_nnzs = p_nnzVal.size();
        unsigned int l_nnzBlocks = l_nnzs / t_ParEntries;
        for (unsigned int i = 0; i < l_nnzBlocks; ++i) {
            memcpy(reinterpret_cast<char*>(p_rdHbmDatAddr), reinterpret_cast<char*>(&(p_rowIdx[i * t_ParEntries])),
                   t_ParEntries * sizeof(t_IndexType));
            p_rdHbmDatAddr = reinterpret_cast<char*>(p_rdHbmDatAddr) + t_ParEntries * sizeof(t_IndexType);
            memcpy(reinterpret_cast<char*>(p_rdHbmDatAddr), reinterpret_cast<char*>(&(p_nnzVal[i * t_ParEntries])),
                   t_ParEntries * sizeof(t_DataType));
            p_rdHbmDatAddr = reinterpret_cast<char*>(p_rdHbmDatAddr) + t_ParEntries * sizeof(t_DataType);
        }
    }
    void setRefHbmMem(void* p_refHbmAddr,
                      unsigned int p_wrOffset,
                      vector<t_DataType>& p_res,
                      t_KrnRowParDespType& p_desp) {
        unsigned int l_rows = p_desp.rows();
        unsigned int l_sz = l_rows * sizeof(t_DataType);
        unsigned int l_byteOffset = p_wrOffset * t_BytesPerHbmRd;
        char* l_wrAddr = reinterpret_cast<char*>(p_refHbmAddr) + l_byteOffset;
        memcpy(l_wrAddr, reinterpret_cast<char*>(p_res.data()), l_sz);
    }
    void setHbmParamMem(t_KrnRowParDespType& p_desp,
                        unsigned int& p_rdOffset,
                        unsigned int& p_wrOffset,
                        vector<t_ParamRwHbmType>& p_param,
                        void*& p_rdHbmParamAddr) {
        unsigned int l_nnzBlocks = p_desp.nnzBlocks();
        unsigned int l_rowBlocks = p_desp.rowBlocks();
        unsigned int l_rowResBlocks = p_desp.rowResBlocks();

        if (l_nnzBlocks != 0) {
            t_ParamRwHbmType l_param;
            l_param.m_rdOffset = p_rdOffset;
            l_param.m_wrOffset = p_wrOffset;
            l_param.m_nnzBlocks = l_nnzBlocks;
            l_param.m_rowBlocks = l_rowBlocks;
            p_param.push_back(l_param);
            memcpy(reinterpret_cast<char*>(p_rdHbmParamAddr), reinterpret_cast<char*>(&l_param), sizeof(l_param));
            p_rdHbmParamAddr = reinterpret_cast<char*>(p_rdHbmParamAddr) + t_HbmBlocks4HbmParam * t_BytesPerHbmRd;
            p_rdOffset += l_nnzBlocks;
            p_wrOffset += alignedBlock(l_rowResBlocks, 2);
        }
    }

    void setConfigMem(t_RoCooParType& p_par, vector<t_DataType>& p_inVec, t_RunConfigType& p_config) {
        memcpy(reinterpret_cast<char*>(p_config.getColPtrAddr()),
               reinterpret_cast<char*>(&(p_config.colPtrParamBlocks())), sizeof(uint32_t));
        memcpy(reinterpret_cast<char*>(p_config.getColVecAddr()),
               reinterpret_cast<char*>(&(p_config.colVecParamBlocks())), sizeof(uint32_t));
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            memcpy(reinterpret_cast<char*>(p_config.getRdHbmAddr(i)),
                   reinterpret_cast<char*>(&(p_config.paramBlocks()[i])), sizeof(uint32_t));
        }
        unsigned int l_rowPars = p_config.rowPars();
        unsigned int l_colPars = p_config.colPars();
        void* l_colPtrParamAddr = reinterpret_cast<char*>(p_config.getColPtrAddr()) + t_ParamOffset;
        void* l_colPtrDatAddr =
            reinterpret_cast<char*>(l_colPtrParamAddr) + p_config.validPars() * t_DdrBlocks4ColParam * t_BytesPerDdrRd;
        void* l_colVecParamAddr = reinterpret_cast<char*>(p_config.getColVecAddr()) + t_ParamOffset;
        void* l_colVecDatAddr =
            reinterpret_cast<char*>(l_colVecParamAddr) + p_config.validPars() * t_DdrBlocks4ColParam * t_BytesPerDdrRd;
        assert(p_inVec.size() == p_config.cols());
        unsigned long long l_colVecSz = p_config.cols() * sizeof(t_DataType);
        memcpy(reinterpret_cast<char*>(l_colVecDatAddr), reinterpret_cast<char*>(p_inVec.data()), l_colVecSz);

        void* l_rdHbmParamAddr[t_HbmChannels];
        void* l_rdHbmDatAddr[t_HbmChannels];
        unsigned int l_rdOffset[t_HbmChannels];
        unsigned int l_wrOffset[t_HbmChannels];

        unsigned int l_colPtrMemBlockOffset =
            (t_ParamOffset / t_BytesPerDdrRd) + p_config.validPars() * t_DdrBlocks4ColParam;
        unsigned int l_colVecMemBlockOffset = l_colPtrMemBlockOffset;

        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            void* l_addr = p_config.getRdHbmAddr(i);
            l_rdHbmParamAddr[i] = reinterpret_cast<char*>(l_addr) + t_ParamOffset;
            l_rdHbmDatAddr[i] = reinterpret_cast<char*>(l_rdHbmParamAddr[i]) +
                                p_config.validRowPars()[i] * t_HbmBlocks4HbmParam * t_BytesPerHbmRd;
            l_rdOffset[i] = (t_ParamOffset / t_BytesPerHbmRd) + p_config.validRowPars()[i] * t_HbmBlocks4HbmParam;
            l_wrOffset[i] = 0;
        }
        for (unsigned int i = 0; i < l_rowPars; ++i) {
            vector<t_DataType> l_rowRes[t_HbmChannels];
            for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
                l_rowRes[ch].resize(p_config.rows(), 0);
            }
            for (unsigned int j = 0; j < l_colPars; ++j) {
                unsigned int l_parId = i * l_colPars + j;
                if (p_par[l_parId].size() != 0) {
                    setColPtrParamMem(p_config.krnColParDesp(l_parId), l_colPtrMemBlockOffset, p_config.paramColPtr(),
                                      l_colPtrParamAddr);
                    setColVecParamMem(p_config.krnColParDesp(l_parId), l_colVecMemBlockOffset, p_config.paramColVec(),
                                      l_colVecParamAddr);
                    void* l_blockColPtrDatAddr = l_colPtrDatAddr;
                    for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
                        vector<t_DataType> l_nnzVal;
                        vector<t_IndexType> l_rowIdx;
                        vector<t_IndexType> l_colPtr;
                        unsigned int l_minRowId = p_config.krnRowParDesps(ch, i).minRowId();
                        t_ChBlockDespType l_chBlockDesp = p_config.krnRowParDesps(ch, i)[j];
                        if (l_chBlockDesp.m_nnzs == 0) {
                            continue;
                        }
                        genCsc(p_par[l_parId], l_chBlockDesp, l_minRowId, l_nnzVal, l_rowIdx, l_colPtr);
                        genRowRes(p_par[l_parId], p_inVec, l_chBlockDesp, l_minRowId, l_rowRes[ch]);
                        setColPtrDatMem(l_colPtr, l_blockColPtrDatAddr);
                        setHbmDatMem(l_nnzVal, l_rowIdx, l_rdHbmDatAddr[ch]);
                    }
                    l_colPtrDatAddr = reinterpret_cast<char*>(l_colPtrDatAddr) +
                                      p_config.krnColParDesp(l_parId).m_nnzColMemBlocks * t_BytesPerDdrRd;
                }
            }
            for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
                setRefHbmMem(p_config.getRefHbmAddr(ch), l_wrOffset[ch], l_rowRes[ch], p_config.krnRowParDesps(ch, i));
                setHbmParamMem(p_config.krnRowParDesps(ch, i), l_rdOffset[ch], l_wrOffset[ch], p_config.paramRwHbm(ch),
                               l_rdHbmParamAddr[ch]);
            }
        }
    }
    void genRunConfig(t_ProgramType& p_program,
                      vector<t_NnzUnitType>& p_nnzUnits,
                      vector<t_DataType>& p_inVec,
                      unsigned int l_rows,
                      unsigned int l_cols,
                      t_RunConfigType& p_config) {
        t_GenMatParType l_genMatPar;
        t_MatParType l_matPar;
        l_genMatPar.genMatPar(p_nnzUnits, l_rows, l_cols, l_matPar);
        genConfigParams(l_matPar, p_config);
        genConfigMem(p_program, p_config);
        setConfigMem(l_matPar.roCooPar(), p_inVec, p_config);
    }
};
} // end namespace sparse
} // end namespace xf
#endif
