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
 * @file L2_types.hpp
 * @brief header file for data types used in L2/L3 host code.
 *
 * This file is part of Vitis SPARSE Library.
 */
#ifndef XF_SPARSE_L2_TYPES_HPP
#define XF_SPARSE_L2_TYPES_HPP

#include <algorithm>
#include <cassert>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include "L2_utils.hpp"
#include "program.hpp"
#include "nnz_worker.hpp"

using namespace std;

namespace xf {
namespace sparse {

template <typename t_DataType, unsigned int t_MemBits>
class ColVec {
   public:
    static const unsigned int t_MemWords = t_MemBits / (8 * sizeof(t_DataType));

   public:
    ColVec() : m_entries(0), m_valAddr(nullptr) {}
    ColVec(unsigned int p_entries) : m_entries(p_entries), m_valAddr(nullptr) {}
    inline unsigned int getEntries() { return m_entries; }
    inline void setEntries(unsigned int p_entries) { m_entries = p_entries; }
    inline void setValAddr(void* p_valAddr) { m_valAddr = p_valAddr; }
    inline void* getValAddr() { return m_valAddr; }
    inline void setEntryVal(unsigned int p_idx, t_DataType p_val) {
        t_DataType* l_valPtr = reinterpret_cast<t_DataType*>(m_valAddr);
        l_valPtr[p_idx] = p_val;
    }
    inline t_DataType getEntryVal(unsigned int p_idx) {
        t_DataType* l_valPtr = reinterpret_cast<t_DataType*>(m_valAddr);
        return (l_valPtr[p_idx]);
    }
    void storeVal(vector<t_DataType>& p_colVec) {
        if (m_entries == 0) {
            m_entries = p_colVec.size();
        } else {
            assert(m_entries == p_colVec.size());
        }
        assert(m_entries % t_MemWords == 0);
        unsigned int l_colBlocks = m_entries / t_MemWords;
        for (unsigned int i = 0; i < l_colBlocks; ++i) {
            t_DataType l_colWord[t_MemWords];
            for (unsigned int j = 0; j < t_MemWords; ++j) {
                l_colWord[j] = p_colVec[i * t_MemWords + j];
            }
            unsigned int l_bytes = t_MemWords * sizeof(t_DataType);
            uint8_t* l_valAddr = reinterpret_cast<uint8_t*>(m_valAddr) + i * t_MemWords * sizeof(t_DataType);
            memcpy(l_valAddr, reinterpret_cast<uint8_t*>(&l_colWord[0]), l_bytes);
        }
    }
    void loadVal(vector<t_DataType>& p_colVec) {
        assert(m_entries % t_MemWords == 0);
        p_colVec.resize(m_entries);
        unsigned int l_colBlocks = m_entries / t_MemWords;

        for (unsigned int i = 0; i < l_colBlocks; ++i) {
            t_DataType l_colWord[t_MemWords];
            unsigned int l_bytes = t_MemWords * sizeof(t_DataType);
            uint8_t* l_valAddr = reinterpret_cast<uint8_t*>(m_valAddr) + i * t_MemWords * sizeof(t_DataType);
            memcpy(reinterpret_cast<uint8_t*>(&l_colWord[0]), l_valAddr, l_bytes);
            for (unsigned int j = 0; j < t_MemWords; ++j) {
                p_colVec[i * t_MemWords + j] = l_colWord[j];
            }
        }
    }

   private:
    unsigned int m_entries;
    void* m_valAddr;
};

template <typename t_DataType, typename t_IndexType, unsigned int t_LineBreakEntries = 8>
class MatCsc {
   public:
    MatCsc() : m_rows(0), m_cols(0), m_nnzs(0), m_valAddr(nullptr), m_rowAddr(nullptr), m_colPtrAddr(nullptr) {
        assert(sizeof(t_DataType) == sizeof(t_IndexType));
    }
    inline unsigned int getRows() { return m_rows; }
    inline unsigned int getCols() { return m_cols; }
    inline unsigned int getNnzs() { return m_nnzs; }
    inline void setRows(unsigned int p_rows) { m_rows = p_rows; }
    inline void setCols(unsigned int p_cols) { m_cols = p_cols; }
    inline void setNnzs(unsigned int p_nnzs) { m_nnzs = p_nnzs; }

    inline void setValAddr(void* p_valAddr) { m_valAddr = p_valAddr; }
    inline void* getValAddr() { return m_valAddr; }

    inline void setRowAddr(void* p_rowAddr) { m_rowAddr = p_rowAddr; }
    inline void* getRowAddr() { return m_rowAddr; }

    inline void setColPtrAddr(void* p_colPtrAddr) { m_colPtrAddr = p_colPtrAddr; }
    inline void* getColPtrAddr() { return m_colPtrAddr; }

    bool isEqual(MatCsc& p_mat) {
        bool l_res = true;
        l_res = l_res && (m_nnzs == p_mat.getNnzs());
        l_res = l_res && (m_rows == p_mat.getRows());
        l_res = l_res && (m_cols == p_mat.getCols());
        int l_memCmp = 0;
        l_memCmp = memcmp(m_valAddr, p_mat.getValAddr(), m_nnzs * sizeof(t_DataType));
        l_res = l_res && (l_memCmp == 0);
        l_memCmp = memcmp(m_rowAddr, p_mat.getRowAddr(), m_nnzs * sizeof(t_IndexType));
        l_res = l_res && (l_memCmp == 0);
        l_memCmp = memcmp(m_colPtrAddr, p_mat.getColPtrAddr(), m_cols * sizeof(t_IndexType));
        l_res = l_res && (l_memCmp == 0);
        return l_res;
    }
    bool write2BinFile(string p_valFileName, string p_rowFileName, string p_colPtrFileName) {
        bool l_res = true;
        ofstream l_ofVal(p_valFileName.c_str(), ios::binary);
        if (!l_ofVal.is_open()) {
            cout << "ERROR: Open " << p_valFileName << endl;
            return false;
        }
        ofstream l_ofRow(p_rowFileName.c_str(), ios::binary);
        if (!l_ofRow.is_open()) {
            cout << "ERROR: Open " << p_rowFileName << endl;
            return false;
        }
        ofstream l_ofColPtr(p_colPtrFileName.c_str(), ios::binary);
        if (!l_ofColPtr) {
            cout << "ERROR: Open " << p_colPtrFileName << endl;
        }
        unsigned long long l_valSz = m_nnzs * sizeof(t_DataType);
        unsigned long long l_colPtrSz = m_cols * sizeof(t_IndexType);
        l_ofVal.write(reinterpret_cast<char*>(m_valAddr), l_valSz);
        l_ofRow.write(reinterpret_cast<char*>(m_rowAddr), l_valSz);
        l_ofColPtr.write(reinterpret_cast<char*>(m_colPtrAddr), l_colPtrSz);
        l_ofVal.close();
        l_ofRow.close();
        l_ofColPtr.close();
        return l_res;
    }
    void print(ostream& os) {
        os << "CSC Mat: rows=" << m_rows << " cols= " << m_cols << " nnzs= " << m_nnzs << endl;
        os << "NNZ Vals:" << endl;
        t_DataType* l_val = reinterpret_cast<t_DataType*>(m_valAddr);
        for (unsigned int i = 0; i < m_nnzs; ++i) {
            os << l_val[i] << "  ";
            if ((i + 1) % t_LineBreakEntries == 0) os << endl;
        }
        os << endl;
        os << "NNZ Row indices:" << endl;
        t_IndexType* l_rowIdx = reinterpret_cast<t_IndexType*>(m_rowAddr);
        for (unsigned int i = 0; i < m_nnzs; ++i) {
            os << l_rowIdx[i] << "  ";
            if ((i + 1) % t_LineBreakEntries == 0) os << endl;
        }
        os << endl;
        os << "NNZ Col Pointers:" << endl;
        t_IndexType* l_colPtr = reinterpret_cast<t_IndexType*>(m_colPtrAddr);
        for (unsigned int i = 0; i < m_cols; ++i) {
            os << l_colPtr[i] << "  ";
            if ((i + 1) % t_LineBreakEntries == 0) os << endl;
        }
        os << endl;
    }
    friend ostream& operator<<(ostream& os, MatCsc& p_mat) {
        p_mat.print(os);
        return (os);
    }

   private:
    unsigned int m_rows, m_cols, m_nnzs;
    void* m_valAddr;
    void* m_rowAddr;
    void* m_colPtrAddr;
};

struct ChBlockDesp {
    unsigned int m_startId;
    unsigned int m_minRowId, m_maxRowId;
    unsigned int m_minColId, m_maxColId;
    unsigned int m_rows, m_cols, m_nnzs;
    unsigned int m_nnzBlocks, m_colBlocks, m_rowBlocks, m_rowResBlocks;
    ChBlockDesp() {
        m_startId = 0;
        m_minRowId = 0;
        m_maxRowId = 0;
        m_minColId = 0;
        m_maxColId = 0;
        m_rows = 0;
        m_cols = 0;
        m_nnzs = 0;
        m_nnzBlocks = 0;
        m_colBlocks = 0;
        m_rowBlocks = 0;
        m_rowResBlocks = 0;
    }
    ChBlockDesp& operator=(const ChBlockDesp& p_desp) {
        m_startId = p_desp.m_startId;
        m_minRowId = p_desp.m_minRowId;
        m_maxRowId = p_desp.m_maxRowId;
        m_rows = p_desp.m_rows;
        m_cols = p_desp.m_cols;
        m_nnzs = p_desp.m_nnzs;
        m_nnzBlocks = p_desp.m_nnzBlocks;
        m_colBlocks = p_desp.m_colBlocks;
        m_rowBlocks = p_desp.m_rowBlocks;
        m_rowResBlocks = p_desp.m_rowResBlocks;
        return *this;
    }
};

template <unsigned t_HbmChannels>
struct ColParDesp {
    unsigned int m_minColId[t_HbmChannels];
    unsigned int m_maxColId[t_HbmChannels];
    unsigned int m_cols[t_HbmChannels];
    unsigned int m_colBlocks[t_HbmChannels];
    unsigned int m_nnzs[t_HbmChannels];
    unsigned int m_nnzBlocks[t_HbmChannels];
    unsigned int m_parMinColId, m_parMaxColId;
    unsigned int m_colVecMemBlocks, m_nnzColMemBlocks;
    ColParDesp() {
        m_parMinColId = 0;
        m_parMaxColId = 0;
        m_colVecMemBlocks = 0;
        m_nnzColMemBlocks = 0;
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            m_minColId[i] = 0;
            m_maxColId[i] = 0;
            m_cols[i] = 0;
            m_colBlocks[i] = 0;
            m_nnzs[i] = 0;
            m_nnzBlocks[i] = 0;
        }
    }
    ColParDesp& operator=(const ColParDesp& p_desp) {
        m_parMinColId = p_desp.m_parMinColId;
        m_parMaxColId = p_desp.m_parMaxColId;
        m_colVecMemBlocks = p_desp.m_colVecMemBlocks;
        m_nnzColMemBlocks = p_desp.m_nnzColMemBlocks;
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            m_minColId[i] = p_desp.m_minColId[i];
            m_maxColId[i] = p_desp.m_maxColId[i];
            m_cols[i] = p_desp.m_cols[i];
            m_colBlocks[i] = p_desp.m_colBlocks[i];
            m_nnzs[i] = p_desp.m_nnzs[i];
            m_nnzBlocks[i] = p_desp.m_nnzBlocks[i];
        }
        return *this;
    }
};

template <unsigned t_HbmChannels>
class KrnColParDesp {
   public:
    typedef struct ColParDesp<t_HbmChannels> t_ColParDespType;

   public:
    KrnColParDesp() {}
    void resize(unsigned int p_size) { m_desp.resize(p_size); }
    vector<t_ColParDespType>& desp() { return m_desp; }
    t_ColParDespType& operator[](unsigned int p_id) {
        assert(p_id < m_desp.size());
        return (m_desp[p_id]);
    }
    unsigned int size() { return (m_desp.size()); }
    vector<t_ColParDespType> getDesp() const { return m_desp; }
    KrnColParDesp& operator=(const KrnColParDesp& p_desp) {
        m_desp = p_desp.getDesp();
        return *this;
    }
    void writeBinFile(ofstream& p_of) {
        unsigned int l_pars = m_desp.size();
        p_of.write(reinterpret_cast<char*>(&l_pars), sizeof(uint32_t));
        for (unsigned int i = 0; i < l_pars; ++i) {
            p_of.write(reinterpret_cast<char*>(&(m_desp[i].m_parMinColId)), sizeof(uint32_t));
            p_of.write(reinterpret_cast<char*>(&(m_desp[i].m_parMaxColId)), sizeof(uint32_t));
            p_of.write(reinterpret_cast<char*>(&(m_desp[i].m_colVecMemBlocks)), sizeof(uint32_t));
            p_of.write(reinterpret_cast<char*>(&(m_desp[i].m_nnzColMemBlocks)), sizeof(uint32_t));
            for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
                p_of.write(reinterpret_cast<char*>(&(m_desp[i].m_minColId[ch])), sizeof(uint32_t));
                p_of.write(reinterpret_cast<char*>(&(m_desp[i].m_maxColId[ch])), sizeof(uint32_t));
                p_of.write(reinterpret_cast<char*>(&(m_desp[i].m_cols[ch])), sizeof(uint32_t));
                p_of.write(reinterpret_cast<char*>(&(m_desp[i].m_colBlocks[ch])), sizeof(uint32_t));
                p_of.write(reinterpret_cast<char*>(&(m_desp[i].m_nnzs[ch])), sizeof(uint32_t));
                p_of.write(reinterpret_cast<char*>(&(m_desp[i].m_nnzBlocks[ch])), sizeof(uint32_t));
            }
        }
    }
    void readBinFile(ifstream& p_if) {
        unsigned int l_pars;
        p_if.read(reinterpret_cast<char*>(&l_pars), sizeof(uint32_t));
        m_desp.resize(l_pars);
        for (unsigned int i = 0; i < l_pars; ++i) {
            p_if.read(reinterpret_cast<char*>(&(m_desp[i].m_parMinColId)), sizeof(uint32_t));
            p_if.read(reinterpret_cast<char*>(&(m_desp[i].m_parMaxColId)), sizeof(uint32_t));
            p_if.read(reinterpret_cast<char*>(&(m_desp[i].m_colVecMemBlocks)), sizeof(uint32_t));
            p_if.read(reinterpret_cast<char*>(&(m_desp[i].m_nnzColMemBlocks)), sizeof(uint32_t));
            for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
                p_if.read(reinterpret_cast<char*>(&(m_desp[i].m_minColId[ch])), sizeof(uint32_t));
                p_if.read(reinterpret_cast<char*>(&(m_desp[i].m_maxColId[ch])), sizeof(uint32_t));
                p_if.read(reinterpret_cast<char*>(&(m_desp[i].m_cols[ch])), sizeof(uint32_t));
                p_if.read(reinterpret_cast<char*>(&(m_desp[i].m_colBlocks[ch])), sizeof(uint32_t));
                p_if.read(reinterpret_cast<char*>(&(m_desp[i].m_nnzs[ch])), sizeof(uint32_t));
                p_if.read(reinterpret_cast<char*>(&(m_desp[i].m_nnzBlocks[ch])), sizeof(uint32_t));
            }
        }
    }
    void print(ostream& p_os) {
        p_os << "INFO: KrnColParDesp" << endl;
        unsigned int l_pars = m_desp.size();
        p_os << "total number of partitions is: " << l_pars << endl;
        for (unsigned int i = 0; i < l_pars; ++i) {
            t_ColParDespType l_desp = m_desp[i];
            p_os << "Partition " << i << ":" << endl;
            p_os << "parMinColId, parMaxColId, colVecMemBlocks, nnzColMemBlocks" << endl;
            p_os << l_desp.m_parMinColId << "  ,  " << l_desp.m_parMaxColId << "  ,  " << l_desp.m_colVecMemBlocks
                 << "  ,  " << l_desp.m_nnzColMemBlocks << endl;
            for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
                p_os << "    Channel " << ch << endl;
                p_os << "        minColId, maxColId, cols, colBlocks, nnzs, nnzBlocks" << endl;
                p_os << "        " << l_desp.m_minColId[ch] << " , " << l_desp.m_maxColId[ch] << " , "
                     << l_desp.m_cols[ch] << " , ";
                p_os << l_desp.m_colBlocks[ch] << " , " << l_desp.m_nnzs[ch] << " , " << l_desp.m_nnzBlocks[ch] << endl;
            }
        }
        p_os << endl;
    }
    friend ostream& operator<<(ostream& p_os, KrnColParDesp& p_desp) {
        p_desp.print(p_os);
        return (p_os);
    }

   private:
    vector<t_ColParDespType> m_desp;
};

template <unsigned int t_HbmChannels>
class KrnRowParDesp {
   public:
    typedef struct ChBlockDesp t_ChBlockDespType;

   public:
    KrnRowParDesp() {}
    inline unsigned int& minRowId() { return m_minRowId; }
    inline unsigned int& maxRowId() { return m_maxRowId; }
    inline unsigned int& rows() { return m_rows; }
    inline unsigned int& nnzs() { return m_nnzs; }
    inline unsigned int& nnzBlocks() { return m_nnzBlocks; }
    inline unsigned int& rowBlocks() { return m_rowBlocks; }
    inline unsigned int& rowResBlocks() { return m_rowResBlocks; }
    vector<t_ChBlockDespType>& rowDesp() { return m_rowDesp; }
    t_ChBlockDespType& operator[](unsigned int p_id) {
        assert(p_id < m_rowDesp.size());
        return (m_rowDesp[p_id]);
    }
    inline unsigned int getMinRowId() const { return m_minRowId; }
    inline unsigned int getMaxRowId() const { return m_maxRowId; }
    inline unsigned int getRows() const { return m_rows; }
    inline unsigned int getNnzs() const { return m_nnzs; }
    inline unsigned int getNnzBlocks() const { return m_nnzBlocks; }
    inline unsigned int getRowBlocks() const { return m_rowBlocks; }
    inline unsigned int getRowResBlocks() const { return m_rowResBlocks; }
    vector<t_ChBlockDespType> getRowDesp() const { return m_rowDesp; }
    KrnRowParDesp& operator=(const KrnRowParDesp& p_desp) {
        m_minRowId = p_desp.getMinRowId();
        m_maxRowId = p_desp.getMaxRowId();
        m_rows = p_desp.getRows();
        m_nnzs = p_desp.getNnzs();
        m_nnzBlocks = p_desp.getNnzBlocks();
        m_rowBlocks = p_desp.getRowBlocks();
        m_rowResBlocks = p_desp.getRowResBlocks();
        m_rowDesp = p_desp.getRowDesp();
        return *this;
    }
    unsigned int size() { return m_rowDesp.size(); }
    void resize(unsigned int p_size) { m_rowDesp.resize(p_size); }
    void addChBlockDesp(t_ChBlockDespType p_desp) { m_rowDesp.push_back(p_desp); }
    void writeBinFile(ofstream& p_of) {
        p_of.write(reinterpret_cast<char*>(&m_minRowId), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_maxRowId), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_rows), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_nnzs), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_nnzBlocks), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_rowBlocks), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_rowResBlocks), sizeof(uint32_t));
        unsigned int l_colPars = m_rowDesp.size();
        p_of.write(reinterpret_cast<char*>(&l_colPars), sizeof(uint32_t));
        for (unsigned int i = 0; i < l_colPars; ++i) {
            p_of.write(reinterpret_cast<char*>(&(m_rowDesp[i].m_startId)), sizeof(uint32_t));
            p_of.write(reinterpret_cast<char*>(&(m_rowDesp[i].m_minRowId)), sizeof(uint32_t));
            p_of.write(reinterpret_cast<char*>(&(m_rowDesp[i].m_maxRowId)), sizeof(uint32_t));
            p_of.write(reinterpret_cast<char*>(&(m_rowDesp[i].m_minColId)), sizeof(uint32_t));
            p_of.write(reinterpret_cast<char*>(&(m_rowDesp[i].m_maxColId)), sizeof(uint32_t));
            p_of.write(reinterpret_cast<char*>(&(m_rowDesp[i].m_rows)), sizeof(uint32_t));
            p_of.write(reinterpret_cast<char*>(&(m_rowDesp[i].m_cols)), sizeof(uint32_t));
            p_of.write(reinterpret_cast<char*>(&(m_rowDesp[i].m_nnzs)), sizeof(uint32_t));
            p_of.write(reinterpret_cast<char*>(&(m_rowDesp[i].m_nnzBlocks)), sizeof(uint32_t));
            p_of.write(reinterpret_cast<char*>(&(m_rowDesp[i].m_colBlocks)), sizeof(uint32_t));
            p_of.write(reinterpret_cast<char*>(&(m_rowDesp[i].m_rowBlocks)), sizeof(uint32_t));
            p_of.write(reinterpret_cast<char*>(&(m_rowDesp[i].m_rowResBlocks)), sizeof(uint32_t));
        }
    }
    void readBinFile(ifstream& p_if) {
        p_if.read(reinterpret_cast<char*>(&m_minRowId), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_maxRowId), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_rows), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_nnzs), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_nnzBlocks), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_rowBlocks), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_rowResBlocks), sizeof(uint32_t));
        unsigned int l_colPars = m_rowDesp.size();
        p_if.read(reinterpret_cast<char*>(&l_colPars), sizeof(uint32_t));
        m_rowDesp.resize(l_colPars);
        for (unsigned int i = 0; i < l_colPars; ++i) {
            p_if.read(reinterpret_cast<char*>(&(m_rowDesp[i].m_startId)), sizeof(uint32_t));
            p_if.read(reinterpret_cast<char*>(&(m_rowDesp[i].m_minRowId)), sizeof(uint32_t));
            p_if.read(reinterpret_cast<char*>(&(m_rowDesp[i].m_maxRowId)), sizeof(uint32_t));
            p_if.read(reinterpret_cast<char*>(&(m_rowDesp[i].m_minColId)), sizeof(uint32_t));
            p_if.read(reinterpret_cast<char*>(&(m_rowDesp[i].m_maxColId)), sizeof(uint32_t));
            p_if.read(reinterpret_cast<char*>(&(m_rowDesp[i].m_rows)), sizeof(uint32_t));
            p_if.read(reinterpret_cast<char*>(&(m_rowDesp[i].m_cols)), sizeof(uint32_t));
            p_if.read(reinterpret_cast<char*>(&(m_rowDesp[i].m_nnzs)), sizeof(uint32_t));
            p_if.read(reinterpret_cast<char*>(&(m_rowDesp[i].m_nnzBlocks)), sizeof(uint32_t));
            p_if.read(reinterpret_cast<char*>(&(m_rowDesp[i].m_colBlocks)), sizeof(uint32_t));
            p_if.read(reinterpret_cast<char*>(&(m_rowDesp[i].m_rowBlocks)), sizeof(uint32_t));
            p_if.read(reinterpret_cast<char*>(&(m_rowDesp[i].m_rowResBlocks)), sizeof(uint32_t));
        }
    }
    void print(ostream& p_os) {
        p_os << "INFO: KrnRowParDesp" << endl;
        p_os << "minRowId, maxRowId, m_rows, m_nnzs, m_nnzBlocks, m_rowBlocks, m_rowResBlocks" << endl;
        p_os << m_minRowId << "  ,  " << m_maxRowId << " , " << m_rows << " , ";
        p_os << m_nnzs << " , " << m_nnzBlocks << " , " << m_rowBlocks << " , " << m_rowResBlocks << endl;
        unsigned int l_colPars = m_rowDesp.size();
        p_os << "total number of partitions in the row block is: " << l_colPars << endl;
        for (unsigned int i = 0; i < l_colPars; ++i) {
            t_ChBlockDespType l_desp = m_rowDesp[i];
            p_os << "    Partition " << i << ":" << endl;
            p_os << "        startId = " << l_desp.m_startId << endl;
            p_os << "        minRowId, maxRowId, minColId, maxColId" << endl;
            p_os << "        " << l_desp.m_minRowId << " , " << l_desp.m_maxRowId << " , " << l_desp.m_minColId << " , "
                 << l_desp.m_maxColId << endl;
            p_os << "        rows, cols, nnzs, nnzBlocks, rowBlocks, rowResBlocks" << endl;
            p_os << "        " << l_desp.m_rows << " , " << l_desp.m_cols << " , " << l_desp.m_nnzs << " , ";
            p_os << l_desp.m_nnzBlocks << " , " << l_desp.m_rowBlocks << " , " << l_desp.m_rowResBlocks << endl;
        }
        p_os << endl;
    }
    friend ostream& operator<<(ostream& p_os, KrnRowParDesp& p_desp) {
        p_desp.print(p_os);
        return (p_os);
    }

   private:
    unsigned int m_minRowId, m_maxRowId;
    unsigned int m_rows, m_nnzs, m_nnzBlocks, m_rowBlocks, m_rowResBlocks;
    vector<t_ChBlockDespType> m_rowDesp;
};

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxRowsPerPar,
          unsigned int t_MaxColsPerPar,
          unsigned int t_HbmChannels>
class RoCooPar {
   public:
    typedef NnzUnit<t_DataType, t_IndexType> t_NnzUnitType;

   public:
    RoCooPar() : m_rows(0), m_cols(0), m_nnzs(0) {}
    void init(unsigned int p_rows, unsigned int p_cols) {
        m_rows = p_rows;
        m_cols = p_cols;
        m_colPars = alignedBlock(p_cols, t_MaxColsPerPar);
        m_rowPars = alignedBlock(p_rows, t_MaxRowsPerPar);
        m_totalPars = m_rowPars * m_colPars;
        if (m_matPars.size() == 0) {
            m_matPars.resize(m_totalPars);
        }
        assert(m_matPars.size() == m_totalPars);
    }
    unsigned int& rows() { return m_rows; }
    unsigned int& cols() { return m_cols; }
    unsigned int& nnzs() { return m_nnzs; }
    unsigned int& colPars() { return m_colPars; }
    unsigned int& rowPars() { return m_rowPars; }
    unsigned int& totalPars() { return m_totalPars; }
    vector<vector<t_NnzUnitType> >& matPars() { return m_matPars; }
    vector<t_NnzUnitType>& operator[](unsigned int p_id) {
        assert(p_id < m_matPars.size());
        return (m_matPars[p_id]);
    }
    void addNnzUnit(unsigned int p_id, t_NnzUnitType p_nnzUnit) {
        assert(m_matPars.size() > p_id);
        m_matPars[p_id].push_back(p_nnzUnit);
    }
    bool check(unsigned int p_rows, unsigned int p_cols, vector<t_NnzUnitType>& p_nnzUnits) {
        bool l_res = true;
        unsigned int l_nnzs = p_nnzUnits.size();
        l_res = l_res && (m_rows == p_rows);
        if (!l_res) {
            cout << "ERROR: "
                 << "number of rows in RoCooPar is " << m_rows << " != " << p_rows << " in the original mat." << endl;
            return false;
        }
        l_res = l_res && (m_cols == p_cols);
        if (!l_res) {
            cout << "ERROR: "
                 << "number of cols in RoCooPar is " << m_cols << " != " << p_cols << " in the original mat." << endl;
            return false;
        }
        l_res = l_res && (m_nnzs == l_nnzs);
        if (!l_res) {
            cout << "ERROR: "
                 << "number of nnzs in RoCooPar is " << m_nnzs << " != " << l_nnzs << " in the original mat." << endl;
            return false;
        }

        vector<t_NnzUnitType> l_nnzUnits(p_nnzUnits);
        // sort(l_nnzUnits.begin(), l_nnzUnits.end(), compareRow<t_DataType, t_IndexType>());
        sort(l_nnzUnits.begin(), l_nnzUnits.end());
        vector<t_NnzUnitType> l_parNnzUnits;
        for (unsigned int i = 0; i < m_totalPars; ++i) {
            unsigned int l_parSize = m_matPars[i].size();
            for (unsigned int j = 0; j < l_parSize; ++j) {
                l_parNnzUnits.push_back(m_matPars[i][j]);
            }
        }
        // sort(l_parNnzUnits.begin(), l_parNnzUnits.end(),  compareRow<t_DataType, t_IndexType>());
        sort(l_parNnzUnits.begin(), l_parNnzUnits.end());
        l_res = l_res && (l_nnzs == l_parNnzUnits.size());
        if (!l_res) {
            cout << "ERROR: "
                 << "number of accumulated nnzs in RoCooPar is " << l_parNnzUnits.size() << " != " << l_nnzs
                 << " in the original mat." << endl;
            return false;
        }
        unsigned int l_errs = 0;
        for (unsigned int i = 0; i < l_nnzs; ++i) {
            if (!(l_parNnzUnits[i] == l_nnzUnits[i])) {
                cout << "ERROR: at entry " << i << ", orignal entry is: " << l_nnzUnits[i]
                     << " entry in the partiton is: " << l_parNnzUnits[i] << endl;
                l_errs++;
            }
        }
        cout << "Total errors: " << l_errs << endl;
        l_res = l_res && (l_errs == 0);
        return l_res;
    }

    void writeBinFile(ofstream& p_of) {
        p_of.write(reinterpret_cast<char*>(&m_rows), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_cols), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_nnzs), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_colPars), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_rowPars), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_totalPars), sizeof(uint32_t));
        for (unsigned int i = 0; i < m_totalPars; ++i) {
            unsigned int l_parSize = m_matPars[i].size();
            p_of.write(reinterpret_cast<char*>(&l_parSize), sizeof(uint32_t));
            for (unsigned int j = 0; j < l_parSize; ++j) {
                m_matPars[i][j].writeBinFile(p_of);
            }
        }
    }
    void readBinFile(ifstream& p_if) {
        p_if.read(reinterpret_cast<char*>(&m_rows), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_cols), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_nnzs), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_colPars), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_rowPars), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_totalPars), sizeof(uint32_t));
        m_matPars.resize(m_totalPars);
        for (unsigned int i = 0; i < m_totalPars; ++i) {
            unsigned int l_parSize = 0;
            p_if.read(reinterpret_cast<char*>(&l_parSize), sizeof(uint32_t));
            m_matPars[i].resize(l_parSize);
            for (unsigned int j = 0; j < l_parSize; ++j) {
                m_matPars[i][j].readBinFile(p_if);
            }
        }
    }
    void print(ostream& p_os) {
        p_os << "INFO: RoCooPar " << endl;
        p_os << "rows, cols, nnzs, colPars, rowPars, totalPars" << endl;
        p_os << m_rows << ", " << m_cols << ", " << m_nnzs << ", " << m_colPars << ", " << m_rowPars << ", "
             << m_totalPars << endl;
        for (unsigned int i = 0; i < m_totalPars; ++i) {
            p_os << "Partition " << i << " has " << m_matPars[i].size() << " nnzs" << endl;
#if DEBUG_dumpData
            unsigned int l_nnzs = m_matPars[i].size();
            if (l_nnzs != 0) {
                p_os << " rowId, colId, val" << endl;
            }
            for (unsigned int j = 0; j < l_nnzs; ++j) {
                p_os << m_matPars[i][j] << endl;
            }
#endif
        }
        p_os << endl;
    }
    friend ostream& operator<<(ostream& p_os, RoCooPar& p_roCooPar) {
        p_roCooPar.print(p_os);
        return (p_os);
    }

   private:
    unsigned int m_rows, m_cols, m_nnzs;
    unsigned int m_colPars, m_rowPars, m_totalPars;
    vector<vector<t_NnzUnitType> > m_matPars;
};

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxRowsPerPar,
          unsigned int t_MaxColsPerPar,
          unsigned int t_HbmChannels>
class MatPar {
   public:
    typedef NnzUnit<t_DataType, t_IndexType> t_NnzUnitType;
    typedef struct ColParDesp<t_HbmChannels> t_ColParDespType;
    typedef RoCooPar<t_DataType, t_IndexType, t_MaxRowsPerPar, t_MaxColsPerPar, t_HbmChannels> t_RoCooParType;
    typedef KrnColParDesp<t_HbmChannels> t_KrnColParDespType;
    typedef KrnRowParDesp<t_HbmChannels> t_KrnRowParDespType;

   public:
    MatPar() {
        m_validPars = 0;
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            m_validRowPars[i] = 0;
            m_nnzBlocks[i] = 0;
            m_rowResBlocks[i] = 0;
        }
    }
    inline unsigned int& validPars() { return m_validPars; }
    inline unsigned int* validRowPars() { return m_validRowPars; }
    inline unsigned int* nnzBlocks() { return m_nnzBlocks; }
    inline unsigned int* rowResBlocks() { return m_rowResBlocks; }

    void init(unsigned int p_rows, unsigned int p_cols) {
        m_roCooPar.init(p_rows, p_cols);
        unsigned int l_rowPars = alignedBlock(p_rows, t_MaxRowsPerPar);
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            m_krnRowParDesps[i].resize(l_rowPars);
        }
        unsigned int l_totalPars = m_roCooPar.totalPars();
        m_krnColParDesp.resize(l_totalPars);
    }
    t_RoCooParType& roCooPar() { return m_roCooPar; }
    vector<t_NnzUnitType>& roCooPar(unsigned int p_id) { return (m_roCooPar[p_id]); }
    t_KrnColParDespType& krnColParDesp() { return m_krnColParDesp; }
    t_ColParDespType& krnColParDesp(unsigned int p_id) { return m_krnColParDesp[p_id]; }
    vector<t_KrnRowParDespType>& krnRowParDesps(unsigned int p_ch) {
        assert(p_ch < t_HbmChannels);
        return m_krnRowParDesps[p_ch];
    }
    t_KrnRowParDespType& krnRowParDesps(unsigned int p_ch, unsigned int p_rowParId) {
        assert(p_ch < t_HbmChannels);
        assert(p_rowParId < m_krnRowParDesps[p_ch].size());
        return (m_krnRowParDesps[p_ch][p_rowParId]);
    }
    bool check(unsigned int p_rows, unsigned int p_cols, vector<t_NnzUnitType>& p_nnzUnits) {
        bool l_res = true;
        l_res = l_res && m_roCooPar.check(p_rows, p_cols, p_nnzUnits);
        if (!l_res) {
            return false;
        }
        unsigned int l_rowPars = m_roCooPar.rowPars();
        unsigned int l_colPars = m_roCooPar.colPars();
        for (unsigned int i = 0; i < l_rowPars; ++i) {
            for (unsigned int j = 0; j < l_colPars; ++j) {
                unsigned int l_parId = i * l_colPars + j;
                unsigned int l_parNnzs = m_roCooPar[l_parId].size();
                unsigned int l_preMaxRowId = 0;
                for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
                    if ((m_krnRowParDesps[ch][i][j].m_minRowId < l_preMaxRowId) &&
                        (m_krnRowParDesps[ch][i][j].m_nnzs != 0)) {
                        cout << "ERROR: minRowId " << m_krnRowParDesps[ch][i][j].m_minRowId;
                        cout << " at channel " << ch << " < ";
                        cout << " maxRowId " << l_preMaxRowId << " in the previous channel " << endl;
                        l_res = false;
                    }
                    l_preMaxRowId = m_krnRowParDesps[ch][i][j].m_maxRowId;
                    l_parNnzs -= m_krnRowParDesps[ch][i][j].m_nnzs;
                }
                if (l_parNnzs != 0) {
                    cout << "Number of Nnzs in Partition [" << i << "][" << j
                         << "] is not equal to the sum of Nnzs in the channels." << endl;
                    l_res = false;
                }
            }
        }
        return l_res;
    }
    void writeBinFile(ofstream& p_of) {
        p_of.write(reinterpret_cast<char*>(&m_validPars), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(m_validRowPars), t_HbmChannels * sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(m_nnzBlocks), t_HbmChannels * sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(m_rowResBlocks), t_HbmChannels * sizeof(uint32_t));
        m_roCooPar.writeBinFile(p_of);
        m_krnColParDesp.writeBinFile(p_of);
        unsigned int l_rowPars = m_roCooPar.rowPars();
        p_of.write(reinterpret_cast<char*>(&l_rowPars), sizeof(uint32_t));
        for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
            for (unsigned int i = 0; i < l_rowPars; ++i) {
                m_krnRowParDesps[ch][i].writeBinFile(p_of);
            }
        }
    }
    void readBinFile(ifstream& p_if) {
        p_if.read(reinterpret_cast<char*>(&m_validPars), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(m_validRowPars), t_HbmChannels * sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(m_nnzBlocks), t_HbmChannels * sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(m_rowResBlocks), t_HbmChannels * sizeof(uint32_t));
        m_roCooPar.readBinFile(p_if);
        m_krnColParDesp.readBinFile(p_if);
        unsigned int l_rowPars;
        p_if.read(reinterpret_cast<char*>(&l_rowPars), sizeof(uint32_t));
        for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
            m_krnRowParDesps[ch].resize(l_rowPars);
            for (unsigned int i = 0; i < l_rowPars; ++i) {
                m_krnRowParDesps[ch][i].readBinFile(p_if);
            }
        }
    }
    void print(ostream& p_os) {
        p_os << "MatPar INFO: validPars" << endl;
        p_os << m_validPars << endl;
        p_os << "MatPar INFO: validRowPars, nnzBlocks, rowResBlocks for " << t_HbmChannels << " channels" << endl;
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            p_os << "CH " << i << " : " << m_validRowPars[i] << "  ,  " << m_nnzBlocks[i] << "  ,  "
                 << m_rowResBlocks[i] << endl;
        }
        p_os << m_roCooPar;
        p_os << m_krnColParDesp;
        p_os << "INFO: Row wise Kernel Partitions" << endl;
        for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
            p_os << "Channel " << ch;
            unsigned int l_rowPars = m_krnRowParDesps[ch].size();
            p_os << " has " << l_rowPars << " KrnRowParDesp" << endl;
            for (unsigned int i = 0; i < l_rowPars; ++i) {
                p_os << "Row partition " << i << endl;
                p_os << m_krnRowParDesps[ch][i];
            }
        }
    }
    friend ostream& operator<<(ostream& p_os, MatPar& p_matPar) {
        p_matPar.print(p_os);
        return (p_os);
    }

   private:
    unsigned int m_validPars;
    unsigned int m_validRowPars[t_HbmChannels];
    unsigned int m_nnzBlocks[t_HbmChannels];
    unsigned int m_rowBlocks[t_HbmChannels];
    unsigned int m_rowResBlocks[t_HbmChannels];
    t_RoCooParType m_roCooPar;
    t_KrnColParDespType m_krnColParDesp;
    vector<t_KrnRowParDespType> m_krnRowParDesps[t_HbmChannels];
};

template <unsigned int t_HbmChannels>
struct ParamColPtr {
    unsigned int m_offset;
    unsigned int m_memBlocks;
    unsigned int m_parBlocks[t_HbmChannels];
    unsigned int m_nnzBlocks[t_HbmChannels];
    void print(ostream& p_os) {
        p_os << "ParamColPtr:" << endl;
        p_os << "offset = " << m_offset << " memblocks = " << m_memBlocks << endl;
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            p_os << "parBlocks[" << i << "] = " << m_parBlocks[i];
            p_os << " nnzBlocks[" << i << "] = " << m_nnzBlocks[i] << endl;
        }
    }
    friend ostream& operator<<(ostream& p_os, ParamColPtr& p_param) {
        p_param.print(p_os);
        return p_os;
    }
};

template <unsigned int t_HbmChannels>
struct ParamColVec {
    unsigned int m_offset;
    unsigned int m_memBlocks;
    unsigned int m_minId[t_HbmChannels];
    unsigned int m_maxId[t_HbmChannels];
    void print(ostream& p_os) {
        p_os << "ParamColVec:" << endl;
        p_os << "offset = " << m_offset << " memblocks = " << m_memBlocks << endl;
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            p_os << "minId[" << i << "] = " << m_minId[i];
            p_os << " maxId[" << i << "] = " << m_maxId[i] << endl;
        }
    }
    friend ostream& operator<<(ostream& p_os, ParamColVec& p_param) {
        p_param.print(p_os);
        return p_os;
    }
};

struct ParamRwHbm {
    unsigned int m_rdOffset, m_wrOffset;
    unsigned int m_nnzBlocks, m_rowBlocks;
    ParamRwHbm() {
        m_rdOffset = 0;
        m_wrOffset = 0;
        m_nnzBlocks = 0;
        m_rowBlocks = 0;
    }
    void print(ostream& p_os) {
        p_os << "ParamRwHbm:" << endl;
        p_os << "rdOffset = " << m_rdOffset << "  wrOffset = " << m_wrOffset << endl;
        p_os << "nnzBlocks = " << m_nnzBlocks << "  rowBlocks = " << m_rowBlocks << endl;
    }
    friend ostream& operator<<(ostream& p_os, ParamRwHbm& p_param) {
        p_param.print(p_os);
        return p_os;
    }
};

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_ParEntries,
          unsigned int t_ParGroups,
          unsigned int t_DdrMemBits,
          unsigned int t_HbmMemBits,
          unsigned int t_HbmChannels,
          unsigned int t_ParamOffset = 1024,
          unsigned int t_PageSize = 4096>
class RunConfig {
   public:
    typedef struct ParamColPtr<t_HbmChannels> t_ParamColPtrType;
    typedef struct ParamColVec<t_HbmChannels> t_ParamColVecType;
    typedef struct ParamRwHbm t_ParamRwHbmType;

    typedef NnzUnit<t_DataType, t_IndexType> t_NnzUnitType;
    typedef Program<t_PageSize> t_ProgramType;
    typedef struct ColParDesp<t_HbmChannels> t_ColParDespType;
    typedef KrnColParDesp<t_HbmChannels> t_KrnColParDespType; // used by host code
    typedef KrnRowParDesp<t_HbmChannels> t_KrnRowParDespType; // used by host code

    static const unsigned int t_BytesPerHbmRd = t_HbmMemBits / 8;
    static const unsigned int t_BytesPerDdrRd = t_DdrMemBits / 8;
    static const unsigned int t_DatasPerDdrRd = t_DdrMemBits / (8 * sizeof(t_DataType));
    static const unsigned int t_IntsPerDdrRd = t_DdrMemBits / (8 * sizeof(uint32_t));
    static const unsigned int t_IntsPerColParam = 2 + t_HbmChannels * 2;
    static const unsigned int t_DdrBlocks4ColParam = (t_IntsPerColParam + t_IntsPerDdrRd - 1) / t_IntsPerDdrRd;

   public:
    inline unsigned int& rows() { return m_rows; }
    inline unsigned int& cols() { return m_cols; }
    inline unsigned int& nnzs() { return m_nnzs; }
    inline unsigned int& rowPars() { return m_rowPars; }
    inline unsigned int& colPars() { return m_colPars; }
    inline unsigned int& totalPars() { return m_totalPars; }
    inline unsigned int& validPars() { return m_validPars; }
    inline unsigned int& nnzColMemBlocks() { return m_nnzColMemBlocks; }
    inline unsigned int* validRowPars() { return m_validRowPars; }
    inline unsigned int* nnzBlocks() { return m_nnzBlocks; }
    inline unsigned int* rowResBlocks() { return m_rowResBlocks; }
    inline unsigned int& colPtrParamBlocks() { return m_colPtrParamBlocks; }
    inline unsigned int& colVecParamBlocks() { return m_colVecParamBlocks; }
    inline unsigned int* paramBlocks() { return m_paramBlocks; }
    inline vector<t_ParamColPtrType>& paramColPtr() { return m_paramColPtr; }
    inline vector<t_ParamColVecType>& paramColVec() { return m_paramColVec; }
    inline vector<t_ParamRwHbmType>& paramRwHbm(unsigned int p_id) { return m_paramRwHbm[p_id]; }
    inline void* getColPtrAddr() { return m_colPtrAddr; }
    inline void setColPtrAddr(void* p_colPtrAddr) { m_colPtrAddr = p_colPtrAddr; }
    inline void* getColVecAddr() { return m_colVecAddr; }
    inline void setColVecAddr(void* p_colVecAddr) { m_colVecAddr = p_colVecAddr; }
    inline void* getRdHbmAddr(unsigned int p_id) {
        assert(p_id < t_HbmChannels);
        return m_rdHbmAddr[p_id];
    }
    inline void setRdHbmAddr(void* p_rdHbmAddr, unsigned int p_id) {
        assert(p_id < t_HbmChannels);
        m_rdHbmAddr[p_id] = p_rdHbmAddr;
    }
    inline void* getWrHbmAddr(unsigned int p_id) {
        assert(p_id < t_HbmChannels);
        return m_wrHbmAddr[p_id];
    }
    inline void* getRefHbmAddr(unsigned int p_id) {
        assert(p_id < t_HbmChannels);
        return m_refHbmAddr[p_id];
    }
    inline void setWrHbmAddr(void* p_wrHbmAddr, unsigned int p_id) {
        assert(p_id < t_HbmChannels);
        m_wrHbmAddr[p_id] = p_wrHbmAddr;
    }
    inline void setRefHbmAddr(void* p_refHbmAddr, unsigned int p_id) {
        assert(p_id < t_HbmChannels);
        m_refHbmAddr[p_id] = p_refHbmAddr;
    }

    inline t_KrnColParDespType& krnColParDesp() { return m_krnColParDesp; }
    inline t_ColParDespType& krnColParDesp(unsigned int p_id) { return m_krnColParDesp[p_id]; }
    inline vector<t_KrnRowParDespType>& krnRowParDesps(unsigned int p_ch) {
        assert(p_ch < t_HbmChannels);
        return m_krnRowParDesps[p_ch];
    }
    inline t_KrnRowParDespType& krnRowParDesps(unsigned int p_ch, unsigned int p_rowParId) {
        assert(p_ch < t_HbmChannels);
        assert(p_rowParId < m_krnRowParDesps[p_ch].size());
        return (m_krnRowParDesps[p_ch][p_rowParId]);
    }
    bool genNnzRowIdx(void* p_datAddr,
                      unsigned int& p_byteOffset,
                      unsigned int p_nnzBlocks,
                      unsigned int p_nnzs,
                      vector<t_DataType>& p_val,
                      vector<t_IndexType>& p_rowIdx) {
        const unsigned int t_HbmBytes = t_HbmMemBits / 8;
        unsigned int l_alignedNnzs = p_nnzBlocks * t_ParEntries;
        p_val.resize(l_alignedNnzs);
        p_rowIdx.resize(l_alignedNnzs);

        for (unsigned int i = 0; i < p_nnzBlocks; ++i) {
            unsigned int l_sz = t_ParEntries * sizeof(t_DataType);
            char* l_dstVal = reinterpret_cast<char*>(p_val.data()) + i * t_ParEntries * sizeof(t_DataType);
            char* l_dstRowIdx = reinterpret_cast<char*>(p_rowIdx.data()) + i * t_ParEntries * sizeof(t_DataType);
            char* l_srcVal = reinterpret_cast<char*>(p_datAddr) + p_byteOffset + i * t_HbmBytes + t_HbmBytes / 2;
            char* l_srcRowIdx = reinterpret_cast<char*>(p_datAddr) + p_byteOffset + i * t_HbmBytes;

            memcpy(l_dstRowIdx, l_srcRowIdx, l_sz);
            memcpy(l_dstVal, l_srcVal, l_sz);
        }
        for (unsigned int i = p_nnzs; i < l_alignedNnzs; ++i) {
            if (p_val[i] != 0) {
                cout << "ERROR: padded val is not zero" << endl;
                return false;
            }
        }
        p_val.resize(p_nnzs);
        p_rowIdx.resize(p_nnzs);
        p_byteOffset += p_nnzBlocks * t_ParEntries * 2 * sizeof(t_DataType);
        return true;
    }
    bool genColPtr(void* p_datAddr,
                   unsigned int& p_byteOffset,
                   unsigned int p_parBlocks,
                   unsigned int p_cols,
                   unsigned int p_nnzs,
                   unsigned int p_nnzBlocks,
                   vector<t_IndexType>& p_colPtr) {
        unsigned int l_alignedCols = p_parBlocks * t_ParEntries;
        unsigned int l_alignedNnzs = p_nnzBlocks * t_ParEntries;
        unsigned int l_sz = l_alignedCols * sizeof(t_IndexType);
        p_colPtr.resize(l_alignedCols);
        if (l_alignedCols == 0) {
            return true;
        }
        char* l_srcAddr = reinterpret_cast<char*>(p_datAddr) + p_byteOffset;
        memcpy(reinterpret_cast<char*>(p_colPtr.data()), l_srcAddr, l_sz);
        if ((p_colPtr[p_cols - 1] != p_nnzs) && (p_cols != l_alignedCols)) {
            cout << "ERROR: colPtr value at " << p_cols - 1 << " is not equal to nnzs" << endl;
            return false;
        }
        for (unsigned int i = p_cols; i < l_alignedCols; ++i) {
            if (p_colPtr[i] != l_alignedNnzs) {
                cout << "ERROR: colPtr value at " << i << " is not equal to alignedNnzs" << endl;
                return false;
            }
        }
        p_colPtr.resize(p_cols);
        if (p_cols == l_alignedCols) {
            p_colPtr[p_cols - 1] = p_nnzs;
        }
        p_byteOffset += p_parBlocks * t_ParEntries * sizeof(t_DataType);
        return true;
    }
    bool checkRowPar(unsigned int& p_rowParId,
                     unsigned int& p_validRowParId,
                     unsigned int& p_validParId,
                     vector<vector<t_NnzUnitType> >& p_pars) {
        const unsigned int t_HbmBytes = t_HbmMemBits / 8;
        const unsigned int t_DdrBytes = t_DdrMemBits / 8;

        bool l_res = true;

        unsigned int l_minRowIdx[t_HbmChannels];
        unsigned int l_nnzIdxOff[t_HbmChannels];
        NnzWorker<t_DataType, t_IndexType> l_nnzWorker;

        for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
            if (m_krnRowParDesps[ch][p_rowParId].nnzs() == 0) {
                // p_rowParId++;
                // return true;
                continue;
            }
            l_nnzIdxOff[ch] = m_paramRwHbm[ch][p_validRowParId].m_rdOffset * t_HbmBytes;
            l_minRowIdx[ch] = m_krnRowParDesps[ch][p_rowParId].getMinRowId();
        }
        for (unsigned int i = 0; i < m_colPars; ++i) {
            vector<t_NnzUnitType> l_nnzUnits(0);
            unsigned int l_colPtrOff;
            unsigned int l_minColIdx;

            unsigned int l_parId = p_rowParId * m_colPars + i;
            if (m_krnColParDesp[l_parId].m_colVecMemBlocks == 0) {
                continue;
            }
            l_colPtrOff = m_paramColPtr[p_validParId].m_offset * t_DdrBytes;
            // unsigned int l_colVecOff = m_paramColVec[p_validParId].m_offset;
            // unsigned int l_parMinColDdrBlock = l_colVecOff - t_ParamOffset/t_BytesPerDdrRd -
            // m_validPars*t_DdrBlocks4ColParam;
            unsigned int l_parMinColDdrBlock = m_krnColParDesp[l_parId].m_parMinColId / t_DatasPerDdrRd;
            unsigned int l_parMinColId = l_parMinColDdrBlock * t_DatasPerDdrRd;
            for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
                if (m_krnColParDesp[l_parId].m_nnzs[ch] == 0) {
                    continue;
                }
                vector<t_DataType> l_val(0);
                vector<t_IndexType> l_rowIdx(0);
                vector<t_IndexType> l_colPtr(0);
                unsigned int l_nnzBlocks = m_paramColPtr[p_validParId].m_nnzBlocks[ch];
                unsigned int l_parBlocks = m_paramColPtr[p_validParId].m_parBlocks[ch];
                unsigned int l_cols = m_krnColParDesp[l_parId].m_cols[ch];
                unsigned int l_nnzs = m_krnColParDesp[l_parId].m_nnzs[ch];

                l_res = l_res && genNnzRowIdx(m_rdHbmAddr[ch], l_nnzIdxOff[ch], l_nnzBlocks, l_nnzs, l_val, l_rowIdx);
                if (!l_res) break;
                l_res =
                    l_res && genColPtr(m_colPtrAddr, l_colPtrOff, l_parBlocks, l_cols, l_nnzs, l_nnzBlocks, l_colPtr);
                if (!l_res) break;
                l_minColIdx = m_paramColVec[p_validParId].m_minId[ch] + l_parMinColId;
                l_nnzWorker.genNnzUnits(l_val, l_rowIdx, l_colPtr, l_minRowIdx[ch], l_minColIdx, l_nnzUnits);
            }
            if (!l_res) {
                break;
            }
            // sort(l_nnzUnits.begin(), l_nnzUnits.end(), compareRow<t_DataType, t_IndexType>());
            sort(l_nnzUnits.begin(), l_nnzUnits.end());
            l_res = l_res && l_nnzWorker.compareNnzs(l_nnzUnits, p_pars[l_parId]);
            p_validParId++;
        }
        p_rowParId++;
        p_validRowParId++;
        return l_res;
    }

    bool checkPars(vector<vector<t_NnzUnitType> >& p_pars) {
        bool l_res = true;
        unsigned int l_rowParId = 0;
        unsigned int l_validRowParId = 0;
        unsigned int l_validParId = 0;
        do {
            l_res = l_res && checkRowPar(l_rowParId, l_validRowParId, l_validParId, p_pars);
        } while ((l_validParId < m_validPars) && l_res);
        return l_res;
    }
    bool checkInVec(vector<t_DataType>& p_inVec) {
        vector<t_DataType> l_vec(m_cols);
        unsigned int l_offset = t_ParamOffset + m_validPars * t_DdrBlocks4ColParam * t_BytesPerDdrRd;
        char* l_datAddr = reinterpret_cast<char*>(m_colVecAddr) + l_offset;
        unsigned long long l_sz = m_cols * sizeof(t_DataType);
        memcpy(reinterpret_cast<char*>(l_vec.data()), l_datAddr, l_sz);
        NnzWorker<t_DataType, t_IndexType> l_nnzWorker;
        bool l_exactMatch = true;
        (void)compareVec<t_DataType>(l_vec, p_inVec, l_exactMatch);
        return l_exactMatch;
    }
    bool checkOutVec(vector<t_DataType>& p_outVec) {
        vector<t_DataType> l_outVec(m_rows, 0);
        // form l_outVec from m_refHbmAddr;

        unsigned int l_validRowParId = 0;
        for (unsigned int i = 0; i < m_rowPars; ++i) {
            vector<t_DataType> l_rowRes[t_HbmChannels];
            unsigned int l_rows = 0;
            for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
                t_KrnRowParDespType l_krnRowDesp = m_krnRowParDesps[ch][i];
                l_rows = l_krnRowDesp.rows();
                unsigned int l_rowResBlocks = l_krnRowDesp.rowResBlocks();
                unsigned int l_minRowId = l_krnRowDesp.minRowId();
                if (l_rows != 0) {
                    unsigned int l_wrOffset = m_paramRwHbm[ch][l_validRowParId].m_wrOffset;
                    unsigned int l_wrByteOffset = l_wrOffset * t_BytesPerHbmRd;
                    char* l_refAddr = reinterpret_cast<char*>(m_refHbmAddr[ch]) + l_wrByteOffset;
                    l_rowRes[ch].resize(l_rowResBlocks * t_ParEntries);
                    unsigned long long l_sz = l_rowResBlocks * t_ParEntries * sizeof(t_DataType);
                    memcpy(reinterpret_cast<char*>(l_rowRes[ch].data()), l_refAddr, l_sz);
                }
                for (unsigned int j = 0; j < l_rows; ++j) {
                    l_outVec[l_minRowId + j] += l_rowRes[ch][j];
                }
            }
            if (l_rows != 0) {
                l_validRowParId++;
            }
        }
        bool l_exactMatch = true;
        bool l_res = compareVec<t_DataType>(l_outVec, p_outVec, l_exactMatch);
        l_res = l_res || l_exactMatch;
        return l_res;
    }

    bool check(vector<vector<t_NnzUnitType> >& p_pars, vector<t_DataType>& p_inVec, vector<t_DataType>& p_outVec) {
        bool l_res = true;
        l_res = l_res && checkPars(p_pars);
        if (l_res) {
            l_res = l_res && checkInVec(p_inVec);
        }
        if (l_res) {
            l_res = l_res && checkOutVec(p_outVec);
        }
        return l_res;
    }

    unsigned int checkRowRes() {
        unsigned int l_totalErrs = 0;
        unsigned int l_validRowParId = 0;
        for (unsigned int i = 0; i < m_rowPars; ++i) {
            vector<t_DataType> l_rowRes[t_HbmChannels];
            vector<t_DataType> l_rowRef[t_HbmChannels];
            unsigned int l_rows = 0;
            for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
                t_KrnRowParDespType l_krnRowDesp = m_krnRowParDesps[ch][i];
                l_rows = l_krnRowDesp.rows();
                unsigned int l_rowResBlocks = l_krnRowDesp.rowResBlocks();
                if (l_rows != 0) {
                    unsigned int l_wrOffset = m_paramRwHbm[ch][l_validRowParId].m_wrOffset;
                    unsigned int l_wrByteOffset = l_wrOffset * t_BytesPerHbmRd;
                    char* l_resAddr = reinterpret_cast<char*>(m_wrHbmAddr[ch]) + l_wrByteOffset;
                    char* l_refAddr = reinterpret_cast<char*>(m_refHbmAddr[ch]) + l_wrByteOffset;
                    l_rowRes[ch].resize(l_rowResBlocks * t_ParEntries);
                    l_rowRef[ch].resize(l_rowResBlocks * t_ParEntries);
                    unsigned long long l_sz = l_rowResBlocks * t_ParEntries * sizeof(t_DataType);
                    memcpy(reinterpret_cast<char*>(l_rowRes[ch].data()), l_resAddr, l_sz);
                    memcpy(reinterpret_cast<char*>(l_rowRef[ch].data()), l_refAddr, l_sz);
                    unsigned int l_errs = compareVec<t_DataType>(l_rowRef[ch], l_rowRef[ch]);
                    if (l_errs != 0) {
                        cout << "Channel " << ch << " has " << l_errs << " mismatches in row block " << i << endl;
                    }
                    l_totalErrs += l_errs;
                }
            }
            if (l_rows != 0) {
                l_validRowParId++;
            }
        }
        return l_totalErrs;
    }
    void writeBinFile(ofstream& p_of, t_ProgramType& p_prog) {
        p_of.write(reinterpret_cast<char*>(&m_rows), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_cols), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_nnzs), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_rowPars), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_colPars), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_totalPars), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_validPars), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_nnzColMemBlocks), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(m_validRowPars), t_HbmChannels * sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(m_nnzBlocks), t_HbmChannels * sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(m_rowResBlocks), t_HbmChannels * sizeof(uint32_t));
        m_krnColParDesp.writeBinFile(p_of);
        for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
            for (unsigned int i = 0; i < m_rowPars; ++i) {
                m_krnRowParDesps[ch][i].writeBinFile(p_of);
            }
        }
        p_of.write(reinterpret_cast<char*>(&m_colPtrParamBlocks), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(&m_colVecParamBlocks), sizeof(uint32_t));
        unsigned int l_paramColPtrSize = m_paramColPtr.size();
        p_of.write(reinterpret_cast<char*>(&l_paramColPtrSize), sizeof(uint32_t));
        unsigned int l_paramColPtrSz = l_paramColPtrSize * sizeof(t_ParamColPtrType);
        p_of.write(reinterpret_cast<char*>(m_paramColPtr.data()), l_paramColPtrSz);
        unsigned int l_paramColVecBlocks = m_paramColVec.size();
        p_of.write(reinterpret_cast<char*>(&l_paramColVecBlocks), sizeof(uint32_t));
        unsigned int l_paramColVecSz = l_paramColVecBlocks * sizeof(t_ParamColVecType);
        p_of.write(reinterpret_cast<char*>(m_paramColVec.data()), l_paramColVecSz);
        p_of.write(reinterpret_cast<char*>(m_paramBlocks), t_HbmChannels * sizeof(uint32_t));
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            unsigned int l_sz = m_paramBlocks[i] * sizeof(t_ParamRwHbmType);
            if (l_sz != 0) p_of.write(reinterpret_cast<char*>(m_paramRwHbm[i].data()), l_sz);
        }
        unsigned int l_colPtrSz = p_prog.getBufSz(m_colPtrAddr);
        p_of.write(reinterpret_cast<char*>(&l_colPtrSz), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(m_colPtrAddr), l_colPtrSz);
        unsigned int l_colVecSz = p_prog.getBufSz(m_colVecAddr);
        p_of.write(reinterpret_cast<char*>(&l_colVecSz), sizeof(uint32_t));
        p_of.write(reinterpret_cast<char*>(m_colVecAddr), l_colVecSz);
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            unsigned int l_sz = p_prog.getBufSz(m_rdHbmAddr[i]);
            p_of.write(reinterpret_cast<char*>(&l_sz), sizeof(uint32_t));
            if (l_sz != 0) p_of.write(reinterpret_cast<char*>(m_rdHbmAddr[i]), l_sz);
        }
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            unsigned int l_sz = p_prog.getBufSz(m_wrHbmAddr[i]);
            p_of.write(reinterpret_cast<char*>(&l_sz), sizeof(uint32_t));
            if (l_sz != 0) p_of.write(reinterpret_cast<char*>(m_wrHbmAddr[i]), l_sz);
        }
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            unsigned int l_sz = p_prog.getBufSz(m_refHbmAddr[i]);
            p_of.write(reinterpret_cast<char*>(&l_sz), sizeof(uint32_t));
            if (l_sz != 0) p_of.write(reinterpret_cast<char*>(m_refHbmAddr[i]), l_sz);
        }
    }
    void readBinFile(ifstream& p_if, t_ProgramType& p_program) {
        p_if.read(reinterpret_cast<char*>(&m_rows), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_cols), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_nnzs), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_rowPars), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_colPars), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_totalPars), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_validPars), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_nnzColMemBlocks), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(m_validRowPars), t_HbmChannels * sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(m_nnzBlocks), t_HbmChannels * sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(m_rowResBlocks), t_HbmChannels * sizeof(uint32_t));
        m_krnColParDesp.readBinFile(p_if);
        for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
            m_krnRowParDesps[ch].resize(m_rowPars);
            for (unsigned int i = 0; i < m_rowPars; ++i) {
                m_krnRowParDesps[ch][i].readBinFile(p_if);
            }
        }
        p_if.read(reinterpret_cast<char*>(&m_colPtrParamBlocks), sizeof(uint32_t));
        p_if.read(reinterpret_cast<char*>(&m_colVecParamBlocks), sizeof(uint32_t));
        unsigned int l_paramColPtrSize;
        p_if.read(reinterpret_cast<char*>(&l_paramColPtrSize), sizeof(uint32_t));
        unsigned int l_paramColPtrSz = l_paramColPtrSize * sizeof(t_ParamColPtrType);
        m_paramColPtr.resize(l_paramColPtrSize);
        p_if.read(reinterpret_cast<char*>(m_paramColPtr.data()), l_paramColPtrSz);
        unsigned int l_paramColVecBlocks;
        p_if.read(reinterpret_cast<char*>(&l_paramColVecBlocks), sizeof(uint32_t));
        unsigned int l_paramColVecSz = l_paramColVecBlocks * sizeof(t_ParamColVecType);
        m_paramColVec.resize(l_paramColVecBlocks);
        p_if.read(reinterpret_cast<char*>(m_paramColVec.data()), l_paramColVecSz);
        p_if.read(reinterpret_cast<char*>(m_paramBlocks), t_HbmChannels * sizeof(uint32_t));
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            m_paramRwHbm[i].resize(m_paramBlocks[i]);
            unsigned int l_sz = m_paramBlocks[i] * sizeof(t_ParamRwHbmType);
            if (l_sz != 0) p_if.read(reinterpret_cast<char*>(m_paramRwHbm[i].data()), l_sz);
        }
        unsigned int l_colPtrSz;
        p_if.read(reinterpret_cast<char*>(&l_colPtrSz), sizeof(uint32_t));
        m_colPtrAddr = p_program.allocMem(l_colPtrSz);
        p_if.read(reinterpret_cast<char*>(m_colPtrAddr), l_colPtrSz);
        unsigned int l_colVecSz;
        p_if.read(reinterpret_cast<char*>(&l_colVecSz), sizeof(uint32_t));
        m_colVecAddr = p_program.allocMem(l_colVecSz);
        p_if.read(reinterpret_cast<char*>(m_colVecAddr), l_colVecSz);
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            unsigned int l_sz;
            p_if.read(reinterpret_cast<char*>(&l_sz), sizeof(uint32_t));
            m_rdHbmAddr[i] = p_program.allocMem(l_sz);
            if (l_sz != 0) p_if.read(reinterpret_cast<char*>(m_rdHbmAddr[i]), l_sz);
        }
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            unsigned int l_sz;
            p_if.read(reinterpret_cast<char*>(&l_sz), sizeof(uint32_t));
            m_wrHbmAddr[i] = p_program.allocMem(l_sz);
            if (l_sz != 0) p_if.read(reinterpret_cast<char*>(m_wrHbmAddr[i]), l_sz);
        }
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            unsigned int l_sz;
            p_if.read(reinterpret_cast<char*>(&l_sz), sizeof(uint32_t));
            m_refHbmAddr[i] = p_program.allocMem(l_sz);
            if (l_sz != 0) p_if.read(reinterpret_cast<char*>(m_refHbmAddr[i]), l_sz);
        }
    }
    void print(ostream& p_os) {
        p_os << "INFO: RunConfig" << endl;
        p_os << "rows, cols, nnzs, rowPars, colPars, totalPars" << endl;
        p_os << m_rows << " , " << m_cols << " , " << m_nnzs << " , ";
        p_os << m_rowPars << " , " << m_colPars << " , " << m_totalPars << endl;

        p_os << "validPars = " << m_validPars << " nnzColMemBlocks = " << m_nnzColMemBlocks << endl;
        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            p_os << "    validRowPars[" << i << "] = " << m_validRowPars[i] << endl;
            p_os << "    nnzBlocks[" << i << "] = " << m_nnzBlocks[i] << endl;
            p_os << "    rowResBlocks[" << i << "] = " << m_rowResBlocks[i] << endl;
        }

        // definitions for CU run
        p_os << "m_colPtrParamBlocks, m_colVecParamBlocks" << endl;
        p_os << m_colPtrParamBlocks << " , " << m_colVecParamBlocks << endl;
        p_os << "Number of ParamColPtr is: " << m_paramColPtr.size() << endl;
        for (unsigned int i = 0; i < m_paramColPtr.size(); ++i) {
            cout << "ParamColPtr " << i << endl;
            p_os << m_paramColPtr[i] << endl;
        }
        p_os << "Number of ParamColVec is: " << m_paramColVec.size() << endl;
        for (unsigned int i = 0; i < m_paramColVec.size(); ++i) {
            cout << "ParamColVec " << i << endl;
            p_os << m_paramColVec[i];
        }
        p_os << endl;

        for (unsigned int i = 0; i < t_HbmChannels; ++i) {
            p_os << "paramBlocks[" << i << "] = " << m_paramBlocks[i] << endl;
            p_os << "Number of paramRwHbm[" << i << "] = " << m_paramRwHbm[i].size() << endl;
            for (unsigned int j = 0; j < m_paramRwHbm[i].size(); ++j) {
                p_os << m_paramRwHbm[i][j];
            }
            p_os << endl;
        }
    }
    friend ostream& operator<<(ostream& p_os, RunConfig& p_conf) {
        p_conf.print(p_os);
        return (p_os);
    }

   private:
    unsigned int m_rows, m_cols, m_nnzs, m_rowPars, m_colPars, m_totalPars;
    unsigned int m_validPars;
    unsigned int m_nnzColMemBlocks;
    unsigned int m_validRowPars[t_HbmChannels];
    unsigned int m_nnzBlocks[t_HbmChannels];
    unsigned int m_rowResBlocks[t_HbmChannels];

    t_KrnColParDespType m_krnColParDesp;
    vector<t_KrnRowParDespType> m_krnRowParDesps[t_HbmChannels];

    // definitions for CU run
    unsigned int m_colPtrParamBlocks, m_colVecParamBlocks;
    vector<t_ParamColPtrType> m_paramColPtr;
    vector<t_ParamColVecType> m_paramColVec;
    unsigned int m_paramBlocks[t_HbmChannels];
    vector<t_ParamRwHbmType> m_paramRwHbm[t_HbmChannels];
    void* m_colPtrAddr;
    void* m_colVecAddr;
    void* m_rdHbmAddr[t_HbmChannels];
    void* m_wrHbmAddr[t_HbmChannels];
    void* m_refHbmAddr[t_HbmChannels];
};
} // end namespace sparse
} // end namespace xf
#endif
