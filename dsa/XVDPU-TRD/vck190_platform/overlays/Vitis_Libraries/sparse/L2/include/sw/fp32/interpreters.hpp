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
 * @file interpreters.hpp
 * @brief header file for data types used in L2/L3 host code.
 *
 * This file is part of Vitis SPARSE Library.
 */
#ifndef XF_SPARSE_INTERPRETERS_HPP
#define XF_SPARSE_INTERPRETERS_HPP

#include <cassert>
#include <fstream>
#include <cstring>
#include <vector>

using namespace std;

namespace xf {
namespace sparse {

template <typename t_DataType, typename t_IndexType, unsigned int t_ParEntries, unsigned int t_ParGroups>
class CscRowInt {
   public:
    CscRowInt() : m_nnzBlocks(0), m_rowBlocks(0), m_rowResBlocks(0) {}
    bool interToCscRowFile(string p_fileName) {
        m_rowResBlocks = 0;
        ifstream l_if(p_fileName.c_str(), ios::binary);
        if (!l_if.is_open()) {
            cout << "ERROR: failed to open file " << p_fileName << endl;
            return false;
        }
        l_if.read(reinterpret_cast<char*>(&m_nnzBlocks), sizeof(uint32_t));
        l_if.read(reinterpret_cast<char*>(&m_rowBlocks), sizeof(uint32_t));
        unsigned int l_nnzs = m_nnzBlocks * t_ParEntries;
        m_nnzVal.resize(l_nnzs);
        m_nnzRowIdx.resize(l_nnzs);
        m_colVal.resize(l_nnzs);
        for (unsigned int i = 0; i < m_nnzBlocks; ++i) {
            l_if.read(reinterpret_cast<char*>(m_nnzRowIdx.data() + i * t_ParEntries),
                      t_ParEntries * sizeof(t_IndexType));
            l_if.read(reinterpret_cast<char*>(m_nnzVal.data() + i * t_ParEntries), t_ParEntries * sizeof(t_DataType));
        }
        l_if.read(reinterpret_cast<char*>(m_colVal.data()), l_nnzs * sizeof(t_DataType));
        l_if.close();
        return true;
    }
    bool interFromCscRowFile(string p_fileName) {
        m_nnzBlocks = 0;
        m_rowBlocks = 0;
        ifstream l_if(p_fileName.c_str(), ios::binary);
        if (!l_if.is_open()) {
            cout << "ERROR: failed to open file " << p_fileName << endl;
            return false;
        }
        unsigned int l_nnzBlocks, l_rowBlocks;
        l_if.read(reinterpret_cast<char*>(&l_nnzBlocks), sizeof(uint32_t));
        l_if.read(reinterpret_cast<char*>(&l_rowBlocks), sizeof(uint32_t));
        m_rowResBlocks = l_rowBlocks * t_ParGroups;
        unsigned int l_rows = m_rowResBlocks * t_ParEntries;
        m_cscRowRes.resize(l_rows);
        l_if.read(reinterpret_cast<char*>(m_cscRowRes.data()), l_rows * sizeof(t_DataType));
        l_if.close();
        return true;
    }
    void print(ostream& os) {
        if (m_nnzBlocks != 0) {
            unsigned int l_nnzs = m_nnzBlocks * t_ParEntries;
            cout << "INFO: data transmitted to cscRowPktKernel" << endl;
            cout << "nnzs = " << l_nnzs << endl;
            cout << "rowIdx     nnz_val     col_val" << endl;
            for (unsigned int i = 0; i < l_nnzs; ++i) {
                os << m_nnzRowIdx[i] << "  " << m_nnzVal[i] << "  " << m_colVal[i] << endl;
            }
        } else if (m_rowResBlocks != 0) {
            unsigned int l_rows = m_rowResBlocks * t_ParEntries;
            cout << "INFO: data received from cscRowPktKernel" << endl;
            cout << "rows = " << l_rows << endl;
            for (unsigned int i = 0; i < l_rows; ++i) {
            }
        }
    }
    friend ostream& operator<<(ostream& os, CscRowInt& p_val) {
        p_val.print(os);
        return (os);
    }

   private:
    unsigned int m_nnzBlocks, m_rowBlocks, m_rowResBlocks;
    vector<t_DataType> m_nnzVal;
    vector<t_IndexType> m_nnzRowIdx;
    vector<t_DataType> m_colVal;
    vector<t_DataType> m_cscRowRes;
};

} // end namespace sparse
} // end namespace xf
#endif
